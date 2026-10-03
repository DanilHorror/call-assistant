"""
Финальный тест "как при реальном звонке" — тот же набор вопросов, что и в
batch_test_final.py (515 kb + 515 context + ~499 search + 500 situational,
один и тот же перемешанный порядок при том же --seed), но прогоняется НЕ
изолированным консольным циклом в CSV, а через РЕАЛЬНЫЙ вживую видимый
конвейер: этот файл — почти дословная копия wiring'а live-подсказок из
main.py (hint_seq_box/hint_session_box/pending_clarification_box/
own_answers, hint_worker, партиал/финал/cancel протокол в hint_queue) плюс
тот же OverlayWindow и тот же TranscriptStore, что и на настоящем звонке.
С Ollama НИЧЕГО не подделывается — каждый вопрос реально уходит в модель
(и в базу знаний/поиск, если они включены в .env, БЕЗ принудительного
переопределения — ровно то, что стоит в настройках сейчас), реальное время
ответа никак не подгоняется и не берётся из старого прогона.

Отличие от batch_test_final.py — принципиальное и намеренное: там каждая
единица (особенно kb/search/situational) считалась ИЗОЛИРОВАННО, чтобы
чисто измерить качество каждого конкретного ответа отдельно. Здесь же
own_answers и pending_clarification НЕ сбрасываются между вопросами — они
копятся точно так же естественно, как копились бы на настоящем звонке с
одним живым собеседником, который просто подряд задаёт вопрос за вопросом
(в реальности main.py тоже понятия не имеет о "типах" и "единицах" —
он просто получает очередную строку транскрипта). Единственное отличие от
настоящего звонка — вместо распознанной живой речи по одной "проговариваются"
вопросы из тестового набора (--interval между ними), а не реальный микрофон
и STT.

Каждый вопрос дожидается своего ПОЛНОГО ответа (final/cancel в hint_queue)
прежде чем "произносится" следующий — то есть вопросы не перекрывают друг
друга (это единственная сознательная пауза во всём файле: между ответом на
один вопрос и постановкой следующего, просто ради удобства просмотра,
--interval, по умолчанию 1.5с).

ВНИМАНИЕ: это МЕДЛЕННЫЙ прогон — по-настоящему дёргает Ollama на каждый из
~2500+ вопросов (515 kb + 515×2 context + ~499 search + 500 situational),
столько же реальных обращений, сколько было в ночном прогоне
batch_test_final.py. Используйте --limit/--start для проверки на части
набора, не обязательно гнать всё целиком.

Запуск (из корня проекта — откроется настоящее окно оверлея):
    venv\\Scripts\\python.exe src\\overlay_final_test.py --limit 20
    venv\\Scripts\\python.exe src\\overlay_final_test.py --limit 20 --start 100
    venv\\Scripts\\python.exe src\\overlay_final_test.py                     (весь набор целиком, часы)
    venv\\Scripts\\python.exe src\\overlay_final_test.py --interval 3
    venv\\Scripts\\python.exe src\\overlay_final_test.py --model llama3.1:8b

Esc в оверлее останавливает тест (текущий в работе запрос при этом
обрывается тем же abort_inflight_hint(), что и на настоящем звонке).
Транскрипт "звонка" (с пометкой _final-overlay-test) сохраняется в
sessions/, как и на настоящем звонке — можно потом перечитать.

ПРИМЕЧАНИЕ ДЛЯ БУДУЩИХ ПРАВОК: hint_worker/abort_inflight_hint/
_make_on_partial ниже — намеренная копия того же кода из main.py (не
переиспользование одной функции), чтобы ни в коем случае не трогать и не
рисковать сломать реальный конвейер настоящего звонка ради тестового
скрипта. Если меняете логику hint_worker в main.py — продублируйте
изменение и сюда, иначе поведение этого теста разъедется с настоящим
звонком.
"""
from __future__ import annotations

import argparse
import datetime as dt
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from config import Settings
from transcript import TranscriptStore
from live_hints import _split_current_line, generate_hint, warmup_model, preload_kb
from overlay import OverlayWindow
from batch_test_final import DEFAULT_SEED, _build_units


def _flatten_questions(units: list[dict[str, Any]]) -> list[str]:
    """
    Разворачивает "единицы" batch_test_final.py в плоский список вопросов,
    которые по одному "произносятся" собеседником — kb/search/situational
    дают один вопрос, context даёт ДВА последовательных (сам вопрос и
    уточнение), оба добавляются как отдельные реплики в один и тот же
    растущий транскрипт (см. докстринг модуля про own_answers/историю).
    """
    questions: list[str] = []
    for unit in units:
        t = unit["type"]
        data = unit["data"]
        if t in ("kb", "search", "situational"):
            questions.append(data.get("question", ""))
        elif t == "context":
            questions.append(data.get("question", ""))
            questions.append(data.get("followup", ""))
        else:
            raise AssertionError(f"неизвестный тип единицы: {t!r}")
    return [q for q in questions if q]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"seed перемешивания вопросов — тот же порядок, что и в batch_test_final.py при том же seed (по умолчанию {DEFAULT_SEED})",
    )
    parser.add_argument("--start", type=int, default=0, help="начать не с 0-го вопроса, а с N-го")
    parser.add_argument("--limit", type=int, default=None, help="прогнать только N вопросов (после разворачивания context-пар в две реплики)")
    parser.add_argument(
        "--interval", type=float, default=1.5,
        help="пауза между ПОЛНЫМ ответом на вопрос и постановкой следующего, сек — только ради удобства просмотра (по умолчанию 1.5)",
    )
    parser.add_argument("--model", type=str, default=None, help="модель Ollama вместо той, что в .env (OLLAMA_MODEL)")
    args = parser.parse_args()

    settings = Settings.load()
    model = args.model or settings.ollama_model
    kb_path_str = str(settings.kb_path) if (settings.kb_enabled and settings.kb_path) else None

    print(f"[final] модель: {model} ({settings.ollama_host})")
    print(f"[final] база знаний: {'включена' if kb_path_str else 'выключена'} — ровно как в .env, без переопределений")
    print("[final] веб-поиск: включён (как в main.py — тумблера на выключение там нет)")
    if not settings.live_hints_enabled:
        print("[final] ПРЕДУПРЕЖДЕНИЕ: LIVE_HINTS_ENABLED=false в .env — на настоящем звонке подсказок бы не было вообще, но здесь это единственный смысл теста, так что подсказки всё равно включены принудительно.")

    print("[final] прогреваю модель Ollama...")
    warmup_model(model=model, host=settings.ollama_host)
    if kb_path_str:
        print(f"[final] база знаний: {kb_path_str}")
        preload_kb(kb_path_str)

    units = _build_units(args.seed)
    questions = _flatten_questions(units)[args.start:]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        print("Нечего прогонять.")
        sys.exit(0)
    print(f"[final] вопросов к прогону: {len(questions)}\n")
    print("Открываю оверлей... (Esc в окне — закончить тест)\n")

    session_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_final-overlay-test"
    session_dir = settings.output_dir / session_id
    transcript_store = TranscriptStore(session_dir)

    # --- Ниже — дословная копия wiring'а live-подсказок из main.py, см.
    # предупреждение в докстринге модуля про то, почему это копия, а не
    # переиспользование. ---
    hint_request_queue: "queue.Queue[tuple[int, str, str | None]]" = queue.Queue(maxsize=1)
    hint_queue: "queue.Queue[tuple]" = queue.Queue()
    stop_event = threading.Event()
    # Сигналит фидеру, что hint_worker полностью закончил текущий запрос
    # (final/cancel уже положены в hint_queue) — этого не было в main.py
    # (там process_loop просто шлёт следующую строку транскрипта, как только
    # она распознана, не дожидаясь ответа на предыдущую), но здесь нужно:
    # мы сами "проговариваем" вопросы и хотим увидеть каждый ответ целиком,
    # а не обрывать один вопрос другим, как могло бы случиться на настоящем
    # быстром звонке.
    request_done = threading.Event()

    pending_clarification_box = {"text": None, "lock": threading.Lock()}
    hint_session_box = {"session": None, "lock": threading.Lock()}
    hint_seq_box = {"latest": 0, "lock": threading.Lock()}
    own_answers: list[str] = []
    OWN_ANSWERS_KEEP = 6

    def abort_inflight_hint() -> None:
        with hint_session_box["lock"]:
            session = hint_session_box["session"]
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def _make_on_partial(seq: int, question: str):
        def on_partial(reply: str, context: str) -> None:
            with hint_seq_box["lock"]:
                is_latest = seq == hint_seq_box["latest"]
            if is_latest:
                hint_queue.put(("partial", seq, question, reply, context))
        return on_partial

    def hint_worker() -> None:
        while not stop_event.is_set():
            try:
                seq, recent_text, pending = hint_request_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            _, question = _split_current_line(recent_text)
            with hint_seq_box["lock"]:
                is_latest = seq == hint_seq_box["latest"]
            if is_latest:
                hint_queue.put(("partial", seq, question, "", ""))
            try:
                t0 = time.time()
                reply, context = generate_hint(
                    recent_text,
                    model=model,
                    host=settings.ollama_host,
                    session_box=hint_session_box,
                    pending_clarification=pending,
                    kb_enabled=bool(kb_path_str),
                    kb_path=kb_path_str,
                    kb_top_k=settings.kb_top_k,
                    kb_min_score=settings.kb_min_score,
                    own_answers=list(own_answers),
                    on_partial=_make_on_partial(seq, question),
                )
                elapsed = time.time() - t0
                with hint_seq_box["lock"]:
                    is_latest = seq == hint_seq_box["latest"]
                if not is_latest:
                    print("[hints] ответ устарел (уже есть более новый вопрос) — пропускаю")
                    hint_queue.put(("cancel", seq))
                elif reply:
                    hint_queue.put(("final", seq, question, reply, context, elapsed))
                    with pending_clarification_box["lock"]:
                        pending_clarification_box["text"] = reply if reply.rstrip().endswith("?") else None
                    own_answers.append(reply)
                    del own_answers[:-OWN_ANSWERS_KEEP]
                else:
                    print("[hints] реплика без вопроса/просьбы — подсказка пропущена")
                    hint_queue.put(("cancel", seq))
            except Exception as e:
                hint_queue.put(("cancel", seq))
                print(f"[hints] не удалось сгенерировать подсказку: {e}")
            finally:
                request_done.set()

    def _submit(question: str) -> int | None:
        line = transcript_store.add("Собеседник", question, "final-overlay-test")
        if line is None:
            return None
        recent_text = transcript_store.recent_text(settings.hints_context_lines)
        with hint_seq_box["lock"]:
            hint_seq_box["latest"] += 1
            seq = hint_seq_box["latest"]
        with pending_clarification_box["lock"]:
            pending = pending_clarification_box["text"]
        try:
            hint_request_queue.put_nowait((seq, recent_text, pending))
        except queue.Full:
            try:
                hint_request_queue.get_nowait()
            except queue.Empty:
                pass
            hint_request_queue.put_nowait((seq, recent_text, pending))
        abort_inflight_hint()
        return seq

    def _feeder() -> None:
        total = len(questions)
        for i, question in enumerate(questions, 1):
            if stop_event.is_set():
                return
            request_done.clear()
            seq = _submit(question)
            if seq is None:
                continue
            print(f"\n>>> [{i}/{total}] {question}")
            while not request_done.wait(timeout=0.5):
                if stop_event.is_set():
                    return
            if stop_event.is_set():
                return
            if stop_event.wait(args.interval):
                return
        print("\nВсе вопросы заданы. Оверлей остаётся открытым — прокрутите историю вверх/вниз,")
        print("Esc в окне закрывает тест.")

    hint_thread = threading.Thread(target=hint_worker, daemon=True)
    hint_thread.start()

    def on_close() -> None:
        stop_event.set()
        print("\nОверлей закрыт (Esc) — тест остановлен.")

    overlay = OverlayWindow(hint_queue, on_close)

    feeder_thread = threading.Thread(target=_feeder, daemon=True)
    feeder_thread.start()

    try:
        overlay.run()
    finally:
        stop_event.set()
        transcript_store.close()
        print(f"[final] транскрипт сохранён: {transcript_store.txt_path}")


if __name__ == "__main__":
    main()
