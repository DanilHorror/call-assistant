"""
ВТОРОЙ, отдельный файл от batch_test.py (специально не трогаем его — он уже
может быть запущен и работать со своим собственным CSV).

batch_test.py проверяет каждый вопрос из базы знаний САМ ПО СЕБЕ, без
истории. Но большинство живых багов в этом чате (главный пример —
"расскажи о каждом из них подробнее" после правильного ответа про
конфиденциальность/целостность/доступность, см. own_answers в
live_hints.py) вылезает именно на ВТОРОМ вопросе — уточнении, которое
ссылается на предыдущий ответ местоимением ("их", "это", "подробнее"),
а не называет тему заново. Этот файл специально гоняет такой двухходовый
сценарий на всех 515 вопросах разом.

В отличие от первой версии этого файла, уточняющие вопросы здесь НЕ
придумывает на лету Ollama — они заранее написаны (см. src/gen_followups.py
и data/knowledge_base_followups.json, тот же список из 515 вопросов, что
и в knowledge_base.json, только с добавленным полем "followup" под каждый
пункт). Так и быстрее (2 похода к Ollama на вопрос вместо 3), и
уточняющие вопросы контролируемого качества, а не то, что придумает
маленькая модель под настроение.

Модель на КАЖДОМ из двух вопросов отвечает СВОИМ обычным путём — тем же,
что и в реальном звонке (KB_ENABLED/поиск берутся из .env, как обычно):
сначала пробуется база знаний (файл-шпаргалка), если там нет подходящей
записи — модель может попросить веб-поиск ("погуглить"), а если и это не
нужно — отвечает по своим знаниям. Здесь никто не подсовывает модели
готовый ответ насильно — просто ПОСЛЕ того как она уже ответила, отдельно
(не трогая generate_hint()/live_hints.py) определяем, каким из трёх путей
она в итоге воспользовалась, и пишем это в CSV — колонки "источник_ответа_1"
и "источник_ответа_2" (kb / search / model).

Для каждого пункта из data/knowledge_base_followups.json:
  1. Генерируем подсказку на сам (первый) вопрос — как обычно, то же самое,
     что делает batch_test.py.
  2. Прогоняем ЗАРАНЕЕ ЗАПИСАННЫЙ уточняющий вопрос через generate_hint()
     с историей (первый вопрос) и с own_answers=[первый ответ] — ровно та
     ситуация, под которую чинили баг с "Твои недавние ответы".
  3. Пишем в CSV: оба вопроса, эталонный ответ первого вопроса из базы,
     оба ответа модели, источник каждого ответа и категорию уточняющего
     вопроса (comparison / list / howto / definition / default — см.
     gen_followups.py) — удобно фильтровать в Excel, если, например,
     окажется, что именно списки ("расскажи о каждом из них") проблемнее
     остальных категорий, или что через веб-поиск ответы хуже, чем из базы.

Это МЕДЛЕННЕЕ, чем batch_test.py — на каждый вопрос базы уходит 2 похода
к Ollama вместо 1. Если запускать одновременно с уже работающим
batch_test.py — оба используют один и тот же локальный Ollama-сервер и
GPU, будут делить ресурсы и заметно тормозить друг друга (в этом чате уже
ловили эффект нехватки VRAM/конкуренции за GPU) — по возможности лучше
запускать по очереди, а не параллельно.

Запуск (из корня проекта; если data/knowledge_base_followups.json ещё нет —
сначала один раз: python src\\gen_followups.py):
    venv\\Scripts\\python.exe src\\batch_test_followup.py
    venv\\Scripts\\python.exe src\\batch_test_followup.py --start 200 --limit 20
    venv\\Scripts\\python.exe src\\batch_test_followup.py --model llama3.1:8b
    venv\\Scripts\\python.exe src\\batch_test_followup.py --out followup_results.csv

    # --backend gigachat — проверка того же двухходового сценария (вопрос +
    # уточнение с местоимением, own_answers) через GigaChat вместо Ollama —
    # то есть "понимает ли контекст разговора" для кандидата на роль модели
    # для обычных вопросов (см. batch_test.py/batch_test_hard.py — тот же приём).
    venv\\Scripts\\python.exe src\\batch_test_followup.py --backend gigachat --model GigaChat-2 --limit 15

Флаги — те же, что у batch_test.py:
    --backend ollama|gigachat  какой бэкенд гонять (по умолчанию ollama)
    --model <имя>            модель вместо той, что в .env (OLLAMA_MODEL для
                              --backend ollama, GIGACHAT_MODEL для --backend gigachat)
    --kb / --no-kb           база знаний вкл/выкл вместо .env (для ПЕРВОГО вопроса;
                              на заранее написанный ВТОРОЙ вопрос она обычно и так
                              не сработает — короткое "расскажи подробнее" почти не
                              пересекается словами с формулировками из базы, что и
                              нужно для честной проверки именно own_answers)
    --search / --no-search   веб-поиск вкл/выкл (по умолчанию включён)
    --start N / --limit N    диапазон вопросов (для докрутки после Ctrl+C)
    --out <файл>             CSV с результатами (по умолчанию followup_test_results_<backend>.csv)
"""
from __future__ import annotations

import contextlib
import csv
import io
import sys
import time
from pathlib import Path

from config import Settings
from knowledge_base import search_kb
from live_hints import _CONTINUATION_KB_MIN_SCORE, _SPEAKER_LABEL_RE, _get_kb, generate_hint, generate_hint_gigachat

ROOT_DIR = Path(__file__).resolve().parent.parent
FOLLOWUPS_PATH = ROOT_DIR / "data" / "knowledge_base_followups.json"


class _Tee(io.TextIOBase):
    """Пишет одновременно в реальный stdout (чтобы прогресс по-прежнему было
    видно в консоли) и в буфер — из буфера потом достаём признак "модель
    просила ПОИСК:" (см. _answer_source ниже), не трогая generate_hint()."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _answer_source(
    current_line: str,
    kb_enabled: bool,
    kb_path: str | None,
    kb_top_k: int,
    kb_min_score: float,
    call,
    own_answers_present: bool = False,
) -> tuple:
    """
    Прогоняет call() (обёрнутый вызов generate_hint) и вдобавок определяет,
    ЧЕМ реально ответила модель — из базы знаний (файл-шпаргалка), из
    веб-поиска ("гугление") или из своих же знаний без подсказок. Не меняет
    generate_hint()/live_hints.py — определяем это СНАРУЖИ:
      - kb: заранее (тем же search_kb(), что использует сам generate_hint
        внутри) проверяем, нашлось бы совпадение в базе знаний для ТЕКУЩЕЙ
        строки — если да, по коду live_hints.py поиск в этом случае вообще
        не предлагается (effective_enable_search = enable_search and not
        kb_block), то есть источник гарантированно "kb".
      - search: если совпадения в базе не было, смотрим, напечатала ли
        generate_hint() строку "[hints] ищу в интернете: ..." (она это
        делает именно в момент запроса ПОИСК: от модели) — ловим через
        перехват stdout.
      - model: ни то, ни другое — модель ответила по своим знаниям, без
        КБ и без поиска.
    current_line — именно та строка, на которую в этом вызове отвечает
    generate_hint() (см. _split_current_line в live_hints.py: последняя
    непустая строка хвоста транскрипта) — для первого вопроса это сам
    вопрос, для второго — сгенерированный уточняющий вопрос.

    own_answers_present — ВАЖНО передавать True для второго (уточняющего)
    вопроса: сам generate_hint() при непустом own_answers поднимает порог
    совпадения с базой (см. _CONTINUATION_KB_MIN_SCORE в live_hints.py —
    добавлено в этом чате именно из-за того, что короткие обобщённые
    уточнения вроде "приведи пример" случайно "находили" не связанную
    карточку базы). Если проверять здесь по обычному kb_min_score — эта
    функция будет ошибочно писать "kb" почти на каждый второй вопрос, хотя
    generate_hint() внутри уже отбросил это совпадение и ответила сама/через
    поиск. Поймано на живом прогоне: 117 из 117 строк с source2="kb" в
    followup_test_results.csv не проходили реальный (повышенный) порог —
    метка была полностью устаревшей.
    """
    kb_hit = False
    if kb_enabled and kb_path:
        try:
            kb = _get_kb(kb_path)
            # Снимаем метку говорящего ("Собеседник: "/"Я: ") — иначе она
            # сама попадает в запрос и занижает score (см. _SPEAKER_LABEL_RE
            # и разбор в live_hints.py); повторяем ТОЧНО ту же логику
            # эффективного порога, что и generate_hint(), иначе метка снова
            # разъедется с реальным поведением.
            query = _SPEAKER_LABEL_RE.sub("", current_line, count=1)
            effective_min_score = kb_min_score
            if own_answers_present:
                effective_min_score = max(kb_min_score, _CONTINUATION_KB_MIN_SCORE)
            kb_hit = bool(search_kb(query, kb, top_k=kb_top_k, min_score=effective_min_score))
        except Exception:
            kb_hit = False

    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        result = call()
    captured = buf.getvalue()

    if kb_hit:
        source = "kb"
    elif "ищу в интернете" in captured:
        source = "search"
    else:
        source = "model"

    # Если итоговый ответ оказался пустым — сохраняем диагностическую строку
    # "[hints] модель ..." из _parse_hint() (live_hints.py), чтобы при разборе
    # CSV не пришлось гадать/перезапускать вручную, ПОЧЕМУ ответ пустой
    # (поймано на практике: две строки с пустым ответ_модели_2 без единой
    # зацепки в CSV, откуда взялась пустота).
    diagnostic = ""
    reply = result[0] if isinstance(result, tuple) and result else None
    if not reply:
        diag_lines = [ln for ln in captured.splitlines() if ln.startswith("[hints] модель")]
        diagnostic = diag_lines[-1] if diag_lines else ""

    return result, source, diagnostic


def _load_followups() -> list[dict]:
    import json

    if not FOLLOWUPS_PATH.is_file():
        print(
            f"Не найден {FOLLOWUPS_PATH} — сначала запусти один раз:\n"
            f"    venv\\Scripts\\python.exe src\\gen_followups.py"
        )
        sys.exit(1)
    return json.loads(FOLLOWUPS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    args = sys.argv[1:]
    backend = "ollama"
    model_override: str | None = None
    kb_override: bool | None = None
    search_override: bool | None = None
    start = 0
    limit: int | None = None
    out_path: Path | None = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
        elif a.startswith("--backend="):
            backend = a.split("=", 1)[1]
            i += 1
        elif a == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif a.startswith("--model="):
            model_override = a.split("=", 1)[1]
            i += 1
        elif a == "--kb":
            kb_override = True
            i += 1
        elif a == "--no-kb":
            kb_override = False
            i += 1
        elif a == "--search":
            search_override = True
            i += 1
        elif a == "--no-search":
            search_override = False
            i += 1
        elif a == "--start" and i + 1 < len(args):
            start = int(args[i + 1])
            i += 2
        elif a.startswith("--start="):
            start = int(a.split("=", 1)[1])
            i += 1
        elif a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
            i += 1
        elif a == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
            i += 2
        elif a.startswith("--out="):
            out_path = Path(a.split("=", 1)[1])
            i += 1
        else:
            print(f"Неизвестный аргумент: {a!r}")
            sys.exit(1)

    if backend not in ("ollama", "gigachat"):
        print(f"Неизвестный --backend {backend!r} — допустимо ollama или gigachat.")
        sys.exit(1)

    settings = Settings.load()
    if backend == "gigachat":
        if not settings.gigachat_credentials:
            print("[ошибка] GIGACHAT_CREDENTIALS не задан в .env — нечем идти в GigaChat.")
            print("         См. .env.example или сначала src/check_gigachat_status.py.")
            sys.exit(1)
        model = model_override or settings.gigachat_model
    else:
        model = model_override or settings.ollama_model
    kb_enabled = settings.kb_enabled if kb_override is None else kb_override
    enable_search = True if search_override is None else search_override
    if out_path is None:
        out_path = Path(f"followup_test_results_{backend}.csv")

    followups = _load_followups()

    items = followups[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} вопросов не осталось (всего {len(followups)}).")
        sys.exit(0)

    where = settings.ollama_host if backend == "ollama" else "GigaChat API"
    print(f"Бэкенд: {backend}, модель: {model} ({where})")
    print(f"База знаний (для первого вопроса): {'включена' if kb_enabled else 'выключена'}")
    print(f"Веб-поиск: {'включён' if enable_search else 'выключен'}")
    print(
        f"Вопросов к прогону: {len(items)} (начиная с #{start}, всего {len(followups)}), "
        "по 2 похода к Ollama на каждый"
    )
    print(f"Результат пишется в: {out_path.resolve()}\n")

    append_mode = start > 0 and out_path.exists()
    mode = "a" if append_mode else "w"

    with out_path.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow(
                [
                    "#",
                    "раздел",
                    "бэкенд",
                    "модель",
                    "вопрос_1",
                    "эталонный_ответ_1",
                    "ответ_модели_1",
                    "вопрос_2",
                    "категория_вопроса_2",
                    "ответ_модели_2",
                    "источник_ответа_1",
                    "источник_ответа_2",
                    "диагностика_пустого_ответа",
                    "секунд_1",
                    "секунд_2",
                    "ошибка",
                ]
            )

        kb_path_str = str(settings.kb_path) if settings.kb_path else None

        def _call(transcript: str, own_answers: list[str] | None = None):
            if backend == "gigachat":
                return generate_hint_gigachat(
                    transcript,
                    credentials=settings.gigachat_credentials,
                    model=model,
                    ca_bundle_file=str(settings.gigachat_ca_bundle_file) if settings.gigachat_ca_bundle_file else None,
                    enable_search=enable_search,
                    kb_enabled=kb_enabled,
                    kb_path=kb_path_str,
                    kb_top_k=settings.kb_top_k,
                    kb_min_score=settings.kb_min_score,
                    own_answers=own_answers,
                )
            return generate_hint(
                transcript,
                model=model,
                host=settings.ollama_host,
                enable_search=enable_search,
                kb_enabled=kb_enabled,
                kb_path=kb_path_str,
                kb_top_k=settings.kb_top_k,
                kb_min_score=settings.kb_min_score,
                own_answers=own_answers,
            )

        t_start_all = time.time()
        total = len(items)
        for offset, item in enumerate(items):
            idx = start + offset
            question1 = item.get("question", "")
            reference1 = item.get("answer", "")
            section = item.get("section", "")
            question2 = item.get("followup", "")
            category2 = item.get("followup_category", "")

            reply1 = reply2 = ""
            source1 = source2 = ""
            diag1 = diag2 = ""
            # Раньше был один таймер на весь двухходовый прогон (t0 ДО шага 1,
            # elapsed ПОСЛЕ шага 2) — то есть "секунд" на самом деле мерил
            # сумму обоих походов к Ollama, а не время ответа на каждый вопрос
            # отдельно. Развели на два отдельных замера — так по CSV видно,
            # например, не тормозит ли именно ВТОРОЙ (уточняющий, с добавленным
            # own_answers) ответ сильнее первого, а не только общее время пары.
            elapsed1: float | None = None
            elapsed2: float | None = None
            error = ""
            try:
                # Шаг 1: обычная подсказка на первый вопрос — так же, как в
                # реальном звонке и как в batch_test.py. Модель отвечает
                # САМА, любым способом, каким пользуется в реальном
                # звонке — из базы знаний, гуглением или по своим знаниям;
                # _answer_source() только СНАРУЖИ определяет, какой из этих
                # путей сработал, ничего не подсказывая модели заранее.
                t0 = time.time()
                (reply1, _ctx1), source1, diag1 = _answer_source(
                    question1,
                    kb_enabled,
                    kb_path_str,
                    settings.kb_top_k,
                    settings.kb_min_score,
                    lambda: _call(f"Собеседник: {question1}"),
                    own_answers_present=False,
                )
                elapsed1 = time.time() - t0

                # Шаг 2: главная проверка — заранее написанный уточняющий
                # вопрос, с историей и с own_answers=[reply1], ровно как
                # теперь делает main.py в реальном звонке. Модель снова
                # отвечает сама (КБ по короткому уточнению почти никогда не
                # сработает — и это ожидаемо, см. обсуждение в чате).
                transcript = f"Собеседник: {question1}\nСобеседник: {question2}"
                t0 = time.time()
                (reply2, _ctx2), source2, diag2 = _answer_source(
                    question2,
                    kb_enabled,
                    kb_path_str,
                    settings.kb_top_k,
                    settings.kb_min_score,
                    lambda: _call(transcript, own_answers=[reply1] if reply1 else None),
                    own_answers_present=bool(reply1),
                )
                elapsed2 = time.time() - t0
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

            diagnostic = " | ".join(d for d in (diag1, diag2) if d)
            elapsed1_str = f"{elapsed1:.1f}" if elapsed1 is not None else ""
            elapsed2_str = f"{elapsed2:.1f}" if elapsed2 is not None else ""

            writer.writerow(
                [
                    idx,
                    section,
                    backend,
                    model,
                    question1,
                    reference1,
                    reply1,
                    question2,
                    category2,
                    reply2,
                    source1,
                    source2,
                    diagnostic,
                    elapsed1_str,
                    elapsed2_str,
                    error,
                ]
            )
            f.flush()

            total_elapsed_row = (elapsed1 or 0) + (elapsed2 or 0)
            status = "ОШИБКА" if error else "ok"
            print(
                f"[{offset + 1}/{total}] #{idx} "
                f"(1: {elapsed1_str or '-'}с, 2: {elapsed2_str or '-'}с, всего {total_elapsed_row:.1f}с, {status}) "
                f"{question1[:60]}"
            )

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {total} вопрос(ов) за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}")


if __name__ == "__main__":
    main()
