"""
Проверка ПОЛНОЙ интеграции судьи через консоль, БЕЗ реального звонка (мик,
VB-Cable, оверлей не нужны). В отличие от batch_test_judge.py (проверяет
только сам ПРОМТ судьи на изолированных кейсах с заранее заданными
known_tags) — этот скрипт прогоняет целый СЦЕНАРИЙ (последовательность реплик
подряд) через ТЕ ЖЕ функции, что использует main.py/hint_worker() на реальном
звонке: судья классифицирует каждую реплику по очереди, хранилище тегов
(TrackedTag/_update_known_tags/_known_tags_for_judge — импортируются прямо из
main.py, не переизобретаются здесь) копится и обновляется по ходу сценария
естественным образом, retrieval (_build_retrieved_context) на реальной
"продолжение"-реплике сам подтягивает прошлый обмен, и наконец
generate_hint()/generate_hint_gigachat() генерирует подсказку — тот же самый
порядок вызовов, что в hint_worker().

Отличия от настоящего звонка — только источник реплик (заранее написанный
сценарий вместо транскрипции живого аудио) и однопоточность (нет очереди,
оверлея, pending_clarification_box из "Я: ...?" — эта часть проверялась
отдельно раньше). Вся СУТЬ новой архитектуры (судья -> теги -> retrieval ->
генерация) работает здесь 1:1 как в main.py, потому что использует БУКВАЛЬНО
те же функции, импортированные из main.py, а не копии.

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\test_live_integration.py
    venv\\Scripts\\python.exe src\\test_live_integration.py --scenario data\\my_scenario.json
    venv\\Scripts\\python.exe src\\test_live_integration.py --backend gigachat
    venv\\Scripts\\python.exe src\\test_live_integration.py --model qwen2.5:7b
    venv\\Scripts\\python.exe src\\test_live_integration.py --scenario data\\stress_test_300.json --out stress_results.csv

Формат файла сценария (JSON, список объектов):
    [{"speaker": "Собеседник", "text": "покажи пример SQL-запроса с JOIN"}, ...]
"я"/"Я" тоже допустимый speaker (для собственных реплик пользователя).

Без --scenario используется встроенный пример: два разных вопроса подряд
(SQL, Фибоначчи), потом отсылка назад БЕЗ конкретики (ожидаем ambiguous=true
— тот самый случай, на который ушло 5 раундов при отладке промта), потом
такая же отсылка, но УЖЕ с конкретной деталью (должна разрешиться сама, без
уточнения), и напоследок чистый бэкчаннел (ожидаем "мусор", в модель вообще
не ходим).

--out <файл> (необязательно, по умолчанию live_integration_results.csv в
корне проекта) — помимо консоли, каждая реплика построчно пишется в CSV:
вердикт судьи, теги, сколько кандидатов показано, тайминги (отдельно судья/
генерация) и сам ответ. Нужен для длинных сценариев (сотни реплик, см.
data/stress_test_300.json) — читать такое построчно в консоли нереально, а
в CSV удобно сортировать/фильтровать (например, все строки с judge_seconds >
3 — подозрение на переполнение контекста или уход модели на CPU).

Если во время прогона GigaChat/Ollama вернут ошибку (например, временный
502 от GigaChat — см. проектный документ) — ошибка ловится и пишется в
колонку gen_error/judge_error ЭТОЙ строки, сценарий продолжается со
следующей реплики, а не падает целиком. Это локальное решение только для
этого тестового скрипта — реальный live-путь (main.py/hint_worker →
llm_gigachat.call_gigachat_retrying) пока НЕ ловит такие ошибки, это
отдельный, осознанно отложенный вопрос (см. проектный документ).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import requests

from config import Settings
from transcript import TranscriptStore
from judge import JudgeVerdict, classify_utterance
from live_hints import _split_current_line, generate_hint, generate_hint_gigachat
from main import (
    TrackedTag,
    _known_tags_for_judge,
    _update_known_tags,
    _build_retrieved_context,
    _select_candidate_tags,
    _judge_history_text,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

_DEFAULT_SCENARIO = [
    {"speaker": "Собеседник", "text": "покажи пример SQL-запроса с JOIN"},
    {"speaker": "Собеседник", "text": "а теперь пример с числами Фибоначчи на Python"},
    {"speaker": "Собеседник", "text": "давай вернёмся к твоему примеру"},
    {"speaker": "Собеседник", "text": "я имел в виду тот, что с JOIN"},
    {"speaker": "Собеседник", "text": "ага, спасибо"},
]


def _load_scenario(path: Path | None) -> list[dict]:
    if path is None:
        print("(--scenario не передан, использую встроенный пример)\n")
        return _DEFAULT_SCENARIO
    if not path.exists():
        print(f"[ошибка] не найден файл сценария: {path}")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = sys.argv[1:]
    # None — не задан явно, определим по .env НИЖЕ той же логикой, что и
    # hint_worker() в main.py (use_gigachat = gigachat_enabled и credentials
    # заданы) — раньше тут было жёстко "ollama" по умолчанию, из-за чего
    # первый прогон этого скрипта проверял РЕДКИЙ Ollama-фолбэк-путь вместо
    # реального боевого пути (если GigaChat настроена — а обычно настроена,
    # это основной бэкенд), и нашёл проблему (переполнение _CORE_RULES/
    # num_ctx=2048), которая реальных звонков с работающей GigaChat не
    # касается — см. обсуждение в проектном документе.
    backend: str | None = None
    model_override: str | None = None
    scenario_path: Path | None = None
    out_path: Path = ROOT_DIR / "live_integration_results.csv"

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
        elif a == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif a == "--scenario" and i + 1 < len(args):
            scenario_path = Path(args[i + 1])
            i += 2
        elif a == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
            i += 2
        else:
            print(f"Неизвестный аргумент: {a!r}")
            sys.exit(1)

    if backend is not None and backend not in ("ollama", "gigachat"):
        print(f"Неизвестный --backend {backend!r} — допустимо ollama или gigachat.")
        sys.exit(1)

    settings = Settings.load()
    if backend is None:
        backend = "gigachat" if (settings.gigachat_enabled and settings.gigachat_credentials) else "ollama"
        print(f"(--backend не передан, определил по .env как в hint_worker(): {backend})")
    if backend == "gigachat" and not settings.gigachat_credentials:
        print("[ошибка] GIGACHAT_CREDENTIALS не задан в .env — нечем идти в GigaChat.")
        sys.exit(1)
    gigachat_model = model_override or settings.gigachat_model
    ollama_model = model_override or settings.ollama_model
    # Судья всегда использует OLLAMA_MODEL (решение зафиксировано в проектном
    # документе) — --model переопределяет её тоже, если передан, чтобы можно
    # было прогнать весь сценарий одной моделью для сравнения.
    judge_model = model_override or settings.ollama_model

    scenario = _load_scenario(scenario_path)

    session_dir = ROOT_DIR / "test_sessions" / "live_integration_console"
    transcript_store = TranscriptStore(session_dir)
    known_tags: list[TrackedTag] = []
    own_answers: list[str] = []
    OWN_ANSWERS_KEEP = 6
    judge_session = requests.Session()

    print(f"Судья: {judge_model} ({settings.ollama_host}), num_ctx={settings.judge_num_ctx}")
    print(f"Генерация: {backend}, модель: {gigachat_model if backend == 'gigachat' else ollama_model}")
    print(f"Реплик в сценарии: {len(scenario)}")
    print(f"Результаты (CSV): {out_path.resolve()}\n")

    total = len(scenario)
    t_start_all = time.time()
    csv_file = out_path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "#", "line_index", "speaker", "text",
            "known_tags_total", "candidates_shown",
            "judge_verdict", "judge_tags", "judge_continues_tags", "judge_ambiguous", "judge_options",
            "judge_seconds", "judge_error",
            "reply", "context", "gen_seconds", "gen_error",
        ]
    )

    try:
        for turn_num, turn in enumerate(scenario, start=1):
            speaker = turn.get("speaker", "Собеседник")
            text = turn.get("text", "")
            line = transcript_store.add(speaker, text, engine="test")
            if line is None:
                continue
            print(f"\n=== [{turn_num}/{total}] {speaker}: {text}  [line.index={line.index}] ===")

            recent_text = transcript_store.recent_text(settings.hints_context_lines)
            # history_text (полное окно HINTS_CONTEXT_LINES) идёт ТОЛЬКО в
            # генератор ответа (recent_text ниже) — судья теперь получает
            # отдельную, короткую _judge_history_text и отобранный
            # _select_candidate_tags вместо всего known_tags, ТОЧНО как в
            # main.py/hint_worker() (см. проектный документ, round про
            # переполнение num_ctx судьи и рост промта на длинных звонках).
            _, question = _split_current_line(recent_text)
            judge_history = _judge_history_text(transcript_store, line.index)
            candidate_tags = _select_candidate_tags(known_tags, question)
            if len(candidate_tags) != len(known_tags):
                print(
                    f"[теги] показываю судье {len(candidate_tags)} из {len(known_tags)} накопленных: "
                    f"{[t.tag for t in candidate_tags]}"
                )

            # Ошибка судьи (Ollama недоступна/упала) ловится ЗДЕСЬ и не роняет
            # весь прогон на 300 репликах — пишем её в CSV и считаем реплику
            # как parse_error (тот же путь деградации, что и "не разобрался
            # JSON"), идём дальше без retrieval, как и раньше при parse_error.
            judge_error = ""
            t0 = time.time()
            try:
                verdict = classify_utterance(
                    current_line=question,
                    history=judge_history,
                    known_tags=_known_tags_for_judge(candidate_tags, line.index),
                    model=judge_model,
                    host=settings.ollama_host,
                    timeout=30.0,
                    session=judge_session,
                    num_ctx=settings.judge_num_ctx,
                )
            except Exception as e:  # noqa: BLE001 — намеренно широкий catch, см. комментарий выше
                judge_error = f"{type(e).__name__}: {e}"
                print(f"[судья] ИСКЛЮЧЕНИЕ: {judge_error} — считаем недоступной, идём дальше без retrieval")
                verdict = JudgeVerdict(verdict="", raw="", parse_error=judge_error)
            judge_elapsed = time.time() - t0
            if not judge_error:
                print(
                    f"[судья, {judge_elapsed:.1f}с] verdict={verdict.verdict!r} tags={verdict.tags} "
                    f"continues_tags={verdict.continues_tags} ambiguous={verdict.ambiguous} "
                    f"options={verdict.ambiguous_options}"
                )
            if verdict.parse_error and not judge_error:
                print(f"[судья] parse_error: {verdict.parse_error}")
                print(f"[судья] сырой ответ модели: {verdict.raw!r}")
                print("-> считаем судью недоступной, идём как раньше (без retrieval)")

            if not verdict.parse_error and verdict.is_garbage:
                print("-> МУСОР: в модель не идём, подсказки не будет")
                writer.writerow(
                    [
                        turn_num, line.index, speaker, text,
                        len(known_tags), len(candidate_tags),
                        verdict.verdict, ", ".join(verdict.tags), ", ".join(verdict.continues_tags),
                        verdict.ambiguous, " | ".join(verdict.ambiguous_options),
                        f"{judge_elapsed:.2f}", judge_error,
                        "", "", "", "",
                    ]
                )
                csv_file.flush()
                continue

            retrieved_context = ""
            if not verdict.parse_error and verdict.continues_tags:
                retrieved_context = _build_retrieved_context(verdict.continues_tags, known_tags, transcript_store)
                if retrieved_context:
                    print(f"[retrieval]\n{retrieved_context}")

            if not verdict.parse_error:
                _update_known_tags(known_tags, verdict, line.index, question)
            print(f"[известные теги сейчас] {[(t.tag, t.last_line_index) for t in known_tags]}")

            # Аналогично судье: ошибка генерации (например, временный 502 от
            # GigaChat — см. проектный документ, call_gigachat_retrying пока
            # не ловит такие ошибки на живом пути) не должна ронять весь
            # 300-репличный прогон — ловим, пишем в CSV, идём дальше.
            gen_error = ""
            reply = ""
            context = ""
            t0 = time.time()
            try:
                if not verdict.parse_error and verdict.ambiguous and verdict.ambiguous_options:
                    options_text = " или ".join(f"«{opt}»" for opt in verdict.ambiguous_options)
                    reply = f"Уточни, пожалуйста: ты имеешь в виду {options_text}?"
                    context = "уточнение (неоднозначная отсылка)"
                elif backend == "gigachat":
                    reply, context = generate_hint_gigachat(
                        recent_text,
                        credentials=settings.gigachat_credentials,
                        model=gigachat_model,
                        ca_bundle_file=str(settings.gigachat_ca_bundle_file)
                        if settings.gigachat_ca_bundle_file
                        else None,
                        kb_enabled=settings.kb_enabled,
                        kb_path=str(settings.kb_path) if settings.kb_path else None,
                        kb_top_k=settings.kb_top_k,
                        kb_min_score=settings.kb_min_score,
                        own_answers=list(own_answers),
                        extra_context=settings.hints_extra_context,
                        retrieved_context=retrieved_context,
                    )
                else:
                    reply, context = generate_hint(
                        recent_text,
                        model=ollama_model,
                        host=settings.ollama_host,
                        num_ctx=settings.ollama_num_ctx,
                        kb_enabled=settings.kb_enabled,
                        kb_path=str(settings.kb_path) if settings.kb_path else None,
                        kb_top_k=settings.kb_top_k,
                        kb_min_score=settings.kb_min_score,
                        own_answers=list(own_answers),
                        extra_context=settings.hints_extra_context,
                        retrieved_context=retrieved_context,
                    )
            except Exception as e:  # noqa: BLE001 — намеренно широкий catch, см. комментарий выше
                gen_error = f"{type(e).__name__}: {e}"
                print(f"[генерация] ИСКЛЮЧЕНИЕ: {gen_error} — подсказки не будет, идём дальше")
            gen_elapsed = time.time() - t0

            writer.writerow(
                [
                    turn_num, line.index, speaker, text,
                    len(known_tags), len(candidate_tags),
                    verdict.verdict, ", ".join(verdict.tags), ", ".join(verdict.continues_tags),
                    verdict.ambiguous, " | ".join(verdict.ambiguous_options),
                    f"{judge_elapsed:.2f}", judge_error,
                    reply, context, f"{gen_elapsed:.2f}", gen_error,
                ]
            )
            csv_file.flush()

            print(f"ПОДСКАЗКА ({gen_elapsed:.1f}с): {reply}")
            print(f"КОНТЕКСТ: {context or '-'}")

            if reply:
                transcript_store.add("Ассистент", reply, engine="test")
                own_answers.append(reply)
                del own_answers[:-OWN_ANSWERS_KEEP]
    finally:
        judge_session.close()
        transcript_store.close()
        csv_file.close()
        total_elapsed = time.time() - t_start_all
        print(f"\n(лог сценария сохранён: {transcript_store.txt_path})")
        print(f"(результаты с таймингами: {out_path.resolve()})")
        print(f"Готово: {total} реплик за {total_elapsed / 60:.1f} мин.")


if __name__ == "__main__":
    main()
