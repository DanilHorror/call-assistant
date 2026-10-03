"""
Массовый прогон смешанного набора вопросов (QA-теория, автоматизация тестов,
общая разработка, ситуационные вопросы на уточнение) — data/mixed_qa_dev_questions.json,
100 вопросов вперемешку (сгенерирован по просьбе "давай протестим").

Отличие от остальных batch_test_*.py: этот скрипт умеет включать/выключать
HINTS_EXTRA_CONTEXT (см. .env, config.py, _build_user_message в live_hints.py)
флагом --context/--no-context — специально, чтобы можно было сравнить ОДИН И
ТОТ ЖЕ набор вопросов с контекстом и без, и увидеть разницу своими глазами
(например, на вопросах раздела "Разработка", где язык программирования не
назван явно — без контекста модель может выбрать не тот язык, с контекстом
из HINTS_EXTRA_CONTEXT — должна писать на Java).

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test_mixed.py --backend gigachat
    venv\\Scripts\\python.exe src\\batch_test_mixed.py --backend gigachat --no-context
    venv\\Scripts\\python.exe src\\batch_test_mixed.py --backend gigachat --type development --limit 10
    venv\\Scripts\\python.exe src\\batch_test_mixed.py --backend ollama --limit 20

Флаги:
    --backend ollama|gigachat  какой бэкенд гонять (обязателен)
    --context / --no-context   передавать ли HINTS_EXTRA_CONTEXT из .env в
                        generate_hint*() (по умолчанию --context, то есть как
                        в реальной работе main.py; --no-context имитирует
                        поведение ДО этой фичи — удобно для сравнения)
    --type qa|automation|development|clarification|all   фильтр по типу
                        вопроса (по умолчанию all)
    --model <имя>       модель вместо той, что в .env
    --kb / --no-kb      база знаний вкл/выкл (по умолчанию — как в .env)
    --search / --no-search  веб-поиск вкл/выкл (по умолчанию включён)
    --start N           начать не с 0-го вопроса, а с N-го
    --limit N           прогнать только N вопросов начиная со --start
    --out <файл>        путь к CSV (по умолчанию: mixed_test_<backend>_<context|nocontext>.csv)

Как и остальные скрипты — каждый вопрос прогоняется НЕЗАВИСИМО (без истории,
без own_answers): проверяем реакцию на вопрос сам по себе, а не поведение в
диалоге. Результат пишется в CSV сразу по каждой строке (с flush).
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

from config import Settings
from live_hints import generate_hint, generate_hint_gigachat

ROOT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT_DIR / "data" / "mixed_qa_dev_questions.json"

CSV_COLUMNS = [
    "#", "тип", "раздел", "бэкенд", "модель", "контекст_вкл",
    "вопрос", "ответ", "контекст_модели", "секунд", "поиск_сработал", "ошибка",
]


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _load_items(type_filter: str) -> list[dict]:
    if not QUESTIONS_PATH.is_file():
        print(f"Не найден {QUESTIONS_PATH} — нечего прогонять.")
        sys.exit(1)
    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if type_filter != "all":
        items = [it for it in items if it.get("type") == type_filter]
    return items


def _run_one(
    backend: str,
    question: str,
    model: str,
    settings: Settings,
    kb_enabled: bool,
    kb_path_str: str | None,
    enable_search: bool,
    extra_context: str | None,
) -> tuple[str, str, bool, float, str]:
    # Вопросы clarification уже сами по себе с меткой "Собеседник: " (см.
    # генератор data/mixed_qa_dev_questions.json) — остальные типы без метки,
    # добавляем её тут же, как и другие batch_test_*.py.
    transcript = question if question.startswith("Собеседник:") else f"Собеседник: {question}"
    reply, context, error = "", "", ""
    search_triggered = False

    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            if backend == "gigachat":
                reply, context = generate_hint_gigachat(
                    transcript,
                    credentials=settings.gigachat_credentials,
                    model=model,
                    ca_bundle_file=str(settings.gigachat_ca_bundle_file) if settings.gigachat_ca_bundle_file else None,
                    enable_search=enable_search,
                    kb_enabled=kb_enabled,
                    kb_path=kb_path_str,
                    kb_top_k=settings.kb_top_k,
                    kb_min_score=settings.kb_min_score,
                    extra_context=extra_context,
                )
            else:
                reply, context = generate_hint(
                    transcript,
                    model=model,
                    host=settings.ollama_host,
                    enable_search=enable_search,
                    kb_enabled=kb_enabled,
                    kb_path=kb_path_str,
                    kb_top_k=settings.kb_top_k,
                    kb_min_score=settings.kb_min_score,
                    extra_context=extra_context,
                )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    search_triggered = "ищу в интернете" in buf.getvalue()
    return reply, context or "-", search_triggered, elapsed, error


def main() -> None:
    args = sys.argv[1:]
    backend: str | None = None
    context_override: bool | None = None
    type_filter = "all"
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
        elif a == "--context":
            context_override = True
            i += 1
        elif a == "--no-context":
            context_override = False
            i += 1
        elif a == "--type" and i + 1 < len(args):
            type_filter = args[i + 1]
            i += 2
        elif a.startswith("--type="):
            type_filter = a.split("=", 1)[1]
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
        print("Нужно явно указать --backend ollama или --backend gigachat.")
        sys.exit(1)
    if type_filter not in ("qa", "automation", "development", "clarification", "all"):
        print("--type должен быть qa, automation, development, clarification или all.")
        sys.exit(1)

    settings = Settings.load()

    if backend == "gigachat":
        if not settings.gigachat_credentials:
            print("[ошибка] GIGACHAT_CREDENTIALS не задан в .env — нечем идти в GigaChat.")
            sys.exit(1)
        model = model_override or settings.gigachat_model
    else:
        model = model_override or settings.ollama_model

    use_context = True if context_override is None else context_override
    extra_context = settings.hints_extra_context if use_context else None
    if use_context and not extra_context:
        print("[внимание] --context запрошен, но HINTS_EXTRA_CONTEXT в .env пуст — "
              "фактически тест пройдёт так же, как --no-context.")

    kb_enabled = settings.kb_enabled if kb_override is None else kb_override
    enable_search = True if search_override is None else search_override
    kb_path_str = str(settings.kb_path) if (kb_enabled and settings.kb_path) else None
    if kb_enabled and not settings.kb_path:
        print("KB запрошена, но в .env не задан KB_PATH — база всё равно не будет использована.")
        kb_enabled = False

    if out_path is None:
        out_path = Path(f"mixed_test_{backend}_{'context' if use_context else 'nocontext'}.csv")

    items = _load_items(type_filter)[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} вопросов не осталось.")
        sys.exit(0)

    print(f"Бэкенд: {backend}, модель: {model}")
    print(f"HINTS_EXTRA_CONTEXT: {'включён — ' + repr(extra_context) if extra_context else 'выключен/пуст'}")
    print(f"База знаний: {'включена' if kb_enabled else 'выключена'}")
    print(f"Веб-поиск: {'включён' if enable_search else 'выключен'}")
    print(f"Тип: {type_filter}, вопросов к прогону: {len(items)} (начиная с #{start})")
    print(f"Результат пишется в: {out_path.resolve()}\n")

    append_mode = start > 0 and out_path.exists()
    mode = "a" if append_mode else "w"

    with out_path.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow(CSV_COLUMNS)

        t_start_all = time.time()
        total = len(items)
        for offset, item in enumerate(items):
            idx = start + offset
            qtype = item.get("type", "")
            section = item.get("section", "")
            question = item.get("question", "")

            reply, context, search_triggered, elapsed, error = _run_one(
                backend, question, model, settings, kb_enabled, kb_path_str, enable_search, extra_context
            )

            writer.writerow(
                [idx, qtype, section, backend, model, "да" if use_context else "нет",
                 question, reply, context, f"{elapsed:.1f}", "да" if search_triggered else "нет", error]
            )
            f.flush()

            status = "ОШИБКА" if error else "ok"
            print(f"[{offset + 1}/{total}] #{idx} [{qtype}] ({elapsed:.1f}с, {status}) {question[:60]}")

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {total} вопрос(ов) за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
