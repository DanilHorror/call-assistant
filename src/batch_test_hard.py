"""
Массовый прогон "сложных" вопросов (те же категории, что main.py считает
is_hard — см. _looks_like_hard_question в main.py: алгоритмические задачи по
программированию + решаемые ситуационные вопросы) — НО, в отличие от
batch_test.py/batch_test_situational.py, умеет гонять их через ЛЮБОЙ из
двух бэкендов, локальную Ollama или облачную GigaChat, флагом --backend.

Смысл именно в СРАВНЕНИИ: главный открытый вопрос сейчас — успевает ли
GigaChat отвечать так же быстро, как локальная модель, на тех вопросах,
которые main.py теперь отправляет именно ей. Если тайминги окажутся
сопоставимы (или GigaChat быстрее — вполне вероятно, у неё нет конкуренции
за те же 8ГБ видеопамяти с Whisper/диаризацией, в отличие от Ollama) — это
будет основанием подумать о том, чтобы со временем отправлять в GigaChat не
только "сложные", а вообще все вопросы. Если наоборот ощутимо медленнее
(сетевая задержка до API Сбера + время самой генерации) — оставляем текущую
гибридную схему как есть, она и задумывалась как компромисс.

Время (колонка "секунд") — это ПОЛНОЕ время от вызова generate_hint()/
generate_hint_gigachat() до возврата готового ответа целиком, БЕЗ
стриминга (on_partial не передаём) — то есть ровно то самое "от получения
вопроса до полного ответа", которое просили измерить, а не время до первого
токена.

Вопросы берутся из ДВУХ источников (можно выбрать один флагом --set):
  - data/algorithmic_questions.json — новый набор (см. рядом), написан
    специально под этот тест, аналогов раньше не было в проекте.
  - data/situational_questions.json — уже существующий набор из
    batch_test_situational.py, тут переиспользуется как есть.

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test_hard.py --backend gigachat
    venv\\Scripts\\python.exe src\\batch_test_hard.py --backend ollama
    venv\\Scripts\\python.exe src\\batch_test_hard.py --backend gigachat --set algorithmic
    venv\\Scripts\\python.exe src\\batch_test_hard.py --backend gigachat --limit 5
    venv\\Scripts\\python.exe src\\batch_test_hard.py --backend gigachat --out giga_run.csv

Флаги:
    --backend ollama|gigachat  какой бэкенд гонять (обязателен, без разумного
                        умолчания — слишком легко случайно прогнать не тот
                        и сравнить бэкенд сам с собой)
    --set algorithmic|situational|all   какой набор вопросов (по умолчанию all)
    --model <имя>       модель вместо той, что в .env (OLLAMA_MODEL для
                        --backend ollama, GIGACHAT_MODEL для --backend gigachat)
    --kb / --no-kb      база знаний вкл/выкл (по умолчанию — как в .env;
                        на "сложные" вопросы обычно ничего не находит, но
                        main.py передаёт этот параметр и туда, и сюда одинаково
                        — для честного сравнения тут так же)
    --search / --no-search  веб-поиск вкл/выкл (по умолчанию включён)
    --start N           начать не с 0-го вопроса, а с N-го (докрутка после Ctrl+C)
    --limit N           прогнать только N вопросов начиная со --start
    --out <файл>        путь к CSV (по умолчанию: hard_test_<backend>.csv)

После прогона печатается сводка по таймингам (мин/медиана/среднее/макс) —
отдельно по каждому типу вопросов (algorithmic/situational) и по всем вместе,
это и есть то, что удобно положить рядом с таким же прогоном другого
бэкенда и сравнить на глаз.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import statistics
import sys
import time
from pathlib import Path

from config import Settings
from live_hints import generate_hint, generate_hint_gigachat

ROOT_DIR = Path(__file__).resolve().parent.parent
ALGORITHMIC_PATH = ROOT_DIR / "data" / "algorithmic_questions.json"
SITUATIONAL_PATH = ROOT_DIR / "data" / "situational_questions.json"

CSV_COLUMNS = ["#", "тип", "раздел", "бэкенд", "модель", "вопрос", "ответ", "контекст", "секунд", "поиск_сработал", "ошибка"]


class _Tee(io.TextIOBase):
    """Тот же приём, что и в batch_test_final.py/batch_test_situational.py:
    пишем и в реальный stdout (прогресс виден в консоли), и в буфер — из
    буфера потом достаём признак "модель попросила ПОИСК:" (печатается как
    "ищу в интернете" в live_hints.py), не трогая сами generate_hint*()."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _load_json(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        print(f"Не найден {path} ({label}) — нечего прогонять.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _build_items(which: str) -> list[dict]:
    items: list[dict] = []
    if which in ("algorithmic", "all"):
        for item in _load_json(ALGORITHMIC_PATH, "алгоритмические вопросы"):
            items.append({"type": "algorithmic", **item})
    if which in ("situational", "all"):
        for item in _load_json(SITUATIONAL_PATH, "ситуационные вопросы"):
            items.append({"type": "situational", **item})
    return items


def _run_one(
    backend: str,
    question: str,
    model: str,
    settings: Settings,
    kb_enabled: bool,
    kb_path_str: str | None,
    enable_search: bool,
) -> tuple[str, str, bool, float, str]:
    """Один поход к модели (без истории, без own_answers — проверяем вопрос
    сам по себе, как и остальные batch_test_*.py). Возвращает
    (ответ, контекст, поиск_сработал, секунды, ошибка)."""
    transcript = f"Собеседник: {question}"
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
                )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    search_triggered = "ищу в интернете" in buf.getvalue()
    return reply, context or "-", search_triggered, elapsed, error


def _print_summary(label: str, timings: list[float]) -> None:
    if not timings:
        print(f"  {label}: нет данных")
        return
    print(
        f"  {label}: n={len(timings)}  "
        f"мин={min(timings):.1f}с  медиана={statistics.median(timings):.1f}с  "
        f"среднее={statistics.mean(timings):.1f}с  макс={max(timings):.1f}с"
    )


def main() -> None:
    args = sys.argv[1:]
    backend: str | None = None
    which_set = "all"
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
        elif a == "--set" and i + 1 < len(args):
            which_set = args[i + 1]
            i += 2
        elif a.startswith("--set="):
            which_set = a.split("=", 1)[1]
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
    if which_set not in ("algorithmic", "situational", "all"):
        print("--set должен быть algorithmic, situational или all.")
        sys.exit(1)

    settings = Settings.load()

    if backend == "gigachat":
        if not settings.gigachat_credentials:
            print("[ошибка] GIGACHAT_CREDENTIALS не задан в .env — нечем идти в GigaChat.")
            print("         См. .env.example (GIGACHAT_CREDENTIALS) или проверьте")
            print("         сначала src/check_gigachat_status.py.")
            sys.exit(1)
        model = model_override or settings.gigachat_model
    else:
        model = model_override or settings.ollama_model

    kb_enabled = settings.kb_enabled if kb_override is None else kb_override
    enable_search = True if search_override is None else search_override
    kb_path_str = str(settings.kb_path) if (kb_enabled and settings.kb_path) else None
    if kb_enabled and not settings.kb_path:
        print("KB запрошена, но в .env не задан KB_PATH — база всё равно не будет использована.")
        kb_enabled = False

    if out_path is None:
        out_path = Path(f"hard_test_{backend}.csv")

    items = _build_items(which_set)[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} вопросов не осталось.")
        sys.exit(0)

    print(f"Бэкенд: {backend}, модель: {model}")
    print(f"База знаний: {'включена' if kb_enabled else 'выключена'}")
    print(f"Веб-поиск: {'включён' if enable_search else 'выключен'}")
    print(f"Набор: {which_set}, вопросов к прогону: {len(items)} (начиная с #{start})")
    print(f"Результат пишется в: {out_path.resolve()}\n")

    append_mode = start > 0 and out_path.exists()
    mode = "a" if append_mode else "w"

    timings_by_type: dict[str, list[float]] = {"algorithmic": [], "situational": []}
    timings_all: list[float] = []

    with out_path.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow(CSV_COLUMNS)

        t_start_all = time.time()
        total = len(items)
        for offset, item in enumerate(items):
            idx = start + offset
            utype = item["type"]
            question = item.get("question", "")
            section = item.get("section", "")

            reply, context, search_triggered, elapsed, error = _run_one(
                backend, question, model, settings, kb_enabled, kb_path_str, enable_search
            )

            writer.writerow(
                [idx, utype, section, backend, model, question, reply, context,
                 f"{elapsed:.1f}", "да" if search_triggered else "нет", error]
            )
            f.flush()

            if not error:
                timings_by_type.setdefault(utype, []).append(elapsed)
                timings_all.append(elapsed)

            status = "ОШИБКА" if error else "ok"
            print(f"[{offset + 1}/{total}] #{idx} [{utype}] ({elapsed:.1f}с, {status}) {question[:60]}")

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {total} вопрос(ов) за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}\n")

        print(f"Сводка по таймингам ({backend}, {model}):")
        _print_summary("algorithmic", timings_by_type.get("algorithmic", []))
        _print_summary("situational", timings_by_type.get("situational", []))
        _print_summary("всего", timings_all)


if __name__ == "__main__":
    main()
