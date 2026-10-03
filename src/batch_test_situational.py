"""
ЧЕТВЁРТЫЙ отдельный файл (batch_test.py / batch_test_followup.py /
batch_test_search.py не трогаем). Проверяет ОБОБЩЕНИЕ правила про
"ситуационные вопросы" (см. _CORE_RULES в live_hints.py) на новые
формулировки, которых нет в самом промпте.

В FIRST_PASS_PROMPT/_CORE_RULES зашиты только 2 живых примера такого
класса вопросов ("у меня дублируются записи", "не сходятся цифры в
отчёте") — известное открытое ограничение (см. PROGRESS.md): фикс работает
надёжно на близких по формулировке случаях, но не факт, что обобщается на
разные темы и разные способы задать похожий по сути вопрос. Этот файл — 24
новые ситуации из тех же доменов, что и остальные тесты (QA/QC, AQA,
процесс разработки, мобильное тестирование, сети/API, Brokerpilot),
специально НЕ похожие текстуально на два примера из промпта.

Вопросы взяты из data/situational_questions.json. Поле "ожидаемый_тип" —
только пометка ДЛЯ ЧЕЛОВЕКА при разборе результатов, модели не показывается:
  - "уточнение"      — единственно верного ответа нет, правильное поведение —
                        точный встречный вопрос по СУТИ данной конкретной
                        ситуации, без выдумывания деталей, без общего совета
                        и без похода в поиск.
  - "общий_ответ_ок" — пограничные случаи ("как бы ты подошёл к X") — тут
                        нормально дать общий профессиональный подход без
                        уточнения, это НЕ повод считать ответ плохим.

В отличие от batch_test_search.py, здесь KB/поиск НЕ форсируются выключенными
— наоборот, важно посмотреть на поведение как в реальном звонке (включая то,
не полезет ли модель зачем-то в веб-поиск на ситуационный вопрос — это само
по себе ошибка, поиск для таких вопросов не имеет смысла). По умолчанию оба
берутся из .env, как в batch_test.py; флагами можно переопределить.

Автоматически (без ручного разбора) в CSV считаются два структурных
"красных флажка" — они проверяют ФОРМУ ответа, а не его суть, и служат
только чтобы быстрее найти подозрительные строки глазами:
  - похоже_на_уточнение: эвристика "ответ оканчивается на '?'" (грубо, но
    быстро находит явные промахи, когда ожидали уточнение, а получили
    развёрнутый совет или выдуманный факт без единого вопроса).
  - неожиданный_поиск: источник_ответа == "search" на ситуационном вопросе —
    само по себе подозрительно независимо от ожидаемого_типа, поиск не
    должен быть путём решения таких вопросов.
Финальный вердикт "хороший ли это конкретно вопрос от модели" эти флажки
НЕ ставят — как и везде в этом наборе тестов, смысл ответа проверяется
глазами по колонке "ответ_модели".

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test_situational.py
    venv\\Scripts\\python.exe src\\batch_test_situational.py --start 10 --limit 5
    venv\\Scripts\\python.exe src\\batch_test_situational.py --model llama3.1:8b
    venv\\Scripts\\python.exe src\\batch_test_situational.py --no-kb
    venv\\Scripts\\python.exe src\\batch_test_situational.py --out results.csv

Флаги — те же, что у остальных batch_test_*.py:
    --backend ollama|gigachat  какой бэкенд гонять (по умолчанию ollama)
    --model <имя>     модель вместо той, что в .env (OLLAMA_MODEL для
                      --backend ollama, GIGACHAT_MODEL для --backend gigachat)
    --kb / --no-kb    база знаний вкл/выкл (по умолчанию — как в .env)
    --search / --no-search  веб-поиск вкл/выкл (по умолчанию — включён)
    --start N         начать не с 0-го вопроса, а с N-го (докрутка после Ctrl+C)
    --limit N         прогнать только N вопросов начиная со --start
    --out <файл>      путь к CSV (по умолчанию situational_test_results_<backend>.csv)

    # --backend gigachat — главная проверка "умеет ли уточнять вопрос":
    # эти 24 ситуации специально написаны так, что правильное поведение —
    # точный встречный вопрос по сути ситуации, а не выдуманный совет.
    venv\\Scripts\\python.exe src\\batch_test_situational.py --backend gigachat --model GigaChat-2
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
from knowledge_base import search_kb
from live_hints import _get_kb, generate_hint, generate_hint_gigachat

ROOT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT_DIR / "data" / "situational_questions.json"


class _Tee(io.TextIOBase):
    """См. тот же приём в batch_test_search.py: пишем и в реальный stdout
    (прогресс виден в консоли), и в буфер — из буфера потом достаём признак
    'модель просила ПОИСК:', не трогая generate_hint()."""

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
    current_line: str, kb_enabled: bool, kb_path: str | None, kb_top_k: int, kb_min_score: float, call
) -> tuple:
    """Тот же приём, что в batch_test_followup.py/batch_test_search.py —
    определяем СНАРУЖИ источник ответа, не меняя live_hints.py."""
    kb_hit = False
    if kb_enabled and kb_path:
        try:
            kb = _get_kb(kb_path)
            kb_hit = bool(search_kb(current_line, kb, top_k=kb_top_k, min_score=kb_min_score))
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
    return result, source


def _load_questions() -> list[dict]:
    if not QUESTIONS_PATH.is_file():
        print(f"Не найден {QUESTIONS_PATH} — нечего прогонять.")
        sys.exit(1)
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


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
    kb_path_str = str(settings.kb_path) if (kb_enabled and settings.kb_path) else None
    if out_path is None:
        out_path = Path(f"situational_test_results_{backend}.csv")

    questions = _load_questions()
    items = questions[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} вопросов не осталось (в файле всего {len(questions)}).")
        sys.exit(0)

    where = settings.ollama_host if backend == "ollama" else "GigaChat API"
    print(f"Бэкенд: {backend}, модель: {model} ({where})")
    print(f"База знаний: {'включена' if kb_enabled else 'выключена'}")
    print(f"Веб-поиск: {'включён' if enable_search else 'выключен'}")
    print(f"Вопросов к прогону: {len(items)} (начиная с #{start}, всего в файле {len(questions)})")
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
                    "ожидаемый_тип",
                    "заметка",
                    "вопрос",
                    "ответ_модели",
                    "контекст_модели",
                    "источник_ответа",
                    "похоже_на_уточнение",
                    "неожиданный_поиск",
                    "секунд",
                    "ошибка",
                ]
            )

        t_start_all = time.time()
        total = len(items)
        for offset, item in enumerate(items):
            idx = start + offset
            question = item.get("question", "")
            section = item.get("section", "")
            expected = item.get("ожидаемый_тип", "")
            note = item.get("заметка", "")
            transcript = f"Собеседник: {question}"

            reply, context, source, error = "", "", "", ""
            t0 = time.time()
            try:
                (reply, context), source = _answer_source(
                    question,
                    kb_enabled,
                    kb_path_str,
                    settings.kb_top_k,
                    settings.kb_min_score,
                    lambda: (
                        generate_hint_gigachat(
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
                        if backend == "gigachat"
                        else generate_hint(
                            transcript,
                            model=model,
                            host=settings.ollama_host,
                            enable_search=enable_search,
                            kb_enabled=kb_enabled,
                            kb_path=kb_path_str,
                            kb_top_k=settings.kb_top_k,
                            kb_min_score=settings.kb_min_score,
                        )
                    ),
                )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0

            looks_like_clarification = reply.rstrip().endswith("?")
            unexpected_search = source == "search"

            writer.writerow(
                [
                    idx,
                    section,
                    backend,
                    model,
                    expected,
                    note,
                    question,
                    reply,
                    context or "-",
                    source,
                    "да" if looks_like_clarification else "нет",
                    "ДА" if unexpected_search else "",
                    f"{elapsed:.1f}",
                    error,
                ]
            )
            f.flush()

            flags = []
            if unexpected_search:
                flags.append("ПОИСК?!")
            if expected == "уточнение" and not looks_like_clarification:
                flags.append("не похоже на уточнение")
            status = "ОШИБКА" if error else (", ".join(flags) if flags else "ok")
            print(f"[{offset + 1}/{total}] #{idx} ({elapsed:.1f}с, {status}) {question[:70]}")

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {total} вопрос(ов) за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}")


if __name__ == "__main__":
    main()
