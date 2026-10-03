"""
ТРЕТИЙ, отдельный файл от batch_test.py и batch_test_followup.py (их не
трогаем — оба уже могут быть запущены/содержать свои результаты).

batch_test.py и batch_test_followup.py проверяют, как модель отвечает на
вопросы, которые ЕСТЬ в базе знаний (data/knowledge_base.json) — то есть
по сути "умеет ли она правильно пересказать найденную шпаргалку". Этот
файл проверяет ПРОТИВОПОЛОЖНОЕ: 100 вопросов из тех же тематик (QA/QC,
автоматизация, Agile/DevOps, мобильное тестирование, сети/API, форекс-
домен Brokerpilot), которых В БАЗЕ ЗНАНИЙ НЕТ и которые специально
сформулированы так, что правильный ответ обычно требует "погуглить" —
это не текстбучные определения (те как раз в базе), а актуальные версии
инструментов и стандартов, цены/тарифы, свежая статистика опросов,
регуляторные требования и т.п. — то, что меняется со временем и что
маленькая локальная модель просто не может знать наверняка "из головы".

Вопросы взяты из data/search_questions.json (100 штук, поля question /
section / why_search — why_search это короткая пометка ДЛЯ ЧЕЛОВЕКА при
разборе результатов: "версия/релиз", "цена", "статистика", "регуляторика"
и т.п. — не показывается модели, только для колонки в CSV).

Почему KB здесь по умолчанию ВЫКЛЮЧЕНА (в отличие от batch_test.py) —
это единственное отличие в поведении генерации от обычного звонка, и оно
осознанное: при подготовке вопросов 21 из 100 случайно пересеклись с
базой знаний по паре общих слов (например, "Какая последняя версия
Cypress и чем она отличается от Playwright?" зацепляется за существующий
в базе вопрос "Сравнение Playwright / Selenium / Cypress / Puppeteer
(2025)?" с score 0.217 — выше порога KB_MIN_SCORE=0.15 из .env). Если бы
KB осталась включена, эти 21 вопрос отвечались бы из найденной (в целом
не совсем релевантной) карточки базы, а не через поиск — и тест перестал
бы быть честной проверкой именно веб-поиска, о которой просили. Флагом
--kb это можно включить обратно, если нужно ИМЕННО проверить, не течёт
ли база на вопросы, которых там по замыслу быть не должно (полезно само
по себе — см. вывод скрипта sanity-проверки, которым собирался этот
файл).

Модель на каждый вопрос отвечает СВОИМ обычным путём (как и в реальном
звонке): раз KB выключена — either отвечает по своим знаниям, either сама
просит веб-поиск ("ищу в интернете: ..."). Здесь никто не подсказывает
и не заставляет её искать — просто, как и в batch_test_followup.py,
СНАРУЖИ (не трогая live_hints.py) смотрим, каким путём она в итоге
воспользовалась, и пишем это в CSV колонкой "источник_ответа" (search /
model; "kb" тоже возможен, если явно передать --kb).

Вердикт "правильно/неправильно" скрипт сам не ставит — как и остальные
два файла, здесь только сбор данных для ручного просмотра в Excel:
логичность, полнота, соответствие вопросу и — важно именно для этого
файла — не выдумала ли модель facts (например, номер версии), когда
поиск не нашёл ничего конкретного, вместо того чтобы честно сказать, что
не уверена.

Результат пишется в файл сразу по каждой строке (с flush) — Ctrl+C не
теряет уже посчитанные строки, можно продолжить флагом --start.

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test_search.py
    venv\\Scripts\\python.exe src\\batch_test_search.py --start 40 --limit 20
    venv\\Scripts\\python.exe src\\batch_test_search.py --model llama3.1:8b
    venv\\Scripts\\python.exe src\\batch_test_search.py --kb        (проверить утечку в базу)
    venv\\Scripts\\python.exe src\\batch_test_search.py --no-search (сравнить: что ответит БЕЗ поиска)
    venv\\Scripts\\python.exe src\\batch_test_search.py --out results.csv

    # --backend gigachat — та же проверка, но через GigaChat вместо Ollama
    # (см. batch_test.py/batch_test_hard.py для того же приёма). Главное,
    # что тут смотрим ДОПОЛНИТЕЛЬНО к самим ответам — реально ли модель
    # просит "ПОИСК: <запрос>" в нужный момент, а не просто отвечает по
    # памяти на вопрос про актуальные версии/цены/статистику — колонка
    # "источник_ответа" (search/model) и новая "формат_ответа_соблюден"
    # это и показывают.
    venv\\Scripts\\python.exe src\\batch_test_search.py --backend gigachat --model GigaChat-2 --limit 15

Флаги — те же, что у batch_test.py/batch_test_followup.py:
    --backend ollama|gigachat  какой бэкенд гонять (по умолчанию ollama)
    --model <имя>     модель вместо той, что в .env (OLLAMA_MODEL для
                      --backend ollama, GIGACHAT_MODEL для --backend gigachat)
    --kb / --no-kb    база знаний вкл/выкл (по умолчанию ВЫКЛ — см. выше)
    --search / --no-search  веб-поиск вкл/выкл (по умолчанию включён — это и
                      есть то, что тестируем)
    --start N         начать не с 0-го вопроса, а с N-го (докрутка после Ctrl+C)
    --limit N         прогнать только N вопросов начиная со --start
    --out <файл>      путь к CSV (по умолчанию search_test_results_<backend>.csv)
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
QUESTIONS_PATH = ROOT_DIR / "data" / "search_questions.json"


class _Tee(io.TextIOBase):
    """Пишет одновременно в реальный stdout (прогресс по-прежнему виден в
    консоли) и в буфер — из буфера потом достаём признак "модель просила
    ПОИСК:" (см. _answer_source), не трогая generate_hint()."""

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
    """Тот же приём, что и в batch_test_followup.py: определяем СНАРУЖИ,
    чем реально ответила модель — из базы знаний, из веб-поиска или из
    своих знаний без подсказок — не меняя generate_hint()/live_hints.py.

    Дополнительно (для сравнения бэкендов, см. --backend) достаём отсюда же
    признак "модель вообще соблюла формат ОТВЕТ:/КОНТЕКСТ:" — не по самому
    ли поиску, а по тому же самому предупреждению в консоли, которое уже
    печатает live_hints.py (_parse_hint), просто ловим его СНАРУЖИ, а не
    меняем сам live_hints.py."""
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
    format_ok = "не вывела строку ОТВЕТ" not in captured
    return result, source, format_ok


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
    if out_path is None:
        out_path = Path(f"search_test_results_{backend}.csv")
    # По умолчанию (флаг не передан) — ВЫКЛ, даже если в .env KB_ENABLED=true,
    # см. докстринг модуля про 21 случайное пересечение с базой знаний. Но
    # если флаг передан явно (--kb или --no-kb) — он должен пересиливать и
    # умолчание, и .env, как и в batch_test.py/batch_test_followup.py.
    kb_enabled = False if kb_override is None else kb_override
    enable_search = True if search_override is None else search_override

    kb_path_str = str(settings.kb_path) if (kb_enabled and settings.kb_path) else None
    if kb_override is True and not settings.kb_path:
        print("Запрошен --kb, но в .env не задан KB_PATH — база всё равно не будет использована.")

    questions = _load_questions()
    items = questions[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} вопросов не осталось (в файле всего {len(questions)}).")
        sys.exit(0)

    where = settings.ollama_host if backend == "ollama" else "GigaChat API"
    print(f"Бэкенд: {backend}, модель: {model} ({where})")
    print(f"База знаний: {'включена' if kb_enabled else 'выключена (по умолчанию для этого теста)'}")
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
                    "почему_гуглить",
                    "бэкенд",
                    "модель",
                    "вопрос",
                    "ответ_модели",
                    "контекст_модели",
                    "источник_ответа",
                    "формат_ответа_соблюден",
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
            why_search = item.get("why_search", "")
            transcript = f"Собеседник: {question}"

            reply, context, source, error = "", "", "", ""
            format_ok = True

            def _call():
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
                )

            t0 = time.time()
            try:
                (reply, context), source, format_ok = _answer_source(
                    question, kb_enabled, kb_path_str, settings.kb_top_k, settings.kb_min_score, _call
                )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0

            writer.writerow(
                [idx, section, why_search, backend, model, question, reply, context or "-", source,
                 "да" if format_ok else "нет", f"{elapsed:.1f}", error]
            )
            f.flush()

            status = "ОШИБКА" if error else f"ok/{source}"
            print(f"[{offset + 1}/{total}] #{idx} ({elapsed:.1f}с, {status}) {question[:70]}")

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {total} вопрос(ов) за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}")


if __name__ == "__main__":
    main()
