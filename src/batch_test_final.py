"""
ФИНАЛЬНЫЙ комбинированный прогон — не заменяет остальные 4 batch_test_*.py
(все они остаются рабочими сами по себе), а ОБЪЕДИНЯЕТ все 4 типа вопросов
в один большой перемешанный прогон, чтобы одним разом (на ночь, без
присмотра) проверить модель во всех режимах сразу, а не по отдельности:

  - kb          — вопросы ИЗ базы знаний (data/knowledge_base.json, 515 шт.) —
                  проверяет, правильно ли модель пересказывает найденную
                  карточку, не обрезая и не перевирая её (как batch_test.py).
  - context     — двухходовые вопрос+уточнение ИЗ базы
                  (data/knowledge_base_followups.json, 515 шт.) — проверяет
                  поведение на ВТОРОМ вопросе с историей и own_answers, где
                  чаще всего вылезают баги с местоимениями ("расскажи о них
                  подробнее") — как batch_test_followup.py.
  - search      — вопросы, требующие актуального факта из веб-поиска
                  (data/search_questions.json, ~499 шт., до расширения было
                  100 — большая часть сгенерирована отдельно под этот
                  финальный тест) — как batch_test_search.py.
  - situational — ситуационные вопросы без единственного правильного ответа,
                  где верное поведение — уточняющий вопрос, а не выдуманная
                  конкретика или общий совет (data/situational_questions.json,
                  500 шт., до расширения было 24) — как batch_test_situational.py.

ВАЖНОЕ ОТЛИЧИЕ от search/situational-тестов по отдельности: там KB по
умолчанию либо выключена (batch_test_search.py — чтобы не путать поиск с
попаданием в базу), либо берётся из .env. Здесь — по явной договорённости
(финальный тест должен имитировать РЕАЛЬНЫЕ условия звонка) — база знаний
включена ПРИНУДИТЕЛЬНО для всех 4 типов вопросов, включая search и
situational. Если на какой-то search-вопрос вдруг найдётся (обычно
случайное, по паре общих слов) совпадение в базе — это не баг теста, а
ожидаемое поведение реального звонка: KB всегда проверяется первой, поиск и
модель — это то, что происходит, когда карточки в базе не нашлось. Колонка
"источник_ответа_1"/"источник_ответа_2" в CSV в любом случае показывает,
что реально сработало на каждый конкретный вопрос — так и видно, где
произошло случайное попадание в базу, если оно есть. Отключить это
поведение (заставить KB молчать на search/situational, как раньше) можно
флагом --no-kb, если понадобится сравнить с изолированным тестом.

Порядок вопросов — ОДИН случайный, перемешивающий вместе kb / search /
situational как одиночные единицы, но context — ВСЕГДА как единый,
неразрывный двухходовый блок (два похода к Ollama внутри одной "единицы"
прогона, а не две отдельные строки транскрипта, которые могли бы
разъехаться при перемешивании). Порядок детерминирован фиксированным
seed'ом (--seed, по умолчанию 1337) — то есть при перезапуске с теми же
исходными файлами данных получится ТОТ ЖЕ порядок, а значит --start/--limit
для докрутки после Ctrl+C работают предсказуемо.

Это МЕДЛЕННЫЙ прогон — единиц много (515 kb + 515 context + ~499 search +
500 situational ≈ 2029 единиц), причём context — это 2 похода к Ollama на
единицу, так что реальных обращений к Ollama получается около 2500+.
Рассчитан на прогон "на ночь без присмотра" — при ~3 секундах на ответ это
примерно 2-3 часа, но конкретное время зависит от машины и модели. Пишем в
CSV сразу по каждой единице (с flush) — Ctrl+C не теряет уже посчитанное,
можно продолжить флагом --start (тот же порядок благодаря фиксированному
seed).

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test_final.py
    venv\\Scripts\\python.exe src\\batch_test_final.py --start 500 --limit 100
    venv\\Scripts\\python.exe src\\batch_test_final.py --model llama3.1:8b
    venv\\Scripts\\python.exe src\\batch_test_final.py --no-kb
    venv\\Scripts\\python.exe src\\batch_test_final.py --seed 7 --out night_run.csv

Флаги:
    --model <имя>     модель Ollama вместо той, что в .env (OLLAMA_MODEL)
    --kb / --no-kb    база знаний вкл/выкл (по умолчанию ПРИНУДИТЕЛЬНО ВКЛ
                      для всех 4 типов вопросов — см. выше; --no-kb выключает
                      её везде, включая kb/context-вопросы, для которых это
                      делает тест бессмысленным, но флаг оставлен для
                      симметрии с остальными batch_test_*.py)
    --search / --no-search  веб-поиск вкл/выкл (по умолчанию включён)
    --seed N          seed для перемешивания (по умолчанию 1337, для
                      воспроизводимого порядка между запусками)
    --start N         начать не с 0-й единицы, а с N-й (докрутка после Ctrl+C)
    --limit N         прогнать только N единиц начиная со --start
    --out <файл>      путь к CSV (по умолчанию final_test_results.csv)
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from config import Settings
from knowledge_base import search_kb
from live_hints import _CONTINUATION_KB_MIN_SCORE, _SPEAKER_LABEL_RE, _get_kb, generate_hint

ROOT_DIR = Path(__file__).resolve().parent.parent
KB_PATH = ROOT_DIR / "data" / "knowledge_base.json"
FOLLOWUPS_PATH = ROOT_DIR / "data" / "knowledge_base_followups.json"
SEARCH_PATH = ROOT_DIR / "data" / "search_questions.json"
SITUATIONAL_PATH = ROOT_DIR / "data" / "situational_questions.json"

DEFAULT_SEED = 1337

CSV_COLUMNS = [
    "#",
    "тип",
    "раздел",
    "вопрос_1",
    "эталонный_ответ_1",
    "ответ_модели_1",
    "контекст_модели_1",
    "источник_ответа_1",
    "секунд_1",
    "вопрос_2",
    "категория_вопроса_2",
    "ответ_модели_2",
    "контекст_модели_2",
    "источник_ответа_2",
    "секунд_2",
    "почему_гуглить",
    "ожидаемый_тип_ситуационный",
    "заметка_ситуационная",
    "похоже_на_уточнение",
    "неожиданный_поиск",
    "диагностика_пустого_ответа",
    "ошибка",
]


class _Tee(io.TextIOBase):
    """Тот же приём, что и в остальных batch_test_*.py: пишем и в реальный
    stdout (прогресс виден в консоли), и в буфер — из буфера потом достаём
    признак "модель просила ПОИСК:", не трогая generate_hint()."""

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
    """Определяет СНАРУЖИ источник ответа (kb / search / model), не меняя
    generate_hint()/live_hints.py — см. подробное объяснение в
    batch_test_followup.py, откуда этот приём (и own_answers_present —
    поднятие порога совпадения с базой на втором, уточняющем вопросе)
    перенесён без изменений."""
    kb_hit = False
    if kb_enabled and kb_path:
        try:
            kb = _get_kb(kb_path)
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

    diagnostic = ""
    reply = result[0] if isinstance(result, tuple) and result else None
    if not reply:
        diag_lines = [ln for ln in captured.splitlines() if ln.startswith("[hints] модель")]
        diagnostic = diag_lines[-1] if diag_lines else ""

    return result, source, diagnostic


def _load_json(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        print(f"Не найден {path} ({label}) — нечего прогонять.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _build_units(seed: int) -> list[dict[str, Any]]:
    """Собирает все 4 типа вопросов в один список "единиц" прогона и
    перемешивает их вместе одним случайным порядком (фиксированный seed —
    для воспроизводимости между запусками), КРОМЕ context-единиц: каждая
    из них — это уже целиком одна единица (вопрос+уточнение с историей
    внутри), она никогда не разбивается на два отдельных элемента списка,
    так что после перемешивания вопрос и уточнение физически не могут
    оказаться не рядом — они и так одна запись."""
    kb_items = _load_json(KB_PATH, "база знаний")
    followup_items = _load_json(FOLLOWUPS_PATH, "база знаний с уточнениями")
    search_items = _load_json(SEARCH_PATH, "вопросы для поиска")
    situational_items = _load_json(SITUATIONAL_PATH, "ситуационные вопросы")

    units: list[dict[str, Any]] = []
    for item in kb_items:
        units.append({"type": "kb", "data": item})
    for item in followup_items:
        units.append({"type": "context", "data": item})
    for item in search_items:
        units.append({"type": "search", "data": item})
    for item in situational_items:
        units.append({"type": "situational", "data": item})

    rng = random.Random(seed)
    rng.shuffle(units)

    print(
        f"Единиц прогона: {len(units)} "
        f"(kb={len(kb_items)}, context={len(followup_items)}, "
        f"search={len(search_items)}, situational={len(situational_items)}), "
        f"порядок перемешан с seed={seed}"
    )
    return units


def _run_single(
    question: str,
    model: str,
    settings: Settings,
    kb_enabled: bool,
    kb_path_str: str | None,
    enable_search: bool,
) -> tuple[str, str, str, str, float, str]:
    """Один обычный (без истории) поход к generate_hint() — используется для
    kb / search / situational. Возвращает
    (ответ, контекст, источник, диагностика_пустого_ответа, секунды, ошибка)."""
    transcript = f"Собеседник: {question}"
    reply, context, source, diagnostic, error = "", "", "", "", ""
    t0 = time.time()
    try:
        (reply, context), source, diagnostic = _answer_source(
            question,
            kb_enabled,
            kb_path_str,
            settings.kb_top_k,
            settings.kb_min_score,
            lambda: generate_hint(
                transcript,
                model=model,
                host=settings.ollama_host,
                enable_search=enable_search,
                kb_enabled=kb_enabled,
                kb_path=kb_path_str,
                kb_top_k=settings.kb_top_k,
                kb_min_score=settings.kb_min_score,
            ),
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    return reply, context or "-", source, diagnostic, elapsed, error


def main() -> None:
    args = sys.argv[1:]
    model_override: str | None = None
    kb_override: bool | None = None
    search_override: bool | None = None
    seed = DEFAULT_SEED
    start = 0
    limit: int | None = None
    out_path = Path("final_test_results.csv")

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--model" and i + 1 < len(args):
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
        elif a == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif a.startswith("--seed="):
            seed = int(a.split("=", 1)[1])
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

    settings = Settings.load()
    model = model_override or settings.ollama_model
    # По умолчанию (флаг не передан) — ПРИНУДИТЕЛЬНО ВКЛ для всех 4 типов
    # вопросов, независимо от KB_ENABLED в .env — см. докстринг модуля:
    # финальный тест должен имитировать реальные условия звонка, где база
    # знаний всегда проверяется первой. --no-kb выключает её везде явно.
    kb_enabled = True if kb_override is None else kb_override
    enable_search = True if search_override is None else search_override
    kb_path_str = str(settings.kb_path) if (kb_enabled and settings.kb_path) else None
    if kb_enabled and not settings.kb_path:
        print("KB запрошена, но в .env не задан KB_PATH — база всё равно не будет использована.")
        kb_enabled = False

    units = _build_units(seed)
    items = units[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} единиц не осталось (всего {len(units)}).")
        sys.exit(0)

    print(f"Модель: {model} ({settings.ollama_host})")
    print(f"База знаний: {'включена (принудительно для всех типов)' if kb_enabled else 'выключена'}")
    print(f"Веб-поиск: {'включён' if enable_search else 'выключен'}")
    print(f"Единиц к прогону: {len(items)} (начиная с #{start}, всего {len(units)})")
    print(f"Результат пишется в: {out_path.resolve()}\n")

    append_mode = start > 0 and out_path.exists()
    mode = "a" if append_mode else "w"

    with out_path.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow(CSV_COLUMNS)

        t_start_all = time.time()
        total = len(items)
        for offset, unit in enumerate(items):
            idx = start + offset
            utype = unit["type"]
            data = unit["data"]

            row = {col: "" for col in CSV_COLUMNS}
            row["#"] = idx
            row["тип"] = utype

            if utype == "kb":
                question = data.get("question", "")
                row["раздел"] = data.get("section", "")
                row["вопрос_1"] = question
                row["эталонный_ответ_1"] = data.get("answer", "")
                reply, context, source, diag, elapsed, error = _run_single(
                    question, model, settings, kb_enabled, kb_path_str, enable_search
                )
                row["ответ_модели_1"] = reply
                row["контекст_модели_1"] = context
                row["источник_ответа_1"] = source
                row["секунд_1"] = f"{elapsed:.1f}"
                row["диагностика_пустого_ответа"] = diag
                row["ошибка"] = error
                preview = question

            elif utype == "search":
                question = data.get("question", "")
                row["раздел"] = data.get("section", "")
                row["почему_гуглить"] = data.get("why_search", "")
                row["вопрос_1"] = question
                reply, context, source, diag, elapsed, error = _run_single(
                    question, model, settings, kb_enabled, kb_path_str, enable_search
                )
                row["ответ_модели_1"] = reply
                row["контекст_модели_1"] = context
                row["источник_ответа_1"] = source
                row["секунд_1"] = f"{elapsed:.1f}"
                row["диагностика_пустого_ответа"] = diag
                row["ошибка"] = error
                preview = question

            elif utype == "situational":
                question = data.get("question", "")
                row["раздел"] = data.get("section", "")
                row["ожидаемый_тип_ситуационный"] = data.get("ожидаемый_тип", "")
                row["заметка_ситуационная"] = data.get("заметка", "")
                row["вопрос_1"] = question
                reply, context, source, diag, elapsed, error = _run_single(
                    question, model, settings, kb_enabled, kb_path_str, enable_search
                )
                row["ответ_модели_1"] = reply
                row["контекст_модели_1"] = context
                row["источник_ответа_1"] = source
                row["секунд_1"] = f"{elapsed:.1f}"
                row["диагностика_пустого_ответа"] = diag
                row["ошибка"] = error
                row["похоже_на_уточнение"] = "да" if reply.rstrip().endswith("?") else "нет"
                row["неожиданный_поиск"] = "ДА" if source == "search" else ""
                preview = question

            elif utype == "context":
                question1 = data.get("question", "")
                reference1 = data.get("answer", "")
                section = data.get("section", "")
                question2 = data.get("followup", "")
                category2 = data.get("followup_category", "")
                row["раздел"] = section
                row["вопрос_1"] = question1
                row["эталонный_ответ_1"] = reference1
                row["вопрос_2"] = question2
                row["категория_вопроса_2"] = category2

                reply1 = reply2 = ""
                source1 = source2 = ""
                diag1 = diag2 = ""
                elapsed1 = elapsed2 = None
                error = ""
                try:
                    t0 = time.time()
                    (reply1, ctx1), source1, diag1 = _answer_source(
                        question1,
                        kb_enabled,
                        kb_path_str,
                        settings.kb_top_k,
                        settings.kb_min_score,
                        lambda: generate_hint(
                            f"Собеседник: {question1}",
                            model=model,
                            host=settings.ollama_host,
                            enable_search=enable_search,
                            kb_enabled=kb_enabled,
                            kb_path=kb_path_str,
                            kb_top_k=settings.kb_top_k,
                            kb_min_score=settings.kb_min_score,
                        ),
                        own_answers_present=False,
                    )
                    elapsed1 = time.time() - t0

                    transcript = f"Собеседник: {question1}\nСобеседник: {question2}"
                    t0 = time.time()
                    (reply2, ctx2), source2, diag2 = _answer_source(
                        question2,
                        kb_enabled,
                        kb_path_str,
                        settings.kb_top_k,
                        settings.kb_min_score,
                        lambda: generate_hint(
                            transcript,
                            model=model,
                            host=settings.ollama_host,
                            enable_search=enable_search,
                            kb_enabled=kb_enabled,
                            kb_path=kb_path_str,
                            kb_top_k=settings.kb_top_k,
                            kb_min_score=settings.kb_min_score,
                            own_answers=[reply1] if reply1 else None,
                        ),
                        own_answers_present=bool(reply1),
                    )
                    elapsed2 = time.time() - t0
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    ctx1 = ctx2 = ""

                row["ответ_модели_1"] = reply1
                row["контекст_модели_1"] = ctx1 or "-"
                row["источник_ответа_1"] = source1
                row["секунд_1"] = f"{elapsed1:.1f}" if elapsed1 is not None else ""
                row["ответ_модели_2"] = reply2
                row["контекст_модели_2"] = ctx2 or "-"
                row["источник_ответа_2"] = source2
                row["секунд_2"] = f"{elapsed2:.1f}" if elapsed2 is not None else ""
                row["диагностика_пустого_ответа"] = " | ".join(d for d in (diag1, diag2) if d)
                row["ошибка"] = error
                preview = question1

            else:
                raise AssertionError(f"неизвестный тип единицы: {utype!r}")

            writer.writerow([row[col] for col in CSV_COLUMNS])
            f.flush()

            secs = row["секунд_1"] or "-"
            if utype == "context":
                secs = f"{row['секунд_1'] or '-'}+{row['секунд_2'] or '-'}"
            status = "ОШИБКА" if row["ошибка"] else "ok"
            print(f"[{offset + 1}/{total}] #{idx} [{utype}] ({secs}с, {status}) {preview[:60]}")

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {total} единиц(ы) за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}")


if __name__ == "__main__":
    main()
