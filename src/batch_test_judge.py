"""
Массовый прогон судьи-классификатора (см. src/judge.py) по набору заранее
написанных каверзных случаев — data/judge_test_cases.json.

В отличие от batch_test.py/batch_test_followup.py (проверяют качество
ИТОГОВОГО ответа на вопрос), этот скрипт проверяет промежуточный шаг новой
архитектуры: правильно ли судья определяет тип реплики (новый вопрос /
продолжение / мусор), правильно ли выбирает continues_tags при неоднозначной
отсылке назад (три случая — конкретика в фразе / recency при равнозначных
кандидатах / честное "не знаю, уточни" при реально разных кандидатах — см.
судья.py и проектный документ "call-assistant: архитектура контекста для
live-подсказок") — и НЕ придумывает синонимы для уже существующих тегов.

Судья пока НЕ подключён к живому пути (main.py) — это ПЕРВЫЙ шаг перед
подключением: сначала убеждаемся, что промт ведёт себя предсказуемо на
заранее известных ловушках, и только потом встраиваем в process_loop()/
hint_worker(). Каждый тест-кейс в JSON уже несёт "known_tags" (эмулирует то,
что в реальном звонке накопилось бы по ходу разговора) — реального хранилища
тегов ещё не существует, оно появится отдельным шагом после этого.

В отличие от batch_test.py, тут ЕСТЬ автоматический вердикт "правильно/
неправильно" по каждому кейсу — задача судьи достаточно структурирована
(конечный набор полей: verdict/continues_tags/ambiguous), в отличие от
открытого текста ответа, который в batch_test.py сознательно не оценивается
автоматически.

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test_judge.py
    venv\\Scripts\\python.exe src\\batch_test_judge.py --model qwen2.5:7b
    venv\\Scripts\\python.exe src\\batch_test_judge.py --start 3 --limit 2
    venv\\Scripts\\python.exe src\\batch_test_judge.py --out judge_results.csv

Флаги:
    --model <имя>   модель вместо OLLAMA_MODEL из .env (напоминание: решение —
                     судья использует ТУ ЖЕ модель, что и обычный fallback-
                     генератор ответов, не отдельную маленькую модель)
    --host <url>    Ollama host вместо OLLAMA_HOST из .env
    --num-ctx N     контекстное окно вместо JUDGE_NUM_CTX из .env (сравнить,
                    например, реальное поведение на 2048 против 3072)
    --start N       начать не с 0-го кейса, а с N-го
    --limit N       прогнать только N кейсов начиная со --start
    --out <файл>    CSV с результатами (по умолчанию judge_test_results.csv)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import requests

from config import Settings
from judge import KnownTag, classify_utterance

ROOT_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT_DIR / "data" / "judge_test_cases.json"


def _load_cases() -> list[dict]:
    if not CASES_PATH.exists():
        print(f"[ошибка] не найден файл с тест-кейсами: {CASES_PATH}")
        sys.exit(1)
    with CASES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _known_tags_from_case(case: dict) -> list[KnownTag]:
    return [
        KnownTag(tag=kt["tag"], summary=kt["summary"], turns_ago=kt["turns_ago"])
        for kt in case.get("known_tags", [])
    ]


def _check_case(case: dict, verdict_obj) -> tuple[bool, list[str]]:
    """Сравнивает фактический вердикт судьи с ожиданиями из кейса.
    Возвращает (прошёл_ли, список_причин_несовпадения) — причины пишем даже
    для прошедших кейсов пустым списком, чтобы вывод был единообразным."""
    problems: list[str] = []

    if verdict_obj.parse_error:
        return False, [f"не удалось разобрать JSON: {verdict_obj.parse_error}"]

    expected_verdict = case.get("expected_verdict", "")
    if verdict_obj.verdict != expected_verdict:
        problems.append(f"verdict: ожидали {expected_verdict!r}, получили {verdict_obj.verdict!r}")

    expected_ambiguous = case.get("expected_ambiguous", False)
    if verdict_obj.ambiguous != expected_ambiguous:
        problems.append(f"ambiguous: ожидали {expected_ambiguous}, получили {verdict_obj.ambiguous}")

    # continues_tags сравниваем только когда ожидание задано явно и разговор
    # не помечен как ambiguous (в ambiguous-кейсах continues_tags должен быть
    # пустым по определению — это уже проверяется полем ambiguous выше).
    if "expected_continues_tags" in case and not expected_ambiguous:
        expected_set = set(case["expected_continues_tags"])
        actual_set = set(verdict_obj.continues_tags)
        if expected_set != actual_set:
            problems.append(
                f"continues_tags: ожидали {sorted(expected_set)}, получили {sorted(actual_set)}"
            )

    # tags — мягкая проверка (список тегов для НОВОЙ темы не продиктован
    # заранее, судья вправе сформулировать его по-своему) — достаточно, чтобы
    # хотя бы одна ожидаемая подстрока встретилась хоть в одном из тегов.
    if "expected_tags_contains_any" in case:
        expected_any = [s.lower() for s in case["expected_tags_contains_any"]]
        actual_tags_lower = [t.lower() for t in verdict_obj.tags]
        if not any(exp in tag for exp in expected_any for tag in actual_tags_lower):
            problems.append(
                f"tags: ожидали хоть одно из {case['expected_tags_contains_any']}, получили {verdict_obj.tags}"
            )

    return (len(problems) == 0), problems


def main() -> None:
    args = sys.argv[1:]
    model_override: str | None = None
    host_override: str | None = None
    num_ctx_override: int | None = None
    start = 0
    limit: int | None = None
    out_path: Path | None = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif a.startswith("--model="):
            model_override = a.split("=", 1)[1]
            i += 1
        elif a == "--host" and i + 1 < len(args):
            host_override = args[i + 1]
            i += 2
        elif a.startswith("--host="):
            host_override = a.split("=", 1)[1]
            i += 1
        elif a == "--num-ctx" and i + 1 < len(args):
            num_ctx_override = int(args[i + 1])
            i += 2
        elif a.startswith("--num-ctx="):
            num_ctx_override = int(a.split("=", 1)[1])
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
    host = host_override or settings.ollama_host
    num_ctx = num_ctx_override if num_ctx_override is not None else settings.judge_num_ctx
    if out_path is None:
        out_path = Path("judge_test_results.csv")

    cases = _load_cases()
    items = cases[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} кейсов не осталось (всего в файле {len(cases)}).")
        sys.exit(0)

    print(f"Модель: {model} ({host}), num_ctx={num_ctx}")
    print(f"Кейсов к прогону: {len(items)} (начиная с #{start}, всего в файле {len(cases)})")
    print(f"Результат пишется в: {out_path.resolve()}\n")

    session = requests.Session()
    passed_count = 0

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["#", "id", "статус", "verdict", "continues_tags", "ambiguous", "tags",
             "проблемы", "секунд", "сырой_ответ_модели"]
        )

        t_start_all = time.time()
        total = len(items)
        for offset, case in enumerate(items):
            idx = start + offset
            known_tags = _known_tags_from_case(case)

            t0 = time.time()
            try:
                verdict_obj = classify_utterance(
                    current_line=case["current_line"],
                    history=case.get("history", ""),
                    known_tags=known_tags,
                    model=model,
                    host=host,
                    timeout=30.0,
                    session=session,
                    num_ctx=num_ctx,
                )
                ok, problems = _check_case(case, verdict_obj)
                verdict_text = verdict_obj.verdict
                continues_text = ", ".join(verdict_obj.continues_tags)
                ambiguous_text = str(verdict_obj.ambiguous)
                tags_text = ", ".join(verdict_obj.tags)
                raw_text = verdict_obj.raw
            except Exception as e:
                ok = False
                problems = [f"{type(e).__name__}: {e}"]
                verdict_text = continues_text = ambiguous_text = tags_text = raw_text = ""
            elapsed = time.time() - t0

            if ok:
                passed_count += 1
            status = "OK" if ok else "ПРОВАЛ"

            writer.writerow(
                [idx, case.get("id", ""), status, verdict_text, continues_text, ambiguous_text,
                 tags_text, " | ".join(problems), f"{elapsed:.1f}", raw_text]
            )
            f.flush()

            print(f"[{offset + 1}/{total}] #{idx} {case.get('id', ''):40s} {status} ({elapsed:.1f}с)")
            for p in problems:
                print(f"    - {p}")

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {passed_count}/{total} прошли за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}")


if __name__ == "__main__":
    main()
