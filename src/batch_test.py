"""
Массовый прогон ВСЕХ вопросов из базы знаний через generate_hint() — чтобы
проверить, как реально отвечает текущая связка (модель + промпт + KB) на
весь датасет разом, а не только на то, что случайно попалось на живом
звонке. В отличие от test_hint.py (одна фраза за раз, для отладки
конкретного бага) — тут автоматически проходим по ВСЕМ 515 вопросам из
data/knowledge_base.json и складываем результат в CSV: вопрос, эталонный
ответ из базы, реальный ответ модели, её "контекст" и время генерации.

Вердикт "правильно/неправильно" эта штука сама НЕ ставит — для открытого
текста автоматическая проверка ненадёжна (см. обсуждение в чате), надёжнее
открыть CSV в Excel и пробежаться глазами по колонкам "эталонный_ответ" и
"ответ_модели" рядом. Зато это на порядок быстрее, чем вручную нажимать
515 вопросов в тренажёре и слушать подсказки одну за одной.

Важно: каждый вопрос прогоняется НЕЗАВИСИМО, без истории разговора и без
"твоих предыдущих ответов" (см. own_answers в live_hints.py) — это проверка
"знает ли модель+КБ ответ на КОНКРЕТНЫЙ вопрос сам по себе", а не проверка
поведения в диалоге (уточнения, продолжение темы и т.п. тут не участвуют).

Результат пишется в файл СРАЗУ по каждой строке (с flush) — если прервать
скрипт на середине (Ctrl+C), уже посчитанные строки не потеряются, и можно
продолжить с того места флагом --start.

Судя по тому, что при KB_ENABLED=true база знаний — это тот же самый
датасет из 515 вопросов, для каждого вопроса search_kb() почти наверняка
найдёт точное совпадение (запрос совпадает с вопросом дословно) — то есть
по умолчанию это тест "правильно ли модель пересказывает найденный в базе
знаний ответ, не обрезая и не перевирая его", а не тест сырых знаний модели
без подсказки. Если нужно именно сырые знания — прогнать с --no-kb (тогда
может пойти веб-поиск на вопросы не про быстро меняющиеся факты почти
никогда не сработает, а вот отвечать модель будет по своим знаниям).

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test.py
    venv\\Scripts\\python.exe src\\batch_test.py --start 200 --limit 50
    venv\\Scripts\\python.exe src\\batch_test.py --model llama3.1:8b
    venv\\Scripts\\python.exe src\\batch_test.py --no-kb --no-search
    venv\\Scripts\\python.exe src\\batch_test.py --out results.csv

    # --backend gigachat — тот же прогон, но через облачную модель вместо
    # Ollama (см. batch_test_hard.py для того же приёма на "сложных"
    # вопросах). Тут — проверка кандидата на роль ОБЫЧНОГО уровня (не Ultra,
    # а модель подешевле/побыстрее, например GigaChat-2) на тех же самых 515
    # реальных вопросах из базы знаний, которыми уже проверяли Ollama —
    # чтобы сравнивать было честно, один и тот же датасет для обоих бэкендов.
    venv\\Scripts\\python.exe src\\batch_test.py --backend gigachat --model GigaChat-2 --limit 30
    venv\\Scripts\\python.exe src\\batch_test.py --backend gigachat --model GigaChat-2

Флаги:
    --backend ollama|gigachat  какой бэкенд гонять (по умолчанию ollama —
                      старое поведение без этого флага не меняется)
    --model <имя>     модель вместо той, что в .env (OLLAMA_MODEL для
                      --backend ollama, GIGACHAT_MODEL для --backend gigachat)
    --kb / --no-kb    включить/выключить базу знаний вместо .env (KB_ENABLED)
    --search / --no-search  включить/выключить веб-поиск (по умолчанию включён,
                      но при --kb он почти не будет использоваться — см. выше)
    --start N         начать не с 0-го вопроса, а с N-го (для докрутки после Ctrl+C)
    --limit N         прогнать только N вопросов начиная со --start (для быстрой проверки)
    --out <файл>      путь к CSV-файлу с результатами (по умолчанию
                      batch_test_results_<backend>.csv)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from config import Settings
from knowledge_base import load_kb
from live_hints import generate_hint, generate_hint_gigachat


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
        out_path = Path(f"batch_test_results_{backend}.csv")

    if not settings.kb_path:
        print("В .env не задан KB_PATH — нечего прогонять (нужна база знаний как источник вопросов).")
        sys.exit(1)

    kb = load_kb(str(settings.kb_path))
    if not kb:
        print(f"Не удалось загрузить базу знаний из {settings.kb_path}")
        sys.exit(1)

    items = kb[start:]
    if limit is not None:
        items = items[:limit]

    if not items:
        print(f"С #{start} вопросов не осталось (в базе всего {len(kb)}).")
        sys.exit(0)

    where = settings.ollama_host if backend == "ollama" else "GigaChat API"
    print(f"Бэкенд: {backend}, модель: {model} ({where})")
    print(f"База знаний: {'включена' if kb_enabled else 'выключена'}")
    print(f"Веб-поиск: {'включён' if enable_search else 'выключен'}")
    print(f"Вопросов к прогону: {len(items)} (начиная с #{start}, всего в базе {len(kb)})")
    print(f"Результат пишется в: {out_path.resolve()}\n")

    # Дописываем в существующий файл, только если явно продолжаем прогон
    # (--start > 0) — иначе (обычный запуск с нуля) начинаем файл заново,
    # чтобы старые результаты от предыдущего прогона не перемешались с новыми.
    append_mode = start > 0 and out_path.exists()
    mode = "a" if append_mode else "w"

    with out_path.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow(
                ["#", "раздел", "вопрос", "эталонный_ответ", "ответ_модели", "контекст_модели",
                 "бэкенд", "модель", "секунд", "ошибка"]
            )

        t_start_all = time.time()
        total = len(items)
        for offset, item in enumerate(items):
            idx = start + offset
            question = item.get("question", "")
            reference = item.get("answer", "")
            section = item.get("section", "")
            transcript = f"Собеседник: {question}"

            reply, context, error = "", "", ""
            t0 = time.time()
            try:
                if backend == "gigachat":
                    reply, context = generate_hint_gigachat(
                        transcript,
                        credentials=settings.gigachat_credentials,
                        model=model,
                        ca_bundle_file=str(settings.gigachat_ca_bundle_file) if settings.gigachat_ca_bundle_file else None,
                        enable_search=enable_search,
                        kb_enabled=kb_enabled,
                        kb_path=str(settings.kb_path) if settings.kb_path else None,
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
                        kb_path=str(settings.kb_path) if settings.kb_path else None,
                        kb_top_k=settings.kb_top_k,
                        kb_min_score=settings.kb_min_score,
                    )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0

            writer.writerow(
                [idx, section, question, reference, reply, context or "-", backend, model, f"{elapsed:.1f}", error]
            )
            f.flush()

            status = "ОШИБКА" if error else "ok"
            print(f"[{offset + 1}/{total}] #{idx} ({elapsed:.1f}с, {status}) {question[:70]}")

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {total} вопрос(ов) за {total_elapsed / 60:.1f} мин.")
        print(f"Результат: {out_path.resolve()}")


if __name__ == "__main__":
    main()
