"""
Быстрая проверка live-подсказок без реального звонка: прогоняет
generate_hint() на вручную заданном хвосте транскрипта. Удобно, чтобы
подбирать промпт, не открывая каждый раз Discord и не дожидаясь VAD.

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\test_hint.py "Собеседник: сколько уровней в модели OSI?"

Можно передать несколько строк транскрипта — каждая отдельным аргументом,
они склеятся через перенос строки (последняя = "текущая" фраза, на неё
и даётся подсказка). Это же удобно, чтобы воспроизвести баг "модель путает
вопросы" — передать СТАРЫЙ вопрос первым аргументом, а НОВЫЙ последним,
именно так, как они шли в реальном разговоре:
    venv\\Scripts\\python.exe src\\test_hint.py "Собеседник: Какая река протекает через Рим?" "Собеседник: Какой химический элемент обозначается символом Ал?"

Без аргументов — использует встроенный пример с OSI.

По умолчанию берётся модель из .env (OLLAMA_MODEL) — но можно сравнить
другую модель одной командой, не трогая .env, флагом --model (в любом
месте среди аргументов, до или после текста транскрипта):
    venv\\Scripts\\python.exe src\\test_hint.py --model llama3.1:8b "Собеседник: сколько уровней в модели OSI?"

Флагом --pending "<текст>" можно проверить сценарий "мы только что задали
уточняющий вопрос, а это — ответ на него" (main.py передаёt это
автоматически в реальном звонке, если предыдущая подсказка заканчивалась
на "?"):
    venv\\Scripts\\python.exe src\\test_hint.py --pending "А расхождение в какой строке — и с чем сверяешь?" "Собеседник: В 15-й строке, сверяю с банковской выпиской"

Флагами --kb / --no-kb можно включить/выключить локальную базу знаний
(KB_ENABLED в .env) для одного конкретного запуска, не трогая .env:
    venv\\Scripts\\python.exe src\\test_hint.py --kb "Собеседник: что такое маржин-колл?"
Флаг --kb-debug печатает, какие записи базы знаний нашлись для текущей
фразы и с каким score — ДО похода к модели, отдельно от самой подсказки:
    venv\\Scripts\\python.exe src\\test_hint.py --kb --kb-debug "Собеседник: что такое маржин-колл?"

Флагом --backend gigachat можно прогнать ту же самую фразу через облачную
GigaChat вместо Ollama (см. GIGACHAT_MODEL/GIGACHAT_CREDENTIALS в .env) —
удобно для точечной проверки одной конкретной фразы, например реплики БЕЗ
вопроса, чтобы убедиться, что модель правильно молчит (ОТВЕТ: -), а не
начинает что-то отвечать там, где отвечать не на что:
    venv\\Scripts\\python.exe src\\test_hint.py --backend gigachat "Собеседник: ага, понятно"
    venv\\Scripts\\python.exe src\\test_hint.py --backend gigachat --model GigaChat-2 "Собеседник: сколько уровней в модели OSI?"
"""
from __future__ import annotations

import sys
import time

from config import Settings
from knowledge_base import format_kb_for_prompt, search_kb
from live_hints import _get_kb, _split_current_line, generate_hint, generate_hint_gigachat


def main() -> None:
    args = sys.argv[1:]
    backend = "ollama"
    model_override: str | None = None
    pending_clarification: str | None = None
    kb_override: bool | None = None
    kb_debug = False
    transcript_args: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
        elif args[i].startswith("--backend="):
            backend = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif args[i].startswith("--model="):
            model_override = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--pending" and i + 1 < len(args):
            pending_clarification = args[i + 1]
            i += 2
        elif args[i].startswith("--pending="):
            pending_clarification = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--kb":
            kb_override = True
            i += 1
        elif args[i] == "--no-kb":
            kb_override = False
            i += 1
        elif args[i] == "--kb-debug":
            kb_debug = True
            i += 1
        else:
            transcript_args.append(args[i])
            i += 1

    if not transcript_args:
        transcript = "Собеседник: сколько уровней в модели OSI?"
        print(f"(аргумент не передан, использую пример по умолчанию)\n")
    else:
        transcript = "\n".join(transcript_args)

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

    print(f"Хвост транскрипта, который уйдёт модели:\n---\n{transcript}\n---\n")
    if pending_clarification:
        print(f"Твой предыдущий уточняющий вопрос: {pending_clarification!r}\n")

    if kb_enabled and settings.kb_path:
        if kb_debug:
            kb = _get_kb(str(settings.kb_path))
            _, current_line = _split_current_line(transcript)
            matches = search_kb(current_line, kb, top_k=settings.kb_top_k, min_score=settings.kb_min_score)
            print(f"[kb-debug] запрос: {current_line!r}")
            if matches:
                print(f"[kb-debug] найдено {len(matches)} совпадени(й):\n{format_kb_for_prompt(matches)}\n")
            else:
                print("[kb-debug] совпадений не найдено\n")
    elif kb_enabled and not settings.kb_path:
        print("[kb] KB_ENABLED=true, но KB_PATH не задан в .env — база знаний не используется\n")

    where = settings.ollama_host if backend == "ollama" else "GigaChat API"
    print(f"Бэкенд: {backend}, модель: {model} ({where})" + (" [--model, не из .env]" if model_override else ""))
    print(f"База знаний: {'включена' if kb_enabled else 'выключена'}" + (" [--kb/--no-kb]" if kb_override is not None else ""))
    print("Генерирую подсказку...\n")

    t0 = time.time()
    if backend == "gigachat":
        reply, context = generate_hint_gigachat(
            transcript,
            credentials=settings.gigachat_credentials,
            model=model,
            ca_bundle_file=str(settings.gigachat_ca_bundle_file) if settings.gigachat_ca_bundle_file else None,
            pending_clarification=pending_clarification,
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
            pending_clarification=pending_clarification,
            kb_enabled=kb_enabled,
            kb_path=str(settings.kb_path) if settings.kb_path else None,
            kb_top_k=settings.kb_top_k,
            kb_min_score=settings.kb_min_score,
        )
    elapsed = time.time() - t0

    print(f"ОТВЕТ:    {reply}")
    print(f"КОНТЕКСТ: {context or '-'}")
    print(f"\n(заняло {elapsed:.1f} сек)")


if __name__ == "__main__":
    main()
