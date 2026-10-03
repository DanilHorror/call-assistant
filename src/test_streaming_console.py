"""
Быстрая проверка стриминга БЕЗ оверлея и без звонка — напрямую дёргает
generate_hint() с on_partial, печатает растущий ответ прямо в консоль. Если
тут всё плавно и без ошибок — значит серверная часть (live_hints.py) точно
работает с реальной Ollama, и проблема (если будет) точно в oверлее
(main.py/overlay.py), а не в парсинге стриминга.

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\test_streaming_console.py
    venv\\Scripts\\python.exe src\\test_streaming_console.py "свой вопрос"
"""
from __future__ import annotations

import sys
import time

from config import Settings
from live_hints import generate_hint


def main() -> None:
    settings = Settings.load()
    question = sys.argv[1] if len(sys.argv) > 1 else "Что такое SOLID?"
    transcript = f"Собеседник: {question}"

    print(f"Модель: {settings.ollama_model} ({settings.ollama_host})")
    print(f"Вопрос: {question}\n")

    def on_partial(reply: str, context: str) -> None:
        # \r + очистка строки, чтобы было видно "живой" растущий текст на
        # одной строке, а не портянку повторов.
        line = f"[печатает] {reply}"
        if context:
            line += f"  |  контекст: {context}"
        print("\r" + " " * 160 + "\r" + line[:155], end="", flush=True)

    t0 = time.time()
    reply, context = generate_hint(
        transcript,
        model=settings.ollama_model,
        host=settings.ollama_host,
        kb_enabled=settings.kb_enabled,
        kb_path=str(settings.kb_path) if settings.kb_path else None,
        kb_top_k=settings.kb_top_k,
        kb_min_score=settings.kb_min_score,
        on_partial=on_partial,
    )
    elapsed = time.time() - t0

    print("\n\n--- ГОТОВО ---")
    print(f"Ответ ({elapsed:.1f}с): {reply}")
    print(f"Контекст: {context or '-'}")


if __name__ == "__main__":
    main()
