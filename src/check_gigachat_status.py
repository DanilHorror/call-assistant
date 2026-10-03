"""
Диагностика подключения к GigaChat API (Сбер) — Этап 3, до того как заводить
эту модель в живой пайплайн подсказок (live_hints.py).

Четыре вещи, которые тут проверяются по очереди:
1. Ключ (GIGACHAT_CREDENTIALS) и подключение вообще работают — можно
   авторизоваться и получить список моделей, доступных именно вашему
   аккаунту/тарифу.
2. Модель, указанная в GIGACHAT_MODEL (.env), реально есть в этом списке —
   отдельная и частая ошибка: вписать в .env модель, которой нет на вашем
   тарифе, и потом гадать, почему запросы падают.
3. Обычный текстовый запрос реально доходит и возвращает ответ — то есть вся
   цепочка (ключ -> access token -> сам чат) работает целиком, а не только
   авторизация.
4. Сама функция generate_hint_gigachat() (тот же код, который пойдёт в живой
   пайплайн подсказок) даёт вменяемые ответы на паре сложных вопросов —
   алгоритмическую задачу и решаемый ситуационный вопрос.

Похоже по духу на check_duckduckgo_status.py — это не часть живого пайплайна,
отдельная одноразовая проверка "всё ли готово", результат которой смотрит
человек глазами перед тем, как переходить к следующему шагу.

Запуск (из корня проекта, при активном venv):
    venv\\Scripts\\python.exe src\\check_gigachat_status.py
"""
from __future__ import annotations

import sys

from config import Settings


def main() -> None:
    settings = Settings.load()

    if not settings.gigachat_credentials:
        print("[ошибка] GIGACHAT_CREDENTIALS не задан в .env.")
        print("         Возьмите Authorization key в личном кабинете developers.sber.ru:")
        print("         GigaChat API -> ваш проект (тип \"физическое лицо\") ->")
        print("         \"Авторизационные данные\" -> Сгенерировать ключ.")
        sys.exit(1)

    try:
        from gigachat import GigaChat
    except ImportError:
        print("[ошибка] Пакет gigachat не установлен. Запустите:")
        print("         venv\\Scripts\\pip.exe install -r requirements.txt")
        sys.exit(1)

    print("Подключаюсь к GigaChat...")
    if settings.gigachat_ca_bundle_file:
        print(f"      (использую сертификат: {settings.gigachat_ca_bundle_file})")
    try:
        client = GigaChat(
            credentials=settings.gigachat_credentials,
            model=settings.gigachat_model,
            ca_bundle_file=str(settings.gigachat_ca_bundle_file) if settings.gigachat_ca_bundle_file else None,
        )
    except Exception as e:
        print(f"[ошибка] Не удалось создать клиента: {e}")
        sys.exit(1)

    # --- Шаг 1: список моделей, реально доступных этому ключу/тарифу ---
    print("\n[1/4] Запрашиваю список доступных моделей...")
    model_ids: list[str] = []
    try:
        models = client.get_models()
        model_ids = [m.id_ for m in models.data] if hasattr(models, "data") else [str(m) for m in models]
        if model_ids:
            print(f"      Доступно: {', '.join(model_ids)}")
        else:
            print("      Список пуст (или неожиданный формат ответа) — см. вывод целиком выше.")
    except Exception as e:
        # SSL-ошибка тут — самая частая проблема на новом окружении: GigaChat
        # требует доверенный корневой сертификат Минцифры. Если увидите
        # SSLCertVerificationError — сначала попробуйте установить сертификат
        # (см. README GigaChat API, "Russian Trusted CA"), и только для этой
        # разовой диагностики, НЕ для боевого кода, можно временно проверить
        # verify_ssl_certs=False, чтобы отличить проблему с сертификатом от
        # проблемы с самим ключом.
        print(f"      [ошибка] {e}")
        if "SSL" in str(e) or "CERTIFICATE" in str(e).upper():
            print(
                "      Похоже на проблему с SSL-сертификатом — GigaChat подписан "
                "сертификатом Минцифры России, которого нет в стандартном доверенном "
                "списке. Инструкция и команда для скачивания — в .env.example, "
                "рядом с GIGACHAT_CA_BUNDLE_FILE."
            )

    # --- Шаг 2: модель из .env реально в списке? ---
    if model_ids and settings.gigachat_model not in model_ids:
        print(
            f"\n[внимание] GIGACHAT_MODEL={settings.gigachat_model!r} нет в списке "
            f"доступных моделей выше — впишите в .env одну из них."
        )

    # --- Шаг 3: реальный запрос ---
    print(f"\n[2/4] Пробую отправить тестовый запрос модели {settings.gigachat_model}...")
    try:
        response = client.chat.create("Ответь одним коротким словом: ты на связи?")
        reply = response.messages[0].content[0].text
        print(f"      Ответ: {reply!r}")
    except Exception as e:
        print(f"      [ошибка] {e}")
        print("\nЕсли выше не было проблем с сертификатом — проверьте сам ключ и то, что")
        print("на вашем тарифе включена именно модель из GIGACHAT_MODEL.")
        sys.exit(1)

    print("\n[3/4] Готово — ключ, модель и сам запрос отработали.")

    # --- Шаг 4: та же самая функция генерации подсказки, что пойдёт в живой
    # пайплайн (generate_hint_gigachat в live_hints.py) — с реальным
    # промптом (_CORE_RULES) на паре примеров "сложных" вопросов, ради
    # которых вообще затевался GigaChat. Смотрите глазами, что качество
    # ответа устраивает, прежде чем подключать это к main.py.
    print("\n[4/4] Пробую generate_hint_gigachat() на паре сложных вопросов...")
    from live_hints import generate_hint_gigachat

    test_questions = [
        "Собеседник: напиши функцию, которая проверяет, является ли строка палиндромом",
        "Собеседник: как ты поступишь, если увидишь на графике, что трафик на сервисе "
        "авторизации просел на 40% сразу после последнего деплоя в 14:00?",
    ]
    for i, transcript in enumerate(test_questions, 1):
        print(f"\n      Вопрос {i}: {transcript.split(': ', 1)[1]}")
        try:
            reply, context = generate_hint_gigachat(
                transcript,
                credentials=settings.gigachat_credentials,
                model=settings.gigachat_model,
                ca_bundle_file=str(settings.gigachat_ca_bundle_file) if settings.gigachat_ca_bundle_file else None,
            )
            print(f"      Ответ:   {reply!r}")
            print(f"      Контекст: {context!r}")
        except Exception as e:
            print(f"      [ошибка] {e}")

    print("\nГотово — если ответы выше по делу и без выдумок, можно переходить к следующему шагу.")


if __name__ == "__main__":
    main()
