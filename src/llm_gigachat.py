"""
Облачная модель GigaChat (Сбер) — Этап 3.

Это транспортный слой, аналог _call_ollama / _call_ollama_stream /
_call_ollama_retrying из live_hints.py, но для GigaChat вместо локальной
Ollama. Специально вынесен в отдельный файл, а не встроен прямо в
live_hints.py — по тому же принципу, что stt_hybrid.py не знает деталей
faster-whisper/Deepgram, а просто вызывает stt/local_whisper.py и
stt/cloud_deepgram.py. Здесь то же самое: live_hints.py вызывает
call_gigachat_retrying() и не знает, что внутри пакет gigachat, OAuth,
сертификат Минцифры и т.п.

ВАЖНО: это НЕ замена локальной модели. main.py решает (см. _looks_like_hard_question)
для КАКИХ вопросов вообще стоит сюда обращаться — обычные вопросы всё так же
идут через generate_hint()/Ollama, это второй бэкенд для узкого класса вопросов
(алгоритмические задачи по программированию, решаемые ситуационные вопросы),
где локальной модели на 8ГБ видеокарте не хватает качества рассуждения.
"""
from __future__ import annotations

import random
import time
from typing import Callable

from gigachat import GigaChat
from gigachat.exceptions import ResponseError
from gigachat.models import ChatCompletionRequest, ChatMessage

# GigaChat, в отличие от Ollama (см. _STOP_MARKER/"stop" в live_hints.py), не
# гарантированно поддерживает обрезку генерации по стоп-слову — не полагаемся
# на это тут вообще. "###" в промпте (общий для обоих бэкендов, см.
# FIRST_PASS_PROMPT/FINAL_PROMPT) всё равно остаётся полезным сигналом ДЛЯ
# САМОЙ МОДЕЛИ, что пора остановиться, а _parse_hint()/_parse_hint_partial() в
# live_hints.py и так разбирают только строки "ОТВЕТ:"/"КОНТЕКСТ:"/"ПОИСК:" —
# всё, что модель допишет ПОСЛЕ них, просто не попадёт ни в одно из трёх
# полей и будет тихо проигнорировано, как и на пути Ollama.

# temperature=0 у некоторых облачных API трактуется как "невалидное значение"
# (в отличие от Ollama, где 0 — штатный режим жадного декодирования, см.
# комментарий в _call_ollama). Берём минимально возможное ненулевое значение —
# по факту так же близко к детерминированному ответу, как temperature=0.
_TEMPERATURE = 0.0001
# Round 11 (см. проектный документ): было 1500 (тот же порядок, что num_predict
# в _call_ollama, под слабую локальную модель) — поднято до 4096 вместе с
# _MAX_FIELD_LEN=8192 в live_hints.py, тем же решением пользователя ("этот
# костыль нужен был для слабой Ollama, GigaChat не нужна эта защита"). Числа
# согласованы: 4096 токенов даже при скромной оценке ~2 символа/токен на
# смеси русского текста и кода — это далеко за 8192 символа _MAX_FIELD_LEN,
# так что _MAX_FIELD_LEN, а не max_tokens, остаётся тем местом, которое
# реально решает верхнюю границу (see live_hints.py _clean_field).
_MAX_TOKENS = 4096


def _make_client(
    credentials: str, model: str, ca_bundle_file: str | None, timeout: float | None = None
) -> GigaChat:
    kwargs = dict(
        credentials=credentials,
        model=model,
        # verify_ssl_certs остаётся включённым (по умолчанию SDK) —
        # ca_bundle_file ДОБАВЛЯЕТ доверенный корневой сертификат Минцифры в
        # цепочку проверки, а не отключает саму проверку. См. .env.example,
        # GIGACHAT_CA_BUNDLE_FILE, и check_gigachat_status.py.
        ca_bundle_file=ca_bundle_file,
    )
    if timeout is not None:
        # Таймаут задаётся на клиенте: в новых версиях SDK методы
        # chat.create()/chat.stream() больше не принимают аргумент timeout.
        try:
            return GigaChat(timeout=timeout, **kwargs)
        except TypeError:
            pass
    return GigaChat(**kwargs)


def _call_compat(fn, request, timeout: float):
    """Вызов метода SDK с таймаутом, совместимый со старыми и новыми версиями
    пакета gigachat: старые принимают timeout в самом вызове, новые — нет
    (TypeError: unexpected keyword argument 'timeout')."""
    try:
        return fn(request, timeout=timeout)
    except TypeError as e:
        if "timeout" not in str(e):
            raise
        return fn(request)


def call_gigachat(
    system_prompt: str,
    user_message: str,
    model: str,
    credentials: str,
    ca_bundle_file: str | None,
    timeout: float,
) -> str:
    """Нестриминговый вызов — аналог _call_ollama(). Возвращает текст целиком."""
    with _make_client(credentials, model, ca_bundle_file, timeout) as client:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_message),
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        response = _call_compat(client.chat.create, request, timeout)
        return response.messages[0].content[0].text


def call_gigachat_stream(
    system_prompt: str,
    user_message: str,
    model: str,
    credentials: str,
    ca_bundle_file: str | None,
    timeout: float,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """
    Потоковая версия — аналог _call_ollama_stream(). on_chunk (опционально)
    вызывается после каждого кусочка с уже НАКОПЛЕННЫМ текстом целиком (не
    дельтой), ровно как в _call_ollama_stream — так live_hints.py может
    передать тот же самый _parse_hint_partial без переделок.
    """
    with _make_client(credentials, model, ca_bundle_file, timeout) as client:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_message),
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        pieces: list[str] = []
        for chunk in _call_compat(client.chat.stream, request, timeout):
            if not chunk.messages or not chunk.messages[0].content:
                continue
            delta = chunk.messages[0].content[0].text or ""
            if delta:
                pieces.append(delta)
                if on_chunk is not None:
                    on_chunk("".join(pieces))
        return "".join(pieces)


def call_gigachat_retrying(
    system_prompt: str,
    user_message: str,
    model: str,
    credentials: str,
    ca_bundle_file: str | None,
    timeout: float,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """
    Вызов с повторами (до _MAX_ATTEMPTS попыток всего) в двух случаях:
    - пустой ответ на ровном месте — повтор сразу;
    - временный сбой: сеть (обрыв соединения, таймаут, 5xx) или лимит запросов
      (429) — повтор с растущей паузой _BACKOFF_BASE * 2^(n-1) + случайный
      разброс, чтобы не долбить перегруженный сервер синхронно.
    Ошибки, которые повтор не исправит (401/403 — ключ, 402 — кончились токены,
    прочие), сразу уходят наверх: каскад моделей и уведомление в интерфейсе
    решаются в main.py по классификации classify_gigachat_error.
    """
    raw = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if on_chunk is not None and attempt > 1:
            on_chunk("")  # сбрасываем недопечатанный черновик неудачной попытки
        try:
            if on_chunk is not None:
                raw = call_gigachat_stream(
                    system_prompt, user_message, model, credentials, ca_bundle_file, timeout, on_chunk=on_chunk
                )
            else:
                raw = call_gigachat(system_prompt, user_message, model, credentials, ca_bundle_file, timeout)
        except Exception as e:
            failure = classify_gigachat_error(e)
            if failure.kind not in _RETRYABLE_KINDS or attempt == _MAX_ATTEMPTS:
                raise
            delay = _BACKOFF_BASE * 2 ** (attempt - 1) + random.uniform(0, _BACKOFF_JITTER)
            print(
                f"[hints:gigachat] временный сбой ({failure.kind}: {type(e).__name__}), "
                f"попытка {attempt}/{_MAX_ATTEMPTS}, повтор через {delay:.1f} с"
            )
            time.sleep(delay)
            continue
        if raw.strip():
            return raw
        if attempt < _MAX_ATTEMPTS:
            print(f"[hints] пустой ответ (попытка {attempt}/{_MAX_ATTEMPTS}) — пробую ещё раз")
    return raw


_MAX_ATTEMPTS = 3
# Пауза перед повтором: 0.5 с, 1 с (+ до 0.3 с разброса). Короткая, потому что
# подсказка нужна во время разговора: дольше ждать уже бессмысленно.
_BACKOFF_BASE = 0.5
_BACKOFF_JITTER = 0.3
_RETRYABLE_KINDS = {"network", "rate_limit"}


# Round 11 (см. проектный документ) — классификация ошибок GigaChat.
#
# Раньше (round 10, "Побочная находка", подтверждено живым сбоем в round
# 10i — ReadTimeout уронил кейс) call_gigachat_retrying() ловила только
# ПУСТОЙ ответ, а любое исключение (сеть, сервер, кончились токены, протух
# ключ) вылетало наружу необработанным — main.py в ответ на ЛЮБУЮ такую
# ошибку одинаково тихо откатывался на локальную Ollama, не различая
# причину вообще. Пользователь явно решил: (1) Ollama как фолбэк для
# генерации больше не используется вообще ("Ollama особо не справится"),
# (2) вместо этого — точная классификация причины сбоя и явное уведомление
# в интерфейсе, плюс (3) при исчерпании токенов — автоматический откат на
# модель послабее (см. main.py _call_gigachat_cascaded), а не просто отказ.
# Для этого нужно уверенно отличать "кончились токены" ото всех остальных
# причин — не вслепую (просто printить текст exception), а по факту.
#
# Проверено по официальной документации Сбера и исходникам библиотеки
# gigachat (2026-09): токены закончились -> HTTP 402 "Payment Required",
# и это ЕДИНСТВЕННАЯ причина именно этого кода (сеть/сервер -> обычно 5xx,
# невалидный ключ -> 401, rate limit -> 429). Библиотека gigachat не имеет
# отдельного класса исключения под 402 (есть только BadRequestError/400,
# AuthenticationError/401, ForbiddenError/403, NotFoundError/404,
# RequestEntityTooLargeError/413, UnprocessableEntityError/422,
# RateLimitError/429, ServerError/5xx) — 402 приходит через базовый
# ResponseError, у которого есть атрибут status_code.
class GigaChatFailure(Exception):
    """
    Классифицированная ошибка вызова GigaChat — main.py решает, что делать,
    по .kind, а не разбирает текст исключения вручную. .kind — один из:
    "no_tokens" (HTTP 402, см. выше — единственный случай, где main.py
    пробует следующую модель в каскаде), "auth" (401/403 — протухший или
    невалидный ключ), "rate_limit" (429), "network" (5xx или сетевая
    ошибка транспорта — сервер недоступен), "unknown" (честный catch-all —
    печатаем/показываем как есть, ничего не выдумываем, см. пункт 2
    обсуждения с пользователем: "если ошибка не обработана, пусть просто
    код выведет").
    """

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


def classify_gigachat_error(e: Exception) -> GigaChatFailure:
    """Превращает произвольное исключение из вызова GigaChat в GigaChatFailure
    с понятным .kind. Никогда не бросает исключение сама — на любой
    непонятный случай возвращает kind="unknown" с текстом исходной ошибки,
    чтобы вызывающий код (main.py) мог честно показать пользователю хоть
    что-то, а не упасть молча."""
    if isinstance(e, ResponseError):
        code = getattr(e, "status_code", None)
        if code == 402:
            return GigaChatFailure("no_tokens", f"токены закончились (HTTP 402): {e}")
        if code in (401, 403):
            return GigaChatFailure("auth", f"невалидный/протухший ключ (HTTP {code}): {e}")
        if code == 429:
            return GigaChatFailure("rate_limit", f"превышен лимит запросов (HTTP 429): {e}")
        if isinstance(code, int) and code >= 500:
            return GigaChatFailure("network", f"сервер GigaChat недоступен (HTTP {code}): {e}")
        return GigaChatFailure("unknown", f"HTTP {code}: {e}")
    # Сетевые проблемы на уровне транспорта (httpx: ConnectError/ConnectTimeout/
    # ReadTimeout и т.п., см. round 10i — именно ReadTimeout когда-то уронил
    # тест) обычно вообще не ResponseError — ловим по имени класса исключения,
    # не завязываясь на прямой импорт httpx (тот же приём, что и duck-typed
    # known_topics в live_hints.py).
    name = type(e).__name__
    # RemoteProtocolError / ReadError — обрыв соединения посреди стриминга
    # ("peer closed connection without sending complete message body"),
    # встречено на живом звонке: это временный сетевой сбой, повтор помогает.
    if any(k in name for k in ("Timeout", "Connect", "Network", "Protocol", "ReadError")):
        return GigaChatFailure("network", f"сеть недоступна ({name}): {e}")
    return GigaChatFailure("unknown", f"необработанная ошибка ({name}): {e}")
