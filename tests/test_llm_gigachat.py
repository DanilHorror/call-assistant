"""Повторы и классификация ошибок GigaChat — без сети: вызовы подменены."""
import pytest

import llm_gigachat as lg
from gigachat.exceptions import ResponseError


class RemoteProtocolError(Exception):
    """Имя совпадает с исключением httpx при обрыве стриминга."""


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)


def _fake_call(outcomes):
    calls = {"n": 0}

    def fake(*_args, **_kwargs):
        outcome = outcomes[calls["n"]]
        calls["n"] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return fake, calls


@pytest.mark.parametrize(
    "error, kind",
    [
        (ResponseError(402), "no_tokens"),
        (ResponseError(401), "auth"),
        (ResponseError(429), "rate_limit"),
        (ResponseError(503), "network"),
        (RemoteProtocolError("peer closed connection"), "network"),
        (ValueError("что-то странное"), "unknown"),
    ],
)
def test_classify(error, kind):
    assert lg.classify_gigachat_error(error).kind == kind


def test_retries_network_error_then_succeeds(monkeypatch):
    fake, calls = _fake_call([RemoteProtocolError("drop"), ResponseError(429), "ОТВЕТ: ок"])
    monkeypatch.setattr(lg, "call_gigachat", fake)
    assert lg.call_gigachat_retrying("sys", "user", "m", "key", None, timeout=5) == "ОТВЕТ: ок"
    assert calls["n"] == 3


def test_does_not_retry_auth_or_quota(monkeypatch):
    for code in (401, 402):
        fake, calls = _fake_call([ResponseError(code), "never"])
        monkeypatch.setattr(lg, "call_gigachat", fake)
        with pytest.raises(ResponseError):
            lg.call_gigachat_retrying("sys", "user", "m", "key", None, timeout=5)
        assert calls["n"] == 1


def test_gives_up_after_max_attempts(monkeypatch):
    fake, calls = _fake_call([RemoteProtocolError("drop")] * 5)
    monkeypatch.setattr(lg, "call_gigachat", fake)
    with pytest.raises(RemoteProtocolError):
        lg.call_gigachat_retrying("sys", "user", "m", "key", None, timeout=5)
    assert calls["n"] == lg._MAX_ATTEMPTS


def test_streaming_retry_resets_draft(monkeypatch):
    fake, _ = _fake_call([RemoteProtocolError("drop"), "ОТВЕТ: ок"])
    monkeypatch.setattr(lg, "call_gigachat_stream", fake)
    seen = []
    lg.call_gigachat_retrying("sys", "user", "m", "key", None, timeout=5, on_chunk=seen.append)
    assert seen == [""]  # черновик неудачной попытки стёрт перед повтором
