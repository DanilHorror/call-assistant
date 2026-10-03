"""Общие настройки тестов: путь к src и заглушка пакета gigachat,
чтобы тесты не требовали сети, ключей и установленного SDK."""
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _install_gigachat_stub() -> None:
    if "gigachat" in sys.modules:
        return
    gigachat = types.ModuleType("gigachat")
    exceptions = types.ModuleType("gigachat.exceptions")
    models = types.ModuleType("gigachat.models")

    class ResponseError(Exception):
        def __init__(self, status_code=None, *args):
            super().__init__(*args)
            self.status_code = status_code

    class _Model:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    exceptions.ResponseError = ResponseError
    models.ChatCompletionRequest = _Model
    models.ChatMessage = _Model
    gigachat.GigaChat = object
    gigachat.exceptions = exceptions
    gigachat.models = models
    sys.modules["gigachat"] = gigachat
    sys.modules["gigachat.exceptions"] = exceptions
    sys.modules["gigachat.models"] = models


_install_gigachat_stub()
