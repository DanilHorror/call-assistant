"""
Дешёвые текстовые фильтры реплик, которые срабатывают ДО запроса к модели.
Вынесены из main.py, чтобы их можно было покрыть тестами без загрузки
аудио, STT и моделей.
"""
from __future__ import annotations

import re

# Одно или несколько (до 4) слов-поддакиваний подряд: "ага", "ага, понятно",
# "угу, ясно, спасибо". Живой прогон показал, что составные варианты
# ("Ага, понятно.") раньше проходили мимо фильтра и тратили запрос к модели.
_BACKCHANNEL_WORD = (
    r"(?:ага+|угу+|да|нет|ну+|м+|мм+|эм+|ясно|ясненько|понятно|поняла?|ок|окей|"
    r"хорошо|ладно|отлично|супер|спасибо|"
    r"hmm+|mm+[- ]?hmm|uh[- ]?huh|yeah|yep|ok|okay|thanks)"
)
_BACKCHANNEL_ONLY_RE = re.compile(
    rf"^{_BACKCHANNEL_WORD}(?:[.,!?…\s]+{_BACKCHANNEL_WORD}){{0,3}}[.,!?…\s]*$",
    re.IGNORECASE,
)


def _is_backchannel_only(text: str) -> bool:
    return bool(_BACKCHANNEL_ONLY_RE.match(text.strip()))
