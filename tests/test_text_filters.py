import pytest

from text_filters import _is_backchannel_only


@pytest.mark.parametrize(
    "text",
    ["ага", "Ага, понятно.", "угу, ясно, спасибо!", "Да.", "Хорошо, спасибо.", "Mm-hmm", "ok, thanks"],
)
def test_backchannel_is_filtered(text):
    assert _is_backchannel_only(text)


@pytest.mark.parametrize(
    "text",
    [
        "Да, но почему отключили?",
        "Нет интернета",
        "Ага, а сколько стоит?",
        "Хорошо, а если на год?",
        "Спасибо, а когда мастер придёт?",
    ],
)
def test_real_question_passes(text):
    assert not _is_backchannel_only(text)
