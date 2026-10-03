"""Поиск по демо-базе: нужные карточки находятся, общие уточнения — нет.
Порог продолжения темы (0.5) в live_hints.py подобран по этим же случаям."""
import json
from pathlib import Path

import pytest

from knowledge_base import search_kb

KB = json.loads((Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json").read_text("utf-8"))
CONTINUATION_THRESHOLD = 0.5


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Сколько стоит тариф домашний 100?", "Сколько стоит тариф «Домашний 100»?"),
        ("Почему отключили интернет, если на балансе есть деньги?", "Почему отключили интернет при положительном балансе?"),
        ("А если подключиться на год, есть скидка?", "Есть ли скидка для новых абонентов?"),
        ("Можно ли приостановить договор на время отпуска?", "Можно ли приостановить договор на время отпуска?"),
        ("Как сменить пароль от вайфая?", "Как сменить пароль от Wi-Fi?"),
    ],
)
def test_target_found_above_continuation_threshold(query, expected):
    top = search_kb(query, KB, top_k=1, min_score=0.0)
    assert top and top[0]["question"] == expected
    assert top[0]["score"] >= CONTINUATION_THRESHOLD


@pytest.mark.parametrize(
    "query",
    ["Расскажи подробнее", "А почему это важно?", "А что ещё?", "А сколько это стоит?", "Сколько стоит подключить Netflix?"],
)
def test_generic_followups_stay_below_threshold(query):
    top = search_kb(query, KB, top_k=1, min_score=0.0)
    assert not top or top[0]["score"] < CONTINUATION_THRESHOLD
