"""
Лёгкий веб-поиск для live-подсказок.

Зачем: локальная модель (qwen2.5:7b) хорошо знает фиксированные факты
("сколько уровней в OSI"), но объективно не может знать быстро меняющиеся
вещи — актуальную игровую мету, свежие новости, курсы и т.п. Раньше на
таких вопросах она либо выдумывала, либо (после жёсткого запрета выдумывать)
уходила в бесполезную отговорку "уточню и отвечу". Вместо этого даём ей
несколько реальных сниппетов из поиска как источник правды.

Используем HTML-версию DuckDuckGo без API-ключа и без регистрации — в
отличие от Google/Bing Search API, для которых нужен привязанный к
биллингу аккаунт (а с доступностью таких сервисов из России бывают
проблемы).
"""
from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://html.duckduckgo.com/html/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) call-assistant/1.0"}


def _element_text(el) -> str:
    """
    el.get_text(strip=True) БЕЗ разделителя склеивает текст из разных
    дочерних тегов впритык, без пробела между ними — на практике поймали
    реальный случай: сниппет из нескольких <b>/<span> кусков разобрался в
    "DefinitionofDoneandDefinitionofReady" (потерялись все пробелы между
    словами), модель не смогла нормально прочитать вторую половину и просто
    проигнорировала её в ответе. get_text(" ", strip=True) вставляет пробел
    между кусками текста из разных узлов — а re.sub дальше схлопывает лишние
    пробелы, которые могли и так уже быть в оригинальном тексте.
    """
    if el is None:
        return ""
    text = el.get_text(" ", strip=True)
    return re.sub(r"\s{2,}", " ", text).strip()


def search_web(query: str, max_results: int = 3, timeout: float = 6.0) -> list[dict]:
    """
    Возвращает список {"title", "snippet", "url"}. Пустой список, если поиск
    не удался (сеть, временная блокировка и т.п.) — это не должно ронять
    live-подсказки, максимум подсказка получится чуть менее содержательной.
    """
    try:
        response = requests.post(SEARCH_URL, data={"q": query}, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
    except Exception as e:
        print(f"[web_search] не удалось выполнить поиск '{query}': {e}")
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.select(".result")[:max_results]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one(".result__url")
            title = _element_text(title_el)
            snippet = _element_text(snippet_el)
            url = _element_text(link_el)
            if title or snippet:
                results.append({"title": title, "snippet": snippet, "url": url})
        return results
    except Exception as e:
        print(f"[web_search] не удалось разобрать выдачу для '{query}': {e}")
        return []


def format_results_for_prompt(results: list[dict]) -> str:
    if not results:
        return ""
    lines = [f"{i}. {r['title']} — {r['snippet']}" for i, r in enumerate(results, 1) if r["snippet"] or r["title"]]
    return "\n".join(lines)
