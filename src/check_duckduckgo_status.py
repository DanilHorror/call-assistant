"""
Лёгкая диагностика DuckDuckGo HTML-поиска: не словили ли мы прямо сейчас
rate-limit/антибот-блокировку на том же эндпоинте, что использует
src/web_search.py.

Это НЕ стресс-тест на поиск порога — специально не долбит DuckDuckGo кучей
запросов подряд (это и есть способ самому спровоцировать бан). Вместо этого
шлём буквально несколько запросов на заведомо "живые" темы, у которых при
нормально работающем поиске результат почти гарантированно не пустой, и
смотрим на три вещи:

1. HTTP-статус и исключения (сеть/таймаут/403 и т.п.) — как и в web_search.py.
2. Известные маркеры антибот-защиты DuckDuckGo в теле ответа. По наблюдениям
   из открытых issues (duck-duck-scrape и др.), блокировка часто приходит с
   HTTP 200, но с JS-заглушкой вида `anomalyDetectionBlock(...)` вместо
   обычной выдачи — то есть просто проверки HTTP-статуса недостаточно.
3. Реальное количество распарсенных результатов (`.result` в HTML) — если
   для вопроса "python programming language" внезапно 0 результатов при
   HTTP 200 без явного маркера блокировки, это тоже подозрительно (тот самый
   диагностический пробел, который в live-коде тихо превращается в []).

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src\\check_duckduckgo_status.py
"""
from __future__ import annotations

import time

import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://html.duckduckgo.com/html/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) call-assistant/1.0"}

# Известные текстовые/скриптовые маркеры анти-бот защиты DuckDuckGo. Список
# не претендует на полноту — если появится новый вид блокировки, эти
# маркеры его не поймают, зато его поймает правило "0 результатов на
# заведомо непустой запрос" ниже.
_BLOCK_MARKERS = [
    "anomalyDetectionBlock",
    "unusual traffic",
    "unusual activity",
    "please complete the following challenge",
    "bots use duckduckgo too",
]

# Намеренно универсальные, почти наверняка результативные запросы — если по
# ним 0 результатов, это куда подозрительнее, чем 0 по редкому вопросу.
TEST_QUERIES = [
    "python programming language",
    "погода",
    "who is the president of the united states",
]


def check_once(query: str) -> dict:
    result: dict = {
        "query": query,
        "status_code": None,
        "elapsed": None,
        "error": None,
        "block_marker": None,
        "results_found": 0,
    }
    try:
        t0 = time.time()
        resp = requests.post(SEARCH_URL, data={"q": query}, headers=HEADERS, timeout=8.0)
        result["elapsed"] = round(time.time() - t0, 2)
        result["status_code"] = resp.status_code
        resp.raise_for_status()
    except Exception as e:
        result["error"] = str(e)
        return result

    body = resp.text
    lowered = body.lower()
    for marker in _BLOCK_MARKERS:
        if marker.lower() in lowered:
            result["block_marker"] = marker
            return result

    soup = BeautifulSoup(body, "html.parser")
    result["results_found"] = len(soup.select(".result"))
    return result


def main() -> None:
    print("Проверяю статус DuckDuckGo HTML-поиска (несколько лёгких тестовых запросов, с паузами)...\n")
    all_ok = True
    for q in TEST_QUERIES:
        r = check_once(q)
        if r["error"]:
            all_ok = False
            print(f"[ОШИБКА]        '{q}' -> {r['error']}")
        elif r["block_marker"]:
            all_ok = False
            print(f"[ПОХОЖЕ НА БАН] '{q}' -> HTTP {r['status_code']}, найден маркер блокировки: '{r['block_marker']}'")
        elif r["results_found"] == 0:
            all_ok = False
            print(f"[ПУСТО]         '{q}' -> HTTP {r['status_code']}, 0 результатов (подозрительно для такого запроса)")
        else:
            print(f"[OK]            '{q}' -> HTTP {r['status_code']}, {r['results_found']} результат(ов), {r['elapsed']}с")
        time.sleep(1.5)  # пауза даже в диагностике — не создаём нагрузку сами

    print()
    if all_ok:
        print("Похоже, DuckDuckGo сейчас отвечает нормально — признаков лимита/блокировки не видно.")
    else:
        print("Есть минимум один тревожный результат выше — похоже на временный rate-limit/блок")
        print("(или сетевую проблему). Если это стабильно повторяется через какое-то время —")
        print("стоит рассмотреть смену поискового бэкенда.")


if __name__ == "__main__":
    main()
