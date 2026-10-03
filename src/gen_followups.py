"""
Одноразовый генератор ВТОРОГО файла с вопросами — src/batch_test_followup.py
раньше просил придумывать уточняющий вопрос ту же локальную модель Ollama
в реальном времени, но по факту нужно было, чтобы уточняющие вопросы
были подготовлены заранее, одним файлом-датасетом — без похода к
Ollama на этом шаге вообще. Этот скрипт делает именно это: один раз читает
data/knowledge_base.json (515 вопросов) и пишет data/knowledge_base_followups.json
той же длины, где к каждому вопросу добавлено поле "followup" — уточняющий
вопрос-продолжение к ЭТОМУ конкретному ответу.

Как выбирается формулировка: вопросы в базе делятся на несколько понятных
видов по структуре вопроса/ответа, и под каждый вид — свой пул из
нескольких естественных уточнений (все — с местоимением/отсылкой назад,
БЕЗ повтора термина, ровно то, что нужно для проверки own_answers в
live_hints.py):
  - сравнение ("A vs B", "A или B") -> "когда выбрать одно, а не другое",
    "как отличить на практике" и т.п.
  - перечисление (несколько пунктов через ";" или запятые, "какие типы/
    виды/уровни/сколько") -> "расскажи о каждом подробнее", "какой из них
    встречал на практике" и т.п.
  - вопрос "как бы ты протестировал/проверил X" -> "с чего начать",
    "что автоматизировать, а что вручную"
  - определение одного понятия ("что такое X") -> "приведи пример", "как
    это применяется на практике", "почему это важно"
  - всё остальное -> общий пул ("расскажи подробнее" и т.п.)

Внутри каждой категории формулировка берётся по кругу (по индексу вопроса
в базе) — чтобы все 515 не получили один и тот же текст, но при этом не
плодить сотни уникальных строк вручную, что не добавило бы пользы для
проверки самого механизма (own_answers), а не качества вопросов.

Запуск (один раз, локально — сеть/Ollama не нужны):
    python3 src/gen_followups.py
Печатает статистику по категориям и пишет data/knowledge_base_followups.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT_DIR / "data" / "knowledge_base.json"
OUT_PATH = ROOT_DIR / "data" / "knowledge_base_followups.json"

_COMPARISON_RE = re.compile(
    r"\bvs\b|\bпротив\b| или |плюсы\s+и\s+минусы|за\s+и\s+против|отличия\b", re.IGNORECASE
)
_LIST_QUESTION_RE = re.compile(
    r"какие\s|виды\b|типы\b|уровн|сколько\b|перечисл|мнемоник|критери|принцип|назовите|"
    r"компонент|ключевые|метрики|ценности|опоры|паттерн", re.IGNORECASE
)
# Структурный признак перечисления прямо в ОТВЕТЕ, даже если сам вопрос не
# содержит явных слов-подсказок ("Назовите 5 компонентов Android-приложения"
# формально не попадает под "какие/виды/типы" выше, но ответ явно состоит
# из отдельных пунктов через запятую, часто с пояснением в скобках у
# каждого — "Activities (экраны), Fragments (UI-компонент)...").
_LIST_ANSWER_RE = re.compile(r"(?:[^,;()]+\([^)]+\)\s*[,;]\s*){2,}")
_NUMBERED_LIST_RE = re.compile(r"(?:^|\s)[1-9][.)]\s")
_HOWTO_RE = re.compile(r"как\s+(бы\s+)?(ты\s+)?(протестировал|проверить|протестировать|тестировать)", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"^что такое|— это\?|\bопределение\b", re.IGNORECASE)

_COMPARISON_TEMPLATES = [
    "А в какой ситуации стоит выбрать одно из них, а не другое?",
    "Приведи пример, когда разница между ними реально важна на практике.",
    "А как их легко перепутать — на что смотреть, чтобы отличить?",
]

_LIST_TEMPLATES = [
    "Расскажи о каждом из них подробнее.",
    "С каким из них тебе реально доводилось работать на практике?",
    "А если нужно выбрать только один — какой самый важный и почему?",
    "Приведи пример для каждого из них.",
]

_HOWTO_TEMPLATES = [
    "С чего бы ты начал в первую очередь?",
    "А что из этого стоит автоматизировать, а что оставить вручную?",
    "Какой из этих пунктов чаще всего забывают на практике?",
]

_DEFINITION_TEMPLATES = [
    "Приведи пример.",
    "А как это применяется на практике?",
    "Почему это вообще важно?",
    "А в каких случаях это может не сработать?",
]

_DEFAULT_TEMPLATES = [
    "Расскажи подробнее.",
    "А как это применяется на практике?",
    "Приведи пример.",
]


def _classify(question: str, answer: str) -> str:
    if _COMPARISON_RE.search(question):
        return "comparison"
    if _HOWTO_RE.search(question):
        return "howto"
    if (
        _LIST_QUESTION_RE.search(question)
        or answer.count(";") >= 2
        or _LIST_ANSWER_RE.search(answer)
        or _NUMBERED_LIST_RE.search(answer)
    ):
        return "list"
    if _DEFINITION_RE.search(question):
        return "definition"
    return "default"


_TEMPLATES = {
    "comparison": _COMPARISON_TEMPLATES,
    "list": _LIST_TEMPLATES,
    "howto": _HOWTO_TEMPLATES,
    "definition": _DEFINITION_TEMPLATES,
    "default": _DEFAULT_TEMPLATES,
}


def main() -> None:
    data = json.loads(SRC_PATH.read_text(encoding="utf-8"))

    # Отдельный счётчик на каждую категорию — чтобы формулировки внутри
    # категории шли по кругу независимо от того, сколько вопросов другой
    # категории попалось между ними.
    counters = {k: 0 for k in _TEMPLATES}
    stats = {k: 0 for k in _TEMPLATES}

    result = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        category = _classify(question, answer)
        templates = _TEMPLATES[category]
        followup = templates[counters[category] % len(templates)]
        counters[category] += 1
        stats[category] += 1

        result.append(
            {
                "question": question,
                "answer": answer,
                "section": item.get("section", ""),
                "followup": followup,
                "followup_category": category,
            }
        )

    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Прочитано вопросов: {len(data)}")
    print("Распределение по категориям уточняющего вопроса:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nЗаписано: {OUT_PATH}")


if __name__ == "__main__":
    main()
