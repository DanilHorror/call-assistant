"""
Сохранение сгенерированных вопросов и ответов (live-подсказок) в отдельный
файл сессии — независимо от transcript.txt (см. transcript.py), где
хранится только то, что реально произнесено вслух, БЕЗ самих ответов
модели. Раньше ответ жил только в оверлее поверх экрана и терялся, как
только оверлей закрывался, — этот файл нужен, чтобы после звонка (особенно
после сольной тренировки без второго собеседника, см. HINTS_ON_MIC в
config.py) можно было спокойно пройтись по всем вопросам и подсказкам
текстом.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HintLog:
    session_dir: Path
    count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.md_path = self.session_dir / "questions_and_answers.md"
        # Дозапись построчно (как и transcript.txt) — чтобы ничего не
        # потерялось, если скрипт упадёт посреди звонка.
        self._fh = open(self.md_path, "a", encoding="utf-8")

    def add(self, question: str, answer: str) -> None:
        """question — "ТЕКУЩАЯ ФРАЗА" как есть (с меткой говорящего, см.
        _split_current_line в live_hints.py), answer — сгенерированный
        текст подсказки. Пустой ответ (модель решила, что отвечать не на
        что) не пишем — см. вызов в hint_worker в main.py."""
        if not answer:
            return
        self.count += 1
        now = dt.datetime.now().strftime("%H:%M:%S")
        block = f"## {self.count}. [{now}] {question}\n\n{answer}\n\n---\n\n"
        self._fh.write(block)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
