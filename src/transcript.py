"""
Хранение бегущего транскрипта звонка: печать в консоль в реальном времени
+ запись в файл, чтобы ничего не потерялось при падении скрипта.

Раньше сюда попадало только то, что реально произнесено вслух ("Я"/
"Собеседник") — ответы ИИ жили отдельно (own_answers в main.py, только в
оверлее). Теперь ответы ИИ пишутся СЮДА ЖЕ, третьей ролью "Ассистент" (см.
hint_worker в main.py) — единая хронологическая лента разговора вместо двух
несинхронизированных списков. См. обсуждение в чате: это упрощает и сам код
(отпадает нужда таскать own_answers отдельным параметром), и даёт модели
более естественную картину — связный диалог по порядку, а не "история" +
отдельно приклеенный список "что я говорил".
"""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscriptLine:
    speaker: str  # "Я" / "Собеседник" / "Собеседник 1", "Собеседник 2"... / "Ассистент"
    text: str
    timestamp: dt.datetime
    engine: str
    # Позиция этой строки в TranscriptStore.lines на момент добавления (см.
    # add() ниже). Нужна judge-интеграции в main.py: тег, накопленный за
    # разговор, хранит номер строки, где он последний раз всплыл (см.
    # TrackedTag в main.py) — и по этому же номеру потом достаётся полный
    # кусок диалога (сама реплика + следующий "Ассистент", если есть) для
    # retrieval-блока в промпте GigaChat, а не только короткий summary,
    # которым обходится сам судья при классификации. -1 по умолчанию — на
    # случай прямого создания TranscriptLine в обход add() (сейчас нигде так
    # не делается, но не должно ломаться, если появится).
    index: int = -1


@dataclass
class TranscriptStore:
    session_dir: Path
    lines: list[TranscriptLine] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.txt_path = self.session_dir / "transcript.txt"
        # Открываем в режиме дозаписи построчно, чтобы файл был актуален
        # даже если скрипт упадёт посреди звонка.
        self._fh = open(self.txt_path, "a", encoding="utf-8")
        # add() теперь вызывается из двух разных потоков (process_loop — для
        # "Я"/"Собеседник", hint_worker — для "Ассистент"), раньше писал
        # только один поток. Лок — чтобы не столкнуться на self.lines.append()
        # и записи в файл одновременно.
        self._lock = threading.Lock()

    def add(self, speaker: str, text: str, engine: str) -> TranscriptLine | None:
        if not text:
            return None  # пустое распознавание (шум/тишина) — не сохраняем
        with self._lock:
            # len(self.lines) ДО append — это как раз позиция, на которую
            # строка встанет; вычисляем и присваиваем под тем же локом, что и
            # сам append(), так что index гарантированно совпадает с реальным
            # индексом строки в списке, даже при вызовах из двух потоков.
            index = len(self.lines)
            line = TranscriptLine(
                speaker=speaker, text=text, timestamp=dt.datetime.now(), engine=engine, index=index
            )
            self.lines.append(line)

            formatted = f"[{line.timestamp:%H:%M:%S}] {speaker}: {text}"
            print(formatted)
            self._fh.write(formatted + "\n")
            self._fh.flush()
            return line

    def full_text(self) -> str:
        parts = []
        for line in self.lines:
            parts.append(f"[{line.timestamp:%H:%M:%S}] {line.speaker}: {line.text}")
        return "\n".join(parts)

    def recent_text(self, n: int) -> str:
        """Последние n строк без таймштампов — компактный контекст для live-подсказок."""
        parts = [f"{line.speaker}: {line.text}" for line in self.lines[-n:]]
        return "\n".join(parts)

    def close(self) -> None:
        self._fh.close()
