"""
После звонка: отправляем полный транскрипт в модель и получаем саммари +
список договорённостей. Сохраняем в summary.md рядом с transcript.txt.

Если включена GigaChat (GIGACHAT_ENABLED=true) — саммари делается через неё
(summarize_transcript_gigachat), а локальная Ollama остаётся запасным
вариантом, если облако недоступно. Без GigaChat — сразу через Ollama.

Локально — значит бесплатно, без ключей и без вопросов доступности API из
конкретной страны. Саммари генерируется один раз, после звонка, так что
задержка тут не критична (в отличие от будущих live-подсказок).
"""
from __future__ import annotations

from pathlib import Path

import requests

SYSTEM_PROMPT = """\
Ты помогаешь оператору зафиксировать детали разговора, чтобы ничего не упустить.
Тебе дан транскрипт звонка (Discord/Zoom/Teams/Meet). "Я" — это пользователь.
Остальные участники помечены как "Собеседник 1", "Собеседник 2" и т.д. — это
разные голоса, распознанные автоматически по звуку, а не настоящие имена.
Если по ходу разговора кто-то из них называет своё имя (представляется сам
или его называют по имени) — используй в саммари это имя вместо номера.
Речь может быть распознана с ошибками, транскрипт не идеален — старайся понять
смысл, а не придираться к опечаткам. Иногда одна и та же фраза может по ошибке
получить не тот номер собеседника (автоматическое распознавание голоса не
всегда точное) — если из контекста ясно, что это один и тот же человек,
считай их одним.

Сделай краткое саммари звонка на русском языке в формате:

## О чём был разговор
2-4 предложения, суть разговора.

## Договорённости и задачи
Список конкретных договорённостей, обещаний, дедлайнов — кто что должен сделать.
Если явных договорённостей не было — напиши "Явных договорённостей не зафиксировано".

## Важные детали
Любые цифры, имена, даты, ссылки, которые упоминались и могут понадобиться позже.

Пиши по делу, без воды. Если какого-то раздела реально нечем заполнить — не выдумывай.
Отвечай только текстом саммари в этом формате, без вступлений вроде "Вот саммари:".
"""


def summarize_transcript(
    transcript_text: str,
    output_dir: Path,
    model: str = "qwen2.5:7b",
    host: str = "http://localhost:11434",
) -> Path:
    response = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Транскрипт звонка:\n\n{transcript_text}"},
            ],
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    summary_text = data["message"]["content"].strip()

    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_path


def summarize_transcript_gigachat(
    transcript_text: str,
    output_dir: Path,
    credentials: str,
    model: str,
    ca_bundle_file: str | None = None,
    timeout: float = 120.0,
) -> Path:
    """То же саммари, но через GigaChat (с повторами при сетевых сбоях,
    см. llm_gigachat.call_gigachat_retrying)."""
    from llm_gigachat import call_gigachat_retrying

    summary_text = call_gigachat_retrying(
        SYSTEM_PROMPT,
        f"Транскрипт звонка:\n\n{transcript_text}",
        model,
        credentials,
        ca_bundle_file,
        timeout=timeout,
    ).strip()
    if not summary_text:
        raise RuntimeError("GigaChat вернула пустое саммари")

    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_path
