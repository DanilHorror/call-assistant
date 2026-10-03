"""
Облачный STT через Deepgram.

Важно: мы уже нарезаем аудио на законченные фразы через VAD (vad_segmenter.py),
поэтому вместо постоянного WebSocket-соединения используем synchronous
"prerecorded"-запрос на каждую фразу — по задержке это почти то же самое
(фраза обычно 1-5 секунд), но сильно проще и надёжнее, чем поддерживать
живой сокет с реконнектами. Если в будущем понадобится честный streaming
(например, для подсказок по недоговорённой фразе) — это отдельная доработка.
"""
from __future__ import annotations

import io
import struct

from deepgram import DeepgramClient, PrerecordedOptions, FileSource


def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Заворачивает сырой PCM16 моно в минимальный WAV-контейнер (без файла на диске)."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes


class CloudDeepgramSTT:
    def __init__(self, api_key: str, language: str):
        self.client = DeepgramClient(api_key)
        self.language = None if language == "auto" else language

    def transcribe_pcm16(self, pcm_bytes: bytes) -> str:
        wav_bytes = _pcm16_to_wav_bytes(pcm_bytes)
        payload: FileSource = {"buffer": wav_bytes}

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language=self.language,
        )
        response = self.client.listen.prerecorded.v("1").transcribe_file(payload, options)

        try:
            return response.results.channels[0].alternatives[0].transcript.strip()
        except (AttributeError, IndexError):
            return ""
