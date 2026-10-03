"""
Собирает поток 30мс-кадров в законченные фразы ("утерансы") с помощью VAD
(детектора речи), чтобы не гонять STT на тишине и не резать слова посередине.

Логика: пока в кадрах есть речь — копим их в буфер. Как только подряд
идёт SILENCE_FRAMES_TO_END тишины — считаем фразу законченной и отдаём её
целиком на распознавание.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import webrtcvad

from audio_capture import FRAME_MS, FRAME_SAMPLES, TARGET_SR

# Сколько подряд "тихих" кадров считать концом фразы (700мс)
SILENCE_FRAMES_TO_END = int(700 / FRAME_MS)
# Минимальная длина фразы, чтобы не гонять STT на случайном шуме (300мс)
MIN_SPEECH_FRAMES = int(300 / FRAME_MS)
# Максимальная длина одной фразы, чтобы длинный монолог не копился бесконечно (15с)
MAX_UTTERANCE_FRAMES = int(15000 / FRAME_MS)


@dataclass
class Utterance:
    label: str  # "mic" | "system"
    pcm_bytes: bytes  # 16-бит PCM моно 16kHz
    started_at: float  # time.time() начала фразы


class VadSegmenter:
    """
    Один сегментатор на каждый источник звука (мик / система), т.к. VAD
    и накопление буфера должны идти раздельно по говорящим.
    """

    def __init__(self, label: str, out_queue: "queue.Queue[Utterance]", vad_aggressiveness: int = 3):
        self.label = label
        self.out_queue = out_queue
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self._buffer: list[bytes] = []
        self._silence_run = 0
        self._utterance_start: float | None = None

    def process_frame(self, pcm_bytes: bytes) -> None:
        is_speech = self.vad.is_speech(pcm_bytes, TARGET_SR)

        if is_speech:
            if self._utterance_start is None:
                self._utterance_start = time.time()
            self._buffer.append(pcm_bytes)
            self._silence_run = 0

            if len(self._buffer) >= MAX_UTTERANCE_FRAMES:
                self._flush()
        else:
            if self._buffer:
                self._silence_run += 1
                self._buffer.append(pcm_bytes)  # немного тишины в конце — естественнее
                if self._silence_run >= SILENCE_FRAMES_TO_END:
                    self._flush()
            # если буфер пуст — просто игнорируем тишину

    def _flush(self) -> None:
        if len(self._buffer) >= MIN_SPEECH_FRAMES:
            pcm = b"".join(self._buffer)
            self.out_queue.put(
                Utterance(label=self.label, pcm_bytes=pcm, started_at=self._utterance_start or time.time())
            )
        self._buffer = []
        self._silence_run = 0
        self._utterance_start = None


def run_segmenters(
    frame_queue: "queue.Queue",
    utterance_queue: "queue.Queue[Utterance]",
    stop_event: threading.Event,
) -> None:
    """
    Фоновый поток: читает (label, frame) из frame_queue и раскидывает по
    отдельным VadSegmenter-ам для mic/system.
    """
    segmenters: dict[str, VadSegmenter] = {}

    while not stop_event.is_set():
        try:
            label, frame = frame_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if label not in segmenters:
            segmenters[label] = VadSegmenter(label, utterance_queue)

        segmenters[label].process_frame(frame)
