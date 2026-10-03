"""
Захват аудио с одного устройства в отдельном потоке.

Каждый AudioCapture читает звук с конкретного устройства (индекс из
list_audio_devices.py), приводит его к моно 16kHz int16 — формату,
который нужен и faster-whisper, и Deepgram, и webrtcvad — и кладёт
кадры по 30мс в очередь для дальнейшей обработки (VAD-сегментация).
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

TARGET_SR = 16000
FRAME_MS = 30  # webrtcvad работает с кадрами 10/20/30 мс
FRAME_SAMPLES = TARGET_SR * FRAME_MS // 1000  # 480 сэмплов при 16kHz/30мс


class AudioCapture:
    """
    Пишет в self.frame_queue кортежи (label, pcm_bytes), где pcm_bytes —
    16-бит PCM моно 16kHz, длиной ровно FRAME_SAMPLES сэмплов.
    """

    def __init__(self, device_index: int, label: str, frame_queue: "queue.Queue"):
        self.device_index = device_index
        self.label = label  # "mic" или "system" — чтобы отличать говорящих
        self.frame_queue = frame_queue

        device_info = sd.query_devices(device_index)
        self.device_sr = int(device_info["default_samplerate"])
        self.channels = 1  # берём один канал, даже если устройство стерео

        self._stream: sd.InputStream | None = None
        self._resample_buffer = np.zeros(0, dtype=np.float32)
        self._pcm_leftover = np.zeros(0, dtype=np.int16)
        self._stop_event = threading.Event()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            # Переполнение буфера и т.п. — не фатально, просто лог.
            print(f"[audio:{self.label}] status: {status}")

        mono = indata[:, 0].astype(np.float32)

        if self.device_sr != TARGET_SR:
            # Ресемплинг до 16kHz (rational resample для точного соотношения)
            gcd = np.gcd(self.device_sr, TARGET_SR)
            up = TARGET_SR // gcd
            down = self.device_sr // gcd
            mono = resample_poly(mono, up, down).astype(np.float32)

        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)

        # Склеиваем с "хвостом" от предыдущего вызова и режем на кадры ровно
        # по FRAME_SAMPLES, чтобы webrtcvad получал кадры фиксированной длины.
        buf = np.concatenate([self._pcm_leftover, pcm16])
        n_full_frames = len(buf) // FRAME_SAMPLES
        usable = n_full_frames * FRAME_SAMPLES
        self._pcm_leftover = buf[usable:]

        for i in range(0, usable, FRAME_SAMPLES):
            frame = buf[i : i + FRAME_SAMPLES]
            try:
                self.frame_queue.put_nowait((self.label, frame.tobytes()))
            except queue.Full:
                pass  # обработка не успевает — лучше уронить кадр, чем застрять

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.device_sr,
            dtype="float32",
            blocksize=int(self.device_sr * 0.03),  # ~30мс блоки на входе
            callback=self._callback,
        )
        self._stream.start()
        print(f"[audio:{self.label}] захват запущен (device={self.device_index}, sr={self.device_sr})")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print(f"[audio:{self.label}] захват остановлен")
