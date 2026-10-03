"""
Диагностика: показывает уровень громкости с обоих аудио-источников в
реальном времени, чтобы понять, доходит ли звук от собеседника (через
VB-Cable) до программы вообще — независимо от VAD и распознавания.

Запуск (из корня проекта):
    venv\\Scripts\\python.exe src/check_audio_levels.py

Говорите в микрофон и одновременно проигрывайте/говорите что-то через
Discord (или любое приложение, направленное на CABLE Input) — обе шкалы
должны двигаться. Если "система" стоит на нуле, пока звук точно играет —
проблема в маршрутизации Windows (микшер громкости), а не в коде питона.
Если шкала "система" двигается, а в основном скрипте фраз собеседника
всё равно нет — проблема дальше, в VAD/распознавании.

Ctrl+C для выхода.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd

from config import Settings

BAR_WIDTH = 30


def _level_bar(rms: float) -> str:
    # Речь обычно даёт rms в районе 0.02-0.2 — растягиваем для наглядности,
    # чтобы даже тихий звук был заметен на шкале.
    filled = min(BAR_WIDTH, int(rms * BAR_WIDTH * 8))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


class LevelMeter:
    def __init__(self, device_index: int, label: str):
        self.label = label
        device_info = sd.query_devices(device_index)
        self.samplerate = int(device_info["default_samplerate"])
        self.rms = 0.0
        self._stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            blocksize=int(self.samplerate * 0.05),
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0]
        self.rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


def main() -> None:
    settings = Settings.load()
    if settings.mic_device_index is None or settings.system_audio_device_index is None:
        print("MIC_DEVICE_INDEX / SYSTEM_AUDIO_DEVICE_INDEX не заданы в .env")
        sys.exit(1)

    print(f"мик = устройство #{settings.mic_device_index}, система = устройство #{settings.system_audio_device_index}\n")

    mic_meter = LevelMeter(settings.mic_device_index, "мик")
    sys_meter = LevelMeter(settings.system_audio_device_index, "система")

    mic_meter.start()
    sys_meter.start()

    print("Говорите в микрофон и проигрывайте звук через Discord/приложение, направленное на CABLE Input.")
    print("Ctrl+C для выхода.\n")

    try:
        while True:
            line = (
                f"мик:     {_level_bar(mic_meter.rms)}   "
                f"система: {_level_bar(sys_meter.rms)}"
            )
            print("\r" + line, end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        mic_meter.stop()
        sys_meter.stop()


if __name__ == "__main__":
    main()
