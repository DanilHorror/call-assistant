"""
Утилита: показывает все аудио-устройства, видимые системе, с их индексами.

Запуск:
    python src/list_audio_devices.py

Что искать:
- Ваш обычный микрофон (например "Microphone (Realtek...)") -> MIC_DEVICE_INDEX
- "CABLE Output (VB-Audio Virtual Cable)" -> SYSTEM_AUDIO_DEVICE_INDEX
  (это устройство появляется ПОСЛЕ установки VB-Cable; в него попадает звук
  от того приложения, которому вы в настройках Windows выставили выход
  "CABLE Input (VB-Audio Virtual Cable)", например от Discord/Zoom).
"""
import sounddevice as sd


def main() -> None:
    print("Доступные аудио-устройства:\n")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        kind = []
        if dev["max_input_channels"] > 0:
            kind.append("IN")
        if dev["max_output_channels"] > 0:
            kind.append("OUT")
        marker = "  <-- похоже на VB-Cable" if "cable" in dev["name"].lower() else ""
        print(
            f"[{idx:2d}] {dev['name']!r:50s} "
            f"{'/'.join(kind):6s} "
            f"in_ch={dev['max_input_channels']} "
            f"default_sr={dev['default_samplerate']:.0f}{marker}"
        )

    print("\nПодсказка: устройство с 'CABLE Output' в названии и IN-каналами")
    print("— это то, что нужно в SYSTEM_AUDIO_DEVICE_INDEX.")
    print("Ваш обычный микрофон (Realtek/USB-гарнитура и т.п.) — в MIC_DEVICE_INDEX.")


if __name__ == "__main__":
    main()
