"""
Гибридный STT: пробуем локальный faster-whisper на GPU, и если он не
успевает за реальным временем (rtf > порога несколько раз подряд),
на лету переключаемся на Deepgram — и обратно, если локальный отыгрался.
"""
from __future__ import annotations

from stt.cloud_deepgram import CloudDeepgramSTT
from stt.local_whisper import LocalWhisperSTT

# rtf (real-time factor) = время_обработки / длительность_аудио.
# Если больше RTF_SWITCH_THRESHOLD несколько фраз подряд — модель не
# успевает, переключаемся в облако.
RTF_SWITCH_THRESHOLD = 0.85
CONSECUTIVE_SLOW_TO_SWITCH = 3


class HybridSTT:
    def __init__(
        self,
        mode: str,  # "auto" | "local" | "cloud"
        whisper_model: str,
        whisper_device: str,
        whisper_compute_type: str,
        deepgram_api_key: str | None,
        language: str,
        whisper_initial_prompt: str | None = None,
    ):
        self.mode = mode
        self.language = language

        self.local: LocalWhisperSTT | None = None
        self.cloud: CloudDeepgramSTT | None = None

        if mode in ("auto", "local"):
            self.local = LocalWhisperSTT(
                whisper_model, whisper_device, whisper_compute_type, language, whisper_initial_prompt
            )

        if mode in ("auto", "cloud"):
            if not deepgram_api_key:
                if mode == "cloud":
                    raise RuntimeError("STT_MODE=cloud, но DEEPGRAM_API_KEY не задан")
                print("[stt:hybrid] DEEPGRAM_API_KEY не задан — фолбэк в облако будет недоступен")
            else:
                self.cloud = CloudDeepgramSTT(deepgram_api_key, language)

        self._using_cloud = mode == "cloud"
        self._slow_streak = 0

    def transcribe(self, pcm_bytes: bytes) -> tuple[str, str]:
        """Возвращает (текст, engine_used)."""
        if self.mode == "cloud" or (self._using_cloud and self.cloud is not None):
            if self.cloud is None:
                # Нет ключа Deepgram, а локальный отстаёт — деваться некуда, используем локальный.
                text, _ = self.local.transcribe_pcm16(pcm_bytes)
                return text, "local(no-cloud-fallback)"
            text = self.cloud.transcribe_pcm16(pcm_bytes)
            return text, "cloud"

        # mode == "local" или "auto" с текущим локальным движком
        text, rtf = self.local.transcribe_pcm16(pcm_bytes)

        if self.mode == "auto" and self.cloud is not None:
            if rtf > RTF_SWITCH_THRESHOLD:
                self._slow_streak += 1
                if self._slow_streak >= CONSECUTIVE_SLOW_TO_SWITCH:
                    print(
                        f"[stt:hybrid] локальная модель не успевает (rtf={rtf:.2f}) "
                        f"{CONSECUTIVE_SLOW_TO_SWITCH} фраз подряд -> переключаюсь на Deepgram"
                    )
                    self._using_cloud = True
                    self._slow_streak = 0
            else:
                self._slow_streak = 0

        return text, "local"
