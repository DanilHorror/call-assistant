"""
Локальный STT через faster-whisper (CTranslate2) на GPU.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

# faster_whisper сам логирует каждый вызов transcribe() на уровне INFO
# ("Processing audio with duration ...") — полезно при отладке библиотеки,
# но на каждую фразу звонка это просто шум поверх нашего собственного,
# гораздо более информативного вывода (транскрипт, подсказки). Поднимаем
# порог до WARNING — свои ошибки/предупреждения библиотека всё равно покажет.
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def _add_nvidia_dll_dirs() -> None:
    """
    На Windows ctranslate2 (движок faster-whisper) — скомпилированное
    C++/CUDA-расширение (.pyd), и грузит cuBLAS/cuDNN через WinAPI
    LoadLibrary БЕЗ флага "искать в AddDllDirectory-каталогах" — то есть
    `os.add_dll_directory()` (который добавляет каталог именно в этот
    список) сам по себе ctranslate2 не видит. Он ищет DLL по классическому
    порядку: папка самого exe/pyd, системные папки Windows и — это тут
    ключевое — переменная окружения PATH. Поэтому одного add_dll_directory
    недостаточно: нужно ЕЩЁ И добавить те же папки в os.environ["PATH"]
    для текущего процесса. Известная особенность faster-whisper/ctranslate2
    на Windows (см. их README про ручную установку cuBLAS/cuDNN "в папку
    из PATH") — DLL физически на диске, но LoadLibrary их не находит
    именно по этой причине.

    Если полного CUDA Toolkit нет, но поставлены пакеты nvidia-cublas-cu12 /
    nvidia-cudnn-cu12 через pip — добавляем их bin-папки и в PATH, и в
    add_dll_directory (на случай, если что-то в цепочке всё же использует
    безопасный поиск).

    torch (нужен для диаризации) на Windows везёт с собой свою собственную
    копию тех же DLL внутри torch\\lib — добавляем и её тоже, чтобы наличие
    torch в процессе не сбивало поиск библиотек для ctranslate2. find_spec
    не импортирует сам модуль, только смотрит, где он лежит на диске —
    torch при этом не загружается лишний раз.
    """
    if sys.platform != "win32":
        return
    try:
        candidate_dirs = []
        for pkg in ("nvidia.cublas", "nvidia.cudnn"):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    candidate_dirs.append(Path(loc) / "bin")

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            for loc in torch_spec.submodule_search_locations:
                candidate_dirs.append(Path(loc) / "lib")

        existing_path = os.environ.get("PATH", "")
        path_dirs_to_add = []
        for dll_dir in candidate_dirs:
            if not dll_dir.is_dir():
                continue
            dll_dir_str = str(dll_dir)
            # безопасный поиск (если конкретный загрузчик его использует)
            os.add_dll_directory(dll_dir_str)
            # классический поиск через PATH — то, что реально нужно ctranslate2
            if dll_dir_str not in existing_path:
                path_dirs_to_add.append(dll_dir_str)

        if path_dirs_to_add:
            os.environ["PATH"] = os.pathsep.join(path_dirs_to_add) + os.pathsep + existing_path
    except Exception:
        pass  # не критично — просто не найдём GPU-библиотеки и уйдём в CPU-фолбэк


_add_nvidia_dll_dirs()


class LocalWhisperSTT:
    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: str,
        initial_prompt: str | None = None,
    ):
        self.model_size = model_size
        self.language = None if language == "auto" else language
        self.device = device
        # Список имён/терминов, которые модель иначе может расслышать неверно
        # (например "Данил" -> "Нанил", "умножители" -> "жители"). Whisper
        # использует initial_prompt как смещение по словарю/стилю текста —
        # это не гарантия, но заметно снижает такие ошибки на похожих словах.
        self.initial_prompt = initial_prompt

        print(f"[stt:local] загружаю модель faster-whisper '{model_size}' (device={device}, {compute_type})...")
        self.model = self._load(device, compute_type)
        print(f"[stt:local] модель загружена (device={self.device})")

    def _load(self, device: str, compute_type: str) -> WhisperModel:
        try:
            model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
            self.device = device
            return model
        except Exception as e:
            if device == "cuda":
                print(f"[stt:local] не удалось загрузить модель на CUDA ({e}), пробую CPU...")
                model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                self.device = "cpu"
                return model
            raise

    def _fallback_to_cpu(self, reason: Exception) -> None:
        print(f"[stt:local] ошибка при распознавании на CUDA ({reason}) — переключаюсь на CPU насовсем")
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        self.device = "cpu"

    def _run_model(self, audio: np.ndarray) -> str:
        """
        Прогоняет аудио через модель и отфильтровывает типичные
        галлюцинации Whisper: сегменты с высокой вероятностью "это не речь"
        (no_speech_prob) или низкой уверенностью модели (avg_logprob) —
        именно такие сегменты чаще всего дают "подпишитесь на канал" и
        похожий мусор на тишине/шуме/неразборчивом звуке.
        """
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=False,
            beam_size=5,
            # Не даём модели "цепляться" за собственный предыдущий вывод —
            # иначе одна галлюцинация начинает повторяться по кругу.
            condition_on_previous_text=False,
            # initial_prompt — это не "память" между фразами (та отключена
            # выше), а статичная подсказка словаря/стиля, действует на
            # каждую фразу одинаково. Безопасна для галлюцинаций, т.к. не
            # накапливается и не зависит от предыдущего вывода модели.
            initial_prompt=self.initial_prompt,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.6:
                continue
            if seg.avg_logprob < -1.0:
                continue
            parts.append(seg.text)
        return "".join(parts).strip()

    def transcribe_pcm16(self, pcm_bytes: bytes) -> tuple[str, float]:
        """
        Возвращает (текст, rtf), где rtf (real-time factor) = время_обработки / длительность_аудио.
        rtf < 1 значит модель успевает в реальном времени.
        """
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = len(audio) / 16000

        t0 = time.time()
        try:
            text = self._run_model(audio)
        except RuntimeError as e:
            if self.device != "cuda":
                raise
            # GPU отвалился уже во время работы (например, не найдены CUDA-библиотеки
            # только при первом реальном вызове) — переключаемся на CPU и повторяем.
            self._fallback_to_cpu(e)
            t0 = time.time()
            text = self._run_model(audio)
        elapsed = time.time() - t0

        rtf = elapsed / duration_s if duration_s > 0 else 0.0
        return text, rtf
