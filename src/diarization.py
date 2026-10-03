"""
Различение говорящих внутри "системного" звука.

Zoom/Discord/Teams смешивают голоса всех участников (кроме вас) в один
аудиопоток ещё до того, как он попадает в VB-Cable — раздельных дорожек по
людям у нас нет. Поэтому различаем их не по источнику звука, а по голосу:
для каждой распознанной фразы считаем "голосовой отпечаток" (speaker
embedding) и сравниваем с отпечатками, уже встреченными в этом звонке.
Похож на уже известный голос — та же метка ("Собеседник N"), не похож —
заводим новую метку.

Это не идеально: если два человека говорят одновременно в рамках одной
фразы (VAD склеил их вместе) — отпечаток получится смешанный и может не
совпасть точно ни с одним известным голосом (тогда, скорее всего, заведётся
лишняя метка). Для типичного созвона по очереди работает надёжно.
"""
from __future__ import annotations

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

# Порог косинусного сходства эмбеддингов, выше которого считаем "тот же голос".
# Если один и тот же человек начинает получать разные номера — понизьте порог.
# Если разные люди схлопываются в одну метку — повысьте. Настраивается через
# .env (DIARIZATION_SIMILARITY_THRESHOLD) — это значение по умолчанию, если
# в .env ничего не задано. Для чистого голоса в звонке 0.72 работает надёжно;
# для зашумлённого/продюсированного звука (музыка, эффекты, как на YouTube)
# отпечаток каждой фразы куда менее стабильный — там стоит понизить, скажем,
# до 0.5-0.55, иначе почти каждая фраза заводит нового "собеседника".
SIMILARITY_THRESHOLD = 0.72

# Меньше этой длины (в сэмплах, 16kHz) — фраза слишком короткая для надёжного
# отпечатка, используем последнего известного говорящего "как есть".
MIN_SAMPLES_FOR_EMBEDDING = 1600  # 100 мс


class SpeakerDiarizer:
    def __init__(self, device: str = "cuda", similarity_threshold: float = SIMILARITY_THRESHOLD):
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self._device = "cuda" if use_cuda else "cpu"
        self.similarity_threshold = similarity_threshold

        print(f"[diarize] загружаю модель голосовых отпечатков (speechbrain, {self._device})...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/spkrec-ecapa-voxceleb",
            run_opts={"device": self._device},
        )
        print("[diarize] готово")

        self._centroids: list[np.ndarray] = []  # усреднённый нормированный эмбеддинг на спикера
        self._counts: list[int] = []
        self._last_label = "Собеседник 1"

    def _embed(self, pcm_bytes: bytes) -> np.ndarray | None:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < MIN_SAMPLES_FOR_EMBEDDING:
            return None

        tensor = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(tensor)
        vec = emb.squeeze().cpu().numpy().astype(np.float64)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def identify(self, pcm_bytes: bytes) -> str:
        """Возвращает метку вида "Собеседник N" для данной фразы."""
        vec = self._embed(pcm_bytes)
        if vec is None:
            # Слишком короткая фраза для надёжного отпечатка — не гадаем на
            # ровном месте, приписываем последнему известному говорящему.
            return self._last_label

        best_idx, best_sim = -1, -1.0
        for i, centroid in enumerate(self._centroids):
            sim = float(np.dot(vec, centroid))
            if sim > best_sim:
                best_idx, best_sim = i, sim

        if best_idx >= 0 and best_sim >= self.similarity_threshold:
            n = self._counts[best_idx]
            updated = (self._centroids[best_idx] * n + vec) / (n + 1)
            norm = np.linalg.norm(updated)
            self._centroids[best_idx] = updated / norm if norm > 0 else updated
            self._counts[best_idx] += 1
            self._last_label = f"Собеседник {best_idx + 1}"
            return self._last_label

        self._centroids.append(vec)
        self._counts.append(1)
        self._last_label = f"Собеседник {len(self._centroids)}"
        return self._last_label
