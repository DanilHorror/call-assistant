"""
Текстовые эмбеддинги. История:

Round 9-10O: узкая задача — "насколько по смыслу текущая фраза похожа на
summary вот этого старого тега" (main.py: TrackedTag/_select_candidate_tags),
считалось дёшево на CPU крошечной моделью cointegrated/rubert-tiny2 (~50МБ,
64 токена контекста).

Round 11 (см. проектный документ): пользователь решил полностью заменить
систему тегов эмбеддинг-поиском по ПОЛНОМУ пулу реплик (main.py/
reply_pool.py) — вопросы И собственные ответы ассистента целиком, включая
многострочные примеры кода. У rubert-tiny2 контекст всего 64 токена — этого
достаточно для короткой фразы-триггера темы, но категорически мало для
целого ответа с кодом (см. round 10M, тот же лимит уже мешал
answer_embedding). Плюс gemma2:9b/Ollama, ради которой эмбеддинги
принципиально держали на CPU (не конкурировать за 8ГБ VRAM), теперь
полностью выведена из живого пути (round 11, отказ от Ollama-фолбэка) —
видеопамять фактически свободна для более крупной модели эмбеддингов на GPU.

Выбор модели: deepvk/USER-bge-m3 (Apache 2.0, бесплатная) — форк BAAI/bge-m3,
дообученный на русских деловых/технических текстах, контекст 8192 токена
(см. live_hints.py _MAX_FIELD_LEN=8192 — числа согласованы намеренно, любой
целиком сгенерированный ответ ассистента гарантированно влезает без обрезки).
Требует ~2.3 ГБ VRAM в FP16 — стоит проверить, что модель реально грузится на GPU и укладывается в память
вместе с остальными процессами звонка.

Открытые вопросы, требующие замера на реальных данных:
1) насколько хорошо USER-bge-m3 представляет СМЕШАННЫЙ текст (русская речь +
   куски кода SQL/Python/YAML внутри одной реплики) — модель дообучена на
   деловой/технической ЛЕКСИКЕ, а не специально на исходном коде. Если
   реальные логи похожести покажут, что код "путается" (разные по сути
   ответы с похожим кодом дают неотличимо высокую похожесть, или наоборот)
   — см. проектный документ, запасной вариант Qwen3-Embedding (модели этой
   линейки много обучались на коде как LLM-семейство).
2) реальное распределение косинусной похожести этой модели на русских
   репликах этого проекта — POOL_SIMILARITY_THRESHOLD в config.py оставлен
   как первая, не откалиброванная попытка, требует прогона на реальных
   данных (см. .env.example).

Старая функция для rubert-tiny2 НЕ удалена — доступна как
_embed_text_rubert_tiny2_legacy(), по правилу проекта "не удаляй, а
закомментируй" (тут даже комментировать не пришлось: она просто больше не
вызывается изнутри embed_text() по умолчанию).

Как и раньше — обязательно defensive: любая проблема (нет библиотеки, нет
сети/GPU при первой загрузке модели и т.п.) не проглатывается тут, а летит
исключением наружу — решение "что делать при сбое" остаётся за вызывающим
кодом (main.py/reply_pool.py), фича обязана быть "тихо необязательной": её
поломка не должна ронять живой звонок.
"""
from __future__ import annotations

import numpy as np

# --- Round 11: активная модель --------------------------------------------

_USER_BGE_M3_MODEL_NAME = "deepvk/USER-bge-m3"
_RUBERT_TINY2_MODEL_NAME = "cointegrated/rubert-tiny2"

# Ленивая загрузка (тяжёлые импорты sentence_transformers/torch — только при
# первом реальном вызове, не при импорте модуля) — main.py импортирует этот
# модуль безусловно, незачем грузить модель, которая, возможно, не
# понадобится вовсе в конкретном звонке (например если пул реплик ни разу не
# понадобился до конца разговора).
_st_model = None


def _load_user_bge_m3():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        # device="cuda" — намеренно, см. докстринг файла: gemma2:9b/Ollama
        # больше не в живом пути (round 11), VRAM свободна. Если CUDA
        # недоступна на конкретной машине, sentence-transformers сама
        # откатится на CPU (SentenceTransformer сам определяет устройство по
        # умолчанию, когда cuda недоступна) — но здесь просим GPU явно
        # первым делом, а не полагаемся молча на автоопределение, чтобы
        # ошибка (если её нет) была видна сразу, а не терялась в тихом
        # отвале на CPU с непредсказуемой скоростью.
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        _st_model = SentenceTransformer(_USER_BGE_M3_MODEL_NAME, device=device)
        print(f"[embeddings] deepvk/USER-bge-m3 загружена на устройство: {device}")
    return _st_model


def embed_text(text: str) -> np.ndarray:
    """
    Round 11: основная функция эмбеддинга — deepvk/USER-bge-m3 (8192 токена
    контекста, см. докстринг файла). Возвращает нормализованный (единичной
    длины) вектор смысла фразы — нормализация нужна, чтобы cosine_sim ниже
    была простым скалярным произведением, без деления на нормы на каждом
    сравнении (в пуле реплик может быть много записей за долгий звонок).

    Бросает исключение при любой проблеме (нет библиотеки, нет сети/GPU на
    первой загрузке модели и т.п.) — НЕ проглатывает её сама, см. пояснение
    в шапке файла: решение "что делать при сбое" — за вызывающим кодом.

    БЕЗ инструкционного префикса к тексту — в отличие от более старых
    версий BGE, M3-семейство (и его деривативы вроде USER-bge-m3) в этом не
    нуждается (см. официальный FAQ модели).
    """
    model = _load_user_bge_m3()
    # normalize_embeddings=True — сама библиотека делает L2-нормализацию,
    # эквивалентно ручному делению на норму, как раньше в rubert-tiny2.
    vector = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vector.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Оба вектора уже нормализованы embed_text — это просто скалярное произведение."""
    return float(np.dot(a, b))


# --- Round 9-10O: старая модель, оставлена как откат ------------------------
#
# НЕ вызывается из embed_text() по умолчанию (round 11 переключил основной
# путь на deepvk/USER-bge-m3 выше) — оставлена целиком рабочей на случай,
# если понадобится откат (например если USER-bge-m3 реально не потянет по
# памяти на конкретной машине, или качество на коде окажется хуже, чем у
# старой модели, что маловероятно, но не проверено, см. докстринг файла).
# Векторы этой модели (312 чисел, rubert-tiny2) НЕСОВМЕСТИМЫ по размерности
# с векторами USER-bge-m3 (1024 числа) — если понадобится откатиться,
# нужно либо очистить уже накопленный пул реплик текущего звонка, либо не
# смешивать вызовы этой функции с embed_text() в одном пуле.

_rubert_tokenizer = None
_rubert_model = None


def _load_rubert_tiny2():
    global _rubert_tokenizer, _rubert_model
    if _rubert_model is None:
        from transformers import AutoModel, AutoTokenizer

        _rubert_tokenizer = AutoTokenizer.from_pretrained(_RUBERT_TINY2_MODEL_NAME)
        _rubert_model = AutoModel.from_pretrained(_RUBERT_TINY2_MODEL_NAME)
        _rubert_model.eval()
    return _rubert_tokenizer, _rubert_model


def _embed_text_rubert_tiny2_legacy(text: str) -> np.ndarray:
    """Старая (round 9-10O) реализация embed_text() — см. комментарий выше.
    Обрезает текст до 64 токенов (реальное ограничение этой конкретной
    модели, не было проблемой для коротких тегов, но стало бы проблемой для
    целых ответов с кодом — см. round 10M и докстринг файла, ровно поэтому
    round 11 и перешёл на другую модель)."""
    import torch

    tokenizer, model = _load_rubert_tiny2()
    inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    token_embeddings = output.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    mean_pooled = (summed / counts)[0].numpy()
    norm = np.linalg.norm(mean_pooled)
    if norm > 0:
        mean_pooled = mean_pooled / norm
    return mean_pooled.astype(np.float32)
