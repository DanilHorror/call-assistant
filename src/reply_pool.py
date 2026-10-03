"""
Round 11 (см. проектный документ, раздел "Round 11"): плоский пул реплик
звонка — полная замена системе тегов (round 9-10O: TrackedTag/TagTouch/
_update_known_tags/_build_retrieved_context/_select_candidate_tags в
main.py, плюс GigaChat-механизм "ТЕМА:"/"НЕОДНОЗНАЧНО:" в live_hints.py).

Идея, подтверждённая пользователем буквально: "Эмбдинги и будут тегами
типо" — вместо того чтобы GigaChat САМ называл тему репликой словом
("ТЕМА: SQL"), а потом мы искали реплики с похожим словом-тегом, мы теперь
переводим КАЖДУЮ содержательную реплику (и вопрос собеседника, и
собственный ответ ассистента — раньше эмбеддился только вопрос) целиком в
вектор смысла (см. embeddings.embed_text, deepvk/USER-bge-m3, до 8192
токенов контекста — целый ответ с кодом влезает без обрезки, см. докстринг
embeddings.py) и решаем похожесть СВОИМ кодом (косинусное сходство) ДО
обращения к GigaChat, а не полагаемся на то, что GigaChat правильно назовёт
тему словом (round 10d задокументировал, что GigaChat часто путался —
названная тема не совпадала с содержанием его же ответа).

Пул — плоский список PoolEntry, без дерева тем/подтем: каждая запись это
либо реплика-вопрос собеседника, либо ответ ассистента на неё, со
взаимными перекрёстными ссылками (question_id/answer_id), чтобы при
совпадении по эмбеддингу можно было восстановить ЕСТЕСТВЕННУЮ пару
"вопрос+ответ" целиком, а не отдать GigaChat голый вопрос без ответа на
него или голый ответ без вопроса, на который он отвечал.

Что НЕ попадает в пул (решено с пользователем явно, "да окей"): мусорные/
подтверждающие реплики вроде "угу", "да, именно" — main.py решает, что
считать мусором (см. _looks_like_hard_question/классификация обычных
реплик), сюда прилетают уже только содержательные тексты.

Порог похожести и ОТСУТСТВИЕ верхнего предела на число совпадений — оба
решения явные, от пользователя: порог настраивается синтетическими тестами
позже (см. config.py POOL_SIMILARITY_THRESHOLD, ещё не откалиброван), а
предела на количество совпадений сейчас сознательно нет ("у нас есть
проблема впихнуть все в промт для гигачата? давай потестим, потом если
надо ограничим") — если реальные звонки покажут, что промпт разрастается
слишком сильно, ограничение (например top_k) добавляется здесь одной
строкой в build_retrieved_context, пока это НЕ сделано намеренно.

Пул — состояние ОДНОГО звонка (живёт внутри hint_worker в main.py, не
переживает перезапуск программы) — как и весь остальной live-контекст
проекта (own_answers, известные теги и т.п. до этого раунда).

Как и в embeddings.py — здесь нет try/except "чтобы не уронить звонок":
любая проблема с самим эмбеддингом (нет модели/сети при первой загрузке)
ловится ЛОКАЛЬНО в _embed_safe (печатается и возвращается None — запись
просто не попадает в пул), но сама генерация подсказки (main.py) не должна
падать целиком, если пул временно недоступен — это защищено уже тут.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import embeddings


@dataclass
class PoolEntry:
    id: int
    speaker: str
    text: str
    embedding: np.ndarray
    line_index: int
    # question_id заполнен у записи-ОТВЕТА (ссылка на вопрос, на который
    # ответили); answer_id заполнен у записи-ВОПРОСА (ссылка на свой ответ,
    # если он уже пришёл на момент поиска — может быть None, если поиск
    # происходит раньше, чем ассистент успел ответить на этот вопрос).
    question_id: int | None = None
    answer_id: int | None = None


def _embed_safe(text: str) -> np.ndarray | None:
    """
    Не даёт сбою эмбеддинга (нет библиотеки/модели/GPU и т.п., см.
    embeddings.py — embed_text() сама НЕ проглатывает ошибку) уронить живой
    звонок целиком: пул — необязательная надстройка над генерацией
    подсказки, а не критический путь. Печатаем ошибку и просто не
    добавляем запись — как и раньше в проекте (та же защитная философия,
    что и вокруг любой не-критической фичи, см. main.py).
    """
    try:
        return embeddings.embed_text(text)
    except Exception as e:
        print(f"[pool] не удалось посчитать embedding: {e}")
        return None


@dataclass
class ReplyPool:
    """Плоский пул реплик текущего звонка — см. докстринг модуля."""

    entries: list[PoolEntry] = field(default_factory=list)
    _next_id: int = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def add_question(self, text: str, line_index: int, speaker: str = "Я") -> int | None:
        """Добавляет реплику собеседника в пул. Возвращает id новой записи,
        либо None, если эмбеддинг не удалось посчитать (см. _embed_safe) —
        вызывающий код (build_retrieved_context) в этом случае просто не
        сможет ни искать, ни быть найденным по этой реплике."""
        emb = _embed_safe(text)
        if emb is None:
            return None
        entry = PoolEntry(id=self._new_id(), speaker=speaker, text=text, embedding=emb, line_index=line_index)
        self.entries.append(entry)
        return entry.id

    def add_answer(self, question_id: int | None, text: str, line_index: int | None) -> int | None:
        """Добавляет собственный ответ ассистента в пул, с обратной ссылкой
        на вопрос (question_id), и проставляет обратную ссылку answer_id на
        самом вопросе. text может быть пустым (генерация не удалась) —
        тогда просто ничего не добавляем, как и раньше own_answers.append
        пропускал пустые ответы."""
        if not text:
            return None
        emb = _embed_safe(text)
        if emb is None:
            return None
        entry = PoolEntry(
            id=self._new_id(),
            speaker="Ассистент",
            text=text,
            embedding=emb,
            line_index=line_index if line_index is not None else -1,
            question_id=question_id,
        )
        self.entries.append(entry)
        if question_id is not None:
            for q in self.entries:
                if q.id == question_id:
                    q.answer_id = entry.id
                    break
        return entry.id

    def search(self, query_embedding: np.ndarray, exclude_id: int | None = None) -> list[tuple[PoolEntry, float]]:
        """Полный линейный перебор всех записей пула (см. докстринг модуля —
        без top_k пока намеренно) — отсортировано по убыванию похожести."""
        scored: list[tuple[PoolEntry, float]] = []
        for e in self.entries:
            if e.id == exclude_id:
                continue
            try:
                score = embeddings.cosine_sim(query_embedding, e.embedding)
            except Exception as err:
                print(f"[pool] не удалось посчитать похожесть для записи {e.id}: {err}")
                continue
            scored.append((e, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def build_retrieved_context(
        self, question: str, line_index: int, threshold: float, log_top_n: int = 5
    ) -> tuple[str, int | None]:
        """
        Главная точка входа для main.py/hint_worker: добавляет текущий
        вопрос в пул (обязательно, даже если ничего не найдётся — иначе его
        саму никогда не найдут будущие реплики) и ищет среди ВСЕХ прошлых
        записей то, что прошло порог. Возвращает (retrieved_context,
        question_id) — question_id нужен main.py, чтобы потом привязать
        будущий add_answer() к этому конкретному вопросу.

        Топ-N (log_top_n) похожих логируется ВСЕГДА, независимо от порога —
        решено явно пользователем ("прогоним синтетикой и посмотрим") для
        последующей калибровки POOL_SIMILARITY_THRESHOLD по реальным логам,
        а не только по результату уже применённого порога.

        Найденные совпадения резолвятся в ЕСТЕСТВЕННУЮ пару вопрос+ответ
        (через question_id/answer_id), даже если само совпадение попало на
        половину пары — так GigaChat получает связный контекст, а не
        оторванный вопрос без ответа или ответ без вопроса. Дедуп по паре —
        если и вопрос, и его же ответ независимо попали в top-similar, пара
        не дублируется в промпте.
        """
        question_id = self.add_question(question, line_index)
        if question_id is None:
            return "", None
        this_entry = next(e for e in self.entries if e.id == question_id)
        scored = self.search(this_entry.embedding, exclude_id=question_id)
        if scored:
            top_log = scored[:log_top_n]
            log_lines = ", ".join(f"{e.text[:40]!r}~{score:.3f}" for e, score in top_log)
            print(f"[pool] топ-{len(top_log)} похожих на текущую реплику: {log_lines}")
        matched = [(e, score) for e, score in scored if score >= threshold]
        if not matched:
            return "", question_id
        by_id = {e.id: e for e in self.entries}
        seen_pairs: set[tuple[int | None, int | None]] = set()
        blocks: list[str] = []
        for entry, score in matched:
            if entry.question_id is not None:
                q_entry = by_id.get(entry.question_id)
                a_entry = entry
            else:
                q_entry = entry
                a_entry = by_id.get(entry.answer_id) if entry.answer_id is not None else None
            pair_key = (q_entry.id if q_entry else None, a_entry.id if a_entry else None)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            piece = f"{q_entry.speaker}: {q_entry.text}" if q_entry else ""
            if a_entry is not None:
                piece += f"\n{a_entry.speaker}: {a_entry.text}"
            blocks.append(f"Похожий момент из разговора (похожесть {score:.2f}):\n{piece}")
        return "\n\n".join(blocks), question_id
