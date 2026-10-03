"""
Массовый прогон РЫЧАГА 2 (см. проектный документ "call-assistant: архитектура
контекста для live-подсказок", Project "голос", раздел "Round 10c/10d/10e") —
объединённого вызова классификация+ответ в один запрос GigaChat
(live_hints.generate_hint_gigachat_with_topics) — по вырезкам из тестового
сценария (по умолчанию data/stress_test_300.json + data/stress_test_300_
answer_key.json, но можно указать любую другую пару файлов той же схемы
через --scenario/--answer-key — см. round 10f: data/stress_test_v2.json,
свежий набор, не пересекающийся со старым, чтобы не проверять рычаг 2 на
данных, под которые уже подгонялся промт).

Аналог batch_test_judge.py, но для нового пути: там прогоняется judge.py
(Ollama) по маленькому набору судья-кейсов, здесь — generate_hint_gigachat_
with_topics() (GigaChat) по вырезкам из большого сценария, С АКЦЕНТОМ на три
категории:

    ambiguous_backref     — должен вернуть ambiguous_options с правильными
                           кандидатами (или хотя бы честно уточнить текстом),
                           а не гадать и не отвечать уверенно не на то.
    long_distance_recall  — должен найти тему за много реплик до этого
                           (провалено судьёй в round 10: 6/6, из-за
                           консолидации тегов, см. проектный документ).
    self_continuation      (round 10f) — собеседник сам, БЕЗ уточняющего
                           вопроса ассистента между репликами, дополняет
                           СВОЙ ЖЕ предыдущий вопрос новой деталью
                           ("как тестируешь Х?" -> "точнее, если Y") —
                           должен трактоваться как продолжение ОДНОЙ темы,
                           а не как самостоятельный новый вопрос. Ранее это
                           было отмечено в проектном документе как
                           "подтверждённо непроверенный пробел" (открытый
                           вопрос пользователя, ещё до рычага 2).

По умолчанию прогоняются именно эти три категории (--categories задаёт
любой другой список: new_topic, vague_continuation, garbage,
clarify_confirmation, detail_continuation, synonym_reuse; clarify_question
пропускается всегда — это реплики "Я", на них подсказка не генерируется
даже в реальном main.py, см. HINTS_ON_MIC).

ВАЖНО (round 10d, исправлено после первого прогона) — про то, какой именно
список тем видит GigaChat. Первая версия этого скрипта по ошибке отдавала
generate_hint_gigachat_with_topics ВЕСЬ накопленный список тем без отбора —
но реальный hint_worker() в main.py ТАК НЕ ДЕЛАЕТ: он сначала прогоняет
known_tags через _select_candidate_tags() (embedding top_k=6 + recency
recent_k=3, полный список только пока тем <= 10 — см. main.py) и только
ОТОБРАННОЕ подмножество отдаёт в _known_topics_for_gigachat(). Первый прогон
с полным списком (до 118 тем) сильно завышал шум неоднозначности и заметно
занижал точность подбора темы — не отражал то, что реально увидела бы модель
в живом звонке. Этот скрипт теперь повторяет тот же отбор локально
(_select_candidate_topics, см. ниже) — с теми же порогами и алгоритмом, что
и main.py._select_candidate_tags, специально не импортируя main.py напрямую
(тяжёлые Windows-only зависимости — AudioCapture/VAD/STT — как и в остальных
batch_test_*.py в этом проекте, они тоже не импортируют main.py).

Реконструкция known_topics — это НЕ полная симуляция main.py, а снэпшот-тест
отдельных вызовов (тот же принцип, что и в batch_test_judge.py: там
known_tags тоже берутся из заранее записанных ожиданий кейса, а не из живого
прогона через process_loop). Для каждого кейса с индексом i полный (ДО
отбора) список тем восстанавливается ПО ЭТАЛОННОЙ разметке (answer_key), а
не по тому, что реально накопил бы судья/рычаг 2 при живом прогоне:
  - каждая реплика категории "new_topic" заводит тему;
  - реплика с одиночным related_topic (vague_continuation,
    detail_continuation, long_distance_recall, clarify_confirmation,
    synonym_reuse) обновляет recency (last_line_index) той темы, на которую
    ссылается related_topic;
  - ambiguous_backref recency НЕ обновляет ни одной из двух тем — как и в
    main.py (тема осталась неразрешённой, см. комментарий у rychag2_result
    в hint_worker());
  - garbage и clarify_question ни на что не влияют.
Эмбеддинг для отбора считается ОДИН РАЗ при создании темы (как TrackedTag.
embedding в main.py) и переиспользуется — регистр строится ИНКРЕМЕНТАЛЬНО по
возрастанию индекса за один проход по всему сценарию (0..N), а не заново с
нуля на каждый тестируемый кейс, ровно как реальный known_tags накапливается
за один звонок.

Сопоставление факта с ожиданием — МЯГКОЕ (как проверка "tags" в
batch_test_judge.py): "тема" в ответе GigaChat — свободный текст, не enum,
точное совпадение строк не ожидается. Совпадением считается пересечение
значимых слов (>= 2 общих слова длиной от 4 букв, либо >=50% слов короткой
из двух строк) — см. _topic_matches(). Это ИНДИКАТОР для быстрого просмотра
CSV, а не строгий вердикт: финальную оценку (правильно ли по сути отвечено)
всё равно делает человек, читая колонку "ответ" рядом с "reference_answer"
из answer_key — как и во всех остальных batch_test_*.py в этом проекте,
открытый текст ответа не оценивается автоматически.

Запуск (из корня проекта, ключ GigaChat должен быть настроен в .env —
GIGACHAT_ENABLED=true, GIGACHAT_CREDENTIALS=...):
    venv\\Scripts\\python.exe src\\batch_test_gigachat_topics.py
    venv\\Scripts\\python.exe src\\batch_test_gigachat_topics.py --scenario data\\stress_test_v2.json --answer-key data\\stress_test_v2_answer_key.json
    venv\\Scripts\\python.exe src\\batch_test_gigachat_topics.py --categories long_distance_recall
    venv\\Scripts\\python.exe src\\batch_test_gigachat_topics.py --categories all
    venv\\Scripts\\python.exe src\\batch_test_gigachat_topics.py --model GigaChat-2-Max --out r2_results.csv

Флаги:
    --scenario <файл>     JSON со сценарием (speaker/text) вместо
                    data/stress_test_300.json
    --answer-key <файл>   JSON с разметкой ожиданий вместо data/
                    stress_test_300_answer_key.json (должен быть той же
                    схемы и той же длины, что и --scenario)
    --categories <список через запятую или "all">
                    какие категории answer_key прогонять (по умолчанию
                    ambiguous_backref,long_distance_recall,self_continuation)
    --model <имя>   модель GigaChat вместо settings.gigachat_model
    --context-lines N   окно транскрипта вместо HINTS_CONTEXT_LINES из .env
    --limit N       прогнать только первые N отобранных кейсов
    --out <файл>    CSV с результатами (по умолчанию gigachat_topics_results.csv)
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import Settings
from judge import KnownTag
from live_hints import GigaChatTopicResult, generate_hint_gigachat_with_topics

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO_PATH = ROOT_DIR / "data" / "stress_test_300.json"
DEFAULT_ANSWER_KEY_PATH = ROOT_DIR / "data" / "stress_test_300_answer_key.json"

# clarify_question — реплики "Я" (собственный уточняющий вопрос ассистента,
# подставленный в сценарий как готовая история) — на них подсказка не
# генерируется даже в реальном main.py (см. process_loop:
# speaker != "Я" or settings.hints_on_mic, по умолчанию HINTS_ON_MIC=false).
# Прогонять их через generate_hint_gigachat_with_topics бессмысленно — в
# answer_key у них expected_verdict=null и нет смысловой проверки.
_SKIP_ALWAYS = {"clarify_question"}

_OWN_ANSWERS_KEEP = 6  # как в main.py (см. OWN_ANSWERS_KEEP в hint_worker)

# Те же числа, что и main.py: _TAG_SELECTION_THRESHOLD/_TOP_K/_RECENT_K —
# см. main.py._select_candidate_tags и комментарий выше про round 10d.
_SELECTION_THRESHOLD = 10
_SELECTION_TOP_K = 6
_SELECTION_RECENT_K = 3

_BRACKET_NOTE_RE = re.compile(r"^\s*\[[^\]]*\]\s*")
_WORD_RE = re.compile(r"[а-яА-ЯёЁa-zA-Z]{4,}")


@dataclass
class _TopicEntry:
    """Аналог main.py.TrackedTag, но здесь без .tag (короткое имя выдаётся
    отдельно, лениво — см. _short_tag) — только то, что реально нужно для
    отбора кандидатов и для показа GigaChat."""

    text: str  # verbatim исходная фраза — соответствует TrackedTag.summary
    last_line_index: int
    embedding: np.ndarray | None = None


def _short_tag(text: str, j: int) -> str:
    """Короткое "имя" темы для показа модели — приближение того, что в
    реальном звонке было бы либо коротким тегом от judge.py, либо именем,
    которое GigaChat само придумало на предыдущем вызове рычага 2 (см.
    _TOPIC_MEMORY_RULES в live_hints.py — "придумай короткое (1-3 слова)
    новое имя"). У нас нет настоящего инвентаря коротких имён (эталон хранит
    только полную исходную фразу) — берём первые до-3 значимых слов фразы,
    это ближе к реальности, чем непрозрачный "topic_N" (так было в первой
    версии скрипта, round 10c — путало и модель, и человека при чтении CSV,
    см. round 10d)."""
    words = _WORD_RE.findall(text)[:3]
    return " ".join(words) if words else f"тема_{j}"


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ошибка] не найден файл: {path}")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _embed_or_none(text: str) -> np.ndarray | None:
    try:
        import embeddings

        return embeddings.embed_text(text)
    except Exception as e:
        print(f"[embeddings] не удалось посчитать эмбеддинг, тема останется без него: {e}")
        return None


def _select_candidate_topics(
    entries: list[_TopicEntry],
    current_line_text: str,
    threshold: int = _SELECTION_THRESHOLD,
    top_k: int = _SELECTION_TOP_K,
    recent_k: int = _SELECTION_RECENT_K,
) -> list[_TopicEntry]:
    """Точная копия алгоритма main.py._select_candidate_tags (см. round 10d
    в докстринге модуля — почему это обязательно, а не просто "покажем
    GigaChat всё, у неё большое окно"): top_k по смыслу (embedding) +
    recent_k по свежести, объединение, без дублей. Если тем <= threshold —
    возвращаем все, как и в main.py."""
    if len(entries) <= threshold:
        return entries

    query_embedding = _embed_or_none(current_line_text)
    if query_embedding is None:
        return entries

    import embeddings

    scored = [
        (embeddings.cosine_sim(query_embedding, e.embedding), e) for e in entries if e.embedding is not None
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    by_similarity = [e for _, e in scored[:top_k]]

    by_recency = sorted(entries, key=lambda e: e.last_line_index, reverse=True)[:recent_k]

    combined: list[_TopicEntry] = []
    seen_ids: set[int] = set()
    for e in [*by_similarity, *by_recency]:
        if id(e) not in seen_ids:
            seen_ids.add(id(e))
            combined.append(e)

    # Round 10g — та же правка, что и в main.py._select_candidate_tags (см.
    # комментарий там): сортируем итоговый список по свежести (самые недавние
    # темы — первыми), а не оставляем порядок "top_k по похожести, потом
    # recent_k по свежести". Диагноз по реальному прогону на stress_test_v2
    # (8/8 ambiguous_backref не назвали правильных двух кандидатов, хотя они
    # гарантированно попадали в отбор через recent_k) — тема, отобранная
    # ТОЛЬКО по свежести, раньше всегда оказывалась в конце списка, даже если
    # это буквально последняя обсуждённая тема, а на расплывчатую отсылку
    # назад ("вернёмся к твоему примеру") embedding-сходство почти не несёт
    # сигнала. Состав кандидатов не меняется, меняется только порядок показа.
    combined.sort(key=lambda e: e.last_line_index, reverse=True)
    return combined


def _update_registry(registry: dict[str, _TopicEntry], answer_key: list[dict], j: int) -> None:
    """Продвигает регистр тем ровно на одну реплику (индекс j) — вызывается
    один раз за проход по возрастанию j, регистр переиспользуется между
    кейсами (см. докстринг модуля)."""
    entry = answer_key[j]
    category = entry["category"]
    related = entry.get("related_topic")

    if category == "new_topic":
        key = entry["text"]
        if key not in registry:
            registry[key] = _TopicEntry(text=entry["text"], last_line_index=j, embedding=_embed_or_none(entry["text"]))
    elif category in ("ambiguous_backref", "garbage") or category in _SKIP_ALWAYS:
        return
    elif isinstance(related, str) and related:
        if related in registry:
            registry[related].last_line_index = j
        else:
            # защитный случай: первое упоминание почему-то не было
            # размечено как new_topic — заводим тему тут же, чтобы не
            # потерять её из списка известных.
            registry[related] = _TopicEntry(text=related, last_line_index=j, embedding=_embed_or_none(related))


def _known_topics_for_call(
    registry: dict[str, _TopicEntry], current_line_text: str, upto_index: int
) -> list[KnownTag]:
    candidates = _select_candidate_topics(list(registry.values()), current_line_text)
    return [
        KnownTag(
            tag=_short_tag(e.text, e.last_line_index),
            summary=e.text,
            turns_ago=upto_index - e.last_line_index,
        )
        for e in candidates
    ]


def _build_recent_transcript(scenario: list[dict], upto_index: int, context_lines: int) -> str:
    start = max(0, upto_index + 1 - context_lines)
    lines = [f"{scenario[j]['speaker']}: {scenario[j]['text']}" for j in range(start, upto_index + 1)]
    return "\n".join(lines)


def _build_own_answers(answer_key: list[dict], upto_index: int) -> list[str]:
    """Грубое приближение main.py::own_answers — берём последние
    OWN_ANSWERS_KEEP непустых reference_answer из эталона (очищенные от
    пояснительных [...]-пометок, которых в реальном ответе GigaChat не
    было бы). Это не то же самое, что реальные ответы, накопленные в ходе
    именно этого прогона рычага 2 — приближение, влияет только на порог
    подключения базы знаний (см. _CONTINUATION_KB_MIN_SCORE в
    live_hints.py), не на сам топик-механизм, который мы тестируем."""
    collected: list[str] = []
    for j in range(upto_index):
        ref = answer_key[j].get("reference_answer")
        if ref:
            collected.append(_BRACKET_NOTE_RE.sub("", ref).strip())
    return collected[-_OWN_ANSWERS_KEEP:]


def _topic_matches(actual: str | None, expected: str) -> bool:
    if not actual or not expected:
        return False
    actual_words = set(w.lower() for w in _WORD_RE.findall(actual))
    expected_words = set(w.lower() for w in _WORD_RE.findall(expected))
    if not actual_words or not expected_words:
        return False
    overlap = actual_words & expected_words
    shorter = min(len(actual_words), len(expected_words))
    return len(overlap) >= 2 or (shorter and len(overlap) / shorter >= 0.5)


def _check_case(entry: dict, result: GigaChatTopicResult) -> tuple[str, list[str]]:
    """Мягкая оценка — см. докстринг модуля. Возвращает (метка, заметки)."""
    notes: list[str] = []
    expected_ambiguous = bool(entry.get("expected_ambiguous"))
    related = entry.get("related_topic")

    if expected_ambiguous:
        if not result.ambiguous_options:
            return "ПРОВАЛ", ["ожидали НЕОДНОЗНАЧНО с вариантами, получили обычный ответ/тему"]
        expected_list = related if isinstance(related, list) else [related]
        matched = [
            exp for exp in expected_list
            if any(_topic_matches(opt, exp) for opt in result.ambiguous_options)
        ]
        if len(matched) == len(expected_list):
            return "OK", notes
        missing = [exp for exp in expected_list if exp not in matched]
        return "ЧАСТИЧНО", [f"не нашли среди вариантов: {missing}"]

    if result.ambiguous_options:
        return "ПРОВАЛ", [f"неожиданно НЕОДНОЗНАЧНО (не должно было): {result.ambiguous_options}"]

    if isinstance(related, str) and related:
        if _topic_matches(result.topic, related):
            return "OK (тема)", notes
        notes.append(f"тема не совпала мягко: ожидали похоже на {related!r}, получили {result.topic!r}")
        return "ПРОВЕРИТЬ", notes

    # related_topic отсутствует (например new_topic) — тема не проверяется
    # строго, важен сам факт, что ответ вообще есть.
    if not result.reply:
        return "ПРОВЕРИТЬ", ["пустой ответ — либо решили, что это мусор, либо парсинг не собрал ОТВЕТ"]
    return "OK (ответ есть)", notes


def main() -> None:
    args = sys.argv[1:]
    categories_arg = "ambiguous_backref,long_distance_recall,self_continuation"
    model_override: str | None = None
    context_lines_override: int | None = None
    limit: int | None = None
    out_path = Path("gigachat_topics_results.csv")
    scenario_path = DEFAULT_SCENARIO_PATH
    answer_key_path = DEFAULT_ANSWER_KEY_PATH

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--scenario" and i + 1 < len(args):
            scenario_path = Path(args[i + 1])
            i += 2
        elif a.startswith("--scenario="):
            scenario_path = Path(a.split("=", 1)[1])
            i += 1
        elif a == "--answer-key" and i + 1 < len(args):
            answer_key_path = Path(args[i + 1])
            i += 2
        elif a.startswith("--answer-key="):
            answer_key_path = Path(a.split("=", 1)[1])
            i += 1
        elif a == "--categories" and i + 1 < len(args):
            categories_arg = args[i + 1]
            i += 2
        elif a.startswith("--categories="):
            categories_arg = a.split("=", 1)[1]
            i += 1
        elif a == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif a.startswith("--model="):
            model_override = a.split("=", 1)[1]
            i += 1
        elif a == "--context-lines" and i + 1 < len(args):
            context_lines_override = int(args[i + 1])
            i += 2
        elif a.startswith("--context-lines="):
            context_lines_override = int(a.split("=", 1)[1])
            i += 1
        elif a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
            i += 1
        elif a == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
            i += 2
        elif a.startswith("--out="):
            out_path = Path(a.split("=", 1)[1])
            i += 1
        else:
            print(f"Неизвестный аргумент: {a!r}")
            sys.exit(1)

    settings = Settings.load()
    if not (settings.gigachat_enabled and settings.gigachat_credentials):
        print(
            "[ошибка] GigaChat не настроена (GIGACHAT_ENABLED=true и "
            "GIGACHAT_CREDENTIALS нужны в .env) — рычаг 2 без неё не запускается."
        )
        sys.exit(1)

    model = model_override or settings.gigachat_model
    context_lines = context_lines_override if context_lines_override is not None else settings.hints_context_lines
    ca_bundle = str(settings.gigachat_ca_bundle_file) if settings.gigachat_ca_bundle_file else None

    scenario = _load_json(scenario_path)
    answer_key = _load_json(answer_key_path)
    if len(scenario) != len(answer_key):
        print(f"[ошибка] {scenario_path.name} и {answer_key_path.name} разной длины — файлы рассинхронизированы.")
        sys.exit(1)

    if categories_arg.strip().lower() == "all":
        wanted = None  # все, кроме _SKIP_ALWAYS
    else:
        wanted = {c.strip() for c in categories_arg.split(",") if c.strip()}

    wanted_indices = {
        idx for idx, entry in enumerate(answer_key)
        if entry["category"] not in _SKIP_ALWAYS and (wanted is None or entry["category"] in wanted)
    }
    if not wanted_indices:
        print("Нет ни одного кейса под заданные --categories.")
        sys.exit(0)

    ordered_indices = sorted(wanted_indices)
    if limit is not None:
        ordered_indices = ordered_indices[:limit]
    limited_set = set(ordered_indices)
    last_needed = max(ordered_indices)

    print(f"Сценарий: {scenario_path.name} / {answer_key_path.name}")
    print(f"GigaChat: {model}, HINTS_CONTEXT_LINES={context_lines}")
    print(f"Категории: {categories_arg} -> {len(ordered_indices)} кейсов")
    print(f"Отбор кандидатов: threshold={_SELECTION_THRESHOLD}, top_k={_SELECTION_TOP_K}, recent_k={_SELECTION_RECENT_K} (как в main.py)")
    print(f"Результат пишется в: {out_path.resolve()}\n")

    passed = 0
    checked = 0
    registry: dict[str, _TopicEntry] = {}

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["#", "index", "category", "current_line", "ожидаемая_тема",
             "статус", "заметки", "тема_гигачата", "неоднозначные_варианты",
             "ответ", "reference_answer", "показано_тем", "секунд", "ошибка",
             "сырой_ответ_модели"]
        )

        t_start_all = time.time()
        offset = 0
        for idx in range(last_needed + 1):
            if idx not in limited_set:
                _update_registry(registry, answer_key, idx)
                continue

            entry = answer_key[idx]
            recent_transcript = _build_recent_transcript(scenario, idx, context_lines)
            own_answers = _build_own_answers(answer_key, idx)
            known_topics = _known_topics_for_call(registry, entry["text"], idx)

            t0 = time.time()
            error_text = ""
            try:
                result = generate_hint_gigachat_with_topics(
                    recent_transcript,
                    credentials=settings.gigachat_credentials,
                    model=model,
                    known_topics=known_topics,
                    ca_bundle_file=ca_bundle,
                    kb_enabled=settings.kb_enabled,
                    kb_path=str(settings.kb_path) if settings.kb_path else None,
                    kb_top_k=settings.kb_top_k,
                    kb_min_score=settings.kb_min_score,
                    own_answers=own_answers,
                    extra_context=settings.hints_extra_context,
                )
                status, notes = _check_case(entry, result)
                topic_text = result.topic or ""
                ambiguous_text = " | ".join(result.ambiguous_options)
                reply_text = result.reply or ""
                raw_text = result.raw or ""
            except Exception as e:
                status = "ОШИБКА"
                notes = []
                topic_text = ambiguous_text = reply_text = raw_text = ""
                error_text = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0

            checked += 1
            if status.startswith("OK"):
                passed += 1

            related = entry.get("related_topic")
            related_text = " | ".join(related) if isinstance(related, list) else (related or "")

            writer.writerow(
                [offset + 1, idx, entry["category"], entry["text"], related_text,
                 status, " | ".join(notes), topic_text, ambiguous_text, reply_text,
                 entry.get("reference_answer") or "", len(known_topics), f"{elapsed:.1f}", error_text,
                 raw_text]
            )
            f.flush()

            print(f"[{offset + 1}/{len(ordered_indices)}] #{idx} {entry['category']:22s} {status} ({elapsed:.1f}с, показано тем: {len(known_topics)})")
            for n in notes:
                print(f"    - {n}")
            if error_text:
                print(f"    ! {error_text}")

            offset += 1
            # регистр обновляем ПОСЛЕ вызова (как main.py обновляет known_tags
            # после ответа судьи/рычага 2, не до) — но т.к. verdict тут не
            # используется для обновления recency ambiguous_backref/garbage
            # кейсов, применяем то же правило, что и _update_registry.
            _update_registry(registry, answer_key, idx)

        total_elapsed = time.time() - t_start_all
        print(f"\nГотово: {passed}/{checked} помечены как OK за {total_elapsed / 60:.1f} мин.")
        print("Напоминание: это мягкая эвристика для быстрого просмотра, не строгий вердикт —")
        print("колонки 'ответ' и 'reference_answer' в CSV нужно свериться глазами, как и в остальных batch_test_*.py.")
        print(f"Результат: {out_path.resolve()}")


if __name__ == "__main__":
    main()
