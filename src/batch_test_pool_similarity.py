"""
Калибровка порога POOL_SIMILARITY_THRESHOLD (Round 11, см. проектный документ
"call-assistant: архитектура контекста для live-подсказок", Project "голос",
раздел "Round 11", пункт 4 — пул эмбеддингов реплик, reply_pool.py).

Пользователь попросил: прогнать уже существующие вопросные файлы (data/
stress_test_300.json + answer_key, data/stress_test_v2.json + answer_key)
через РЕАЛЬНУЮ модель эмбеддингов (deepvk/USER-bge-m3) и посмотреть, нужно ли
что-то настраивать — БЕЗ обращения к GigaChat и без всего остального
интерфейса (аудио/Tkinter). Это возможно ровно потому, что embeddings.py и
reply_pool.py не тянут ничего из этого — единственная тяжёлая зависимость тут
sentence_transformers/torch (для реальной загрузки модели), при первом
реальном вызове.

Первый реальный прогон (пользователь, на своей машине) показал плохую
картину: похожесть между ДВУМЯ РАЗНЫМИ, ничем не связанными вопросами
(new_topic) оказалась систематически ВЫШЕ, чем похожесть настоящего
продолжения на свою реальную тему (медианы 0.6+ против 0.3-0.5). Разбор (см.
чат, WebSearch по deepvk/USER-bge-m3 и общей литературе про sentence-эмбеддинги)
указал на классическую АНИЗОТРОПИЮ embedding-пространства: у моделей на базе
BERT/трансформеров векторы почти всех предложений в среднем смотрят в одном
"общем" направлении (общий регистр языка/домена), из-за чего сырое косинусное
сходство между ЛЮБЫМИ двумя фразами уже заранее раздуто этим общим
направлением — совпадение доменной лексики (у нас: QA/DevOps-жаргон) даёт
случайно высокую похожесть даже для абсолютно не связанных вопросов. Это не
специфический дефект именно USER-bge-m3 — стандартная особенность подобных
моделей, и стандартное же лекарство — mean-centering: перед сравнением
вычесть из каждого вектора среднее по уже накопленным векторам и заново
отнормировать. Модель НЕ требует специальных инструкционных префиксов
("query: "/"passage: ") — это подтверждено и для родительской BAAI/bge-m3
(автор модели: "No, bge-m3 doesn't need prefix instruction"), и в карточке
самой USER-bge-m3 префиксы тоже не упоминаются — так что дело не в этом.

Скрипт теперь считает ОБА варианта на одних и тех же реальных векторах, без
двух отдельных прогонов модели: "сырую" косинусную похожесть (как сейчас в
проде, reply_pool.search()) и "центрированную" (вычитаем среднее по всем
эмбеддингам, накопленным в пуле К ЭТОМУ МОМЕНТУ разговора — включая текущую
реплику, — из каждого вектора, дальше как обычно). Среднее пересчитывается
заново при каждом поиске (а не хранится один раз) — так среднее всегда честно
отражает "фон" именно на данный момент разговора, и одинаково применяется и к
текущей реплике, и к каждому кандидату, с которым её сравнивают (см. вывод из
Medium-разбора анизотропии — непоследовательное центрирование запроса и
документов портит результат сильнее, чем отсутствие центрирования вовсе).
Это ЧИСТО ВЕКТОРНЫЙ эксперимент — main.py/reply_pool.py НЕ меняются, GigaChat
и видимое окно транскрипта тут вообще не участвуют, только сравнение двух
способов мерить расстояние между уже посчитанными эмбеддингами.

Для реплик, у которых answer_key.related_topic указывает на реальную более
раннюю тему (vague_continuation, detail_continuation, ambiguous_backref
[список из двух], long_distance_recall, synonym_reuse, clarify_confirmation,
self_continuation) — эталонный кандидат(ы) находится по ДОСЛОВНОМУ совпадению
текста с более ранней репликой в этом же сценарии (та реплика, что и создала
тему), берётся МАКСИМУМ похожести среди самой этой реплики-вопроса И её
эталонного ответа (потому что build_retrieved_context в main.py резолвит
совпадение по любой из двух половин пары в естественную пару целиком — см.
докстринг reply_pool.py). Для new_topic/garbage (никакой более ранней темы не
подразумевается) — это, наоборот, ИЗМЕРЕНИЕ БАЗОВОГО ШУМА: максимальная
похожесть на что угодно из уже накопленного пула, где совпадения В ПРИНЦИПЕ
быть не должно.

clarify_question (реплики "Я" — собственный уточняющий вопрос ассистента)
пропускается совсем, как и в batch_test_gigachat_topics.py — на них подсказка
не генерируется даже в реальном main.py по умолчанию (HINTS_ON_MIC=false).

Второй реальный прогон (mean-centering) показал, что анизотропия — НЕ
главная причина: центрирование сделало recall хуже на обоих файлах (см. чат).
Отдельно нашлась настоящая причина похожести=1.000 у части категорий —
буквальные дословные повторы одной и той же фразы-заглушки в тестовых данных
(например "приведи ещё пример" x12, "да, ты прав" x7) для vague_continuation/
clarify_confirmation/ambiguous_backref — эти три категории в принципе не
решаются по содержанию текста (нужна свежесть/recency, не эмбеддинг), это
отдельная, не относящаяся к качеству модели история. По-настоящему годные для
эмбеддинг-поиска категории (long_distance_recall/detail_continuation/
synonym_reuse/self_continuation) уникальны по тексту — по ним и есть смысл
сравнивать модели.

Round "проверка другой модели": пользователь попросил прогнать те же файлы
через Qwen/Qwen3-Embedding-0.6B (упоминалась в docstring embeddings.py как
запасной вариант, сильнее заточена на код/технический текст, обучена как
LLM-семейство) — ДО того, как менять production-код (embeddings.py/
reply_pool.py не трогаем, это опять чисто векторный эксперимент, как и с
центрированием). В отличие от USER-bge-m3, Qwen3-Embedding АСИММЕТРИЧНА:
документы (то, что кладём в пул) эмбеддятся как есть, а запросы (то, чем
ищем) — с инструкционным префиксом (`model.encode(text, prompt_name="query")`
— так рекомендует сама карточка модели на HuggingFace). Скрипт учитывает это:
класс-бэкенд `_Qwen3EmbeddingBackend.embed_query()` применяет префикс,
`.embed_document()` — нет; для USER-bge-m3 (симметричная модель, префикс не
нужен — подтверждено и для родительской BAAI/bge-m3 автором: "No, bge-m3
doesn't need prefix instruction") оба метода совпадают.

ВАЖНО про версии перед запуском с `--model qwen3-embedding-0.6b`: этой модели
нужен transformers>=4.51.0, а в requirements.txt проекта жёстко зафиксирован
transformers==4.44.2 (нужен для legacy rubert-tiny2 и совместимости с
faster-whisper/tokenizers) — тот же класс проблемы, что уже был с
sentence-transformers==6.1.0 (см. проектный документ). Скрипт сам проверяет
версию при старте с этим бэкендом и честно предупреждает, а не падает
непонятным `KeyError: 'qwen3'` — но сам конфликт зависимостей решается
на месте, скорее всего через отдельное окружение/venv для ЭТОГО теста,
чтобы не трогать основной venv проекта до того, как модель докажет, что того
стоит.

Round "контекст в запросе": следующая идея из списка улучшений — пользователь
спросил "буд-то нужен контекст? по сути возвращаемся к тегам?". Ответ — НЕТ:
старая система (Round 9-10O) полагалась на то, что GigaChat САМ называл тему
словом ("ТЕМА: X"), и совпадение искалось по этому текстовому ярлыку (Round
10d задокументировал, что это ненадёжно — GigaChat путался, названная тема не
совпадала с содержанием его же ответа). Идея контекста в запросе не вводит
никакого именования темы и не обращается к GigaChat вообще — это расширение
ТЕКСТА, который эмбеддится для ПОИСКА (query), последними N записями САМОГО
ПУЛА перед вычислением вектора. Документы (то, что хранится в пуле) по-прежнему
эмбеддятся как одна изолированная реплика, как и в проде — меняется только то,
чем мы ищем, а не то, что хранится.

Гипотеза: категории vague_continuation/clarify_confirmation/ambiguous_backref
(см. выше) содержательно ПУСТЫ по тексту ("приведи ещё пример") — эмбеддинг
самой фразы в принципе не может нести сигнал о теме. Если подмешать в вектор
запроса несколько последних реплик реального разговора, пустая фраза
перестаёт быть пустой — вектор впитывает то, что обсуждалось прямо перед ней.

Риск, поднятый в чате: "а если между вопросом и ответом влезет ещё реплика?"
— тот же класс бага, что уже был найден и исправлен в Round 10O для тегов
("вопрос+1" — позиционное угадывание ответа, ломавшееся при вставке чужой
реплики за время генерации GigaChat). Наивное окно по СЫРЫМ строкам
транскрипта унаследовало бы этот риск. Решение — `_build_context_query_text()`
берёт окно из записей САМОГО ПУЛА (`ReplyPool.entries`, уже отфильтрован от
мусора, связь вопрос/ответ по id, не по позиции строки) — наследует защиту
Round 11 бесплатно, без возврата к бухгалтерии тегов (TrackedTag/TagTouch).

`--context-window N` (0 по умолчанию — выключено, полная обратная
совместимость) считается в ТОМ ЖЕ прогоне рядом с RAW/CENTERED, не отдельным
запуском: при N=0 столбцы *_context в CSV побайтово совпадают с *_raw (встроенная
проверка на регресс) — проверено офлайн fake-бэкендом. При N>0 нужно смотреть
margin именно в CONTEXT-варианте для vague_continuation/clarify_confirmation/
ambiguous_backref — в RAW/CENTERED там заведомо нет сигнала (см. выше), сравнивать
там нечего.

Первый реальный прогон с `--context-window` упал рекурсией (баг МОЙ, не
идеи — см. комментарий у `_REAL_EMBED_TEXT` ниже, исправлено). Сам прогон
на реальных цифрах после фикса пользователь ещё не делал — вместо этого
разговор пошёл дальше, к следующему уточнению (см. ниже), поэтому цифр по
голому `--context-window N` (окно из СЫРЫХ последних записей пула, без
разбора, пустые они или нет) пока нет.

Round "контекст без мусора" (следующий шаг того же обсуждения): пользователь
подметил параллель с человеческой памятью — реплики-пустышки ("приведи ещё
пример") мозг не запоминает как отдельное событие, они не создают новую
"запись", а просто продолжают то, что уже обсуждалось. Проверил идею НАПРЯМУЮ
на реальном файле разметки (без каких-либо эмбеддингов — просто текстовый
разбор stress_test_300_answer_key.json, см. чат): если считать "на сколько
реплик назад лежит правильный эталон" по ВСЕМ репликам подряд — для
vague_continuation попадание "с первого раза" (эталон — самая последняя
реплика) всего 14.5%, для ambiguous_backref — 0%. Причина, если посмотреть
руками: часто между настоящей темой и финальной пустой фразой стоит ЦЕЛАЯ
ЦЕПОЧКА других пустых фраз подряд ("распиши поподробнее" → "а можно проще?"
→ "покажи код" → только потом "да, ты прав") — наивное "последнее" в такой
цепочке указывает на другую пустую фразу, а не на тему.

Если же считать это расстояние, ПРОПУСКАЯ реплики категорий
vague_continuation/clarify_confirmation/garbage/ambiguous_backref при подсчёте
(то есть они не создают свою "запись" в счёте, ровно как предложил
пользователь) — картина заметно меняется для части категорий: ambiguous_backref
0% -> 30% "с первого раза" и 50% в пределах двух содержательных реплик назад;
vague_continuation 14.5% -> 22.9%. А вот clarify_confirmation как было 0%, так
и осталось — руками разобрано, что в этих тестовых данных подтверждения
("да, ты прав") специально ссылаются на тему за 10-40 содержательных реплик
назад, с кучей ДРУГИХ новых тем между ними — это структурно тот же класс
задачи, что и long_distance_recall (дальняя память), а не "что сейчас
активно", и окно тут в принципе не поможет, сколько его ни расширяй.

Из этого разбора и родился следующий эксперимент, реализованный в этом
раунде: `_build_context_query_text_content_only()` — то же самое окно контекста
для ЗАПРОСА, что и `_build_context_query_text()`, но собирается не из последних
`window` записей пула ПОДРЯД, а из последних `window` записей, ЧЬЯ КАТЕГОРИЯ
(из answer_key) не входит в `_EMPTY_CATEGORIES` — то есть пропуская пустые/
подтверждающие/мусорные реплики при отборе, как и в ручном разборе выше.

**Честно про то, что это НЕ проверяет и не может проверить**: категория
реплики берётся из ГОТОВОЙ РАЗМЕТКИ теста (answer_key) — это возможно только
потому, что мы точно знаем, какая реплика "пустая", а прод так не умеет. В
реальной работе (main.py/reply_pool.py) для этого нужен отдельный, ещё не
существующий классификатор "это пустая/содержательная фраза" — эвристика по
ключевым словам/длине или отдельный лёгкий вызов модели, это отдельная,
нерешённая часть задачи. Здесь эталонная категория используется только чтобы
проверить саму ГИПОТЕЗУ на цифрах ("если бы мы умели идеально отличать пустое
от содержательного — помогло бы это?"), прежде чем городить классификатор,
который сам может ошибаться и вносить свой собственный шум.

Флаг `--context-filter-empty` включает этот режим (требует `--context-window
N` с N>0, иначе фильтровать нечего). Без флага — старое поведение
`--context-window` не меняется (окно из сырых последних записей пула,
включая пустые), полная обратная совместимость, ничего не удалено.

**Не проверено на реальных данных** — нужен прогон пользователя с
`--context-window 2 --context-filter-empty` (и то же без флага, для сравнения)
на обоих файлах.

Запуск (на машине пользователя, из корня проекта):
    venv\\Scripts\\python.exe src\\batch_test_pool_similarity.py
    venv\\Scripts\\python.exe src\\batch_test_pool_similarity.py --model qwen3-embedding-0.6b
    venv\\Scripts\\python.exe src\\batch_test_pool_similarity.py --context-window 2 --out pool_sim_300_ctx2.csv
    venv\\Scripts\\python.exe src\\batch_test_pool_similarity.py --context-window 2 --context-filter-empty --out pool_sim_300_ctx2_filtered.csv
    venv\\Scripts\\python.exe src\\batch_test_pool_similarity.py --model qwen3-embedding-0.6b --scenario data\\stress_test_v2.json --answer-key data\\stress_test_v2_answer_key.json
    venv\\Scripts\\python.exe src\\batch_test_pool_similarity.py --out pool_sim_300.csv

Флаги:
    --scenario <файл>     JSON со сценарием вместо data/stress_test_300.json
    --answer-key <файл>   JSON с разметкой вместо data/stress_test_300_answer_key.json
    --model <имя>          user-bge-m3 (по умолчанию, текущая прод-модель) или
                           qwen3-embedding-0.6b (эксперимент, см. выше)
    --context-window N     сколько последних записей пула добавлять к тексту
                           запроса перед эмбеддингом (0 по умолчанию — выключено,
                           см. Round "контекст в запросе" выше)
    --context-filter-empty  при построении окна пропускать записи категорий
                           vague_continuation/clarify_confirmation/garbage/
                           ambiguous_backref (см. Round "контекст без мусора"
                           выше) — ТОЛЬКО с готовой разметкой теста, недоступно
                           в проде как есть. Без --context-window > 0 не имеет
                           эффекта.
    --out <файл>          CSV с построчными результатами (по умолчанию
                           pool_similarity_results.csv — стоит явно указывать
                           РАЗНЫЕ файлы для разных моделей/режимов, иначе
                           результаты перезапишут друг друга)
    --limit N              обработать только первые N реплик сценария

ВАЖНО, честно: сам скрипт не требует ни GigaChat, ни Ollama, ни аудио, ни
Tkinter — только embeddings.py/reply_pool.py (плюс sentence_transformers
напрямую для qwen3-бэкенда). Логика (оба бэкенда, центрирование, контекстное окно, поиск дублей)
сначала проверена офлайн на детерминированной fake-модели, реальные цифры —
с прогона на настоящей модели.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

import embeddings
from reply_pool import ReplyPool

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO_PATH = ROOT_DIR / "data" / "stress_test_300.json"
DEFAULT_ANSWER_KEY_PATH = ROOT_DIR / "data" / "stress_test_300_answer_key.json"
DEFAULT_OUT_PATH = ROOT_DIR / "pool_similarity_results.csv"

# "Я"-реплики (собственный уточняющий вопрос ассистента) — на них подсказка
# не генерируется даже в реальном main.py по умолчанию (HINTS_ON_MIC=false),
# как и в batch_test_gigachat_topics.py._SKIP_ALWAYS.
_SKIP_ALWAYS = {"clarify_question"}

# Категории, где related_topic указывает на РЕАЛЬНУЮ более раннюю тему —
# для остальных (new_topic/garbage) совпадение не ожидается вообще, см.
# докстринг файла.
_NO_MATCH_EXPECTED = {"new_topic", "garbage"}

# Round "контекст без мусора" (см. докстринг файла) — категории, которые
# считаем "пустыми по содержанию" при построении контекстного ОКНА (не при
# поиске — сами эти реплики по-прежнему кладутся в пул и участвуют в поиске
# как обычно, меняется только то, что попадает в текст ЗАПРОСА). ЧЕСТНО: это
# работает только потому, что категория здесь берётся из готовой разметки
# теста (answer_key) — прямой аналог такого решения в проде (main.py) не
# существует, там нужен отдельный классификатор "пустая фраза или нет" (см.
# докстринг файла, что именно не проверено).
_EMPTY_CATEGORIES = {"vague_continuation", "clarify_confirmation", "garbage", "ambiguous_backref"}

THRESHOLD_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Захватываем НАСТОЯЩУЮ embeddings.embed_text ДО того, как run() подменит
# embeddings.embed_text на backend.embed_document (см. run() ниже). Без этого
# — реальный баг, найденный на прогоне пользователя ("maximum recursion depth
# exceeded"): _UserBgeM3Backend.embed_document сам вызывает
# embeddings.embed_text(text) для расчёта настоящего эмбеддинга через
# deepvk/USER-bge-m3, но после монкипатча в run() имя embeddings.embed_text
# указывает на backend.embed_document — то есть САМ НА СЕБЯ, и вызов уходит в
# бесконечную рекурсию. У _Qwen3EmbeddingBackend такой проблемы не было
# (она вызывает SentenceTransformer напрямую, а не embeddings.embed_text) —
# поэтому баг не проявлялся на прогонах с --model qwen3-embedding-0.6b и
# всплыл только сейчас, на первом реальном прогоне user-bge-m3 ПОСЛЕ того,
# как была введена абстракция бэкендов. Захват ссылки на функцию здесь, на
# уровне модуля, происходит один раз при импорте — до любого монкипатча.
_REAL_EMBED_TEXT = embeddings.embed_text


class _EmbeddingBackend:
    """
    Абстракция "документ vs запрос" для сравнения моделей эмбеддингов без
    правки embeddings.py/reply_pool.py (см. докстринг модуля). embed_document
    — то, что реально хранится в пуле (add_question/add_answer в проде);
    embed_query — то, чем ищем прямо сейчас (в проде сейчас используется тот
    же самый вектор, что и document, потому что deepvk/USER-bge-m3
    симметрична — см. ниже).
    """

    name = "base"

    def embed_document(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class _UserBgeM3Backend(_EmbeddingBackend):
    """Текущая прод-модель (Round 11). Симметричная — документ и запрос
    эмбеддятся одинаково, без инструкционных префиксов (см. докстринг
    embeddings.py и разбор в чате — подтверждено и для родительской
    BAAI/bge-m3 автором модели)."""

    name = "user-bge-m3"

    def embed_document(self, text: str) -> np.ndarray:
        # _REAL_EMBED_TEXT, не embeddings.embed_text — см. комментарий у
        # объявления _REAL_EMBED_TEXT выше (иначе бесконечная рекурсия после
        # монкипатча в run()).
        return _REAL_EMBED_TEXT(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_document(text)


class _Qwen3EmbeddingBackend(_EmbeddingBackend):
    """
    Qwen/Qwen3-Embedding-0.6B — эксперимент (см. докстринг модуля). В отличие
    от USER-bge-m3, АСИММЕТРИЧНАЯ: запрос эмбеддится с инструкционным
    префиксом (`prompt_name="query"`, встроен в ST-конфиг модели), документ —
    без префикса. Требует transformers>=4.51.0 — проверяется в _check_ready().
    """

    name = "qwen3-embedding-0.6b"
    _MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self) -> None:
        self._model = None

    def _check_ready(self) -> None:
        import transformers

        installed = tuple(int(p) for p in transformers.__version__.split(".")[:2])
        if installed < (4, 51):
            raise RuntimeError(
                f"Qwen3-Embedding требует transformers>=4.51.0, установлен {transformers.__version__}. "
                f"В requirements.txt проекта transformers жёстко зафиксирован на 4.44.2 (нужен "
                f"legacy rubert-tiny2 пути и совместимости с faster-whisper) — для ЭТОГО теста "
                f"придётся временно поставить более новый transformers (например "
                f"`pip install \"transformers>=4.51.0,<5.0.0\"` в ОТДЕЛЬНОМ venv, чтобы не сломать "
                f"основное окружение проекта до того, как модель докажет, что того стоит)."
            )

    def _load(self):
        if self._model is None:
            self._check_ready()
            from sentence_transformers import SentenceTransformer

            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            self._model = SentenceTransformer(self._MODEL_ID, device=device)
            print(f"[embeddings] {self._MODEL_ID} загружена на устройство: {device}")
        return self._model

    def embed_document(self, text: str) -> np.ndarray:
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        vec = model.encode(text, prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype(np.float32)


_BACKENDS: dict[str, type[_EmbeddingBackend]] = {
    _UserBgeM3Backend.name: _UserBgeM3Backend,
    _Qwen3EmbeddingBackend.name: _Qwen3EmbeddingBackend,
}


def _load(scenario_path: Path, answer_key_path: Path) -> tuple[list[dict], list[dict]]:
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    if len(scenario) != len(answer_key):
        raise ValueError(
            f"scenario ({len(scenario)} реплик) и answer_key ({len(answer_key)} реплик) "
            f"разной длины — файлы должны быть парой одной схемы"
        )
    return scenario, answer_key


def _centered_search(pool: ReplyPool, query_id: int, query_vec_raw: np.ndarray) -> list[tuple[object, float]]:
    """
    Тот же поиск, что pool.search(), но с коррекцией анизотропии: перед
    сравнением из КАЖДОГО вектора (и запроса, и кандидатов) вычитается
    среднее по ВСЕМ эмбеддингам, накопленным в пуле на данный момент
    (плюс сырой вектор запроса — важно для асимметричных моделей вроде
    Qwen3, где вектор запроса и вектор документа с тем же текстом РАЗНЫЕ, и
    среднее без учёта query было бы нечестным), и результат заново
    нормализуется. Среднее пересчитывается заново при каждом вызове —
    намеренно, см. докстринг модуля (согласованность между запросом и
    документами важнее, чем стабильность среднего между вызовами).
    """
    doc_vecs = [e.embedding for e in pool.entries]
    mean_vec = np.stack(doc_vecs + [query_vec_raw]).mean(axis=0)

    def _center_normalize(v: np.ndarray) -> np.ndarray:
        c = v - mean_vec
        norm = np.linalg.norm(c)
        return c / norm if norm > 0 else c

    query_vec = _center_normalize(query_vec_raw)

    scored: list[tuple[object, float]] = []
    for e in pool.entries:
        if e.id == query_id:
            continue
        cv = _center_normalize(e.embedding)
        scored.append((e, float(np.dot(query_vec, cv))))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _build_context_query_text(pool: ReplyPool, qid: int, text: str, speaker: str, window: int) -> str:
    """
    Round "контекст в запросе" (см. обсуждение в чате): строит текст
    ЗАПРОСА (только для поиска — то, что хранится в пуле как ДОКУМЕНТ,
    по-прежнему эмбеддится как одна изолированная реплика, ровно как в
    проде) с добавлением последних `window` записей САМОГО ПУЛА перед
    текущей репликой.

    Намеренно берём записи ПУЛА, а не сырые строки транскрипта по номеру:
    пул уже отфильтрован от мусорных/вставленных реплик (см. докстринг
    reply_pool.py, "что НЕ попадает в пул"), и связь вопрос/ответ внутри
    него идёт по id (question_id/answer_id), а не по позиции строки. Это
    сознательный выбор именно из-за риска, который пользователь поднял в
    чате: "а если между вопросом и ответом влезет ещё реплика?" — тот же
    класс проблемы уже был найден и исправлен в Round 10O для старой
    системы тегов ("вопрос+1" — позиционное угадывание ответа, когда за
    время генерации GigaChat, 1-3+ секунды, собеседник успевает сказать
    что-то ещё, и это вклинивается между вопросом и ещё не записанным
    ответом). Наивное окно по сырым строкам транскрипта унаследовало бы
    именно этот риск (посторонняя реплика попадёт в контекст); окно по
    записям пула — нет, потому что мусорные/вставленные реплики либо не
    проходят фильтр до пула вообще, либо попадают туда отдельной
    content-полной записью со своей собственной парой вопрос+ответ, не
    путая существующую пару.

    window <= 0 — без изменений, возвращает text как есть (старое
    поведение, полная обратная совместимость с прогонами без контекста).
    """
    if window <= 0:
        return text
    prior = [e for e in pool.entries if e.id != qid][-window:]
    lines = [f"{e.speaker}: {e.text}" for e in prior]
    lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _build_context_query_text_content_only(
    pool: ReplyPool,
    content_entry_ids: list[int],
    text: str,
    speaker: str,
    window: int,
) -> str:
    """
    Round "контекст без мусора" (см. докстринг файла) — тот же принцип, что
    _build_context_query_text() выше (окно строится из ЗАПИСЕЙ ПУЛА по id,
    не из сырых строк транскрипта по номеру — то же наследование защиты от
    Round 10O), но берёт не последние `window` записей пула ПОДРЯД, а
    последние `window` записей ИЗ content_entry_ids — списка id, которые
    вызывающий код (run()) заполняет только для реплик "содержательных"
    категорий (не входящих в _EMPTY_CATEGORIES), в порядке появления.

    Иначе говоря: пустые/подтверждающие/мусорные реплики пропускаются при
    отборе окна — как будто их не было — вместо того, чтобы окно из двух
    "последних записей" наполовину состояло из других пустых фраз (см.
    докстринг файла, разбор реальных примеров вроде цепочки "распиши
    поподробнее" -> "а можно проще?" -> "покажи код" -> "да, ты прав").

    window <= 0 — без изменений, возвращает text как есть (та же гарантия
    обратной совместимости, что и у _build_context_query_text()).
    """
    if window <= 0:
        return text
    ids = content_entry_ids[-window:]
    by_id = {e.id: e for e in pool.entries}
    lines = [f"{by_id[cid].speaker}: {by_id[cid].text}" for cid in ids if cid in by_id]
    lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def run(
    scenario: list[dict],
    answer_key: list[dict],
    backend: _EmbeddingBackend,
    limit: int | None = None,
    context_window: int = 0,
    context_filter_empty: bool = False,
) -> list[dict]:
    # Монкипатчим embeddings.embed_text на выбранный бэкенд — reply_pool.py
    # вызывает именно эту функцию внутри add_question/add_answer (см.
    # docstring модуля), так их не нужно трогать вообще, чтобы поменять
    # модель для этого изолированного теста.
    embeddings.embed_text = backend.embed_document

    # Разовая проверка ДО цикла: reply_pool._embed_safe ловит любую ошибку
    # эмбеддинга молча (по одной строке за раз, чтобы не ронять звонок в
    # проде) — если модель вообще не грузится (не тот transformers, нет
    # сети, опечатка в имени), без этой проверки та же самая ошибка
    # напечаталась бы по разу на КАЖДУЮ из ~300 реплик подряд, что бесполезно
    # шумно. Явно пробуем один раз и останавливаемся с понятным сообщением.
    try:
        backend.embed_document("тестовая фраза для проверки загрузки модели")
    except Exception as e:
        print(f"\nМодель '{backend.name}' не загрузилась: {e}")
        print("Прерываю прогон — нет смысла повторять одну и ту же ошибку на каждой реплике.")
        return []

    pool = ReplyPool()
    # Первое появление дословного текста реплики -> id её записи в пуле.
    # related_topic всегда ссылается на ДОСЛОВНЫЙ текст более ранней реплики
    # (так собран весь набор — см. проектный документ, round 10/10f) — это
    # НЕ имитация судьи/GigaChat, а прямое чтение эталонной разметки.
    entry_id_by_text: dict[str, int] = {}
    rows_out: list[dict] = []
    # Round "контекст без мусора" — id записей пула, чья категория (из
    # answer_key) НЕ входит в _EMPTY_CATEGORIES, в порядке появления. Только
    # для --context-filter-empty (см. _build_context_query_text_content_only).
    content_entry_ids: list[int] = []

    n = len(scenario) if limit is None else min(limit, len(scenario))
    for i in range(n):
        srow = scenario[i]
        krow = answer_key[i]
        category = krow.get("category", "?")
        text = srow["text"]
        speaker = srow.get("speaker", "Собеседник")

        if category in _SKIP_ALWAYS:
            continue

        qid = pool.add_question(text, line_index=i, speaker=speaker)
        if qid is None:
            print(f"[{i}] не удалось посчитать embedding для вопроса — пропускаю строку")
            continue
        entry_id_by_text.setdefault(text, qid)

        # entry.embedding — это DOCUMENT-вектор этой реплики (посчитан внутри
        # add_question через пропатченный embeddings.embed_text выше). Для
        # симметричной модели (user-bge-m3) query-вектор совпадает с ним; для
        # асимметричной (qwen3) — нет, поэтому явно считаем отдельно.
        query_vec = backend.embed_query(text)
        by_id = {e.id: e for e in pool.entries}
        scored_raw = pool.search(query_vec, exclude_id=qid)
        scored_centered = _centered_search(pool, qid, query_vec) if len(pool.entries) > 1 else []

        # Контекстный вариант (см. _build_context_query_text) — считается
        # ВСЕГДА в том же прогоне, рядом с RAW/CENTERED, а не отдельным
        # запуском: при context_window=0 query_text совпадает с text и
        # scored_context будет побайтово равен scored_raw (удобная проверка
        # на регресс), при context_window>0 — прямое сравнение "с контекстом
        # vs без" на одних и тех же данных и той же модели.
        if context_filter_empty:
            query_text_ctx = _build_context_query_text_content_only(
                pool, content_entry_ids, text, speaker, context_window
            )
        else:
            query_text_ctx = _build_context_query_text(pool, qid, text, speaker, context_window)
        query_vec_ctx = backend.embed_query(query_text_ctx) if context_window > 0 else query_vec
        scored_context = pool.search(query_vec_ctx, exclude_id=qid)

        # content_entry_ids копится ПОСЛЕ того, как текущая реплика уже
        # использовала его состояние ДО себя (см. вызов выше) — иначе реплика
        # включала бы саму себя в собственное контекстное окно.
        if category not in _EMPTY_CATEGORIES:
            content_entry_ids.append(qid)

        expected_match = category not in _NO_MATCH_EXPECTED
        related = krow.get("related_topic")
        candidates = related if isinstance(related, list) else ([related] if related else [])

        correct_ids: set[int] = set()
        missing_candidates: list[str] = []
        if expected_match:
            for cand_text in candidates:
                cand_qid = entry_id_by_text.get(cand_text)
                if cand_qid is None:
                    missing_candidates.append(cand_text)
                    continue
                correct_ids.add(cand_qid)
                cand_entry = by_id[cand_qid]
                if cand_entry.answer_id is not None:
                    correct_ids.add(cand_entry.answer_id)

        def _best_correct_noise(scored: list[tuple[object, float]]) -> tuple[float | None, float | None]:
            correct_scores = [score for e, score in scored if e.id in correct_ids]
            noise_scores = [score for e, score in scored if e.id not in correct_ids]
            return (
                max(correct_scores) if correct_scores else None,
                max(noise_scores) if noise_scores else None,
            )

        best_correct_raw, best_noise_raw = _best_correct_noise(scored_raw)
        best_correct_c, best_noise_c = _best_correct_noise(scored_centered)
        best_correct_ctx, best_noise_ctx = _best_correct_noise(scored_context)

        # Похожесть >= 0.999 между РАЗНЫМИ репликами практически всегда значит,
        # что тексты дословно (или почти дословно) совпадают — эмбеддинг не
        # "угадывает" смысл тут, он просто получает на вход два одинаковых
        # текста. Центрирование эту причину не лечит (см. докстринг модуля —
        # если два вектора идентичны, вычитание одного и того же среднего из
        # обоих и повторная нормализация оставит их идентичными, косинус
        # снова будет 1.0) — значит это отдельная, не геометрическая причина,
        # стоит увидеть её отдельно от общей сводки по категориям.
        near_dup = [(e.text, e.speaker) for e, s in scored_raw if e.id not in correct_ids and s >= 0.999]

        def _fmt(v: float | None) -> str | float:
            return round(v, 4) if v is not None else ""

        def _margin(a: float | None, b: float | None) -> str | float:
            return round(a - b, 4) if (a is not None and b is not None) else ""

        rows_out.append(
            {
                "index": i,
                "category": category,
                "speaker": speaker,
                "text": text,
                "expected_match": expected_match,
                "related_topic": " | ".join(candidates) if candidates else "",
                "candidate_missing_in_pool": "; ".join(missing_candidates) if missing_candidates else "",
                "best_correct_raw": _fmt(best_correct_raw),
                "best_noise_raw": _fmt(best_noise_raw),
                "margin_raw": _margin(best_correct_raw, best_noise_raw),
                "best_correct_centered": _fmt(best_correct_c),
                "best_noise_centered": _fmt(best_noise_c),
                "margin_centered": _margin(best_correct_c, best_noise_c),
                "best_correct_context": _fmt(best_correct_ctx),
                "best_noise_context": _fmt(best_noise_ctx),
                "margin_context": _margin(best_correct_ctx, best_noise_ctx),
                "top5_raw": "; ".join(f"{e.text[:50]!r}~{s:.3f}" for e, s in scored_raw[:5]),
                "top5_centered": "; ".join(f"{e.text[:50]!r}~{s:.3f}" for e, s in scored_centered[:5]),
                "top5_context": "; ".join(f"{e.text[:50]!r}~{s:.3f}" for e, s in scored_context[:5]),
                "near_duplicate_text": " || ".join(f"{spk}: {txt[:80]!r}" for txt, spk in near_dup[:3]) if near_dup else "",
            }
        )

        ref_answer = krow.get("reference_answer")
        if ref_answer:
            aid = pool.add_answer(qid, ref_answer, line_index=i)
            # Ответ на содержательную реплику — тоже содержательная запись
            # (та же пара, что и correct_ids выше резолвит question_id/
            # answer_id в одну тему целиком, см. докстринг reply_pool.py).
            if aid is not None and category not in _EMPTY_CATEGORIES:
                content_entry_ids.append(aid)

    return rows_out


def _percentiles(values: list[float]) -> str:
    if not values:
        return "нет данных"
    values = sorted(values)
    return (
        f"min={values[0]:.3f} p25={values[len(values)//4]:.3f} "
        f"median={statistics.median(values):.3f} p75={values[3*len(values)//4]:.3f} "
        f"max={values[-1]:.3f} (n={len(values)})"
    )


def _threshold_sweep(rows_out: list[dict], correct_field: str, noise_field: str) -> None:
    expected_rows = [r for r in rows_out if r["expected_match"] and r[correct_field] != ""]
    baseline_rows = [r for r in rows_out if not r["expected_match"] and r[noise_field] != ""]
    print(f"{'порог':>6} | {'recall (найден верный)':>24} | {'ложный шум (new_topic/garbage выше порога)':>44}")
    for t in THRESHOLD_SWEEP:
        recall_hits = sum(1 for r in expected_rows if r[correct_field] >= t)
        recall = recall_hits / len(expected_rows) if expected_rows else float("nan")
        fp_hits = sum(1 for r in baseline_rows if r[noise_field] >= t)
        fp_rate = fp_hits / len(baseline_rows) if baseline_rows else float("nan")
        print(f"{t:>6.2f} | {recall_hits:>3}/{len(expected_rows):<3} = {recall:>6.1%}      | {fp_hits:>3}/{len(baseline_rows):<3} = {fp_rate:>6.1%}")


def _report_near_duplicates(rows_out: list[dict], limit: int = 20) -> None:
    dup_rows = [r for r in rows_out if r["near_duplicate_text"]]
    print(f"\n=== Дословные (или почти дословные) дубли текста — похожесть >= 0.999 на ДРУГУЮ, не связанную по теме реплику ===")
    print(f"Найдено строк: {len(dup_rows)} из {len(rows_out)}")
    if not dup_rows:
        print("(нет ни одной — дубли не объясняют картину, причина в другом)")
        return
    by_cat: dict[str, int] = {}
    for r in dup_rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print("По категориям:", ", ".join(f"{c}={n}" for c, n in sorted(by_cat.items(), key=lambda kv: -kv[1])))
    print(f"\nПримеры (первые {min(limit, len(dup_rows))}):")
    for r in dup_rows[:limit]:
        print(f"  [{r['index']}] {r['category']} | {r['speaker']}: {r['text'][:70]!r}")
        print(f"        совпало с: {r['near_duplicate_text']}")


def summarize(rows_out: list[dict], context_window: int = 0) -> None:
    _report_near_duplicates(rows_out)

    print("\n=== По категориям: RAW (сырой косинус, как сейчас в проде) vs CENTERED (с коррекцией анизотропии) vs CONTEXT (запрос с контекстом пула) ===")
    if context_window <= 0:
        print(
            "(context_window=0 — колонки CONTEXT побайтово совпадают с RAW, "
            "это ожидаемо; передай --context-window N > 0, чтобы увидеть эффект)"
        )
    categories = sorted({r["category"] for r in rows_out})
    for cat in categories:
        cat_rows = [r for r in rows_out if r["category"] == cat]
        correct_raw = [r["best_correct_raw"] for r in cat_rows if r["best_correct_raw"] != ""]
        noise_raw = [r["best_noise_raw"] for r in cat_rows if r["best_noise_raw"] != ""]
        correct_c = [r["best_correct_centered"] for r in cat_rows if r["best_correct_centered"] != ""]
        noise_c = [r["best_noise_centered"] for r in cat_rows if r["best_noise_centered"] != ""]
        correct_ctx = [r["best_correct_context"] for r in cat_rows if r["best_correct_context"] != ""]
        noise_ctx = [r["best_noise_context"] for r in cat_rows if r["best_noise_context"] != ""]
        missing = sum(1 for r in cat_rows if r["candidate_missing_in_pool"])
        noise_label = "шум (baseline)" if cat in _NO_MATCH_EXPECTED else "ближайший ПОСТОРОННИЙ конкурент"
        print(f"\n[{cat}] n={len(cat_rows)}")
        if correct_raw:
            print(f"  RAW      правильный кандидат: {_percentiles(correct_raw)}")
        if noise_raw:
            print(f"  RAW      {noise_label}:        {_percentiles(noise_raw)}")
        if correct_c:
            print(f"  CENTERED правильный кандидат: {_percentiles(correct_c)}")
        if noise_c:
            print(f"  CENTERED {noise_label}:        {_percentiles(noise_c)}")
        if correct_ctx:
            print(f"  CONTEXT  правильный кандидат: {_percentiles(correct_ctx)}")
        if noise_ctx:
            print(f"  CONTEXT  {noise_label}:        {_percentiles(noise_ctx)}")
        if missing:
            print(f"  !! {missing} строк(и) не нашли эталонный кандидат дословно в пройденной части сценария — проверить вручную")

    print("\n=== Подбор порога — RAW (сырой косинус) ===")
    _threshold_sweep(rows_out, "best_correct_raw", "best_noise_raw")

    print("\n=== Подбор порога — CENTERED (с коррекцией анизотропии) ===")
    _threshold_sweep(rows_out, "best_correct_centered", "best_noise_centered")

    print(f"\n=== Подбор порога — CONTEXT (запрос + последние {context_window} записи пула) ===")
    _threshold_sweep(rows_out, "best_correct_context", "best_noise_context")

    print(
        "\nКак читать: сравните три таблицы подбора порога выше. Если CONTEXT даёт "
        "заметно более высокий recall при том же (или более низком) уровне ложного "
        "шума, чем RAW — особенно смотри на категории vague_continuation/"
        "clarify_confirmation/ambiguous_backref в разбивке по категориям выше (там "
        "текст сам по себе пуст по смыслу, поэтому именно там контекст и должен "
        "помочь, если гипотеза верна). Если разницы почти нет или CONTEXT хуже — "
        "гипотеза не подтвердилась, и добавление контекста в запрос не решает "
        "задачу само по себе. Про CENTERED (коррекция анизотропии) вывод уже "
        "сделан отдельно — см. проектный документ: она систематически ухудшает "
        "recall на реальных данных, приведена тут только для полноты картины."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--answer-key", type=Path, default=DEFAULT_ANSWER_KEY_PATH)
    parser.add_argument("--model", choices=sorted(_BACKENDS), default=_UserBgeM3Backend.name)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Сколько последних записей ПУЛА (не сырых строк транскрипта — см. "
            "докстринг _build_context_query_text) добавлять к тексту текущей "
            "реплики перед вычислением query-вектора. 0 (по умолчанию) — "
            "без контекста, старое поведение. Столбцы *_context в CSV при "
            "этом побайтово совпадают со столбцами *_raw."
        ),
    )
    parser.add_argument(
        "--context-filter-empty",
        action="store_true",
        help=(
            "Round 'контекст без мусора' (см. докстринг файла) — при "
            "построении окна пропускать записи категорий vague_continuation/"
            "clarify_confirmation/garbage/ambiguous_backref (по разметке "
            "answer_key, ТОЛЬКО для этого теста, недоступно в проде как "
            "есть). Без --context-window > 0 не имеет эффекта."
        ),
    )
    args = parser.parse_args()

    print(f"Сценарий: {args.scenario}")
    print(f"Эталон:   {args.answer_key}")
    print(f"Модель:   {args.model}")
    if args.context_window > 0:
        mode = "без мусора (--context-filter-empty)" if args.context_filter_empty else "сырое окно последних записей пула"
        print(f"Контекст запроса: {args.context_window} посл. запис(и/ей) пула, режим: {mode}")
    else:
        print("Контекст запроса: выключен (0)")
        if args.context_filter_empty:
            print("(--context-filter-empty указан, но без --context-window > 0 не имеет эффекта)")
    scenario, answer_key = _load(args.scenario, args.answer_key)
    backend = _BACKENDS[args.model]()

    print("Загружаю модель эмбеддингов (первый вызов может занять время — "
          "скачивание с HuggingFace, если модель ещё не в кеше)...")
    rows_out = run(
        scenario,
        answer_key,
        backend,
        limit=args.limit,
        context_window=args.context_window,
        context_filter_empty=args.context_filter_empty,
    )

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nCSV записан: {args.out} ({len(rows_out)} строк)")

    summarize(rows_out, context_window=args.context_window)


if __name__ == "__main__":
    main()
