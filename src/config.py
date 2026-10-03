"""
Загрузка настроек из .env в один объект Settings.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Ищем .env в корне проекта (на уровень выше src/)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get_int(name: str) -> int | None:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return None
    return int(val)


def _get_model_list(name: str, default: str) -> list[str]:
    """Round 11: список моделей GigaChat через запятую (каскад отката при
    исчерпании токенов, см. Settings.gigachat_model_cascade) — пустые
    элементы (двойная запятая, случайный пробел) отбрасываются."""
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    deepgram_api_key: str | None

    mic_device_index: int | None
    system_audio_device_index: int | None
    # Полностью отключить захват микрофона (свой голос вообще не пишется и
    # не попадает в транскрипт/подсказки — только собеседник из системного
    # звука). MIC_DEVICE_INDEX при этом не нужен и не проверяется.
    mic_capture_enabled: bool

    stt_mode: str  # auto | local | cloud
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_initial_prompt: str | None

    language: str
    output_dir: Path
    diarization_enabled: bool
    diarization_similarity_threshold: float

    # Саммари после звонка — локальная модель через Ollama, без ключей.
    ollama_model: str
    ollama_host: str
    # Размер контекстного окна (в токенах) для ЖИВЫХ подсказок во время
    # звонка (не для саммари после звонка — там нужен весь транскрипт целиком,
    # см. summarizer.py, туда это не передаётся). Меньше контекст — меньше
    # видеопамяти под KV-кэш модели, и больше шансов, что вся модель целиком
    # поместится на GPU вместо частичного ухода на CPU (см. историю про 8ГБ
    # видеокарту и gemma2:9b — на дефолтных 4096 модель не помещалась целиком,
    # на 2048 стала помещаться). Подсказкам хватает с большим запасом — в
    # промпт и так уходит только HINTS_CONTEXT_LINES строк транскрипта.
    ollama_num_ctx: int
    # Отдельный, СВОЙ размер контекста для судьи (judge.py) — НЕ ollama_num_ctx
    # выше, хотя судья тоже всегда ходит в Ollama (см. judge.py). Промт судьи
    # (правила + 9 примеров) сам по себе занимает ~2000 расчётных токенов —
    # даже после сжатия формулировок (round 8) реальные звонки показывают
    # переполнение при ollama_num_ctx=2048 нестабильными симптомами (то один,
    # то другой кейс проваливается между прогонами при неизменном промте —
    # признак обрезки контекста, а не проблемы с качеством правил). Значение
    # по умолчанию (3072) — ПЕРВАЯ ПОПЫТКА, не проверенная на реальном железе:
    # тот же 8ГБ GPU/gemma2:9b, что уже один раз не поместился целиком на
    # num_ctx=4096 (см. комментарий у ollama_num_ctx) — не помещение на 3072
    # тоже возможно, и увидеть это можно либо по явной ошибке Ollama, либо по
    # резкому росту задержки судьи (модель частично ушла на CPU). Меняйте
    # JUDGE_NUM_CTX в .env, если понадобится подобрать другое значение опытным
    # путём.
    judge_num_ctx: int

    # Live-подсказки поверх экрана во время звонка (Этап 2).
    live_hints_enabled: bool
    hints_context_lines: int

    # ВРЕМЕННЫЙ флаг для сольной тренировки без второго собеседника: если true,
    # подсказки генерируются и на фразы из "Я" (мик), а не только на "Собеседник"
    # (система/VB-Cable). По умолчанию false — обычное поведение звонка не меняется.
    hints_on_mic: bool

    # Писать ли каждый сгенерированный вопрос+ответ отдельным файлом сессии
    # (session_dir/questions_and_answers.md, см. hint_log.py) — до сих пор
    # ответ жил только в оверлее и терялся при его закрытии, в transcript.txt
    # самих ответов модели нет вообще (там только то, что произнесено вслух).
    # По умолчанию включено — это чисто аддитивная запись, поведение самих
    # подсказок никак не меняет, можно выключить в .env, если файл не нужен.
    hints_log_enabled: bool

    # Свободный текст про сам разговор, который не меняется от
    # вопроса к вопросу (например, язык программирования по умолчанию, стек,
    # позиция) — подставляется в промпт live-подсказок отдельным блоком
    # "Контекст этого разговора" перед историей транскрипта (см.
    # _build_user_message в live_hints.py). Придумано из-за конкретного
    # случая: GigaChat, отвечая на вопрос "напиши функцию для проверки
    # палиндрома" (без указания языка), один раз выдала код на C/C++ вместо
    # ожидаемого Python — модели попросту неоткуда было знать, какой язык
    # вообще имеется в виду. По умолчанию не задан (None) — ничего не
    # меняется в поведении, пока не впишете в .env.
    hints_extra_context: str | None

    # Локальная база знаний для live-подсказок (заранее подготовленные
    # пары вопрос/ответ — например, справочник по продукту) — необязательна,
    # по умолчанию выключена, чтобы не менять поведение без явного включения.
    kb_enabled: bool
    kb_path: Path | None
    kb_top_k: int
    kb_min_score: float

    # Облачная модель GigaChat (Сбер) — Этап 3 начинался с идеи "только для
    # сложных вопросов", но по итогам сравнительных прогонов (см. историю в
    # чате: batch_test.py/batch_test_search.py/batch_test_followup.py/
    # batch_test_situational.py на реальных наборах вопросов) GigaChat-2
    # оказалась быстрее и не хуже локальной Ollama на ОБЫЧНЫХ вопросах тоже —
    # поэтому теперь GigaChat, если включена, отвечает НА ВСЕ вопросы, просто
    # разными своими моделями: gigachat_model (сильная/дорогая, Ultra) — для
    # вопросов, которые main.py считает "сложными" (см.
    # _looks_like_hard_question), gigachat_normal_model (лёгкая/дешёвая,
    # Lite/GigaChat-2) — для всех остальных. Ollama осталась ЕДИНСТВЕННЫМ
    # запасным вариантом — не по классификатору "сложный/обычный" (это
    # слишком нечёткое понятие для выбора МЕЖДУ облаком и локалью, см.
    # обсуждение в чате), а чисто механически: любая ошибка обращения к
    # GigaChat (сеть, лимит токенов, ключ) — откат на локальную модель, см.
    # main.py hint_worker(). По умолчанию GigaChat выключена целиком, чтобы
    # поведение не менялось без явного включения; см. .env.example.
    gigachat_enabled: bool
    gigachat_credentials: str | None
    gigachat_model: str
    gigachat_normal_model: str
    # Путь к корневому сертификату "Минцифры России" (Russian Trusted Root
    # CA) — без него запросы к GigaChat падают с SSL: CERTIFICATE_VERIFY_FAILED
    # (self-signed certificate in certificate chain), т.к. GigaChat подписан
    # этим удостоверяющим центром, а его нет в стандартном списке доверенных
    # корневых сертификатов. Официальный файл: см. .env.example.
    gigachat_ca_bundle_file: Path | None

    # Round 10M (см. проектный документ) — семантический recall "варианта Б":
    # находить старый ОТВЕТ по смыслу текущего вопроса (через эмбеддинги,
    # см. main.py TrackedTag.answer_embedding), а не только по буквальным
    # словам "повтори"/"тот же" (round 10L, _looks_like_repeat_request).
    # ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНО — пользователь явно попросил внедрять аккуратно,
    # с возможностью отката без правки кода. Похожесть при этом СЧИТАЕТСЯ И
    # ЛОГИРУЕТСЯ в любом случае (см. main.py hint_worker, "[recall] похожесть
    # ...") — это чисто диагностика с нулевым риском, нужна, чтобы накопить
    # реальные числа с живых звонков и осознанно подобрать порог, прежде чем
    # разрешать этому влиять на настоящие подсказки. Включайте (true) только
    # после того, как посмотрите реальные значения в logs\latest.log.
    semantic_recall_enabled: bool
    # Порог косинусной похожести (0..1), при котором semantic_recall_enabled
    # выше срабатывает. 0.72 — ПЕРВАЯ ПОПЫТКА (как в своё время JUDGE_NUM_CTX,
    # см. комментарий выше) — не откалибровано на реальных данных этого
    # проекта, только общее ожидание от модели rubert-tiny2. Подбирайте по
    # реальным цифрам из logs\latest.log.
    semantic_recall_threshold: float

    # Round 10N (см. проектный документ) — по итогам разговора с пользователем
    # переосмыслили round 10L/10M: не нужно ГАДАТЬ по фразе, хочет ли
    # собеседник именно ПОВТОР старого ответа — если GigaChat и так уже решил,
    # что текущий вопрос ПРОДОЛЖАЕТ известную тему (не новая), ему в любом
    # случае полезнее реальный текст старого обмена репликами, а не только имя
    # темы, вне зависимости от того, что именно просят сделать с этим
    # содержимым (повторить/доработать/сравнить/что угодно — это решает сама
    # модель, см. main.py hint_worker). ПО УМОЛЧАНИЮ true — это теперь основной
    # путь. false — откат к СТАРОЙ, узкой логике round 10L/10M (regex "повтори"
    # ИЛИ семантическая похожесть выше SEMANTIC_RECALL_THRESHOLD) — код старой
    # логики не удалён, доступен через этот же флаг, без правки кода.
    always_retrieve_on_continuation: bool

    # --- Round 11 (см. проектный документ) ---------------------------------
    #
    # Пользователь решил: полностью работаем с GigaChat, Ollama как фолбэк
    # для генерации больше НЕ используется вообще (ни на сетевой сбой, ни на
    # исчерпание токенов, ни на протухший ключ) — вместо тихого отката на
    # слабую локальную модель теперь явная классификация причины сбоя (см.
    # llm_gigachat.classify_gigachat_error) и уведомление в интерфейсе (см.
    # overlay.py, новый тип сообщения "error"). Старый код отката на Ollama
    # (main.py, judge.py+generate_hint_gigachat/_call_local) НЕ удалён —
    # просто больше не вызывается автоматически при сбое GigaChat, доступен
    # только когда GIGACHAT_ENABLED=false целиком (GigaChat вообще не
    # настроена), это отдельный, осознанно не тронутый сценарий.
    #
    # Каскад моделей при исчерпании токенов (HTTP 402) — от сильной к
    # слабой. main.py начинает с той модели, которую и так выбрал бы (Ultra
    # для сложных вопросов, GigaChat-2 для обычных, см.
    # _looks_like_hard_question), и при 402 у ТЕКУЩЕЙ модели продолжает ВНИЗ
    # по этому списку начиная с её позиции (см. main.py
    # _resolve_gigachat_cascade). Порядок и точные строки взяты по названиям
    # тарифов Сбера (Ultra/Max/Pro/Lite) и по уже существующему в проекте
    # соглашению об именовании (GIGACHAT_MODEL=GigaChat-3-Ultra,
    # GIGACHAT_NORMAL_MODEL=GigaChat-2) — ТОЧНОЕ написание строк для
    # Max/Pro стоит сверить со скриптом check_gigachat_status.py и
    # поправить в .env при необходимости, если API вернёт 400/404 на
    # неверное имя модели — это не сломает саму логику каскада, только
    # порядок/состав списка.
    gigachat_model_cascade: list[str]

    # Порог семантического сходства (0..1) для эмбеддинг-поиска по пулу
    # реплик (main.py reply_pool.py, ReplyPool.build_retrieved_context) —
    # заменяет собой всю систему тегов round 9-10O (TrackedTag/touches
    # остаются в коде как откат, не вызываются). ПЕРВАЯ ПОПЫТКА (как когда-то
    # JUDGE_NUM_CTX/SEMANTIC_RECALL_THRESHOLD) — НЕ откалибровано на реальных
    # данных именно для эмбеддинг-модели deepvk/USER-bge-m3 (у неё, как и у
    # всего семейства BGE, распределение похожести может ощутимо отличаться
    # от rubert-tiny2, на котором подбирался старый SEMANTIC_RECALL_THRESHOLD
    # =0.72 — тот старый номер сюда НЕ переносится "по аналогии"). Похожесть
    # для топ-N кандидатов печатается в logs\latest.log ВСЕГДА, независимо от
    # порога — пользователь планирует прогнать существующие тестовые наборы
    # (data/stress_test_300*.json, data/stress_test_v2*.json — там уже
    # размечено, какие пары реплик реально связаны) синтетически и подобрать
    # число по факту, а не вслепую.
    pool_similarity_threshold: float

    @classmethod
    def load(cls) -> "Settings":
        output_dir = Path(os.getenv("OUTPUT_DIR", "./sessions"))
        if not output_dir.is_absolute():
            output_dir = ROOT_DIR / output_dir

        return cls(
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY") or None,
            mic_device_index=_get_int("MIC_DEVICE_INDEX"),
            system_audio_device_index=_get_int("SYSTEM_AUDIO_DEVICE_INDEX"),
            mic_capture_enabled=os.getenv("MIC_CAPTURE_ENABLED", "true").strip().lower() != "false",
            stt_mode=os.getenv("STT_MODE", "auto").lower(),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            whisper_initial_prompt=os.getenv("WHISPER_INITIAL_PROMPT") or None,
            language=os.getenv("LANGUAGE", "ru"),
            output_dir=output_dir,
            diarization_enabled=os.getenv("DIARIZATION_ENABLED", "true").strip().lower() != "false",
            diarization_similarity_threshold=float(os.getenv("DIARIZATION_SIMILARITY_THRESHOLD", "0.72")),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048")),
            judge_num_ctx=int(os.getenv("JUDGE_NUM_CTX", "3072")),
            live_hints_enabled=os.getenv("LIVE_HINTS_ENABLED", "true").strip().lower() != "false",
            hints_context_lines=int(os.getenv("HINTS_CONTEXT_LINES", "20")),
            hints_on_mic=os.getenv("HINTS_ON_MIC", "false").strip().lower() == "true",
            hints_log_enabled=os.getenv("HINTS_LOG_ENABLED", "true").strip().lower() != "false",
            hints_extra_context=os.getenv("HINTS_EXTRA_CONTEXT") or None,
            kb_enabled=os.getenv("KB_ENABLED", "false").strip().lower() == "true",
            kb_path=cls._resolve_kb_path(os.getenv("KB_PATH")),
            kb_top_k=int(os.getenv("KB_TOP_K", "3")),
            kb_min_score=float(os.getenv("KB_MIN_SCORE", "0.15")),
            gigachat_enabled=os.getenv("GIGACHAT_ENABLED", "false").strip().lower() == "true",
            gigachat_credentials=os.getenv("GIGACHAT_CREDENTIALS") or None,
            gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
            gigachat_normal_model=os.getenv("GIGACHAT_NORMAL_MODEL", "GigaChat-2"),
            gigachat_ca_bundle_file=cls._resolve_optional_path(os.getenv("GIGACHAT_CA_BUNDLE_FILE")),
            semantic_recall_enabled=os.getenv("SEMANTIC_RECALL_ENABLED", "false").strip().lower() == "true",
            semantic_recall_threshold=float(os.getenv("SEMANTIC_RECALL_THRESHOLD", "0.72")),
            always_retrieve_on_continuation=os.getenv("ALWAYS_RETRIEVE_ON_CONTINUATION", "true").strip().lower()
            != "false",
            gigachat_model_cascade=_get_model_list(
                "GIGACHAT_MODEL_CASCADE",
                "GigaChat-3-Ultra,GigaChat-2-Max,GigaChat-2-Pro,GigaChat-2",
            ),
            pool_similarity_threshold=float(os.getenv("POOL_SIMILARITY_THRESHOLD", "0.75")),
        )

    @staticmethod
    def _resolve_optional_path(raw: str | None) -> "Path | None":
        """Путь из .env — либо абсолютный, либо относительно корня проекта."""
        if not raw or not raw.strip():
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT_DIR / p
        return p

    _resolve_kb_path = _resolve_optional_path

    def validate_for_run(self) -> list[str]:
        """Возвращает список проблем, которые не дадут скрипту нормально запуститься."""
        problems = []
        if self.mic_capture_enabled and self.mic_device_index is None:
            problems.append(
                "MIC_DEVICE_INDEX не задан в .env — запустите "
                "python src/list_audio_devices.py и впишите индекс микрофона."
            )
        if self.system_audio_device_index is None:
            problems.append(
                "SYSTEM_AUDIO_DEVICE_INDEX не задан в .env — впишите индекс "
                "устройства VB-Cable (CABLE Output), см. python src/list_audio_devices.py."
            )
        if self.stt_mode in ("cloud", "auto") and not self.deepgram_api_key:
            if self.stt_mode == "cloud":
                problems.append("STT_MODE=cloud, но DEEPGRAM_API_KEY не задан.")
        if self.stt_mode == "local" and self.whisper_device == "cuda":
            pass  # проверка доступности CUDA — в stt/local_whisper.py при загрузке модели
        if self.kb_enabled and (self.kb_path is None or not self.kb_path.is_file()):
            problems.append(
                f"KB_ENABLED=true, но файл базы знаний не найден по пути "
                f"KB_PATH={self.kb_path!r} — проверьте .env."
            )
        if self.gigachat_enabled and not self.gigachat_credentials:
            problems.append(
                "GIGACHAT_ENABLED=true, но GIGACHAT_CREDENTIALS не задан — "
                "возьмите Authorization key в личном кабинете developers.sber.ru "
                "(GigaChat API -> ваш проект -> Авторизационные данные)."
            )
        if self.gigachat_ca_bundle_file is not None and not self.gigachat_ca_bundle_file.is_file():
            problems.append(
                f"GIGACHAT_CA_BUNDLE_FILE={self.gigachat_ca_bundle_file!r} указан, но файла "
                f"нет — см. инструкцию по сертификату Минцифры в .env.example."
            )
        return problems
