"""
Round "на глаз" (см. проектный документ, обсуждение окна контекста и его
ограничений): пользователь попросил не агрегированную статистику (как в
batch_test_pool_similarity.py — проценты recall/шума по категориям), а
ПОСТРОЧНЫЙ человекочитаемый отчёт: прогнать data/stress_test_300.json через
РЕАЛЬНУЮ deepvk/USER-bge-m3 (embeddings.py, без моков и без подмены модели —
это ровно то, что использует прод) и для каждой реплики показать НАБОР РЕПЛИК,
который реально увидела бы GigaChat в промпте — чтобы оценить на глаз, где
система попадает, а где промахивается, а не только смотреть на проценты.

Формат набора (по прямому уточнению пользователя): "набор не тем, а реплик,
которые и составляют промт для гигачата. первой репликой идет реплика
триггер" — то есть каждый блок начинается с ТЕКУЩЕЙ (триггерной) реплики,
дальше идёт ранжированный по похожести список реплик из пула ДО неё, с
пометкой, какие из них реально прошли бы POOL_SIMILARITY_THRESHOLD и попали бы
в промпт вместе с триггером.

Использует РЕАЛЬНЫЕ embeddings.py/reply_pool.py напрямую, БЕЗ абстракции
бэкендов batch_test_pool_similarity.py (та нужна для СРАВНЕНИЯ моделей; здесь
модель всегда ровно та, что в проде, — deepvk/USER-bge-m3, без подмены) и без
монкипатчей — reply_pool.ReplyPool сама вызывает embeddings.embed_text
изнутри, этого достаточно.

Показывает РАНЖИРОВАННЫЙ список отдельных реплик (вопросы и ответы по
отдельности), а не резолвнутые пары вопрос+ответ, которые реально строит
ReplyPool.build_retrieved_context() (см. reply_pool.py — там ещё дедуп пар и
склейка вопроса с его ответом в один блок). Это сознательное упрощение ради
читаемости построчного отчёта — если реплика-вопрос прошла порог, то в
реальном промпте она ушла бы ВМЕСТЕ со своим ответом одним блоком, а не сама
по себе. Отчёт это не показывает явно, но оба элемента пары будут в списке
рядом (соседние по похожести, так как эмбеддятся из связанного контекста), и
если нужна 1:1 точность промпта — см. build_retrieved_context() в reply_pool.py.

Требует sentence_transformers и сеть/кеш HuggingFace для deepvk/USER-bge-m3.
Логика (skip категорий,
ранжирование, пометка порога, вердикт) проверена офлайн детерминированной
fake-моделью — реальные результаты — с прогона на
машине.

Запуск (на машине пользователя, из корня проекта):
    venv\\Scripts\\python.exe src\\inspect_pool_retrievals.py
    venv\\Scripts\\python.exe src\\inspect_pool_retrievals.py --out review_300.txt
    venv\\Scripts\\python.exe src\\inspect_pool_retrievals.py --top 10 --threshold 0.70
    venv\\Scripts\\python.exe src\\inspect_pool_retrievals.py --scenario data\\stress_test_v2.json --answer-key data\\stress_test_v2_answer_key.json

Флаги:
    --scenario <файл>     JSON со сценарием (по умолчанию data/stress_test_300.json)
    --answer-key <файл>   JSON с разметкой (по умолчанию data/stress_test_300_answer_key.json)
    --threshold N          порог отсечения (по умолчанию 0.75 — см. config.py
                           Settings.pool_similarity_threshold, тот же дефолт,
                           что в .env/POOL_SIMILARITY_THRESHOLD)
    --top N                сколько кандидатов показывать в ранжированном
                           списке на каждую реплику (по умолчанию 8)
    --out <файл>          текстовый файл отчёта (по умолчанию
                           pool_retrieval_review.txt в корне проекта)
    --limit N              обработать только первые N реплик сценария
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reply_pool import ReplyPool

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO_PATH = ROOT_DIR / "data" / "stress_test_300.json"
DEFAULT_ANSWER_KEY_PATH = ROOT_DIR / "data" / "stress_test_300_answer_key.json"
DEFAULT_OUT_PATH = ROOT_DIR / "pool_retrieval_review.txt"
DEFAULT_THRESHOLD = 0.75  # см. config.py Settings.pool_similarity_threshold (дефолт)

# Те же соглашения, что в batch_test_pool_similarity.py — см. её докстринг.
_SKIP_ALWAYS = {"clarify_question"}
_NO_MATCH_EXPECTED = {"new_topic", "garbage"}


def _load(scenario_path: Path, answer_key_path: Path) -> tuple[list[dict], list[dict]]:
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    if len(scenario) != len(answer_key):
        raise ValueError(
            f"scenario ({len(scenario)} реплик) и answer_key ({len(answer_key)} реплик) "
            f"разной длины — файлы должны быть парой одной схемы"
        )
    return scenario, answer_key


def run(
    scenario: list[dict],
    answer_key: list[dict],
    threshold: float,
    top_n: int,
    limit: int | None = None,
) -> list[str]:
    pool = ReplyPool()
    entry_id_by_text: dict[str, int] = {}
    report_lines: list[str] = []

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
            report_lines.append(f"[{i:03d}] embedding не посчитан — пропущено\n")
            continue
        entry_id_by_text.setdefault(text, qid)

        this_entry = next(e for e in pool.entries if e.id == qid)
        # Реальный поиск ReplyPool.search() — та же функция, что и в проде
        # (build_retrieved_context вызывает её же). Полный ранжированный
        # список, не только то, что прошло порог — специально, чтобы видеть
        # "почти попадания" чуть ниже порога.
        scored = pool.search(this_entry.embedding, exclude_id=qid)

        expected_match = category not in _NO_MATCH_EXPECTED
        related = krow.get("related_topic")
        candidates = related if isinstance(related, list) else ([related] if related else [])
        correct_ids: set[int] = set()
        if expected_match:
            for cand_text in candidates:
                cand_qid = entry_id_by_text.get(cand_text)
                if cand_qid is not None:
                    correct_ids.add(cand_qid)
                    cand_entry = next(e for e in pool.entries if e.id == cand_qid)
                    if cand_entry.answer_id is not None:
                        correct_ids.add(cand_entry.answer_id)

        top = scored[:top_n]
        correct_rank = next((rank for rank, (e, s) in enumerate(scored, start=1) if e.id in correct_ids), None)
        passed = [(e, s) for e, s in top if s >= threshold]

        lines: list[str] = []
        lines.append("=" * 88)
        lines.append(f"НАБОР #{i:03d}  (категория: {category}, порог: {threshold:.2f})")
        lines.append("-" * 88)
        lines.append(f"  [ТРИГГЕР]      {speaker}: {text!r}")
        if not top:
            lines.append("  (пул пуст — это первая содержательная реплика разговора)")
        for e, s in top:
            mark = "✓" if s >= threshold else " "
            correct_tag = "  <- ЭТАЛОН" if e.id in correct_ids else ""
            lines.append(f"  [{s:.3f} {mark}]      {e.speaker}: {e.text[:90]!r}{correct_tag}")
        lines.append(f"  (✓ = прошло порог {threshold:.2f} — реально ушло бы в промпт GigaChat вместе с триггером)")
        lines.append("")

        if candidates:
            lines.append(f"Эталон (что должно быть найдено): {' | '.join(candidates)}")
        if expected_match:
            if correct_rank is None:
                verdict = "ПРОМАХ — эталонный кандидат не найден дословно в пройденной части сценария (проверить вручную)"
            elif any(e.id in correct_ids for e, _ in passed):
                verdict = f"ВЕРНО — эталон попал в набор (место #{correct_rank} по похожести, прошёл порог)"
            elif correct_rank <= top_n:
                verdict = f"ПРОМАХ (близко) — эталон в топ-{top_n} (место #{correct_rank}), но НЕ прошёл порог {threshold:.2f}"
            else:
                verdict = f"ПРОМАХ — эталон есть в пуле, но вне топ-{top_n} (место #{correct_rank})"
        else:
            verdict = "ЛОЖНОЕ СРАБАТЫВАНИЕ — категория 'шум', но что-то прошло порог" if passed else "ОК — шумовая категория, ничего не прошло порог"
        lines.append(f"Вердикт: {verdict}")
        lines.append("")

        report_lines.extend(lines)

        ref_answer = krow.get("reference_answer")
        if ref_answer:
            pool.add_answer(qid, ref_answer, line_index=i)

    return report_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--answer-key", type=Path, default=DEFAULT_ANSWER_KEY_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top", type=int, default=8, dest="top_n")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Сценарий: {args.scenario}")
    print(f"Эталон:   {args.answer_key}")
    print(f"Порог:    {args.threshold}")
    print("Загружаю модель эмбеддингов deepvk/USER-bge-m3 (первый вызов может занять "
          "время — скачивание с HuggingFace, если модель ещё не в кеше)...")

    scenario, answer_key = _load(args.scenario, args.answer_key)
    report_lines = run(scenario, answer_key, args.threshold, args.top_n, limit=args.limit)

    args.out.write_text("\n".join(report_lines), encoding="utf-8-sig")
    print(f"\nОтчёт записан: {args.out} ({len(report_lines)} строк)")
    print("Открой файл текстовым редактором и посмотри блоки НАБОР #NNN — каждый "
          "начинается с реплики-триггера, дальше ранжированный список похожих "
          "реплик из пула (✓ = реально попало бы в промпт), в конце — вердикт.")


if __name__ == "__main__":
    main()
