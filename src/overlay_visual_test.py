"""
Визуальный тест оверлея (`overlay.py`) — открывает ТО ЖЕ САМОЕ окно, что и на
живом звонке, но НЕ дёргает Ollama и не слушает микрофон/систему. Вместо этого
по очереди "показывает" в оверлее уже реально сгенерированные ответы модели —
взятые прямо из `final_test_results.csv` (ночной комбинированный прогон на
2029 вопросах) — как будто они только что пришли с реального звонка.

Смысл ровно тот, о котором договорились: посмотреть ГЛАЗАМИ на реальный
интерфейс с реальными (не выдуманными) ответами и проверить, что ничего не
обрезается и не ломается — перенос длинных строк, прокрутка истории,
раскраска времени по скорости (быстро/средне/медленно), поведение при
пустом ответе, стриминг "печатает…", отмена черновика при устаревшем вопросе.
Вопросы модели заново НЕ задаются — экономим время и не гоняем Ollama просто
чтобы посмотреть на вёрстку того, что она уже один раз ответила.

Набор случаев — `data/overlay_visual_test_cases.json`, 14 штук, подобраны под
конкретные крайние ситуации (см. комментарий в конце каждого случая в самом
файле): самый длинный ответ во всём датасете (1364 символа, "расскажи о
каждом подробнее"), самый быстрый и самый медленный (для проверки цвета
времени), уточняющие вопросы (оканчиваются на "?"), реальный артефакт с
переносом строки внутри текста ответа (из карточки базы знаний), серия
коротких вопросов подряд (накопление истории/автопрокрутка), симуляция
стриминга по словам и симуляция отменённого черновика. Один (и только один)
случай — синтетика, а не реальный ответ: блок "📌 контекст" ни разу не
появился за все 2029 прогонов батч-теста (там нет истории разговора, только
одиночные вопросы) — специально помечен в данных как придуманный, чтобы
проверить хотя бы вёрстку этого блока, раз его нечем заполнить реальным
примером.

Каждая запись теперь показывает и вопрос собеседника ("🗣", над ответом), и
сам ответ ("💬") — как в реальном оверлее после соответствующей правки
main.py/overlay.py, чтобы визуально было понятно, на какой именно вопрос это
подсказка, а не только что ответила модель.

Вопрос и ответ — два ОТДЕЛЬНЫХ сообщения в очереди, а не одно "final" сразу
с обоими сразу (совсем как в main.py: сначала ("partial", seq, question, "",
""), потом уже готовый ("final", ...)), но без какой-либо придуманной паузы
между ними — "секунд_1"/"секунд_2" из CSV это время ОДНОГО конкретного
запроса к Ollama в ту ночь, случайное и ничем не гарантированное для любого
другого раза, так что подавать его как "настоящую" паузу перед ответом
было бы обманом, а не реализмом. Здесь вопрос и ответ просто идут подряд.
Единственные паузы, которые в файле остались, — это чисто зрительный темп
просмотра (между случаями и внутри "печатает по словам"/демонстрации
отмены), не претендующий на то, что это чьё-то реальное время генерации.

Кроме куцего набора из 14 случаев есть ПОЛНЫЙ — data/overlay_visual_test_cases_full.json,
2543 реальных ответа из ночного прогона подряд, В ТОМ ЖЕ ПОРЯДКЕ, что и был на
самом деле (context-пары идут двумя последовательными записями, как и в живом
звонке — это не склеено в одну запись, как в CSV, а показано ровно так, как
оверлей реально показал бы такой разговор). Это для стресс-теста вёрстки на
объёме, а не для разбора каждого конкретного случая — с 2543 записями разумно
сильно снизить --interval, иначе прогон растянется на часы, либо ограничить
через --limit.

Запуск (из корня проекта — откроется настоящее окно оверлея):
    venv\\Scripts\\python.exe src\\overlay_visual_test.py
    venv\\Scripts\\python.exe src\\overlay_visual_test.py --interval 6   (медленнее пролистывать)
    venv\\Scripts\\python.exe src\\overlay_visual_test.py --cases my_cases.json  (свой набор случаев)
    venv\\Scripts\\python.exe src\\overlay_visual_test.py --cases data\\overlay_visual_test_cases_full.json --interval 1.5
    venv\\Scripts\\python.exe src\\overlay_visual_test.py --cases data\\overlay_visual_test_cases_full.json --interval 1.5 --limit 200

В консоли, из которой запущен скрипт, перед каждым случаем печатается его
название — так видно, какая именно ситуация сейчас показывается в оверлее, не
нужно гадать по содержимому. Esc в оверлее закрывает окно как обычно (это НЕ
завершает звонок и не пишет summary — тут его просто нет, это чистая проверка
интерфейса).
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

from overlay import OverlayWindow

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = ROOT_DIR / "data" / "overlay_visual_test_cases.json"

# Ниже — чисто зрительный темп ПОКАЗА (чтобы глаз успевал следить), а не
# попытка смоделировать чьё-то реальное время генерации/поиска — договорились
# явно этого не делать (elapsed из CSV — случайное число одного конкретного
# запроса в ту ночь, а не характеристика "как долго вообще думает модель").

# Пауза по словам в симуляции стриминга — просто чтобы черновик рос заметно
# для глаза, а не всей строкой разом.
_STREAM_WORD_DELAY = 0.12
# Пауза между partial-черновиком и cancel в демонстрации отмены — достаточно,
# чтобы успеть увидеть сам черновик перед тем, как он пропадёт.
_CANCEL_DEMO_DELAY = 1.4


def _load_cases(path: Path) -> list[dict]:
    import json

    if not path.is_file():
        print(f"Не найден {path} — нечего показывать.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _run_hint(hint_queue: "queue.Queue[tuple]", seq: int, case: dict) -> None:
    # Сначала — только вопрос (оверлей сам покажет "печатает…" вместо ответа,
    # см. _build_hint_parts в overlay.py), совсем как в main.py: там тоже
    # сначала кладётся ("partial", seq, question, "", "") сразу после
    # вопроса, а сам ответ прилетает отдельным "final" чуть позже. Никакой
    # паузы между ними мы не выдумываем — идут подряд, одно за другим.
    question = case.get("question", "")
    hint_queue.put(("partial", seq, question, "", ""))
    hint_queue.put((
        "final", seq, question, case.get("reply", ""),
        case.get("context", ""), case.get("elapsed", 3.0),
    ))


def _run_stream(
    hint_queue: "queue.Queue[tuple]", seq: int, case: dict, stop_event: threading.Event, speed: float
) -> None:
    question = case.get("question", "")
    full_reply = case.get("reply", "")
    hint_queue.put(("partial", seq, question, "", ""))
    words = full_reply.split(" ") if full_reply else []
    shown = ""
    for word in words:
        if stop_event.is_set():
            return
        shown = f"{shown} {word}".strip()
        hint_queue.put(("partial", seq, question, shown, ""))
        if stop_event.wait(_STREAM_WORD_DELAY / speed):
            return
    hint_queue.put(("final", seq, question, full_reply, case.get("context", ""), case.get("elapsed", 3.0)))


def _run_cancel_demo(
    hint_queue: "queue.Queue[tuple]", seq: int, case: dict, stop_event: threading.Event, speed: float
) -> None:
    # Вопрос и недопечатанный черновик показываются вместе — тут не вопрос
    # "когда появится ответ", а демонстрация того, что при устаревшем
    # вопросе черновик убирается без следа. Пауза до cancel — чтобы успеть
    # увидеть сам черновик глазами, не более того.
    question = case.get("question", "")
    hint_queue.put(("partial", seq, question, case.get("partial_reply", ""), ""))
    if stop_event.wait(_CANCEL_DEMO_DELAY / speed):
        return
    hint_queue.put(("cancel", seq))


def _run_burst(hint_queue: "queue.Queue[tuple]", seq_start: int, case: dict, stop_event: threading.Event, speed: float) -> int:
    seq = seq_start
    for item in case.get("items", []):
        if stop_event.is_set():
            return seq
        seq += 1
        question = item.get("question", "")
        hint_queue.put(("partial", seq, question, "", ""))
        hint_queue.put((
            "final", seq, question, item.get("reply", ""),
            item.get("context", ""), item.get("elapsed", 2.0),
        ))
        if stop_event.wait(item.get("interval_after", 0.9) / speed):
            return seq
    return seq


def _feeder(
    hint_queue: "queue.Queue[tuple]",
    cases: list[dict],
    interval: float,
    stop_event: threading.Event,
    speed: float,
) -> None:
    seq = 0
    for case in cases:
        if stop_event.is_set():
            return
        label = case.get("label", "(без названия)")
        kind = case.get("kind", "hint")
        print(f"\n>>> [{kind}] {label}")

        if kind == "hint":
            seq += 1
            _run_hint(hint_queue, seq, case)
        elif kind == "stream":
            seq += 1
            _run_stream(hint_queue, seq, case, stop_event, speed)
        elif kind == "cancel_demo":
            seq += 1
            _run_cancel_demo(hint_queue, seq, case, stop_event, speed)
        elif kind == "burst":
            seq = _run_burst(hint_queue, seq, case, stop_event, speed)
        else:
            print(f"    неизвестный тип случая {kind!r} — пропущено")
            continue

        if stop_event.wait(case.get("interval_after", interval) / speed):
            return

    print("\nВсе случаи показаны. Оверлей остаётся открытым — прокрутите историю вверх/вниз,")
    print("Esc в окне закрывает тест.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval", type=float, default=4.5,
        help="пауза между случаями по умолчанию, сек (у некоторых случаев своя, из JSON)",
    )
    parser.add_argument(
        "--cases", type=str, default=None,
        help=(
            f"свой JSON с тестовыми случаями вместо встроенного ({DEFAULT_CASES_PATH.name}, 14 "
            "отобранных крайних случаев). Для прогона на ВСЕХ реальных вопросах ночного теста "
            "(2543 случая подряд, context-пары как в реальном звонке — двумя последовательными "
            "записями) укажи data\\overlay_visual_test_cases_full.json — с таким объёмом разумно "
            "снизить --interval, иначе прогон растянется на часы."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="показать только первые N случаев из файла (удобно для быстрой проверки большого набора)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help=(
            "множитель на зрительный темп показа — паузы между случаями, скорость 'печатает по "
            "словам' и демонстрации отмены (по умолчанию 1.0; 2.0 — вдвое быстрее для беглого "
            "просмотра большого набора). Между вопросом и обычным готовым ответом паузы нет "
            "вообще, так что на неё --speed не влияет — там нечего ускорять."
        ),
    )
    args = parser.parse_args()

    cases_path = Path(args.cases) if args.cases else DEFAULT_CASES_PATH
    cases = _load_cases(cases_path)
    if args.limit is not None:
        cases = cases[: args.limit]
    print(f"Загружено случаев: {len(cases)} из {cases_path}")
    print("Открываю оверлей... (Esc в окне — закончить тест)\n")

    hint_queue: "queue.Queue[tuple]" = queue.Queue()
    stop_event = threading.Event()

    def on_close() -> None:
        stop_event.set()
        print("\nОверлей закрыт (Esc) — тест остановлен.")

    overlay = OverlayWindow(hint_queue, on_close)

    feeder_thread = threading.Thread(
        target=_feeder, args=(hint_queue, cases, args.interval, stop_event, args.speed), daemon=True
    )
    feeder_thread.start()

    overlay.run()


if __name__ == "__main__":
    main()
