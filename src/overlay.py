"""
Полупрозрачное окно поверх экрана с live-подсказками во время звонка.
Tkinter обязан жить в главном потоке — окно создаётся и его mainloop()
запускается там же, а новые подсказки приходят из фоновых потоков через
очередь и подхватываются периодическим опросом (root.after).
История подсказок: раньше окно просто затирало предыдущую подсказку новой,
как только собеседник произносил следующую фразу — если не успел дочитать,
подсказка терялась безвозвратно. Теперь это прокручиваемый список: каждая
новая подсказка ДОБАВЛЯЕТСЯ СВЕРХУ, а более старые уходят вниз — самое
актуальное всегда видно сразу, без прокрутки. Хотите перечитать историю —
промотайте вниз; пока вы там, новые подсказки не дёргают вас обратно
наверх, пока сами не вернётесь к верху списка.
Печатающийся ответ: пока модель ещё генерирует ответ, самая верхняя запись
в истории — "черновик", который растёт по мере поступления кусочков текста
от Ollama (см. on_partial в live_hints.py / hint_worker в main.py), вместо
того чтобы 2-5 секунд просто молчать, а потом появиться сразу целиком.
Технически это одна и та же запись, которая многократно перерисовывается
НА МЕСТЕ (между "1.0" и меткой self._stream_mark) — уже готовая история
ниже этой записи не трогается вообще. Если пришёл более новый вопрос (или
модель решила, что отвечать не на что, или генерация оборвалась ошибкой) —
недопечатанный черновик убирается без следа ("cancel"), а не остаётся в
истории как есть.
Каждая запись показывает и саму фразу собеседника ("🗣", приглушённым
курсивом), и ответ модели ("💬") под ней — для наглядности, на какой именно
вопрос это подсказка (main.py вычисляет фразу той же функцией, что и
generate_hint() внутри себя, и прокидывает её в hint_queue отдельным полем).
Управление:
Окно можно перетаскивать мышкой за полосу заголовка сверху.
Окно можно растягивать мышкой за уголок "◢" снизу справа (мин. 320x220).
Колесо мыши / выделение текста — как в обычном чате.
Esc — закрыть оверлей и корректно завершить звонок (то же самое, что Ctrl+C).
"""
from __future__ import annotations
import datetime as dt
import queue
import tkinter as tk

# Минимальный размер окна при ручном растягивании (см. _on_resize) — чтобы
# случайно не утянуть окно до нуля и не потерять его на экране.
_MIN_WIDTH = 320
_MIN_HEIGHT = 220


class OverlayWindow:
    def __init__(self, hint_queue: "queue.Queue[tuple]", on_close):
        """
        on_close: вызывается один раз, когда пользователь закрывает оверлей
        (Esc) — сюда вешаем корректную остановку звонка и генерацию саммари.

        hint_queue теперь несёт тегированные сообщения ("partial"/"final"/
        "cancel"/"error", см. main.py) вместо старого одного вида кортежа —
        формат разобран в _poll_queue. "error" (round 11) — постоянное
        уведомление о классифицированном сбое GigaChat (кончились токены на
        всех моделях каскада, протух ключ, нет сети и т.п.) — см.
        main.py _notify_gigachat_failure/classify_gigachat_error.
        """
        self.hint_queue = hint_queue
        self.on_close = on_close
        self._closed = False
        # Состояние текущего "печатающегося" черновика (см. докстринг модуля
        # выше). None = сейчас ничего не стримится, история статична.
        self._streaming_seq: int | None = None
        self._streaming_had_content: bool = False
        self._stream_mark = "stream_end"
        self.root = tk.Tk()
        self.root.title("call-assistant")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        # Обычное окно Windows с рамкой и заголовком: поверх остальных окон,
        # но видно и в панели задач, и при демонстрации экрана.
        self.root.configure(bg="#1e1e1e")
        width, height = 460, 420
        screen_w = self.root.winfo_screenwidth()
        x = screen_w - width - 24
        y = 24
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.update()  # окно должно быть полностью создано до правки стилей


        # Полоска для перетаскивания окна — раньше можно было тащить за любое
        # место, но с появлением истории подсказок (Text-виджет, там теперь
        # работает выделение текста мышкой) это стало бы конфликтовать с
        # выделением. Поэтому окно теперь двигается только за эту полосу —
        # делаем её повыше и заметнее, чтобы было легко ухватиться.
        header = tk.Frame(self.root, bg="#2a2a2a", cursor="fleur")
        header.pack(fill="x")
        header_label = tk.Label(
            header, text="⠿  call-assistant — тащить сюда, Esc чтобы закончить звонок",
            fg="#999999", bg="#2a2a2a", font=("Segoe UI", 9), cursor="fleur",
        )
        header_label.pack(side="left", padx=12, pady=6)
        # История подсказок — прокручиваемый лог, а не одна затирающаяся строка.
        body = tk.Frame(self.root, bg="#1e1e1e")
        body.pack(fill="both", expand=True, padx=(12, 0), pady=(0, 8))
        scrollbar = tk.Scrollbar(body)
        scrollbar.pack(side="right", fill="y")
        self.text = tk.Text(
            body,
            bg="#1e1e1e",
            fg="#f0f0f0",
            font=("Segoe UI", 10),
            wrap="word",
            relief="flat",
            padx=4,
            pady=2,
            yscrollcommand=scrollbar.set,
            state="disabled",
            cursor="arrow",
        )
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text.yview)
        # Три варианта цвета для времени генерации — чтобы на живом звонке
        # медленный ответ (когда модель "задумалась" дольше обычного) было
        # видно сразу по цвету, не вчитываясь в цифры.
        self.text.tag_configure("time", foreground="#777777", font=("Segoe UI", 8))
        self.text.tag_configure("time_fast", foreground="#6fcf70", font=("Segoe UI", 8))
        self.text.tag_configure("time_slow", foreground="#e06c75", font=("Segoe UI", 8, "bold"))
        # Вопрос собеседника — над ответом, приглушённым курсивом, чтобы
        # глаз сразу отличал "что спросили" от "что ответить" (ответ ниже —
        # основной, крупный и не курсивом).
        self.text.tag_configure("question_label", foreground="#9aa0a6", font=("Segoe UI", 9, "bold"))
        self.text.tag_configure("question_text", foreground="#9aa0a6", font=("Segoe UI", 10, "italic"))
        self.text.tag_configure("reply_label", foreground="#8ab4f8", font=("Segoe UI", 9, "bold"))
        self.text.tag_configure("reply_text", foreground="#f0f0f0", font=("Segoe UI", 11))
        self.text.tag_configure("context_label", foreground="#8ab4f8", font=("Segoe UI", 9, "bold"))
        self.text.tag_configure("context_text", foreground="#c8c8c8", font=("Segoe UI", 10))
        # Round 11 (см. проектный документ): раньше ЛЮБОЙ сбой GigaChat тихо
        # уводил на локальную Ollama — пользователь ничего не видел, даже
        # если сбой был постоянным (кончились токены/протух ключ). Теперь
        # Ollama-фолбэка нет вообще, поэтому сбой обязан быть ВИДЕН прямо в
        # оверлее, а не только в консольном логе. Переиспользуем ту же
        # красную палитру, что и у "time_slow" (медленный ответ) — для
        # единого визуального языка "тут что-то не так, обрати внимание".
        self.text.tag_configure("error_label", foreground="#e06c75", font=("Segoe UI", 9, "bold"))
        self.text.tag_configure("error_text", foreground="#e06c75", font=("Segoe UI", 10, "bold"))

        # Уголок для растягивания окна мышкой — дублирует системную рамку,
        # удобен, когда окно прижато к краю экрана. Маленький треугольник внизу
        # справа, тащится как и полоса заголовка, но меняет размер, а не положение.
        resize_grip = tk.Label(
            self.root, text="◢", fg="#555555", bg="#1e1e1e",
            font=("Segoe UI", 10), cursor="sizing",
        )
        resize_grip.place(relx=1.0, rely=1.0, anchor="se", x=-1, y=-1)
        resize_grip.bind("<ButtonPress-1>", self._start_resize)
        resize_grip.bind("<B1-Motion>", self._on_resize)
        self._resize_start = (0, 0, 0, 0)

        self._append_placeholder("Слушаю...")
        # Перетаскивание окна мышкой — только за полосу сверху, а не за любое
        # место (иначе конфликтовало бы с выделением текста в истории ниже).
        for widget in (header, header_label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)
        self._drag_offset = (0, 0)
        self.root.bind("<Escape>", lambda _event: self._handle_close())
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._poll_queue()

    def _append_placeholder(self, text: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", f"{text}\n", ("reply_text",))
        self.text.configure(state="disabled")

    def _is_scrolled_to_top(self) -> bool:
        # yview() -> (доля видимого сверху, доля видимого снизу), 0.0 = самый верх.
        return self.text.yview()[0] <= 0.001

    def _append_hint(self, question: str, reply: str, context: str, elapsed: float | None = None) -> None:
        was_at_top = self._is_scrolled_to_top()
        had_content = self.text.index("end-1c") != "1.0"
        parts = self._build_hint_parts(question, reply, context, elapsed, had_content)
        self.text.configure(state="normal")
        self.text.insert("1.0", *parts)
        self.text.configure(state="disabled")
        if was_at_top:
            self.text.see("1.0")

    def _build_hint_parts(
        self, question: str, reply: str, context: str, elapsed: float | None, had_content: bool
    ) -> list:
        """
        Общая сборка "тело записи" — и для готовой подсказки (_append_hint,
        elapsed известен или её вообще нет), и для растущего "черновика" во
        время стриминга (_render_stream, elapsed всегда None, пока не готово).
        Время генерации рядом с меткой времени — сколько реально заняло у
        модели ответить на этот вопрос (от вызова generate_hint до готового
        ответа, см. hint_worker в main.py). Порог "быстро/медленно" —
        ориентир на глаз, не точная наука: до 2с — быстро (зелёным), больше
        5с — заметно медленно (красным, жирным), между ними — обычный серый.
        Пока модель ещё печатает (elapsed=None), время не показываем вообще —
        не показывать ложную "секунду в секунду" цифру, которой ещё нет.

        question — сама фраза собеседника (main.py вычисляет её тем же
        способом, что и generate_hint() внутри себя, см. _split_current_line
        в live_hints.py), показывается НАД ответом отдельной приглушённой
        строкой — для наглядности, на какой именно вопрос это подсказка.
        Может быть пустой (например, если main.py почему-то не смог её
        вычислить) — тогда просто не показываем эту строку вообще, ответ
        рисуется как раньше, без пустого места на месте вопроса.
        """
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        time_label = f"[{timestamp}]"
        time_tag = "time"
        if elapsed is not None:
            time_label = f"[{timestamp} · {elapsed:.1f}с]"
            if elapsed <= 2.0:
                time_tag = "time_fast"
            elif elapsed > 5.0:
                time_tag = "time_slow"
        # Пока ещё ничего не напечатано (самый первый тик сразу после
        # вопроса) — показываем "печатает…" вместо пустой строки, чтобы было
        # видно, что модель уже работает, а не просто завис оверлей.
        reply_display = reply or ("печатает…" if elapsed is None else "…")
        # Собираем всю запись одной вставкой (Text.insert принимает несколько
        # пар текст/теги за раз) — так порядок кусков не зависит от того, что
        # каждая вставка в "1.0" сдвигает уже вставленное вниз.
        parts: list = [f"{time_label}\n", (time_tag,)]
        if question:
            parts += ["🗣 ", ("question_label",), f"{question}\n", ("question_text",)]
        parts += ["💬 ", ("reply_label",), f"{reply_display}\n", ("reply_text",)]
        if context:
            parts += ["📌 ", ("context_label",), f"{context}\n", ("context_text",)]
        if had_content:
            parts += ["\n", ()]  # пустая строка-разделитель перед уже существующей историей
        return parts

    # --- Стриминг ("печатает…") ------------------------------------------
    #
    # Три обработчика ниже соответствуют трём тегам сообщений из hint_queue
    # (см. main.py): "partial" растит черновик, "final" превращает его в
    # обычную готовую запись (с раскрашенным временем), "cancel" убирает его
    # без следа. self._streaming_seq — какому запросу принадлежит текущий
    # черновик наверху истории; self._stream_mark — граница между ним и уже
    # существующей (неприкасаемой) историей ниже, автоматически сдвигается
    # Tkinter при вставке/удалении текста перед ней.

    def _handle_partial(self, seq: int, question: str, reply: str, context: str) -> None:
        was_at_top = self._is_scrolled_to_top()
        if self._streaming_seq != seq:
            # Новый черновик — раньше здесь либо не было ничего, либо лежала
            # уже ЗАВЕРШЁННАЯ запись предыдущего вопроса (started с "final",
            # streaming_seq сброшен в None) — в обоих случаях просто отмечаем
            # начало новой границы на самом верху, существующая история ниже
            # никак не трогается.
            self._streaming_had_content = self.text.index("end-1c") != "1.0"
            self.text.mark_set(self._stream_mark, "1.0")
            self.text.mark_gravity(self._stream_mark, "right")
            self._streaming_seq = seq
        self._render_stream(question, reply, context, elapsed=None, was_at_top=was_at_top)

    def _handle_final(self, seq: int, question: str, reply: str, context: str, elapsed: float) -> None:
        if self._streaming_seq == seq:
            was_at_top = self._is_scrolled_to_top()
            self._render_stream(question, reply, context, elapsed, was_at_top)
            self._streaming_seq = None
        else:
            # На практике недостижимо при текущей архитектуре (hint_worker
            # обрабатывает запросы строго по одному, и на каждый либо
            # приходит хотя бы начальный "partial" перед этим самым "final",
            # либо вместо "final" приходит "cancel") — но на всякий случай не
            # теряем готовый ответ молча, а показываем его как обычную новую
            # запись, как было до стриминга.
            self._append_hint(question, reply, context, elapsed)

    def _handle_error(self, message: str) -> None:
        """
        Round 11: постоянная (не стримящаяся, не привязанная к seq) запись
        об ошибке GigaChat — main.py решает ЧТО написать (уже человекочитаемый
        русский текст, см. main.py _notify_gigachat_failure), тут только
        отрисовка. Вставляется как новая запись сверху, ровно как обычная
        подсказка (_append_hint) — история ниже не трогается.
        """
        was_at_top = self._is_scrolled_to_top()
        had_content = self.text.index("end-1c") != "1.0"
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        parts: list = [f"[{timestamp}]\n", ("time",)]
        parts += ["⚠ ", ("error_label",), f"{message}\n", ("error_text",)]
        if had_content:
            parts += ["\n", ()]
        self.text.configure(state="normal")
        self.text.insert("1.0", *parts)
        self.text.configure(state="disabled")
        if was_at_top:
            self.text.see("1.0")

    def _handle_cancel(self, seq: int) -> None:
        if self._streaming_seq != seq:
            return
        self.text.configure(state="normal")
        self.text.delete("1.0", self._stream_mark)
        self.text.configure(state="disabled")
        self._streaming_seq = None

    def _render_stream(
        self, question: str, reply: str, context: str, elapsed: float | None, was_at_top: bool
    ) -> None:
        parts = self._build_hint_parts(question, reply, context, elapsed, self._streaming_had_content)
        self.text.configure(state="normal")
        self.text.delete("1.0", self._stream_mark)
        self.text.insert("1.0", *parts)
        self.text.configure(state="disabled")
        if was_at_top:
            self.text.see("1.0")

    def _start_drag(self, event) -> None:
        self._drag_offset = (event.x, event.y)

    def _on_drag(self, event) -> None:
        x = self.root.winfo_pointerx() - self._drag_offset[0]
        y = self.root.winfo_pointery() - self._drag_offset[1]
        self.root.geometry(f"+{x}+{y}")

    def _start_resize(self, event) -> None:
        # Запоминаем стартовую точку мыши и стартовый размер окна ОДИН раз,
        # на нажатии кнопки — дальше в _on_resize считаем новый размер как
        # смещение мыши от этой точки, а не от предыдущего кадра (иначе
        # накопилась бы ошибка округления на каждом маленьком движении).
        self._resize_start = (
            self.root.winfo_pointerx(),
            self.root.winfo_pointery(),
            self.root.winfo_width(),
            self.root.winfo_height(),
        )

    def _on_resize(self, event) -> None:
        start_x, start_y, start_w, start_h = self._resize_start
        dx = self.root.winfo_pointerx() - start_x
        dy = self.root.winfo_pointery() - start_y
        new_w = max(_MIN_WIDTH, start_w + dx)
        new_h = max(_MIN_HEIGHT, start_h + dy)
        # Тащим только за нижне-правый угол — левый верхний угол (x, y окна)
        # не трогаем, поэтому в geometry() не указываем "+x+y", иначе окно
        # заодно поехало бы с места.
        self.root.geometry(f"{new_w}x{new_h}")

    def _handle_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.root.destroy()
        self.on_close()

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.hint_queue.get_nowait()
                tag = msg[0]
                if tag == "partial":
                    _, seq, question, reply, context = msg
                    self._handle_partial(seq, question, reply, context)
                elif tag == "final":
                    _, seq, question, reply, context, elapsed = msg
                    self._handle_final(seq, question, reply, context, elapsed)
                elif tag == "cancel":
                    _, seq = msg
                    self._handle_cancel(seq)
                elif tag == "error":
                    _, message = msg
                    self._handle_error(message)
        except queue.Empty:
            pass
        if not self._closed:
            self.root.after(200, self._poll_queue)

    def run(self) -> None:
        """Блокирует поток до закрытия окна (Esc) или KeyboardInterrupt."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._handle_close()