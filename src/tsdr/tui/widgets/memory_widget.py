from __future__ import annotations

from rich.markup import escape
from rich.segment import Segment, Segments
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from tsdr.core.bandplans import contrast_fg
from tsdr.core.events.events import MemoriesChangedEvent
from tsdr.core.memories import Memory, get_memory_store, memory_color, recall_memory
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.units import format_hz
from tsdr.tui.commands._format import freq_mhz
from tsdr.tui.inline_edit import InlineEditor
from tsdr.tui.widgets.panel import PanelWidget, set_orientation_classes

_ROW_LINES = 3  # each memory option is name / details / button lines
_BUTTON_LINE = 2  # line offset within an option where hover buttons render
_BUTTON_GAP = "   "

# Fixed line-2 column widths so a varying mode/bandwidth never shifts the tags.
# Sized to the widest values: mode "NAVTEX" (6), bandwidth "156.2k" (6).
_MODE_W = 6
_BW_W = 6
_COL_GAP = "  "

_BUTTONS: list[tuple[str, str, str]] = [
    ("recall", "Recall", "cyan"),
    ("rename", "Rename", "green"),
    ("remove", "Remove", "red"),
]
_CONFIRM_REMOVE = ("remove", "Confirm?", "red bold")


def _buttons(armed_remove: bool) -> list[tuple[str, str, str]]:
    if armed_remove:
        return [*_BUTTONS[:-1], _CONFIRM_REMOVE]
    return _BUTTONS


class _MemoryList(OptionList):
    """OptionList with per-row hover buttons; body-click recalls the memory."""

    can_focus = False  # click-only; the app owns the keyboard

    class RowAction(Message):
        def __init__(self, action: str, memory_id: str) -> None:
            self.action = action
            self.memory_id = memory_id
            super().__init__()

    class RowHover(Message):
        def __init__(self, current: int | None) -> None:
            self.current = current
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hover_index: int | None = None
        self._button_ranges: list[tuple[int, int, str]] = []

    def set_button_ranges(self, ranges: list[tuple[int, int, str]]) -> None:
        self._button_ranges = ranges

    def reset_hover(self) -> None:
        self._hover_index = None
        self._button_ranges = []

    async def _on_click(self, event: events.Click) -> None:
        # Never call super(): OptionList's own action_select/highlight would double-fire.
        event.stop()
        content_y = event.y + self.scroll_offset.y
        lines = self._lines
        if not (0 <= content_y < len(lines)):
            return
        option_index, line_offset = lines[content_y]
        option_id = self.get_option_at_index(option_index).id
        if option_id is None:  # disabled placeholder row ("No memories")
            return
        if line_offset == _BUTTON_LINE and option_index == self._hover_index:
            for start, end, action in self._button_ranges:
                if start <= event.x < end:
                    self.post_message(self.RowAction(action, option_id))
                    return
        self.post_message(self.RowAction("recall", option_id))

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        super()._on_mouse_move(event)
        self._set_hover(self._mouse_hovering_over)

    def _on_leave(self, event: events.Leave) -> None:
        super()._on_leave(event)
        self._set_hover(None)

    def _set_hover(self, index: int | None) -> None:
        if index != self._hover_index:
            self._hover_index = index
            self.post_message(self.RowHover(index))


class _FilterBar(Static):
    """The line above the list: shows the filter, or the live buffer while the
    inline editor is active. Clicking it starts a filter-edit session (the app
    owns the keyboard, so there is no focusable Input)."""

    class Clicked(Message):
        pass

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(self.Clicked())


class MemoryWidget(Vertical, PanelWidget):
    """Docked panel listing frequency memories (bookmarks).

    Click-based: clicking a row recalls it (tunes the focused device to the
    memory's frequency and mode); hovering a row reveals its Recall · Rename ·
    Remove buttons on the line below. Filter and rename text are edited inline
    via `InlineEditor` in the top bar (no focusable Textual Input — the app owns
    keys). Adding memories and editing tags/mode/color stay in the `memory`
    command / `m` hotkey.
    """

    dock_edge = reactive(None)

    def __init__(self) -> None:
        super().__init__()
        self._filter = ""
        self._memories: list[Memory] = []
        self._rows: list[Memory] = []
        self._pos: dict[str, int] = {}
        self._hover_id: str | None = None
        self._active_id: str | None = None
        self._centered = False
        self._pending_remove: str | None = None
        self._remove_timer: Timer | None = None
        self._last_width = -1
        self._filter_bar = _FilterBar(id="memory-filter")
        self._list = _MemoryList(id="memory-list")
        self._editor = InlineEditor(self)

    def compose(self):
        yield self._filter_bar
        yield self._list

    def on_mount(self) -> None:
        self._memories = get_memory_store().all()
        self._rebuild()
        self._refresh_filter_bar()
        # Centering happens on the first sized resize (viewport height is still
        # 0 at mount-time refresh); see on_resize.

    def on_unmount(self) -> None:
        self._editor.cancel()
        self._cancel_remove_timer()

    def watch_dock_edge(self, edge) -> None:
        set_orientation_classes(self, edge)

    def on_resize(self, event: events.Resize) -> None:
        if event.size.width != self._last_width:
            self._last_width = event.size.width
            self.call_after_refresh(lambda: self._rebuild(preserve_scroll=True))
        if not self._centered:
            # Initial layout emits several resizes; center after the last one
            # settles the viewport, not on the premature mount-time refresh.
            self.call_after_refresh(self._center_on_current)

    def update_memories(self, event: MemoriesChangedEvent) -> None:
        self._memories = list(event.memories)  # type: ignore[arg-type]
        ctx = self._editor.context
        if (
            self._editor.active
            and isinstance(ctx, tuple)
            and ctx[0] == "rename"
            and not any(m.id == ctx[1] for m in self._memories)
        ):
            self._editor.cancel()
        self._rebuild(preserve_scroll=True)

    def update_active(self) -> None:
        """Re-evaluate which memory matches the tuned frequency and repaint the
        changed rows. Called on ConfigChanged / FocusChanged."""
        new_id = self._active_in(self._rows)
        if new_id == self._active_id:
            return
        old_id, self._active_id = self._active_id, new_id
        if old_id is not None:
            self._repaint(old_id)
        if new_id is not None:
            self._repaint(new_id)

    def _focused_freq(self) -> float | None:
        device = get_engine().get_focused_device()
        return device.config.tuned_frequency if device is not None else None

    def _active_in(self, memories: list[Memory]) -> str | None:
        freq = self._focused_freq()
        if freq is None:
            return None
        return next((m.id for m in memories if abs(freq - m.frequency) < 1.0), None)

    def _center_on_current(self) -> None:
        if self._centered or not self._rows or self._list.content_size.height <= 0:
            return
        freq = self._focused_freq()
        if freq is None:
            return
        idx = min(range(len(self._rows)), key=lambda i: abs(freq - self._rows[i].frequency))
        viewport = self._list.content_size.height
        target = idx * _ROW_LINES - max(0, (viewport - _ROW_LINES) // 2)
        self._list.scroll_to(y=max(0, target), animate=False)
        self._centered = True

    def _width(self) -> int:
        return int(max(8, self._list.content_size.width - 1))

    def _visible(self) -> list[Memory]:
        q = self._filter.casefold()
        if not q:
            return self._memories
        return [
            m
            for m in self._memories
            if q in m.name.casefold()
            or q in m.audio_spec.mode.casefold()
            or any(q in t.casefold() for t in m.tags)
            or q in f"{m.frequency / 1e6:.3f}"
        ]

    def _rebuild(self, *, preserve_scroll: bool = False) -> None:
        scroll_y = self._list.scroll_offset.y if preserve_scroll else 0
        self._list.clear_options()
        self._rows = []
        self._pos = {}
        self._hover_id = None
        self._list.reset_hover()
        self._cancel_remove_timer()
        self._pending_remove = None
        width = self._width()
        visible = self._visible()
        self._active_id = self._active_in(visible)
        for m in visible:
            self._pos[m.id] = len(self._rows)
            self._rows.append(m)
            row = _render_row(m, hovered=False, width=width, active=m.id == self._active_id)
            self._list.add_option(Option(row[0], id=m.id))
        if not self._rows:
            self._list.add_option(_message_option("No matches" if self._filter else "No memories"))
        if scroll_y:
            self.call_after_refresh(lambda: self._list.scroll_to(y=scroll_y, animate=False))

    def _repaint(self, memory_id: str) -> None:
        pos = self._pos.get(memory_id)
        if pos is None:
            return
        text, ranges = _render_row(
            self._rows[pos],
            hovered=memory_id == self._hover_id,
            width=self._width(),
            armed_remove=memory_id == self._pending_remove,
            active=memory_id == self._active_id,
        )
        self._list.replace_option_prompt_at_index(pos, text)
        if memory_id == self._hover_id:
            self._list.set_button_ranges(ranges)

    def _memory_id_at(self, index: int | None) -> str | None:
        if index is None or not (0 <= index < len(self._rows)):
            return None
        return self._rows[index].id

    @on(_MemoryList.RowHover)
    def _on_row_hover(self, event: _MemoryList.RowHover) -> None:
        previous = self._hover_id
        self._hover_id = self._memory_id_at(event.current)
        if previous is not None:
            self._repaint(previous)
        if self._hover_id is not None:
            self._repaint(self._hover_id)
        else:
            self._list.set_button_ranges([])

    @on(_MemoryList.RowAction)
    def _on_row_action(self, event: _MemoryList.RowAction) -> None:
        pos = self._pos.get(event.memory_id)
        if pos is None:
            return
        memory = self._rows[pos]
        confirming = event.action == "remove" and self._pending_remove == memory.id
        if self._pending_remove is not None and not confirming:
            self._disarm_remove(self._pending_remove)
        if event.action == "recall":
            self._recall(memory)
        elif event.action == "rename":
            self._begin_rename(memory.id)
        elif event.action == "remove":
            if confirming:
                self._remove(memory)
            else:
                self._arm_remove(memory.id)

    def _arm_remove(self, memory_id: str) -> None:
        # First click arms; the button relabels to "Confirm?" for one second,
        # then reverts. A second click within the window actually removes.
        self._cancel_remove_timer()
        self._pending_remove = memory_id
        self._repaint(memory_id)
        self._remove_timer = self.set_timer(1.0, lambda: self._disarm_remove(memory_id))

    def _disarm_remove(self, memory_id: str) -> None:
        if self._pending_remove != memory_id:
            return
        self._cancel_remove_timer()
        self._pending_remove = None
        self._repaint(memory_id)

    def _cancel_remove_timer(self) -> None:
        if self._remove_timer is not None:
            self._remove_timer.stop()
            self._remove_timer = None

    def _remove(self, memory: Memory) -> None:
        self._cancel_remove_timer()
        self._pending_remove = None
        get_memory_store().remove(memory.id)
        self._publish_changed()
        self.app.show_status(f"Removed memory '{escape(memory.name)}'")

    def _recall(self, memory: Memory) -> None:
        did = get_engine().focused_device
        if did is None:
            self.app.show_status("[red]No device focused[/]")
            return
        try:
            recall_memory(memory, did)
        except SDRException as exc:
            self.app.show_status(f"[red]{escape(str(exc))}[/]")
            return
        self.app.show_status(f"Recalled '{escape(memory.name)}' @ {freq_mhz(memory.frequency)}")

    @on(_FilterBar.Clicked)
    def _on_filter_bar_clicked(self, event: _FilterBar.Clicked) -> None:
        self._begin_filter()

    def _refresh_filter_bar(self) -> None:
        self._filter_bar.update(self._filter_bar_content())

    def _filter_bar_content(self) -> Segments | Text:
        editor = self._editor
        if editor.active and editor.buffer is not None:
            context = editor.context
            if isinstance(context, tuple):  # ("rename", memory_id)
                prefix = f"rename {self._memory_name(str(context[1]))}: "
            else:
                prefix = "filter: "  # match the committed "filter: …" display
            segments = [
                Segment(prefix, Style(color="cyan")),
                *editor.buffer.render_segments(Style()),
            ]
            return Segments(segments)
        if self._filter:
            return Text(f"filter: {self._filter}", style="dim", no_wrap=True)
        return Text("filter memories… (click to type)", style="dim", no_wrap=True)

    def _memory_name(self, memory_id: str) -> str:
        pos = self._pos.get(memory_id)
        return self._rows[pos].name[:24] if pos is not None else memory_id

    def _begin_filter(self) -> None:
        self._editor.start(
            self._filter,
            redraw=self._refresh_filter_bar,
            on_change=self._filter_changed,
            on_commit=lambda _value: None,
            context="filter",
        )

    def _filter_changed(self, value: str) -> None:
        self._filter = value.strip()  # keep the typed case; matching casefolds in _visible
        self._rebuild()

    def _begin_rename(self, memory_id: str) -> None:
        memory = get_memory_store().get(memory_id)
        if memory is None:
            return
        self._editor.start(
            memory.name,
            redraw=self._refresh_filter_bar,
            on_commit=lambda value: self._commit_rename(memory_id, value),
            context=("rename", memory_id),
        )

    def _commit_rename(self, memory_id: str, value: str) -> None:
        value = value.strip()
        if not value:
            return
        get_memory_store().rename(memory_id, value)
        self._publish_changed()

    def _publish_changed(self) -> None:
        store = get_memory_store()
        get_engine().event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))


def _message_option(text: str) -> Option:
    return Option(Text(text, style="dim italic"), disabled=True)


def _line1(m: Memory, width: int, *, active: bool = False) -> Text:
    color = memory_color(m)
    if active:
        bar = Style(bgcolor=color, color=contrast_fg(color), bold=True)
        line = Text(no_wrap=True, end="", style=bar)
        marker_style = name_style = freq_style = bar
    else:
        line = Text(no_wrap=True, end="")
        marker_style = Style(color=color)
        name_style = Style(color=color, bold=True)
        freq_style = Style(color="cyan")
    line.append("▼ ", style=marker_style)
    freq = Text(f"{m.frequency / 1e6:.3f} MHz", style=freq_style, end="")
    name = Text(m.name, style=name_style, end="")
    avail = max(1, width - line.cell_len - freq.cell_len - 1)
    name.truncate(avail, overflow="ellipsis")
    line.append_text(name)
    line.append(" " * max(1, width - line.cell_len - freq.cell_len))
    line.append_text(freq)
    return line


def _line2(m: Memory, width: int) -> Text:
    line = Text(no_wrap=True, end="")
    line.append(m.audio_spec.mode.ljust(_MODE_W), style="green")
    line.append(_COL_GAP)
    line.append(format_hz(m.bandwidth, decimals=1).ljust(_BW_W), style="yellow")
    if m.tags:
        line.append(_COL_GAP)
        tags = Text(" ".join(f"#{t}" for t in m.tags), style="dim", end="")
        tags.truncate(max(0, width - line.cell_len), overflow="ellipsis")
        line.append_text(tags)
    return line


def _buttons_line(
    buttons: list[tuple[str, str, str]],
) -> tuple[Text, list[tuple[int, int, str]]]:
    """Render the hover-button line and its click-hit column ranges together, so
    the painted labels and the clickable spans measure from the same text."""
    line = Text(no_wrap=True, end="")
    ranges: list[tuple[int, int, str]] = []
    for i, (action, label, color) in enumerate(buttons):
        if i:
            line.append(_BUTTON_GAP)
        start = line.cell_len
        line.append(label, style=f"{color} underline")
        ranges.append((start, line.cell_len, action))
    return line, ranges


def _render_row(
    m: Memory, *, hovered: bool, width: int, armed_remove: bool = False, active: bool = False
) -> tuple[Text, list[tuple[int, int, str]]]:
    # The button line is always present (blank when idle) so hover never reflows the list.
    button_line = Text(no_wrap=True, end="")
    ranges: list[tuple[int, int, str]] = []
    if hovered:
        button_line, ranges = _buttons_line(_buttons(armed_remove))
    lines = [_line1(m, width, active=active), _line2(m, width), button_line]
    text = Text(no_wrap=True, end="")
    for i, line in enumerate(lines):
        if i:
            text.append("\n")
        text.append_text(line)
    return text, ranges
