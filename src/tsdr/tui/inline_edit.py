from __future__ import annotations

from collections.abc import Callable

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.timer import Timer
from textual.widget import Widget

_CURSOR_STYLE = Style(reverse=True)
_BLINK_INTERVAL = 0.53


class InlineEditBuffer:
    """Text buffer with cursor, for inline editing in any widget.

    Handles text manipulation and cursor-aware segment rendering.
    Cursor blink timing is managed by the hosting widget.
    """

    __slots__ = ("value", "cursor_pos", "cursor_visible")

    def __init__(self, value: str, cursor_pos: int | None = None) -> None:
        self.value = value
        self.cursor_pos = cursor_pos if cursor_pos is not None else len(value)
        self.cursor_visible = True

    def insert(self, char: str) -> None:
        self.value = self.value[: self.cursor_pos] + char + self.value[self.cursor_pos :]
        self.cursor_pos += 1

    def backspace(self) -> None:
        if self.cursor_pos > 0:
            self.value = self.value[: self.cursor_pos - 1] + self.value[self.cursor_pos :]
            self.cursor_pos -= 1

    def delete(self) -> None:
        if self.cursor_pos < len(self.value):
            self.value = self.value[: self.cursor_pos] + self.value[self.cursor_pos + 1 :]

    def move_left(self) -> None:
        if self.cursor_pos > 0:
            self.cursor_pos -= 1

    def move_right(self) -> None:
        if self.cursor_pos < len(self.value):
            self.cursor_pos += 1

    def home(self) -> None:
        self.cursor_pos = 0

    def end(self) -> None:
        self.cursor_pos = len(self.value)

    def toggle_cursor(self) -> None:
        self.cursor_visible = not self.cursor_visible

    def reset_cursor(self) -> None:
        self.cursor_visible = True

    def render_segments(self, style: Style) -> list[Segment]:
        """Render text with cursor as Rich Segments.

        When cursor_visible is False, renders plain text without cursor highlight.
        When cursor is at end of text, a trailing space shows the cursor block.
        """
        if not self.cursor_visible:
            return [Segment(self.value, style)] if self.value else []

        before = self.value[: self.cursor_pos]
        if self.cursor_pos < len(self.value):
            cursor_char = self.value[self.cursor_pos]
            after = self.value[self.cursor_pos + 1 :]
        else:
            cursor_char = " "
            after = ""

        segments: list[Segment] = []
        if before:
            segments.append(Segment(before, style))
        segments.append(Segment(cursor_char, _CURSOR_STYLE))
        if after:
            segments.append(Segment(after, style))
        return segments


class InlineEditor:
    """An inline text-edit session hosted inside a widget.

    Owns the edit buffer, the cursor-blink timer, and key handling, so any widget
    gets the same inline editing the spectrum uses for memory labels - without a
    focusable Textual `Input` (the app owns the keyboard and routes keys here via
    `TSDRApp.active_inline_editor`). Per-session behaviour (what to redraw, what to
    do on commit/change/cancel) is passed to `start()`.

    Only one editor is active app-wide: starting a session tears down whatever was
    active (a silent takeover - neither a commit nor a cancel), so the host must
    register its instance's activity through `host.app.active_inline_editor`.
    """

    def __init__(self, host: Widget) -> None:
        self._host = host
        self.buffer: InlineEditBuffer | None = None
        self.context: object = None
        self._timer: Timer | None = None
        self._redraw: Callable[[], None] | None = None
        self._on_commit: Callable[[str], None] | None = None
        self._on_change: Callable[[str], None] | None = None
        self._on_cancel: Callable[[], None] | None = None

    @property
    def active(self) -> bool:
        return self.buffer is not None

    def start(
        self,
        value: str,
        *,
        redraw: Callable[[], None],
        on_commit: Callable[[str], None],
        on_change: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        context: object = None,
    ) -> None:
        app = self._host.app
        current: InlineEditor | None = app.active_inline_editor
        if current is not None:
            current._teardown()
        self.buffer = InlineEditBuffer(value)
        self.context = context
        self._redraw = redraw
        self._on_commit = on_commit
        self._on_change = on_change
        self._on_cancel = on_cancel
        self._timer = self._host.set_interval(_BLINK_INTERVAL, self._blink)
        app.active_inline_editor = self
        redraw()

    def handle_key(self, event: events.Key) -> None:
        buf = self.buffer
        if buf is None:
            return
        if event.key == "enter":
            self._commit()
        elif event.key == "escape":
            self.cancel()
        elif event.key == "backspace":
            buf.backspace()
            self._changed()
        elif event.key == "delete":
            buf.delete()
            self._changed()
        elif event.key == "left":
            buf.move_left()
            self._moved()
        elif event.key == "right":
            buf.move_right()
            self._moved()
        elif event.key == "home":
            buf.home()
            self._moved()
        elif event.key == "end":
            buf.end()
            self._moved()
        elif event.character and event.is_printable:
            buf.insert(event.character)
            self._changed()
        event.prevent_default()

    def cancel(self) -> None:
        on_cancel = self._on_cancel
        self._teardown()
        if on_cancel is not None:
            on_cancel()

    def _commit(self) -> None:
        assert self.buffer is not None
        value = self.buffer.value
        on_commit = self._on_commit
        self._teardown()
        if on_commit is not None:
            on_commit(value)

    def _changed(self) -> None:
        assert self.buffer is not None
        self._reset_cursor()
        if self._on_change is not None:
            self._on_change(self.buffer.value)
        if self._redraw is not None:
            self._redraw()

    def _moved(self) -> None:
        self._reset_cursor()
        if self._redraw is not None:
            self._redraw()

    def _reset_cursor(self) -> None:
        if self.buffer is not None:
            self.buffer.reset_cursor()
        if self._timer is not None:
            self._timer.reset()

    def _blink(self) -> None:
        if self.buffer is None:
            return
        self.buffer.toggle_cursor()
        if self._redraw is not None:
            self._redraw()

    def _teardown(self) -> None:
        """Drop the session without firing commit/cancel and repaint once so the
        cursor disappears. Both explicit end (commit/cancel) and silent takeover
        route through here."""
        redraw = self._redraw
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.buffer = None
        self.context = None
        self._redraw = None
        self._on_commit = None
        self._on_change = None
        self._on_cancel = None
        app = self._host.app
        if app.active_inline_editor is self:
            app.active_inline_editor = None
        if redraw is not None:
            redraw()
