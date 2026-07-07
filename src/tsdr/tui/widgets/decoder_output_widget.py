from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.markup import escape
from textual.reactive import reactive
from textual.strip import Strip
from textual.widgets import RichLog

from tsdr.core.events.events import DecodedMessage, DecoderOutputEvent
from tsdr.tui.widgets.panel import PanelWidget, set_orientation_classes

if TYPE_CHECKING:
    from textual.selection import Selection
    from textual.timer import Timer

    from tsdr.tui.model import Edge

# Cap in-place redraws of the live line so a decoder streaming many partials/sec
# can't flood the message loop; a trailing timer still draws the final state.
_TAIL_REDRAW_INTERVAL = 0.05


class DecoderOutputWidget(RichLog, PanelWidget):
    """Decoder console: a scrolling log whose last line is 'live'.

    A decoder streaming `partial=True` messages redraws the last line in place
    (delete its strips, re-write); the line seals into permanent history when a
    distinct line supersedes it, and identical lines fold into `… ×N`. Keeping the
    live line as the log's own last line (rather than a separate widget) means it
    is always flush against the history and selection spans the whole log. Rich
    formatting is opt-in per message via `markup`.

    RichLog is also outside Textual's text-selection protocol (its `render_line`
    applies no cell offsets and consults no selection); this adds both, modelled on
    the `Log` widget.
    """

    dock_edge: reactive[Edge | None] = reactive(None)
    # Non-focusable so digit keys reach the panel toggles instead of scrolling.
    can_focus = False

    def __init__(self) -> None:
        super().__init__(max_lines=500, markup=True, wrap=True, min_width=0, auto_scroll=False)
        self._live_strips = 0
        self._state = "empty"  # "empty" | "streaming" | "sealed"
        self._stream_line = ""
        self._sealed_key: tuple[str, str, bool] | None = None
        self._sealed_line = ""
        self._sealed_count = 0
        self._dirty = False
        self._timer: Timer | None = None
        self._last_render = 0.0

    def watch_dock_edge(self, edge: Edge | None) -> None:
        set_orientation_classes(self, edge)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def update_decoder(self, event: DecoderOutputEvent) -> None:
        for msg in event.messages:
            line = _format_line(event.protocol, msg)
            key = (event.protocol, msg.text, msg.markup)
            if msg.partial:
                if self._state == "sealed":
                    self._commit()
                self._state = "streaming"
                self._stream_line = line
            elif self._state == "sealed" and self._sealed_key == key:
                self._sealed_count += 1
            else:
                if self._state == "sealed":
                    self._commit()
                self._state = "sealed"
                self._sealed_key = key
                self._sealed_line = line
                self._sealed_count = 1
            self._dirty = True
        self._request_render()

    def _live_text(self) -> str:
        if self._state == "streaming":
            return f"{self._stream_line}[dim]█[/dim]"
        if self._state == "sealed":
            if self._sealed_count > 1:
                return f"{self._sealed_line} [dim]×{self._sealed_count}[/dim]"
            return self._sealed_line
        return ""

    def _commit(self) -> None:
        # Draw a pending redraw first, then freeze it so the next line appends below.
        self._draw_live()
        self._live_strips = 0

    def _request_render(self) -> None:
        if self._timer is not None:
            return
        elapsed = time.monotonic() - self._last_render
        if elapsed >= _TAIL_REDRAW_INTERVAL:
            self._flush()
        else:
            self._timer = self.set_timer(_TAIL_REDRAW_INTERVAL - elapsed, self._flush)

    def _flush(self) -> None:
        self._timer = None
        self._last_render = time.monotonic()
        self._draw_live()

    def _draw_live(self) -> None:
        if not self._dirty:
            return
        if not self._size_known:
            return  # can't measure pre-layout; stays dirty, redrawn on the next tick
        self._dirty = False
        content = self._live_text()
        follow = self.is_vertical_scroll_end
        if self._live_strips:
            del self.lines[-self._live_strips :]
            self._line_cache.clear()  # replaced indices would otherwise render stale
        before = len(self.lines)
        self.write(content, scroll_end=follow)
        self._live_strips = len(self.lines) - before
        # RichLog.write only repaints as a side effect of virtual_size changing or a
        # scroll. An in-place last-line edit keeps line count and width the same, so
        # nothing would repaint (the line would only advance on an unrelated refresh,
        # e.g. mouse move). Force it.
        self.refresh()

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        scroll_x, scroll_y = self.scroll_offset
        doc_y = scroll_y + y
        selection = self.text_selection
        if selection is not None:
            span = selection.get_span(doc_y)
            if span is not None:
                start, end = span
                if end == -1:
                    end = strip.cell_length
                start = max(0, min(start, strip.cell_length))
                end = max(start, min(end, strip.cell_length))
                style = self.screen.get_component_rich_style("screen--selection")
                strip = Strip.join(
                    [
                        strip.crop(0, start),
                        strip.crop(start, end).apply_style(style),
                        strip.crop(end, strip.cell_length),
                    ]
                )
        return strip.apply_offsets(scroll_x, doc_y)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        text = "\n".join(line.text for line in self.lines)
        return selection.extract(text), "\n"

    def selection_updated(self, selection: Selection | None) -> None:
        self.refresh()


def _format_line(protocol: str, msg: DecodedMessage) -> str:
    t = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
    # Markup-flagged messages carry decoder-escaped Rich markup; others are plain
    # text and must be escaped so stray '[' don't render as tags.
    body = msg.text if msg.markup else escape(msg.text)
    return f"[dim]{t}[/dim] [cyan]{escape(protocol)}[/cyan] {body}"
