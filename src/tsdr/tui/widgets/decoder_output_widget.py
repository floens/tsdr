import time

from rich.markup import escape
from textual.widgets import RichLog

from tsdr.core.events.events import DecoderOutputEvent
from tsdr.tui.widgets.panel import PanelWidget


class DecoderOutputWidget(RichLog, PanelWidget):
    """Scrollable text widget showing decoded protocol messages."""

    # RichLog is focusable by default; that would let the widget grab keyboard
    # focus when mounted and swallow panel-toggle digit keys (e.g. pressing 6
    # while the panel is open would scroll the log instead of closing it).
    can_focus = False

    def __init__(self) -> None:
        # min_width=0 so wrapping follows the (narrow) panel width; RichLog's
        # default min_width=78 would render wider than the panel and add a
        # horizontal scrollbar instead of wrapping.
        super().__init__(max_lines=500, auto_scroll=True, markup=True, wrap=True, min_width=0)

    def on_mount(self) -> None:
        self.border_title = "Decoder Output"

    def update_decoder(self, event: DecoderOutputEvent) -> None:
        """Append new decoded messages."""
        for msg in event.messages:
            t = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
            # Markup-flagged messages carry decoder-escaped Rich markup; others
            # are plain text and must be escaped so stray '[' don't render as tags.
            body = msg.text if msg.markup else escape(msg.text)
            self.write(f"[dim]{t}[/dim] [cyan]{escape(event.protocol)}[/cyan] {body}")
