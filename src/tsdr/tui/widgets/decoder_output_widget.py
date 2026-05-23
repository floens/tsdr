import time

from rich.markup import escape
from textual.widgets import RichLog

from tsdr.core.events.events import DecoderOutputEvent


class DecoderOutputWidget(RichLog):
    """Scrollable text widget showing decoded protocol messages."""

    def __init__(self) -> None:
        super().__init__(max_lines=500, auto_scroll=True, markup=True, wrap=True)

    def on_mount(self) -> None:
        self.border_title = "Decoder Output"

    def update_decoder(self, event: DecoderOutputEvent) -> None:
        """Append new decoded messages."""
        for msg in event.messages:
            t = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
            self.write(f"[dim]{t}[/dim] [cyan]{escape(event.protocol)}[/cyan] {escape(msg.text)}")
