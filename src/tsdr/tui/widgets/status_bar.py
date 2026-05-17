import logging
from typing import Literal

from textual.timer import Timer
from textual.widgets import Static

from tsdr.core.events.events import JitterBufferUpdateEvent

logger = logging.getLogger(__name__)

DisplayMode = Literal["output", "error"]


class StatusBar(Static):
    """Status bar that displays general output, errors, or a buffering indicator.

    Three priority layers, highest wins:
      1. error — transient, auto-clears after DISPLAY_DURATION
      2. buffering — present while any device's jitter buffer is rebuffering
      3. output — last command output, auto-clears after DISPLAY_DURATION

    Buffering preempts output (and reverts to it once the rebuffer clears),
    but errors preempt buffering (and on error-clear, fall through to
    buffering or output via _refresh()).
    """

    DEFAULT_CLASSES = "status-bar"
    DISPLAY_DURATION = 30.0  # seconds

    def __init__(self) -> None:
        super().__init__("")
        self._display_mode: DisplayMode = "output"
        self._last_output: str = ""
        self._error_timer: Timer | None = None
        self._output_timer: Timer | None = None
        # (device_id, fill_seconds, target_seconds), or None when not buffering.
        self._buffering_state: tuple[str, float, float] | None = None

    def show_output(self, message: str) -> None:
        """Show command output (auto-clears after DISPLAY_DURATION)."""
        if self._error_timer is not None:
            self._error_timer.stop()
            self._error_timer = None
        if self._output_timer is not None:
            self._output_timer.stop()
            self._output_timer = None

        self._display_mode = "output"
        self._last_output = message
        self._refresh()

        if message:
            self._output_timer = self.set_timer(self.DISPLAY_DURATION, self._clear_output)

    def show_error(self, message: str) -> None:
        if self._error_timer is not None:
            self._error_timer.stop()

        self._display_mode = "error"
        self._render_error(message)

        # Auto-revert to whatever layer is now top (buffering or output).
        self._error_timer = self.set_timer(self.DISPLAY_DURATION, self._revert_from_error)

    def update_jitter_buffer(self, event: JitterBufferUpdateEvent) -> None:
        """Wired from JitterBufferUpdate messages; tracks rebuffering state."""
        if event.rebuffering:
            self._buffering_state = (
                event.device_id,
                event.fill_seconds,
                event.target_seconds,
            )
        else:
            self._buffering_state = None
        self._refresh()

    def _revert_from_error(self) -> None:
        self._error_timer = None
        self._display_mode = "output"
        self._refresh()

    def _clear_output(self) -> None:
        self._output_timer = None
        self._last_output = ""
        if self._display_mode == "output":
            self._refresh()

    def _refresh(self) -> None:
        """Render whichever layer wins. Errors preempt buffering; buffering preempts output."""
        if self._display_mode == "error":
            # Error message was already rendered by _render_error and stays
            # until its timer fires.
            return
        if self._buffering_state is not None:
            dev, fill, target = self._buffering_state
            self.update(f"[yellow]Buffering[/yellow] {dev}… {fill:.2f}s / {target:.2f}s")
            return
        self._render_output(self._last_output)

    def _render_output(self, message: str) -> None:
        self.update(message if message else "")

    def _render_error(self, message: str) -> None:
        self.update(f"[red]Warning[/red] {message}")
