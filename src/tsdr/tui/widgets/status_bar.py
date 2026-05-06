import logging
from typing import Literal

from textual.timer import Timer
from textual.widgets import Static

logger = logging.getLogger(__name__)

DisplayMode = Literal["output", "error"]


class StatusBar(Static):
    """Status bar widget that displays general status output or errors.

    The status bar has two display modes:
    - output: Shows the last command output (auto-clears after DISPLAY_DURATION)
    - error: Shows transient error messages (auto-clears after DISPLAY_DURATION)
    """

    DEFAULT_CLASSES = "status-bar"
    DISPLAY_DURATION = 30.0  # seconds

    def __init__(self) -> None:
        super().__init__("")
        self._display_mode: DisplayMode = "output"
        self._last_output: str = ""
        self._error_timer: Timer | None = None
        self._output_timer: Timer | None = None

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
        self._render_output(message)

        if message:
            self._output_timer = self.set_timer(self.DISPLAY_DURATION, self._clear_output)

    def show_error(self, message: str) -> None:
        if self._error_timer is not None:
            self._error_timer.stop()

        self._display_mode = "error"
        self._render_error(message)

        # Auto-revert to output mode after duration
        self._error_timer = self.set_timer(self.DISPLAY_DURATION, self._revert_to_output)

    def _revert_to_output(self) -> None:
        self._error_timer = None
        self.show_output(self._last_output)

    def _clear_output(self) -> None:
        self._output_timer = None
        self._last_output = ""
        if self._display_mode == "output":
            self._render_output("")

    def _render_output(self, message: str) -> None:
        self.update(message if message else "")

    def _render_error(self, message: str) -> None:
        self.update(f"[red]Warning[/red] {message}")
