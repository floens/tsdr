from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import RichLog, Static

from tsdr.core.sdr.engine import get_engine
from tsdr.core.units import format_hz
from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.console.highlight import highlight_command
from tsdr.tui.console.terminal_input import TerminalInput
from tsdr.tui.model import ConsoleUIState

_MATCH_STYLE = Style(bold=True, color="yellow")
_DESC_STYLE = Style(dim=True)
_SELECTED_STYLE = Style(reverse=True)
_MORE_STYLE = Style(dim=True, italic=True)


def _format_frequency(hz: float) -> str:
    """Format frequency compactly: 227.36M, 1.5G, 430k."""
    return format_hz(hz, decimals=2)


class _HistoryLog(RichLog):
    can_focus = False


class ConsoleWidget(Vertical):
    """Shell-like console with scrollable history, autocomplete overlay, and inline prompt.

    Reactive props:
      console_state — full autocomplete state (visible/items/index). One reactive
        instead of three avoids stale-companion reads when multiple fields
        change in the same reconcile batch.
    """

    console_state: reactive[ConsoleUIState] = reactive(ConsoleUIState())

    def compose(self) -> ComposeResult:
        yield _HistoryLog(id="console-history", wrap=True, markup=True)
        yield Static("", id="autocomplete-overlay")
        yield TerminalInput(
            placeholder=" ` console  space run  ←→ tune  ↑↓ bw  d demod  gG gain  ⇧↕ vol  1-9 band  mM^m mem  kj zoom  hl/HL db  i image",
            id="command-input",
        )

    def watch_console_state(self, state: ConsoleUIState) -> None:
        if state.autocomplete_visible and state.autocomplete_items:
            self.show_autocomplete(list(state.autocomplete_items), state.selected_index)
        else:
            self.clear_autocomplete()

    def write_command(self, cmd: str, output: str, prompt: list[tuple[str, Style]]) -> None:
        """Echo a command and its output into the history log."""
        history = self.query_one("#console-history", RichLog)
        line = Text()
        for text, style in prompt:
            line.append(text, style=style)
        for span_text, span_style in highlight_command(cmd):
            line.append(span_text, style=span_style + Style(bold=True))
        history.write(line)
        if output:
            history.write(output)

    def write_info(self, message: str) -> None:
        """Write a line of informational output to the console history."""
        self.query_one("#console-history", RichLog).write(message)

    def show_autocomplete(self, items: list[MenuItem], selected_index: int) -> None:
        """Show autocomplete suggestions as a floating overlay positioned above the input."""
        overlay = self.query_one("#autocomplete-overlay", Static)

        max_items = 10
        total = len(items)
        display_count = min(total, max_items)

        start = max(0, min(selected_index - max_items + 1, total - display_count))
        end = start + display_count

        rendered = Text()
        line_count = 0
        for i in range(start, end):
            item = items[i]
            desc = item.description
            if len(desc) > 60:
                desc = desc[:57] + "..."
            line = Text(" ")
            match_set = set(item.match_indices)
            for j, ch in enumerate(item.value):
                line.append(ch, style=_MATCH_STYLE if j in match_set else None)
            pad = max(1, 16 - len(item.value))
            line.append(" " * pad)
            line.append(desc, style=_DESC_STYLE)
            if i == selected_index:
                line.stylize(_SELECTED_STYLE)
            if line_count > 0:
                rendered.append("\n")
            rendered.append_text(line)
            line_count += 1

        if total > max_items:
            remaining = total - max_items
            rendered.append("\n")
            rendered.append(f" ... {remaining} more matches", style=_MORE_STYLE)
            line_count += 1

        overlay.update(rendered)

        # Position above the input line within ConsoleWidget.
        # With position:absolute + overlay:screen, offset is relative to
        # the container (ConsoleWidget). Place popup so its bottom edge
        # sits just above the input (last row of ConsoleWidget).
        popup_height = line_count + 2  # content + border
        container_height = self.region.height

        # Align horizontally with the typed text (after prompt)
        cmd_input = self.query_one("#command-input", TerminalInput)
        prompt_len = cmd_input.prompt_len
        text_len = len(cmd_input.value)
        available = cmd_input.size.width - prompt_len
        scroll_offset = max(0, text_len - available + 1) if text_len >= available else 0
        x = prompt_len + text_len - scroll_offset

        overlay.styles.offset = (x, container_height - 3 - popup_height)

        overlay.add_class("visible")

    def clear_autocomplete(self) -> None:
        """Hide the autocomplete overlay."""
        overlay = self.query_one("#autocomplete-overlay", Static)
        overlay.update("")
        overlay.remove_class("visible")

    def sync_prompt(self) -> None:
        """Update the terminal prompt to reflect current focused device state."""
        dim = Style(color="white", dim=True)
        device = get_engine().get_focused_device()
        if device is None:
            segments = [("$ ", dim)]
        else:
            freq = _format_frequency(device.config.tuned_frequency)
            segments = [
                (device.active_mode.lower(), Style(color="cyan")),
                (f"@{device.device_id}", dim),
                (f":{freq}", dim),
                ("$ ", dim),
            ]
        self.query_one("#command-input", TerminalInput).prompt_segments = segments

    def clear_history(self) -> None:
        """Clear the history log."""
        self.query_one("#console-history", RichLog).clear()
