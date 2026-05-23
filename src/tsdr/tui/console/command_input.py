from __future__ import annotations

import logging

from textual import on

from tsdr.tui._mixin_base import MixinBase
from tsdr.tui.commands import registry
from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.console.history import get_history
from tsdr.tui.console.terminal_input import TerminalInput
from tsdr.tui.console.widget import ConsoleWidget
from tsdr.tui.model.store import get_ui_store

logger = logging.getLogger(__name__)


def _arg_prefix_len(value: str) -> int:
    """Length of the last in-progress token in `value` (the chars after the last space)."""
    if value.endswith(" "):
        return 0
    parts = value.rsplit(" ", 1)
    return len(parts[-1])


def _menu_action(items: list[MenuItem], auto_apply: bool) -> str:
    """Decide what to do given the computed completion items.

    Auto-apply is reserved for user-initiated Tab: silently completing during
    auto-open would undo edits the user is making (e.g. backspacing back to
    `focus ` would re-fill the single device choice).
    """
    if not items:
        return "none"
    if auto_apply and len(items) == 1:
        return "apply"
    return "show"


def _should_auto_open(value: str, dismissed_for: str, has_completions: bool) -> bool:
    """Decide whether the completion menu should auto-open for this input."""
    if not value.strip():
        return False
    if value == dismissed_for:
        return False
    return has_completions


class CommandInputMixin(MixinBase):
    """Handles command input, autocomplete, and command registration."""

    _auto_open_dismissed_for: str = ""

    @on(TerminalInput.Changed, "#command-input")
    def on_input_changed(self, event: TerminalInput.Changed) -> None:
        if get_ui_store().model.console.autocomplete_visible:
            self._clear_preview()
        self._maybe_auto_open(event.value)

    def _maybe_auto_open(self, value: str) -> None:
        has_completions = bool(registry.get_completions(value))
        if not _should_auto_open(value, self._auto_open_dismissed_for, has_completions):
            return
        self._open_autocomplete(auto_apply=False)

    @on(TerminalInput.Submitted, "#command-input")
    def on_submit(self, event: TerminalInput.Submitted) -> None:
        console_state = get_ui_store().model.console
        if console_state.autocomplete_visible and console_state.selected_index >= 0:
            self._apply_preview()
            return

        value = event.value.strip()
        prompt = event.input.prompt_segments

        event.input.value = ""
        event.input.reset_history_cursor()
        self._auto_open_dismissed_for = ""
        self._clear_preview()

        console = self.query_one(ConsoleWidget)

        if not value:
            # Empty Enter: echo an empty prompt line so the user sees the terminal is alive.
            console.write_command("", "", prompt)
            return

        get_history().add(value)

        # Echo the command immediately so the prompt line paints before the
        # command blocks; run the command after the next refresh.
        console.write_command(value, "", prompt)
        self.call_after_refresh(self._execute_submitted, value)

    def _execute_submitted(self, value: str) -> None:
        result = registry.execute(value)
        console = self.query_one(ConsoleWidget)
        if result:
            console.write_info(result)
        console.sync_prompt()

    def _run_startup_commands(self) -> None:
        self._startup_index = 0
        self._execute_next_startup_command()

    def _execute_next_startup_command(self) -> None:
        if self._startup_index >= len(self.startup_commands):
            return

        i = self._startup_index + 1
        total = len(self.startup_commands)
        command = self.startup_commands[self._startup_index]

        try:
            result = registry.execute(command)
            registry.last_command_output = f"[Startup {i}/{total}] {result}"
            self.show_status(registry.last_command_output)
        except Exception as e:
            logger.error(
                "startup_command_failed index=%d total=%d error=%r", i, total, e, exc_info=True
            )
            registry.last_command_output = f"[Startup {i}/{total}] Error: {e}"
            self._show_error(str(e))

        self._startup_index += 1
        if self._startup_index < len(self.startup_commands):
            self.call_after_refresh(self._execute_next_startup_command)
        else:
            self.query_one(ConsoleWidget).sync_prompt()

    def _focus_command_input(self) -> None:
        cmd_input = self.query_one("#command-input", TerminalInput)
        cmd_input.active = True
        self.query_one(ConsoleWidget).add_class("focused")

    def _blur_command_input(self) -> None:
        cmd_input = self.query_one("#command-input", TerminalInput)
        if not cmd_input.value.strip():
            cmd_input.value = ""
        cmd_input.active = False
        self.query_one(ConsoleWidget).remove_class("focused")

    def _open_autocomplete(self, *, auto_apply: bool = True) -> None:
        cmd_input = self.query_one("#command-input", TerminalInput)
        value = cmd_input.value

        if " " in value:
            completions = registry.get_completions(value)
            prefix_len = _arg_prefix_len(value)
            items = [
                MenuItem(c.value, c.description, tuple(range(prefix_len))) for c in completions
            ]
        else:
            items = registry.get_filtered_commands(value.strip())

        action = _menu_action(items, auto_apply=auto_apply)
        if action == "none":
            return
        store = get_ui_store()
        if action == "apply":
            store.update_console(
                autocomplete_visible=True,
                autocomplete_items=tuple(items),
                selected_index=0,
            )
            self._apply_preview()
            return

        store.update_console(
            autocomplete_visible=True,
            autocomplete_items=tuple(items),
            selected_index=-1,
        )

    def _cycle_preview(self, direction: int) -> None:
        store = get_ui_store()
        console = store.model.console
        items = console.autocomplete_items
        if not items:
            return

        if console.selected_index == -1:
            new_index = 0 if direction > 0 else len(items) - 1
        else:
            new_index = (console.selected_index + direction) % len(items)

        store.update_console(selected_index=new_index)

    def _apply_preview(self) -> None:
        console = get_ui_store().model.console
        if not console.autocomplete_items or console.selected_index < 0:
            return

        selected = console.autocomplete_items[console.selected_index].value

        cmd_input = self.query_one("#command-input", TerminalInput)
        value = cmd_input.value

        if " " in value:
            if value.endswith(" "):
                cmd_input.value = f"{value}{selected} "
            else:
                parts = value.rsplit(" ", 1)
                cmd_input.value = f"{parts[0]} {selected} "
        else:
            cmd_input.value = f"{selected} "

        cmd_input.cursor_position = len(cmd_input.value)
        self._auto_open_dismissed_for = ""

        self._clear_preview()

    def _clear_preview(self) -> None:
        get_ui_store().update_console(
            autocomplete_visible=False,
            autocomplete_items=(),
            selected_index=-1,
        )

    def _dismiss_autocomplete(self) -> None:
        cmd_input = self.query_one("#command-input", TerminalInput)
        self._auto_open_dismissed_for = cmd_input.value
        self._clear_preview()
