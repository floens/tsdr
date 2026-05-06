"""Base class for TUI mixins.

At runtime the base is MessagePump (provides Textual's metaclass so @on()
handlers are registered).  Type checkers see App as the base plus the custom
attributes that TSDRApp adds, so mixins get full type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App as _Base

    from tsdr.tui.commands.registry import MenuItem
    from tsdr.tui.state import UIState
else:
    from textual.message_pump import MessagePump as _Base


class MixinBase(_Base):  # type: ignore[misc]
    if TYPE_CHECKING:
        ui_state: UIState
        startup_commands: list[str]
        _startup_index: int

        def show_status(self, message: str) -> None: ...
        def _show_error(self, message: str) -> None: ...
        def _show_autocomplete(self, commands: list[MenuItem], selected_index: int) -> None: ...
        def _cycle_preview(self, direction: int) -> None: ...
        def _apply_preview(self) -> None: ...
        def _clear_preview(self) -> None: ...
        def _dismiss_autocomplete(self) -> None: ...
        def _open_autocomplete(self) -> None: ...
        def _focus_command_input(self) -> None: ...
        def _blur_command_input(self) -> None: ...
        def _notify_image_mode_changed(self) -> None: ...
