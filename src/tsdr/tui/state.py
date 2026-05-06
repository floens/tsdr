from dataclasses import dataclass, field

from tsdr.tui.commands.registry import MenuItem


@dataclass
class UIState:
    """Mutable UI state for the TUI layer."""

    current_hint: str = ""
    hint_index: int = -1
    filtered_commands: list[MenuItem] = field(default_factory=list)
    autocomplete_visible: bool = False
    active_panel: str | None = None
    image_mode: bool = False
    zoom: float = 1.0
    db_min: float = -90.0
    db_max: float = -45.0

    def adjust_zoom(self, direction: int) -> None:
        """Zoom in (direction=1) or out (direction=-1) by factor 1.5."""
        if direction > 0:
            self.zoom = round(min(512.0, self.zoom * 1.5), 1)
        else:
            self.zoom = round(max(1.0, self.zoom / 1.5), 1)

    def adjust_db_min(self, direction: int) -> None:
        """Adjust min dB by ±5, clamped so db_min < db_max - 5."""
        new_min = self.db_min + direction * 5
        if new_min < self.db_max - 5:
            self.db_min = new_min

    def adjust_db_max(self, direction: int) -> None:
        """Adjust max dB by ±5, clamped so db_max > db_min + 5."""
        new_max = self.db_max + direction * 5
        if new_max > self.db_min + 5:
            self.db_max = new_max

    def set_hint(self, hint: str, index: int, commands: list[MenuItem]) -> None:
        self.current_hint = hint
        self.hint_index = index
        self.filtered_commands = commands

    def show_autocomplete(self, commands: list[MenuItem]) -> None:
        self.autocomplete_visible = True
        self.hint_index = -1
        self.filtered_commands = commands

    def clear_autocomplete(self) -> None:
        self.autocomplete_visible = False
        self.current_hint = ""
        self.hint_index = -1
        self.filtered_commands = []
