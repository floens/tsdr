"""The single bottom panel bar.

Renders one horizontal row of buttons — one per docked panel: the hotkey digit
followed by the panel title. Styling lives in app.tcss via the component classes
below (`edge-strip--digit`, `edge-strip--title`, `edge-strip--active`); the
active panel's title uses the active style, inactive ones the title style.
Buttons are padded apart by two spaces.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget


class EdgeStrip(Widget):
    """One button per panel, rendered as a single horizontal row."""

    COMPONENT_CLASSES = {
        "edge-strip--digit",
        "edge-strip--title",
        "edge-strip--active",
    }

    glyphs: reactive[tuple[tuple[str, str, bool], ...]] = reactive(())

    def render(self) -> RenderableType:
        if not self.glyphs:
            return Text("")
        digit_style = self.get_component_rich_style("edge-strip--digit")
        title_style = self.get_component_rich_style("edge-strip--title")
        active_style = self.get_component_rich_style("edge-strip--active")
        text = Text()
        for i, (digit, label, is_active) in enumerate(self.glyphs):
            if i > 0:
                text.append("  ")
            text.append(f"{digit or ' '} ", style=digit_style)
            text.append(label, style=active_style if is_active else title_style)
        return text

    def watch_glyphs(self, _glyphs: tuple[tuple[str, str, bool], ...]) -> None:
        self.refresh()
