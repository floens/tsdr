"""Transposable, scrollable panel content base widget.

A :class:`SectionPanel` describes its content as an ordered list of
:class:`Section`s — each a labeled block of lines. The panel's dock edge chooses
the arrangement: stacked vertically on the tall left/right sidebars, laid out as
side-by-side columns on the wide bottom bar. The panel is a scroll view, so when
its content is taller than the space it is given (a short sidebar, or the bottom
bar's ``max-height``) it scrolls rather than overflowing its dock. Subclasses
implement ``build_sections()`` and call ``self.refresh()`` when their data
changes (never ``self.update()`` — content is rendered by an inner surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Static

from tsdr.tui.widgets.panel import PanelWidget, set_orientation_classes

if TYPE_CHECKING:
    from typing import Self

    from textual.app import ComposeResult, RenderResult
    from textual.geometry import Region

    from tsdr.tui.model import Edge


@dataclass(frozen=True)
class Section:
    """One labeled block of a panel.

    ``body`` is Rich markup (``str``) or a renderable. ``width``/``min_width``
    apply only in the horizontal (bottom-bar) layout; vertical stacking ignores
    them.
    """

    body: str | RenderableType
    title: str | None = None
    width: int | None = None
    min_width: int | None = None


class _SectionSurface(Static):
    """Full-height render surface for a SectionPanel.

    Reports the sections' true height (``height: auto``) instead of clipping to
    the viewport, so the enclosing SectionPanel — a scroll view — can scroll it.
    """

    DEFAULT_CSS = "_SectionSurface { width: 1fr; height: auto; }"

    def render(self) -> RenderResult:
        return cast("SectionPanel", self.parent)._render_sections()


class SectionPanel(VerticalScroll, PanelWidget):
    """Base for panels that show one or more columns of labeled rows.

    ``dock_edge`` drives orientation: ``"bottom"`` lays sections side by side,
    ``"left"``/``"right"`` stacks them. The panel is a scroll view: content
    taller than its box scrolls. Subclasses implement ``build_sections``.
    """

    dock_edge: reactive[Edge | None] = reactive(None)

    def compose(self) -> ComposeResult:
        yield _SectionSurface()

    def build_sections(self) -> list[Section]:
        raise NotImplementedError

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> Self:
        # Content lives in the inner surface; re-render and re-measure it
        # (layout=True) so the scrollable extent tracks the content height.
        if self.is_mounted:
            try:
                self.query_one(_SectionSurface).refresh(layout=True)
            except NoMatches:
                pass
        super().refresh(*regions, repaint=repaint, layout=layout, recompose=recompose)
        return self

    def _render_sections(self) -> RenderResult:
        sections = self.build_sections()
        if not sections:
            return Text("")
        if self.dock_edge == "bottom":
            return self._render_horizontal(sections)
        return self._render_vertical(sections)

    def _render_vertical(self, sections: list[Section]) -> RenderResult:
        blocks: list[RenderableType] = []
        for i, s in enumerate(sections):
            if i:
                blocks.append(Text(""))
            if s.title is not None:
                blocks.append(Text.from_markup(s.title))
            blocks.append(_as_renderable(s.body))
        return Group(*blocks)

    def _render_horizontal(self, sections: list[Section]) -> RenderResult:
        grid = Table.grid(padding=(0, 1))
        cells: list[RenderableType] = []
        for s in sections:
            grid.add_column(width=s.width, min_width=s.min_width, vertical="top")
            cells.append(_cell(s))
        grid.add_row(*cells)
        return grid

    def watch_dock_edge(self, edge: Edge | None) -> None:
        set_orientation_classes(self, edge)
        self.refresh()

    def on_resize(self) -> None:
        # Some panels reflow their columns from content height; recompute on resize.
        self.refresh()


def _as_renderable(body: str | RenderableType) -> RenderableType:
    return Text.from_markup(body) if isinstance(body, str) else body


def _cell(s: Section) -> RenderableType:
    body = _as_renderable(s.body)
    if s.title is None:
        return body
    return Group(Text.from_markup(s.title), body)
