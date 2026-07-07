"""Contract for widgets mounted as a panel's content, and the wrapper that hosts them.

A panel widget inherits :class:`PanelWidget` so the dock framework can hand it
context uniformly. Widgets may ignore it; it's the hook for adapting internal
layout (a vertical column on the left/right docks vs a horizontal row on the
bottom bar).

:class:`PanelContent` is the per-edge wrapper the reconciler mounts around the
active panel's content. It owns the panel border and the title shown on it. The
static title comes from ``derive_tree`` (the panel registry); a panel can push a
dynamic title via :meth:`PanelWidget.set_panel_title`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive

if TYPE_CHECKING:
    from textual.widget import Widget

    from tsdr.tui.model import Edge


class PanelTitleChanged(Message):
    """Bubbled by a :class:`PanelWidget` to update its panel's border title.

    ``title`` of ``None`` reverts to the static title from ``derive_tree``.
    """

    def __init__(self, title: str | None) -> None:
        self.title = title
        super().__init__()


class PanelWidget:
    """Mixin marking a widget as a panel's content widget.

    Mix in alongside the Textual base, e.g. ``class StatsWidget(Static, PanelWidget)``.
    The reconciler sets ``dock_edge`` to the edge the panel is docked on. It is a
    plain attribute by default; widgets that adapt their layout promote it to a
    ``reactive`` (see :class:`~tsdr.tui.widgets.section_panel.SectionPanel`) with a
    ``watch_dock_edge`` handler, typically calling :func:`set_orientation_classes`.
    """

    dock_edge: Edge | None = None

    def set_panel_title(self, title: str | None) -> None:
        """Update the title shown in the enclosing panel's border.

        Bubbles to the :class:`PanelContent` wrapper; call again to change it, or
        pass ``None`` to revert to the static title. An override does not survive a
        panel/dock switch, so a panel using this must (re)post from ``on_mount`` or
        its own state changes.
        """
        cast("Widget", self).post_message(PanelTitleChanged(title))


class PanelContent(Vertical):
    """Per-edge wrapper that draws the panel border and its title.

    ``panel_title`` is the static title set by ``derive_tree`` from the panel
    registry. A child :class:`PanelWidget` may override it dynamically via
    :meth:`PanelWidget.set_panel_title`; the override is dropped whenever the
    static title changes (a different panel became active on this edge).
    """

    panel_title: reactive[str] = reactive("")

    def __init__(self) -> None:
        super().__init__()
        self._override: str | None = None

    def watch_panel_title(self, title: str) -> None:
        self._override = None
        self._apply_title()

    def on_panel_title_changed(self, event: PanelTitleChanged) -> None:
        self._override = event.title
        self._apply_title()
        event.stop()

    def _apply_title(self) -> None:
        self.border_title = self._override if self._override is not None else self.panel_title


def set_orientation_classes(widget: Widget, edge: Edge | None) -> None:
    """Toggle the ``panel-tall`` / ``panel-wide`` classes from the dock edge.

    ``panel-tall`` on the side docks (the tall sidebar), ``panel-wide`` on the
    bottom bar. Orientation-aware CSS keys off these classes.
    """
    widget.set_class(edge in ("left", "right"), "panel-tall")
    widget.set_class(edge == "bottom", "panel-wide")
