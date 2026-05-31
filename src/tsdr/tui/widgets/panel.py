"""Contract for widgets mounted as a panel's content.

A panel widget inherits :class:`PanelWidget` so the dock framework can hand it
context uniformly. Today the only context is ``dock_edge`` — which screen edge
the panel is docked on — set by the reconciler from the derived tree. Widgets
may ignore it; it's the hook for adapting internal layout (a vertical column on
the left/right docks vs a horizontal row on the bottom bar) later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsdr.tui.model import Edge


class PanelWidget:
    """Mixin marking a widget as a panel's content widget.

    Mix in alongside the Textual base, e.g. ``class StatsWidget(Static, PanelWidget)``.
    The reconciler sets ``dock_edge`` to the edge the panel is docked on. It is a
    plain attribute for now (no behaviour); when widgets need to adapt their
    layout it can become a ``reactive`` with a ``watch_dock_edge`` handler.
    """

    dock_edge: Edge | None = None
