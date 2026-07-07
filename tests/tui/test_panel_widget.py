"""The panel↔widget contract: every panel-content widget is a PanelWidget."""

from __future__ import annotations

import inspect

import pytest
from textual.reactive import Reactive

from tsdr.tui.view.factory import FACTORY
from tsdr.tui.view.tree import _DEMOD_FACTORY_KINDS
from tsdr.tui.widgets.panel import PanelWidget

# Kinds whose widget is mounted as a panel's content: the demod multiplexer's
# decoder widgets plus the standalone panels.
_PANEL_CONTENT_KINDS = (*_DEMOD_FACTORY_KINDS.values(), "decoder_text", "stats", "performance")


@pytest.mark.parametrize("kind", _PANEL_CONTENT_KINDS)
def test_panel_content_widget_is_panel_widget(kind: str) -> None:
    """A new panel widget that forgets to inherit PanelWidget fails here."""
    assert issubclass(FACTORY[kind], PanelWidget), f"FACTORY[{kind!r}] must inherit PanelWidget"


@pytest.mark.parametrize("kind", _PANEL_CONTENT_KINDS)
def test_dock_edge_default_is_none(kind: str) -> None:
    """The contract's dock_edge hook is present and defaults to None, whether the
    widget keeps the plain attr or promotes it to a reactive (SectionPanel)."""
    dock_edge = inspect.getattr_static(FACTORY[kind], "dock_edge")
    if isinstance(dock_edge, Reactive):
        assert dock_edge._default is None
    else:
        assert dock_edge is None
