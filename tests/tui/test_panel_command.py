"""Tests for the `panel move` command and the alt+digit edge-cycle keybinding."""

from __future__ import annotations

from tsdr.tui.commands.builtin.panel import PanelCommand
from tsdr.tui.keyboard import KeyboardMixin
from tsdr.tui.model import Edge, EdgePanels, UILayout, UIModel
from tsdr.tui.model.store import get_ui_store, init_ui_store


def _seed(layout: UILayout) -> None:
    init_ui_store(UIModel(layout=layout))


def _edge_of(layout: UILayout, panel_id: str) -> Edge | None:
    for edge in ("left", "right", "bottom"):
        if panel_id in getattr(layout, edge).panels:
            return edge  # type: ignore[return-value]
    return None


# --- /panel move command ----------------------------------------------------


def test_move_relocates_and_activates() -> None:
    _seed(UILayout(right=EdgePanels(panels=("stats", "performance"), active="stats")))
    out = PanelCommand().execute(["move", "stats", "left"])
    layout = get_ui_store().model.layout
    assert "stats" not in layout.right.panels
    assert layout.left.panels == ("stats",)
    assert layout.left.active == "stats"
    assert "edge=left" in out


def test_move_unknown_panel_errors() -> None:
    _seed(UILayout())
    out = PanelCommand().execute(["move", "nope", "left"])
    assert "Unknown panel" in out


def test_move_index_inserts_at_position() -> None:
    _seed(
        UILayout(
            left=EdgePanels(panels=("decoder-output",), active="decoder-output"),
            right=EdgePanels(panels=("stats",), active="stats"),
        )
    )
    PanelCommand().execute(["move", "stats", "left", "--index", "0"])
    layout = get_ui_store().model.layout
    assert layout.left.panels == ("stats", "decoder-output")
    assert layout.left.active == "stats"


def test_move_completion_offers_ids_then_edges() -> None:
    cmd = PanelCommand()
    ids = [c.value for c in cmd.get_completions(["move"], "")]
    assert "stats" in ids and "performance" in ids
    edges = {c.value for c in cmd.get_completions(["move", "stats"], "")}
    assert edges == {"left", "right", "bottom"}


# --- alt+digit edge cycle ---------------------------------------------------


def test_cycle_moves_panel_to_next_edge() -> None:
    _seed(
        UILayout(
            right=EdgePanels(panels=("stats",), active="stats"),
            hotkeys=((3, "stats"),),
        )
    )
    KeyboardMixin.__new__(KeyboardMixin)._cycle_panel_edge(3)
    layout = get_ui_store().model.layout
    assert _edge_of(layout, "stats") == "left"
    assert layout.left.active == "stats"


def test_cycle_wraps_through_ring() -> None:
    _seed(
        UILayout(
            left=EdgePanels(panels=("stats",), active="stats"),
            hotkeys=((3, "stats"),),
        )
    )
    kb = KeyboardMixin.__new__(KeyboardMixin)
    seen = []
    for _ in range(4):
        kb._cycle_panel_edge(3)
        seen.append(_edge_of(get_ui_store().model.layout, "stats"))
    assert seen == ["bottom", "right", "left", "bottom"]


def test_cycle_unknown_digit_is_noop() -> None:
    _seed(UILayout(right=EdgePanels(panels=("stats",), active="stats"), hotkeys=((3, "stats"),)))
    KeyboardMixin.__new__(KeyboardMixin)._cycle_panel_edge(9)
    assert _edge_of(get_ui_store().model.layout, "stats") == "right"
