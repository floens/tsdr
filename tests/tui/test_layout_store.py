"""Tests for the UIStore layout-mutation methods: toggle_panel / set_panel_active / move_panel."""

from __future__ import annotations

from tsdr.tui.model import EdgePanels, UILayout, UIModel
from tsdr.tui.model.store import UIStore


def _store_with_layout(layout: UILayout) -> UIStore:
    return UIStore(UIModel(layout=layout))


# --- toggle_panel -----------------------------------------------------------


def test_toggle_panel_activates_inactive() -> None:
    store = _store_with_layout(
        UILayout(bottom=EdgePanels(panels=("rds",), active=None)),
    )
    store.toggle_panel("rds")
    assert store.model.layout.bottom.active == "rds"


def test_toggle_panel_deactivates_active() -> None:
    store = _store_with_layout(
        UILayout(bottom=EdgePanels(panels=("rds",), active="rds")),
    )
    store.toggle_panel("rds")
    assert store.model.layout.bottom.active is None


def test_toggle_panel_replaces_other_active_on_same_edge() -> None:
    """Toggling a docked-but-inactive panel makes it active and displaces the previous active."""
    store = _store_with_layout(
        UILayout(bottom=EdgePanels(panels=("rds", "dab"), active="rds")),
    )
    store.toggle_panel("dab")
    assert store.model.layout.bottom.active == "dab"


def test_toggle_panel_finds_panel_on_correct_edge() -> None:
    store = _store_with_layout(
        UILayout(
            right=EdgePanels(panels=("stats",), active=None),
            bottom=EdgePanels(panels=("rds",), active=None),
        )
    )
    store.toggle_panel("stats")
    assert store.model.layout.right.active == "stats"
    assert store.model.layout.bottom.active is None


def test_toggle_panel_undocked_is_noop(caplog) -> None:
    store = _store_with_layout(UILayout())  # nothing docked
    events: list[object] = []
    store.subscribe(lambda *_: events.append(None))
    store.toggle_panel("rds")
    assert events == []
    assert any("panel_toggle_unknown" in r.message for r in caplog.records)


def test_toggle_panel_works_while_bar_hidden() -> None:
    """Hotkeys are independent of the panel bar: toggling activates a panel even
    when the bar is hidden (strips_visible=False)."""
    store = _store_with_layout(
        UILayout(bottom=EdgePanels(panels=("rds",), active=None), strips_visible=False)
    )
    store.toggle_panel("rds")
    assert store.model.layout.bottom.active == "rds"
    assert store.model.layout.strips_visible is False


# --- set_panel_active -------------------------------------------------------


def test_set_panel_active_succeeds() -> None:
    store = _store_with_layout(
        UILayout(right=EdgePanels(panels=("stats", "performance"), active=None))
    )
    store.set_panel_active("right", "performance")
    assert store.model.layout.right.active == "performance"


def test_set_panel_active_to_none_clears() -> None:
    store = _store_with_layout(UILayout(right=EdgePanels(panels=("stats",), active="stats")))
    store.set_panel_active("right", None)
    assert store.model.layout.right.active is None


def test_set_panel_active_rejects_panel_not_on_edge(caplog) -> None:
    store = _store_with_layout(UILayout(right=EdgePanels(panels=("stats",), active=None)))
    events: list[object] = []
    store.subscribe(lambda *_: events.append(None))
    store.set_panel_active("right", "ghost")
    assert events == []
    assert any("panel_set_active_invalid" in r.message for r in caplog.records)


# --- move_panel -------------------------------------------------------------


def test_move_panel_relocates_between_edges() -> None:
    store = _store_with_layout(
        UILayout(
            right=EdgePanels(panels=("stats", "performance"), active="stats"),
            bottom=EdgePanels(panels=()),
        )
    )
    store.move_panel("stats", "bottom")
    assert "stats" not in store.model.layout.right.panels
    assert store.model.layout.right.active is None  # was the moved panel
    assert store.model.layout.bottom.panels == ("stats",)


def test_move_panel_appends_when_index_omitted() -> None:
    store = _store_with_layout(
        UILayout(
            right=EdgePanels(panels=("rds",)),
            bottom=EdgePanels(panels=("dab", "adsb")),
        )
    )
    store.move_panel("rds", "bottom")
    assert store.model.layout.bottom.panels == ("dab", "adsb", "rds")


def test_move_panel_inserts_at_index() -> None:
    store = _store_with_layout(
        UILayout(
            right=EdgePanels(panels=("rds",)),
            bottom=EdgePanels(panels=("dab", "adsb")),
        )
    )
    store.move_panel("rds", "bottom", index=1)
    assert store.model.layout.bottom.panels == ("dab", "rds", "adsb")


def test_move_panel_within_same_edge() -> None:
    """Moving a panel that's already on the target edge reorders without duplicating."""
    store = _store_with_layout(
        UILayout(bottom=EdgePanels(panels=("rds", "dab", "adsb"), active="rds"))
    )
    store.move_panel("rds", "bottom", index=2)
    assert store.model.layout.bottom.panels == ("dab", "adsb", "rds")
    # Active was "rds" and rds is still docked here, so active stays "rds"
    assert store.model.layout.bottom.active == "rds"


def test_move_panel_clears_active_when_panel_was_active_on_old_edge() -> None:
    store = _store_with_layout(
        UILayout(
            right=EdgePanels(panels=("stats", "performance"), active="stats"),
            bottom=EdgePanels(panels=("rds",), active="rds"),
        )
    )
    store.move_panel("stats", "bottom")
    assert store.model.layout.right.active is None
    # bottom's active doesn't change just because stats moved in
    assert store.model.layout.bottom.active == "rds"


def test_move_panel_brand_new_to_edge() -> None:
    """Moving a panel that was undocked just inserts it onto the target edge."""
    store = _store_with_layout(UILayout())
    store.move_panel("rds", "bottom")
    assert store.model.layout.bottom.panels == ("rds",)
