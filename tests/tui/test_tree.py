"""Snapshot-style tests for derive_tree. Catches structural regressions."""

from __future__ import annotations

from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.model import (
    DEFAULT_LAYOUT,
    ConsoleUIState,
    DeviceUIState,
    EdgePanels,
    UILayout,
    UIModel,
)
from tsdr.tui.view.tree import derive_tree


def _keys(spec) -> list[str]:
    """Flatten a WidgetSpec tree into a list of keys in pre-order."""
    out = [spec.key]
    for c in spec.children:
        out.extend(_keys(c))
    return out


def _docks_row(tree):
    """main-container > docks-row."""
    return tree.children[1].children[0]


def _center(tree):
    """main-container > docks-row > center."""
    return _docks_row(tree).children[1]


def _empty_layout() -> UILayout:
    return UILayout(left=EdgePanels(), right=EdgePanels(), bottom=EdgePanels(), hotkeys=())


def _empty_layout_bar_visible() -> UILayout:
    return UILayout(
        left=EdgePanels(),
        right=EdgePanels(),
        bottom=EdgePanels(),
        hotkeys=(),
        strips_visible=True,
    )


# --- skeleton ---------------------------------------------------------------


def test_empty_layout_with_bar_visible_shows_single_panel_bar() -> None:
    """The bar is visible by default: one full-width panel-bar, no side strips."""
    tree = derive_tree(UIModel(layout=_empty_layout_bar_visible()))
    assert _keys(tree) == [
        "root",
        "tuner",
        "main-container",
        "docks-row",
        "dock:left",
        "center",
        "viz-container",
        "spectrum",
        "waterfall",
        "dock:right",
        "panel-bar",
        "status-bar",
        "console",
    ]


def test_bar_hidden_yields_no_panel_bar() -> None:
    """strips_visible=False: no panel-bar, docks empty."""
    layout = UILayout(
        left=EdgePanels(),
        right=EdgePanels(),
        bottom=EdgePanels(),
        hotkeys=(),
        strips_visible=False,
    )
    keys = _keys(derive_tree(UIModel(layout=layout)))
    assert "panel-bar" not in keys


def test_active_panel_mounts_without_bar_when_hidden() -> None:
    """A hidden bar does not block panels opening: panel-content still mounts."""
    layout = UILayout(
        right=EdgePanels(panels=("stats",), active="stats"),
        strips_visible=False,
    )
    keys = _keys(derive_tree(UIModel(layout=layout)))
    assert "panel-bar" not in keys
    assert "panel-content:right" in keys


def test_tuner_has_no_reactive_props() -> None:
    tree = derive_tree(UIModel(clock_visible=False, timezone="UTC", image_mode=True))
    tuner = tree.children[0]
    assert tuner.props == {}


def test_spectrum_props_carry_zoom_db_image_mode() -> None:
    m = UIModel(zoom=4.0, db_min=-80.0, db_max=-30.0, image_mode=True)
    tree = derive_tree(m)
    viz = _center(tree).children[0]
    spectrum = viz.children[0]
    assert spectrum.key == "spectrum"
    assert spectrum.props == {
        "zoom": 4.0,
        "db_min": -80.0,
        "db_max": -30.0,
        "image_mode": True,
    }


def test_waterfall_props_carry_zoom_db_image_mode() -> None:
    m = UIModel(zoom=4.0, db_min=-80.0, db_max=-30.0, image_mode=False)
    tree = derive_tree(m)
    waterfall = _center(tree).children[0].children[1]
    assert waterfall.key == "waterfall"
    assert waterfall.props == {
        "zoom": 4.0,
        "db_min": -80.0,
        "db_max": -30.0,
        "image_mode": False,
    }


# --- panel visibility -------------------------------------------------------


def test_no_panel_widget_when_inactive() -> None:
    """A docked-but-inactive panel does not mount a wrapper or widget."""
    layout = UILayout(
        bottom=EdgePanels(panels=("demod",), active=None),
    )
    keys = _keys(derive_tree(UIModel(layout=layout)))
    assert not any(k.startswith("panel-content:") for k in keys)
    assert not any(k.startswith("panel:") for k in keys)


def test_right_active_panel_wraps_widget() -> None:
    layout = UILayout(right=EdgePanels(panels=("stats",), active="stats"))
    tree = derive_tree(UIModel(layout=layout))
    dock_right = _docks_row(tree).children[2]
    assert [c.key for c in dock_right.children] == ["panel-content:right"]
    wrapper = dock_right.children[0]
    assert [c.key for c in wrapper.children] == ["panel:stats"]


def test_left_active_panel_wraps_widget() -> None:
    layout = UILayout(left=EdgePanels(panels=("decoder-output",), active="decoder-output"))
    tree = derive_tree(UIModel(layout=layout))
    dock_left = _docks_row(tree).children[0]
    assert [c.key for c in dock_left.children] == ["panel-content:left"]
    wrapper = dock_left.children[0]
    assert [c.key for c in wrapper.children] == ["panel:decoder-output"]


def test_panel_widget_receives_its_dock_edge() -> None:
    """Each panel widget's spec carries `dock_edge` = the edge it's docked on."""
    layout = UILayout(
        left=EdgePanels(panels=("decoder-output",), active="decoder-output"),
        right=EdgePanels(panels=("stats",), active="stats"),
    )
    tree = derive_tree(UIModel(layout=layout))
    left_widget = _docks_row(tree).children[0].children[0].children[0]
    right_widget = _docks_row(tree).children[2].children[0].children[0]
    assert left_widget.key == "panel:decoder-output"
    assert left_widget.props["dock_edge"] == "left"
    assert right_widget.key == "panel:stats"
    assert right_widget.props["dock_edge"] == "right"


def test_stats_panel_wrapper_includes_constellation_when_image_mode() -> None:
    """Stats + image_mode → wrapper contains stats and constellation, stacked
    in the wrapper's vertical layout."""
    layout = UILayout(right=EdgePanels(panels=("stats",), active="stats"))
    tree = derive_tree(UIModel(layout=layout, image_mode=True))
    wrapper = _docks_row(tree).children[2].children[0]
    assert wrapper.key == "panel-content:right"
    assert [c.key for c in wrapper.children] == ["panel:stats", "constellation"]


def test_performance_panel_wrapper_does_not_include_constellation() -> None:
    layout = UILayout(right=EdgePanels(panels=("performance",), active="performance"))
    tree = derive_tree(UIModel(layout=layout, image_mode=True))
    wrapper = _docks_row(tree).children[2].children[0]
    keys = [c.key for c in wrapper.children]
    assert "constellation" not in keys


# --- demod multiplexer ------------------------------------------------------


def _demod_active() -> UILayout:
    return UILayout(bottom=EdgePanels(panels=("demod",), active="demod"))


def _bottom_wrapper(tree):
    """main-container > docks-row > center > panel-content:bottom."""
    return _center(tree).children[1]


def test_demod_panel_no_focused_device_yields_no_widget() -> None:
    """demod active but no focused device → no wrapper, no inner widget."""
    keys = _keys(derive_tree(UIModel(layout=_demod_active())))
    assert not any(k.startswith("panel-content:") for k in keys)
    assert not any(k.startswith("panel:") for k in keys)


def test_demod_panel_focused_device_without_decoder_yields_no_widget() -> None:
    """Focused device exists but has no active_decoder_kind → no inner widget."""
    m = UIModel(
        layout=_demod_active(),
        devices=(DeviceUIState(device_id="rtl0"),),
        focused_device_id="rtl0",
    )
    keys = _keys(derive_tree(m))
    assert not any(k.startswith("panel:demod") for k in keys)


def test_demod_panel_rds_kind_mounts_rds_widget() -> None:
    m = UIModel(
        layout=_demod_active(),
        devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="rds"),),
        focused_device_id="rtl0",
    )
    tree = derive_tree(m)
    wrapper = _bottom_wrapper(tree)
    assert wrapper.key == "panel-content:bottom"
    panel = wrapper.children[0]
    assert panel.key == "panel:demod:rds"
    assert panel.kind == "decoder_rds"
    assert panel.props == {"dock_edge": "bottom"}


def test_demod_panel_dab_kind_carries_image_mode_prop() -> None:
    m = UIModel(
        layout=_demod_active(),
        image_mode=True,
        devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="dab"),),
        focused_device_id="rtl0",
    )
    tree = derive_tree(m)
    panel = _bottom_wrapper(tree).children[0]
    assert panel.key == "panel:demod:dab"
    assert panel.kind == "decoder_dab"
    assert panel.props == {"dock_edge": "bottom", "image_mode": True}


def test_demod_panel_swaps_key_when_kind_changes() -> None:
    """Switching active_decoder_kind changes both kind and key so the reconciler
    unmounts the old decoder widget and mounts a fresh one."""

    def kind_key(active: str) -> tuple[str, str]:
        m = UIModel(
            layout=_demod_active(),
            devices=(DeviceUIState(device_id="rtl0", active_decoder_kind=active),),  # type: ignore[arg-type]
            focused_device_id="rtl0",
        )
        panel = _bottom_wrapper(derive_tree(m)).children[0]
        return panel.kind, panel.key

    assert kind_key("rds") == ("decoder_rds", "panel:demod:rds")
    assert kind_key("tetra") == ("decoder_tetra", "panel:demod:tetra")
    assert kind_key("adsb") == ("decoder_adsb", "panel:demod:adsb")


def test_demod_panel_text_kind_yields_no_widget() -> None:
    """`text` decoder kind has its own decoder-output panel; demod ignores it."""
    m = UIModel(
        layout=_demod_active(),
        devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="text"),),
        focused_device_id="rtl0",
    )
    keys = _keys(derive_tree(m))
    assert not any(k.startswith("panel:demod") for k in keys)


def test_unknown_active_id_is_silently_ignored() -> None:
    """If active references a panel id not in PANELS, no wrapper or widget is mounted."""
    layout = UILayout(bottom=EdgePanels(panels=("ghost",), active="ghost"))
    keys = _keys(derive_tree(UIModel(layout=layout)))
    assert not any(k.startswith("panel-content:") for k in keys)
    assert not any(k.startswith("panel:ghost") for k in keys)


# --- panel bar glyphs -------------------------------------------------------


def _panel_bar(tree):
    """main-container > panel-bar (its second child when visible)."""
    return tree.children[1].children[1]


def test_panel_bar_pairs_digit_and_label() -> None:
    """Each docked panel yields a (digit, label, is_active) tuple."""
    layout = UILayout(
        bottom=EdgePanels(panels=("demod", "decoder-output"), active="demod"),
        hotkeys=((1, "demod"), (2, "decoder-output")),
    )
    tree = derive_tree(UIModel(layout=layout))
    bar = _panel_bar(tree)
    assert bar.key == "panel-bar"
    assert bar.props["glyphs"] == (
        ("1", "Demod", True),
        ("2", "Decoder", False),
    )


def test_panel_bar_aggregates_all_edges_in_edge_order() -> None:
    """The single bar lists panels from every edge, walked left → bottom → right
    (regardless of hotkey digit)."""
    layout = UILayout(
        left=EdgePanels(panels=("decoder-output",), active="decoder-output"),
        right=EdgePanels(panels=("stats",), active=None),
        bottom=EdgePanels(panels=("demod",), active=None),
        hotkeys=((1, "demod"), (2, "decoder-output"), (3, "stats")),
    )
    tree = derive_tree(UIModel(layout=layout))
    assert _panel_bar(tree).props["glyphs"] == (
        ("2", "Decoder", True),
        ("1", "Demod", False),
        ("3", "Stats", False),
    )


def test_panel_bar_default_layout_reads_left_bottom_right() -> None:
    """The default bar reads left → bottom → right with ascending hotkeys 1-4."""
    tree = derive_tree(UIModel(layout=DEFAULT_LAYOUT))
    assert _panel_bar(tree).props["glyphs"] == (
        ("1", "Decoder", False),
        ("2", "Demod", False),
        ("3", "Stats", False),
        ("4", "Performance", False),
    )


def test_panel_bar_blank_digit_when_no_hotkey() -> None:
    """Panel docked without a hotkey: digit is empty, label still set from PANELS."""
    layout = UILayout(
        bottom=EdgePanels(panels=("demod",), active=None),
        hotkeys=(),
        strips_visible=True,
    )
    tree = derive_tree(UIModel(layout=layout))
    assert _panel_bar(tree).props["glyphs"] == (("", "Demod", False),)


def test_panel_bar_demod_title_static_without_decoder() -> None:
    """No focused decoder → the demod button keeps its static "Demod" title."""
    tree = derive_tree(UIModel(layout=DEFAULT_LAYOUT))
    labels = {g[1] for g in _panel_bar(tree).props["glyphs"]}
    assert "Demod" in labels


def test_panel_bar_demod_title_reflects_active_decoder() -> None:
    """The demod button shows the focused device's active decoder name."""
    m = UIModel(
        layout=DEFAULT_LAYOUT,
        devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="rds"),),
        focused_device_id="rtl0",
    )
    glyphs = _panel_bar(derive_tree(m)).props["glyphs"]
    # digit 2 is the demod panel in the default layout
    demod = next(g for g in glyphs if g[0] == "2")
    assert demod[1] == "RDS"


# --- console ----------------------------------------------------------------


def test_console_props_pass_through_state_object() -> None:
    item = MenuItem(value="x", description="d", match_indices=())
    state = ConsoleUIState(
        autocomplete_visible=True,
        autocomplete_items=(item,),
        selected_index=2,
    )
    tree = derive_tree(UIModel(console=state))
    console = tree.children[3]
    assert console.props == {"console_state": state}


# --- determinism ------------------------------------------------------------


def test_same_model_yields_equal_trees() -> None:
    layout = UILayout(
        right=EdgePanels(panels=("stats",), active="stats"),
        bottom=EdgePanels(panels=("demod",), active="demod"),
        hotkeys=((1, "demod"), (2, "stats")),
    )
    m = UIModel(
        zoom=2.5,
        layout=layout,
        image_mode=True,
        devices=(
            DeviceUIState(device_id="a", active_decoder_kind="rds"),
            DeviceUIState(device_id="b"),
        ),
        focused_device_id="a",
    )
    assert derive_tree(m) == derive_tree(m)
