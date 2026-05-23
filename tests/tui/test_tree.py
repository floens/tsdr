"""Snapshot-style tests for derive_tree. Catches structural regressions."""

from __future__ import annotations

from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.model import ConsoleUIState, DeviceUIState, UIModel
from tsdr.tui.view.tree import derive_tree


def _keys(spec) -> list[str]:
    """Flatten a WidgetSpec tree into a list of keys in pre-order."""
    out = [spec.key]
    for c in spec.children:
        out.extend(_keys(c))
    return out


# --- always-present skeleton ------------------------------------------------


def test_empty_model_yields_skeleton() -> None:
    tree = derive_tree(UIModel())
    assert _keys(tree) == [
        "root",
        "tuner",
        "main-container",
        "viz-container",
        "spectrum",
        "waterfall",
        "status-bar",
        "console",
    ]


def test_tuner_has_no_reactive_props() -> None:
    """TunerWidget consumes UI state via the ClockWidget's store poll and via
    update_config events; it has no reactive props of its own."""
    tree = derive_tree(UIModel(clock_visible=False, timezone="UTC", image_mode=True))
    tuner = tree.children[0]
    assert tuner.props == {}


def test_spectrum_props_carry_zoom_db_image_mode() -> None:
    m = UIModel(zoom=4.0, db_min=-80.0, db_max=-30.0, image_mode=True)
    tree = derive_tree(m)
    spectrum = tree.children[1].children[0].children[0]
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
    waterfall = tree.children[1].children[0].children[1]
    assert waterfall.key == "waterfall"
    assert waterfall.props == {
        "zoom": 4.0,
        "db_min": -80.0,
        "db_max": -30.0,
        "image_mode": False,
    }


# --- decoder visibility -----------------------------------------------------


def test_no_decoder_widget_when_no_active_demod() -> None:
    m = UIModel(devices=(DeviceUIState(device_id="rtl0"),), focused_device_id="rtl0")
    keys = _keys(derive_tree(m))
    assert not any(k.startswith("decoder:") for k in keys)


def test_rds_decoder_widget_appears_for_focused_device() -> None:
    m = UIModel(
        devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="rds"),),
        focused_device_id="rtl0",
    )
    keys = _keys(derive_tree(m))
    assert "decoder:rtl0:rds" in keys


def test_dab_decoder_carries_image_mode_prop() -> None:
    m = UIModel(
        image_mode=True,
        devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="dab"),),
        focused_device_id="rtl0",
    )
    tree = derive_tree(m)
    viz = tree.children[1].children[0]
    decoder = viz.children[-1]
    assert decoder.key == "decoder:rtl0:dab"
    assert decoder.props == {"image_mode": True}


def test_non_dab_decoder_has_no_props() -> None:
    m = UIModel(
        devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="adsb"),),
        focused_device_id="rtl0",
    )
    tree = derive_tree(m)
    viz = tree.children[1].children[0]
    decoder = viz.children[-1]
    assert decoder.key == "decoder:rtl0:adsb"
    assert decoder.props == {}


def test_decoder_skipped_for_non_focused_device() -> None:
    m = UIModel(
        devices=(
            DeviceUIState(device_id="rtl0", active_decoder_kind="rds"),
            DeviceUIState(device_id="hackrf", active_decoder_kind="dab"),
        ),
        focused_device_id="rtl0",
    )
    keys = _keys(derive_tree(m))
    assert "decoder:rtl0:rds" in keys
    assert "decoder:hackrf:dab" not in keys


# --- sidebar conditionality -------------------------------------------------


def test_no_sidebar_when_panel_none() -> None:
    tree = derive_tree(UIModel(active_panel=None))
    main_children = [c.key for c in tree.children[1].children]
    assert main_children == ["viz-container"]


def test_sidebar_with_stats_panel() -> None:
    tree = derive_tree(UIModel(active_panel="stats"))
    main = tree.children[1]
    sidebar = main.children[1]
    assert sidebar.key == "sidebar"
    assert [c.key for c in sidebar.children] == ["stats"]


def test_sidebar_stats_plus_constellation_when_image_mode() -> None:
    tree = derive_tree(UIModel(active_panel="stats", image_mode=True))
    main = tree.children[1]
    sidebar = main.children[1]
    assert [c.key for c in sidebar.children] == ["stats", "constellation"]


def test_sidebar_performance_panel() -> None:
    tree = derive_tree(UIModel(active_panel="performance"))
    main = tree.children[1]
    sidebar = main.children[1]
    assert [c.key for c in sidebar.children] == ["performance"]


def test_constellation_does_not_appear_in_performance_panel() -> None:
    tree = derive_tree(UIModel(active_panel="performance", image_mode=True))
    sidebar = tree.children[1].children[1]
    keys = [c.key for c in sidebar.children]
    assert "constellation" not in keys


# --- console ----------------------------------------------------------------


def test_console_props_pass_through_state_object() -> None:
    """ConsoleWidget consumes the whole ConsoleUIState as a single reactive so
    one watcher fires with consistent values (no stale-companion reads)."""
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
    m = UIModel(
        zoom=2.5,
        active_panel="stats",
        image_mode=True,
        devices=(
            DeviceUIState(device_id="a", active_decoder_kind="rds"),
            DeviceUIState(device_id="b"),
        ),
        focused_device_id="a",
    )
    assert derive_tree(m) == derive_tree(m)
