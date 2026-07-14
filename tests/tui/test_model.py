import dataclasses

import pytest

from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.model import (
    DEFAULT_LAYOUT,
    ConsoleUIState,
    DeviceUIState,
    EdgePanels,
    UILayout,
    UIModel,
)


def test_defaults() -> None:
    m = UIModel()
    assert m.db_min == -100.0
    assert m.db_max == -30.0
    assert m.image_mode is False
    assert m.layout == DEFAULT_LAYOUT
    assert m.clock_visible is True
    assert m.timezone is None
    assert m.ntp_server is None
    assert m.devices == ()
    assert m.focused_device_id is None
    assert m.console == ConsoleUIState()


def test_default_layout_has_expected_edges() -> None:
    """Default layout: left=decoder-output/memories, right=stats/performance,
    bottom=demod (the demod panel multiplexes RDS/DAB/ADSB/TETRA/DMR based on the
    focused device's active decoder kind)."""
    assert DEFAULT_LAYOUT.left.panels == ("decoder-output", "memories")
    assert DEFAULT_LAYOUT.left.active is None
    assert DEFAULT_LAYOUT.right.panels == ("stats", "performance")
    assert DEFAULT_LAYOUT.right.active is None
    assert DEFAULT_LAYOUT.bottom.panels == ("demod", "directory")
    assert DEFAULT_LAYOUT.bottom.active is None


def test_default_hotkeys_are_unique() -> None:
    digits = [d for d, _ in DEFAULT_LAYOUT.hotkeys]
    assert len(digits) == len(set(digits))


def test_default_panel_bar_is_visible() -> None:
    """The bottom panel bar is the launcher — visible by default."""
    assert DEFAULT_LAYOUT.strips_visible is True
    assert UILayout().strips_visible is True


def test_initial_panel_bar_visible_when_absent_from_prefs() -> None:
    """No `strips_visible` key in prefs → the bar defaults to visible."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": [], "active": ""},
                "right": {"panels": [], "active": ""},
                "bottom": {"panels": [], "active": ""},
                "hotkeys": [],
            }
        }
    }
    m = UIModel.initial(prefs)
    assert m.layout.strips_visible is True


def test_initial_panel_bar_hidden_when_false() -> None:
    """An explicit `strips_visible: false` round-trips to a hidden bar."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": [], "active": ""},
                "right": {"panels": [], "active": ""},
                "bottom": {"panels": [], "active": ""},
                "hotkeys": [],
                "strips_visible": False,
            }
        }
    }
    m = UIModel.initial(prefs)
    assert m.layout.strips_visible is False


def test_frozen() -> None:
    m = UIModel()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.db_min = 2.0  # type: ignore[misc]


def test_replace_returns_new_instance() -> None:
    m = UIModel(db_min=-90.0)
    n = dataclasses.replace(m, db_min=-80.0)
    assert m.db_min == -90.0
    assert n.db_min == -80.0
    assert m is not n


def test_equality_value_based() -> None:
    a = UIModel(db_min=-80.0, db_max=-20.0)
    b = UIModel(db_min=-80.0, db_max=-20.0)
    assert a == b
    assert a is not b


def test_equality_uses_console_subtree() -> None:
    items = (MenuItem(value="x", description="d", match_indices=()),)
    a = UIModel(console=ConsoleUIState(autocomplete_visible=True, autocomplete_items=items))
    b = UIModel(console=ConsoleUIState(autocomplete_visible=True, autocomplete_items=items))
    assert a == b


def test_device_state_defaults() -> None:
    d = DeviceUIState(device_id="rtl0")
    assert d.device_id == "rtl0"
    assert d.has_audio_pipeline is False
    assert d.active_decoder_kind is None


def test_initial_with_empty_prefs() -> None:
    assert UIModel.initial({}) == UIModel()


def test_initial_reads_ui_prefs() -> None:
    prefs = {
        "ui": {
            "db_min": -90.0,
            "db_max": -30.0,
            "image_mode": True,
            "clock_visible": False,
            "timezone": "Europe/Amsterdam",
            "ntp_server": "pool.ntp.org",
        }
    }
    m = UIModel.initial(prefs)
    assert m.db_min == -90.0
    assert m.db_max == -30.0
    assert m.image_mode is True
    assert m.clock_visible is False
    assert m.timezone == "Europe/Amsterdam"
    assert m.ntp_server == "pool.ntp.org"


def test_initial_coerces_empty_strings_to_none() -> None:
    """preferences saves '' for None — must round-trip back to None."""
    prefs = {
        "ui": {
            "timezone": "",
            "ntp_server": "",
        }
    }
    m = UIModel.initial(prefs)
    assert m.timezone is None
    assert m.ntp_server is None


def test_initial_layout_roundtrips_from_prefs() -> None:
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": ["stats"], "active": "stats"},
                "right": {"panels": ["performance"], "active": ""},
                "bottom": {"panels": ["demod", "decoder-output"], "active": "demod"},
                "hotkeys": [
                    {"digit": 1, "panel": "stats"},
                    {"digit": 2, "panel": "demod"},
                ],
            }
        }
    }
    m = UIModel.initial(prefs)
    # 'memories' is absent from the saved layout, so it's augmented onto its
    # default (left) edge.
    assert m.layout.left == EdgePanels(panels=("stats", "memories"), active="stats")
    assert m.layout.right == EdgePanels(panels=("performance",), active=None)
    # 'directory' is absent from the saved layout, so it's augmented onto its
    # default (bottom) edge.
    assert m.layout.bottom == EdgePanels(
        panels=("demod", "decoder-output", "directory"), active="demod"
    )
    # Hotkeys are not persisted — always sourced from the current default.
    assert m.layout.hotkeys == DEFAULT_LAYOUT.hotkeys


def test_initial_layout_falls_back_on_invalid_active() -> None:
    """active that isn't in panels gets coerced to None, not rejected wholesale."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": [], "active": ""},
                "right": {"panels": ["stats"], "active": "ghost"},
                "bottom": {"panels": [], "active": ""},
                "hotkeys": [],
            }
        }
    }
    m = UIModel.initial(prefs)
    assert m.layout.right.active is None


def test_initial_layout_filters_unknown_panel_ids() -> None:
    """Panel ids not in PANELS (typos, prior-schema ids like 'rds') are dropped;
    `active` gets coerced to None when its target was filtered."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": ["decoder-output"], "active": ""},
                "right": {"panels": ["stats", "performance", "ghost"], "active": "ghost"},
                "bottom": {"panels": ["demod"], "active": "demod"},
                "hotkeys": [],
            }
        }
    }
    m = UIModel.initial(prefs)
    assert m.layout.right.panels == ("stats", "performance")
    assert m.layout.right.active is None  # 'ghost' got filtered, so active is invalid


def test_initial_layout_augments_missing_panels() -> None:
    """Saved layout from a prior schema (e.g. only the old per-decoder ids in
    bottom) gets missing PANELS appended to their default edges. The user's
    own customizations to other edges are preserved."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": [], "active": ""},
                "right": {"panels": ["stats"], "active": ""},
                "bottom": {"panels": ["rds", "dab", "adsb"], "active": "rds"},
                "hotkeys": [],
            }
        }
    }
    m = UIModel.initial(prefs)
    # rds/dab/adsb filtered → bottom empty post-filter; demod + directory (both
    # default-bottom) augmented onto it.
    assert m.layout.bottom.panels == ("demod", "directory")
    # decoder-output + memories absent from saved → appended to default edge (left).
    assert m.layout.left.panels == ("decoder-output", "memories")
    # performance was missing from saved right → appended.
    assert m.layout.right.panels == ("stats", "performance")


def test_initial_layout_falls_back_on_garbage() -> None:
    m = UIModel.initial({"ui": {"layout": "not a dict"}})
    assert m.layout == DEFAULT_LAYOUT


def test_initial_ignores_unknown_top_level_keys() -> None:
    m = UIModel.initial({"ui": {"unknown_key": 2.0}, "engine": {"audio_volume": 0.5}})
    assert m == UIModel()


def test_ui_layout_and_edge_panels_frozen() -> None:
    layout = UILayout()
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.left = EdgePanels()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.left.panels = ("x",)  # type: ignore[misc]


def test_devices_tuple_immutable() -> None:
    devices = (DeviceUIState(device_id="a"), DeviceUIState(device_id="b"))
    m = UIModel(devices=devices)
    n = dataclasses.replace(m, devices=(DeviceUIState(device_id="c"),))
    assert m.devices == devices
    assert n.devices == (DeviceUIState(device_id="c"),)
