import dataclasses

import pytest

from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.model import ConsoleUIState, DeviceUIState, UIModel


def test_defaults() -> None:
    m = UIModel()
    assert m.zoom == 1.0
    assert m.db_min == -90.0
    assert m.db_max == -45.0
    assert m.image_mode is False
    assert m.active_panel is None
    assert m.clock_visible is True
    assert m.timezone is None
    assert m.ntp_server is None
    assert m.devices == ()
    assert m.focused_device_id is None
    assert m.console == ConsoleUIState()


def test_frozen() -> None:
    m = UIModel()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.zoom = 2.0  # type: ignore[misc]


def test_replace_returns_new_instance() -> None:
    m = UIModel(zoom=1.0)
    n = dataclasses.replace(m, zoom=2.0)
    assert m.zoom == 1.0
    assert n.zoom == 2.0
    assert m is not n


def test_equality_value_based() -> None:
    a = UIModel(zoom=2.0, db_min=-80.0)
    b = UIModel(zoom=2.0, db_min=-80.0)
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
            "zoom": 4.5,
            "db_min": -100.0,
            "db_max": -30.0,
            "image_mode": True,
            "active_panel": "stats",
            "clock_visible": False,
            "timezone": "Europe/Amsterdam",
            "ntp_server": "pool.ntp.org",
        }
    }
    m = UIModel.initial(prefs)
    assert m.zoom == 4.5
    assert m.db_min == -100.0
    assert m.db_max == -30.0
    assert m.image_mode is True
    assert m.active_panel == "stats"
    assert m.clock_visible is False
    assert m.timezone == "Europe/Amsterdam"
    assert m.ntp_server == "pool.ntp.org"


def test_initial_coerces_empty_strings_to_none() -> None:
    """preferences.save_ui_state writes '' for None — must round-trip back to None."""
    prefs = {
        "ui": {
            "active_panel": "",
            "timezone": "",
            "ntp_server": "",
        }
    }
    m = UIModel.initial(prefs)
    assert m.active_panel is None
    assert m.timezone is None
    assert m.ntp_server is None


def test_initial_ignores_unknown_panel_value() -> None:
    m = UIModel.initial({"ui": {"active_panel": "garbage"}})
    assert m.active_panel is None


def test_initial_ignores_unknown_top_level_keys() -> None:
    m = UIModel.initial({"ui": {"zoom": 2.0}, "engine": {"audio_volume": 0.5}})
    assert m.zoom == 2.0


def test_devices_tuple_immutable() -> None:
    devices = (DeviceUIState(device_id="a"), DeviceUIState(device_id="b"))
    m = UIModel(devices=devices)
    n = dataclasses.replace(m, devices=(DeviceUIState(device_id="c"),))
    assert m.devices == devices
    assert n.devices == (DeviceUIState(device_id="c"),)
