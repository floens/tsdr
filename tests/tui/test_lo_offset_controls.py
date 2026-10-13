"""LO offset controls: `o`/`O` put the LO below/above the dial (the same key
again turns it off); `config --lo-offset/--lo-offset-size` set it from the console."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams, mock
from tsdr.tui.commands.sdr import config as config_command
from tsdr.tui.keybindings import Action, Context, build_lookup
from tsdr.tui.tuning_mixin import TuningMixin

DIAL = 162.55e6


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    caps = replace(mock._MOCK_CAPABILITIES, zero_if=True, frequency_range=(70e6, 6e9))
    monkeypatch.setattr(mock, "_MOCK_CAPABILITIES", caps)
    engine = SDREngine()
    engine.add_device(
        "pluto",
        "mock",
        MockParams(),
        DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, sample_rate=2.4e6),
    )
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    return engine


def _keys(engine: SDREngine) -> SimpleNamespace:
    keys = SimpleNamespace(statuses=[], errors=[])
    keys._focused = lambda: engine.devices["pluto"]
    keys.show_status = keys.statuses.append
    keys._show_error = keys.errors.append
    return keys


def _press(keys: SimpleNamespace, side: str) -> str:
    TuningMixin._toggle_lo_offset(keys, side)  # type: ignore[arg-type]
    return keys.statuses[-1]


def _lo_offset(engine: SDREngine) -> float:
    config = engine.devices["pluto"].config
    return config.center_frequency - config.tuned_frequency


def test_o_and_shift_o_are_bound() -> None:
    lookup = build_lookup()
    assert lookup[(Context.GLOBAL, "o")] is Action.LO_OFFSET_BELOW
    assert lookup[(Context.GLOBAL, "O")] is Action.LO_OFFSET_ABOVE


def test_keys_switch_sides_and_toggle_off(engine: SDREngine) -> None:
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    keys = _keys(engine)
    # Auto already puts a zero-IF radio's LO below, so `o` turns it off.
    assert _press(keys, "below") == "LO offset: off"
    assert _lo_offset(engine) == 0.0
    assert _press(keys, "below") == "LO offset: 50 kHz below the dial"
    assert _lo_offset(engine) == -50e3
    assert _press(keys, "above") == "LO offset: 50 kHz above the dial"
    assert _lo_offset(engine) == 50e3
    assert _press(keys, "above") == "LO offset: off"


def test_key_without_demod_explains_when_it_applies(engine: SDREngine) -> None:
    keys = _keys(engine)
    assert _press(keys, "above") == "LO offset: above the dial while demodulating"
    assert _lo_offset(engine) == 0.0


def test_config_command_sets_side_and_size(engine: SDREngine, monkeypatch) -> None:
    monkeypatch.setattr(config_command, "get_engine", lambda: engine)
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    cmd = config_command.SDRConfigCommand()
    out = cmd.execute(["--device", "pluto", "--lo-offset", "above", "--lo-offset-size", "30k"])
    assert "lo-offset" in out
    assert _lo_offset(engine) == 30e3
    cmd.execute(["--device", "pluto", "--lo-offset-size", "auto"])
    assert engine.devices["pluto"].config.lo_offset_hz is None
    assert _lo_offset(engine) == 50e3


def test_config_command_rejects_bad_size(engine: SDREngine, monkeypatch) -> None:
    monkeypatch.setattr(config_command, "get_engine", lambda: engine)
    out = config_command.SDRConfigCommand().execute(["--device", "pluto", "--lo-offset-size", "0"])
    assert "lo_offset_hz must be positive" in out
