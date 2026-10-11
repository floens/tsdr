"""`demod` accepts mode names in any case, as the README shows (`demod wfm`)."""

from __future__ import annotations

import pytest

from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams
from tsdr.tui.commands.registry import COMMANDS, execute


@pytest.fixture
def engine() -> SDREngine:
    eng = SDREngine()
    eng.add_device("m", "mock", MockParams(), DeviceConfig())
    return eng


@pytest.mark.parametrize("typed", ["wfm", "WFM", "Wfm"])
def test_demod_mode_is_case_insensitive(engine: SDREngine, typed: str) -> None:
    out = execute(f"demod {typed}")
    assert "Enabled" in out
    assert engine.devices["m"].active_mode == "WFM"


@pytest.mark.parametrize("typed", ["off", "OFF"])
def test_demod_off_is_case_insensitive(engine: SDREngine, typed: str) -> None:
    execute("demod nfm")
    out = execute(f"demod {typed}")
    assert "Disabled" in out
    assert engine.devices["m"].active_mode == "RAW"


def test_demod_completion_matches_lowercase_prefix() -> None:
    values = [c.value for c in COMMANDS["demod"].get_completions([], "nf")]
    assert values == ["NFM"]
