"""LO offset through the engine: the demodulator and the offset settings move
a zero-IF radio's LO beside the dial."""

from dataclasses import replace
from types import MappingProxyType

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, StageType
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams, mock

DIAL = 162.55e6


@pytest.fixture
def zero_if(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = replace(mock._MOCK_CAPABILITIES, zero_if=True, frequency_range=(70e6, 6e9))
    monkeypatch.setattr(mock, "_MOCK_CAPABILITIES", caps)


def _engine(**config) -> SDREngine:
    engine = SDREngine()
    defaults = {"tuned_frequency": DIAL, "center_frequency": DIAL, "sample_rate": 2.4e6}
    engine.add_device("pluto", "mock", MockParams(), DeviceConfig(**{**defaults, **config}))
    return engine


def _lo_offset(engine: SDREngine) -> float:
    config = engine.devices["pluto"].config
    return config.center_frequency - config.tuned_frequency


@pytest.mark.usefixtures("zero_if")
def test_demodulating_moves_lo_below_dial() -> None:
    engine = _engine()
    assert _lo_offset(engine) == 0.0
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    assert _lo_offset(engine) == -50e3


@pytest.mark.usefixtures("zero_if")
def test_demod_off_puts_lo_back_on_dial() -> None:
    engine = _engine()
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.remove_pipeline("pluto", "audio")
    assert _lo_offset(engine) == 0.0


@pytest.mark.usefixtures("zero_if")
def test_wfm_offset_clears_its_wider_channel() -> None:
    engine = _engine()
    engine.set_audio_demod("pluto", DemodSpec(mode="WFM"))
    assert _lo_offset(engine) == -200e3


@pytest.mark.usefixtures("zero_if")
def test_tuning_keeps_the_offset() -> None:
    engine = _engine()
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.update_device_config("pluto", tuned_frequency=146.52e6)
    assert engine.devices["pluto"].config.center_frequency == 146.52e6 - 50e3


@pytest.mark.usefixtures("zero_if")
def test_side_and_size_settings() -> None:
    engine = _engine()
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.update_device_config("pluto", lo_offset="above")
    assert _lo_offset(engine) == 50e3
    engine.update_device_config("pluto", lo_offset_hz=30e3)
    assert _lo_offset(engine) == 30e3
    engine.update_device_config("pluto", lo_offset="off")
    assert _lo_offset(engine) == 0.0


@pytest.mark.usefixtures("zero_if")
def test_bottom_of_range_flips_lo_above() -> None:
    engine = _engine(tuned_frequency=70.02e6, center_frequency=70.02e6)
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    assert _lo_offset(engine) == pytest.approx(50e3)


def test_low_if_radio_keeps_lo_on_dial() -> None:
    engine = _engine()
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    assert _lo_offset(engine) == 0.0


def _add_nfm_pipeline_while_running(engine: SDREngine) -> None:
    """The pipelines update set_audio_demod makes, against a device marked
    running (free mode keeps a fitting center only while running) without
    starting workers."""
    context = engine.devices["pluto"]
    context.state = DeviceState.RUNNING
    try:
        audio = PipelineConfig(
            stages=(StageType.FREQUENCY_SHIFT, StageType.DEMODULATOR),
            audio_spec=DemodSpec(mode="NFM"),
        )
        pipelines = MappingProxyType(dict(context.config.pipelines) | {"audio": audio})
        engine.update_device_config("pluto", pipelines=pipelines)
    finally:
        context.state = DeviceState.STOPPED


@pytest.mark.usefixtures("zero_if")
def test_free_mode_moves_lo_off_dial_when_demod_starts() -> None:
    engine = _engine(tuning_mode="free")
    _add_nfm_pipeline_while_running(engine)
    assert _lo_offset(engine) == -50e3


@pytest.mark.usefixtures("zero_if")
def test_free_mode_keeps_distant_lo_on_demod_change() -> None:
    engine = _engine(tuning_mode="free")
    engine.update_device_config("pluto", center_frequency=DIAL - 400e3)
    _add_nfm_pipeline_while_running(engine)
    assert engine.devices["pluto"].config.center_frequency == DIAL - 400e3


def _tune_while_running(engine: SDREngine, freq: float) -> None:
    context = engine.devices["pluto"]
    context.state = DeviceState.RUNNING
    try:
        engine.update_device_config("pluto", tuned_frequency=freq)
    finally:
        context.state = DeviceState.STOPPED


@pytest.mark.usefixtures("zero_if")
def test_free_mode_moves_the_lo_when_the_dial_lands_on_it() -> None:
    engine = _engine(tuning_mode="free")
    engine.update_device_config("pluto", center_frequency=DIAL - 400e3)
    _add_nfm_pipeline_while_running(engine)
    _tune_while_running(engine, DIAL - 400e3)
    assert _lo_offset(engine) == -50e3


@pytest.mark.usefixtures("zero_if")
def test_free_mode_keeps_the_lo_when_the_dial_stays_clear() -> None:
    engine = _engine(tuning_mode="free")
    engine.update_device_config("pluto", center_frequency=DIAL - 400e3)
    _add_nfm_pipeline_while_running(engine)
    _tune_while_running(engine, DIAL - 300e3)
    assert engine.devices["pluto"].config.center_frequency == DIAL - 400e3


@pytest.mark.usefixtures("zero_if")
@pytest.mark.parametrize("tuning_mode", ["center", "free"])
def test_a_bandwidth_change_leaves_a_fixed_width_decoder_on_the_dial(tuning_mode: str) -> None:
    """ADS-B's channel is 2 MHz whatever the bandwidth setting: too wide to sit
    beside the LO. The bandwidth keys lower the setting; a pending 100 kHz was
    taken as the channel's width and moved the LO into the channel."""
    engine = _engine(tuned_frequency=1090e6, center_frequency=1090e6, tuning_mode=tuning_mode)
    engine.set_audio_demod("pluto", DemodSpec(mode="ADSB"))
    assert _lo_offset(engine) == 0.0
    engine.update_device_config("pluto", channel_bandwidth=100e3)
    assert _lo_offset(engine) == 0.0


@pytest.mark.usefixtures("zero_if")
def test_an_audio_modes_pending_bandwidth_sizes_the_offset() -> None:
    engine = _engine()
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.update_device_config("pluto", channel_bandwidth=150e3, tuned_frequency=DIAL + 1e3)
    assert _lo_offset(engine) == -150e3
