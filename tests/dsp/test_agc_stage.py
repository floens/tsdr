from unittest.mock import MagicMock

import numpy as np

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import AGCGainChangeEvent
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.agc_stage import AGCStage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.devices.base import DeviceCapabilities


def _make_context(
    *,
    enable_agc: bool = True,
    auto_gain: bool = False,
    rf_gain: float = 29.7,
    gain_range: tuple[float, float] = (0.0, 49.6),
) -> tuple[PipelineContext, EventBus, list[AGCGainChangeEvent]]:
    """Build a PipelineContext suitable for driving AGCStage in tests."""
    bus = EventBus()
    captured: list[AGCGainChangeEvent] = []
    bus.subscribe(AGCGainChangeEvent, lambda e: captured.append(e))  # type: ignore[arg-type]

    device = MagicMock()
    device.capabilities = DeviceCapabilities(
        frequency_range=None,
        frequency_controllable=True,
        sample_rates=None,
        gain_supported=gain_range != (0.0, 0.0),
        gain_range=gain_range,
        gain_step=1.0,
        gain_unit="dB",
        bias_tee_supported=False,
    )

    device_context = MagicMock()
    device_context.device = device
    device_context.device_id = "rtl0"
    device_context.config = DeviceConfig(
        rf_gain=rf_gain, auto_gain=auto_gain, enable_agc=enable_agc
    )

    return (
        PipelineContext(device_context=device_context, event_bus=bus, config=SDRConfig()),
        bus,
        captured,
    )


def _batch(iq: np.ndarray) -> SamplesBatch:
    return SamplesBatch(iq_samples=iq.astype(np.complex64))


def _clipping_iq(n: int = 4096) -> np.ndarray:
    # All samples saturate the ADC.
    return np.full(n, 1.0 + 1.0j, dtype=np.complex64)


def _weak_iq(n: int = 4096, level: float = 0.001) -> np.ndarray:
    return np.full(n, level + 0j, dtype=np.complex64)


def test_steps_down_on_clipping():
    ctx, _, captured = _make_context(rf_gain=29.7)
    stage = AGCStage(cooldown_s=0.0, gain_step_db=2.5)
    stage.on_config_change(ctx.device_context.config)

    stage.process(_batch(_clipping_iq()), ctx)

    assert len(captured) == 1
    assert captured[0].rf_gain == 29.7 - 2.5


def test_steps_up_on_weak_signal():
    ctx, _, captured = _make_context(rf_gain=29.7)
    stage = AGCStage(cooldown_s=0.0, gain_step_db=2.5)
    stage.on_config_change(ctx.device_context.config)

    stage.process(_batch(_weak_iq()), ctx)

    assert len(captured) == 1
    assert captured[0].rf_gain == 29.7 + 2.5


def test_clamps_to_max_gain():
    ctx, _, captured = _make_context(rf_gain=49.0, gain_range=(0.0, 49.6))
    stage = AGCStage(cooldown_s=0.0, gain_step_db=2.5)
    stage.on_config_change(ctx.device_context.config)

    stage.process(_batch(_weak_iq()), ctx)

    assert len(captured) == 1
    assert captured[0].rf_gain == 49.6


def test_clamps_to_min_gain():
    ctx, _, captured = _make_context(rf_gain=1.0, gain_range=(0.0, 49.6))
    stage = AGCStage(cooldown_s=0.0, gain_step_db=2.5)
    stage.on_config_change(ctx.device_context.config)

    stage.process(_batch(_clipping_iq()), ctx)

    assert len(captured) == 1
    assert captured[0].rf_gain == 0.0


def test_no_event_when_already_at_boundary():
    ctx, _, captured = _make_context(rf_gain=49.6, gain_range=(0.0, 49.6))
    stage = AGCStage(cooldown_s=0.0, gain_step_db=2.5)
    stage.on_config_change(ctx.device_context.config)

    stage.process(_batch(_weak_iq()), ctx)

    assert captured == []


def test_no_op_when_device_has_no_gain_control():
    ctx, _, captured = _make_context(gain_range=(0.0, 0.0))
    stage = AGCStage(cooldown_s=0.0)
    stage.on_config_change(ctx.device_context.config)

    stage.process(_batch(_clipping_iq()), ctx)
    stage.process(_batch(_weak_iq()), ctx)

    assert captured == []


def test_disabled_when_enable_agc_false():
    ctx, _, captured = _make_context(enable_agc=False, rf_gain=20.0)
    stage = AGCStage(cooldown_s=0.0)

    stage.process(_batch(_clipping_iq()), ctx)

    assert captured == []
    # Internal state syncs to config when AGC is off.
    assert stage._current_gain_db == 20.0


def test_disabled_when_auto_gain_active():
    ctx, _, captured = _make_context(enable_agc=True, auto_gain=True, rf_gain=15.0)
    stage = AGCStage(cooldown_s=0.0)

    stage.process(_batch(_clipping_iq()), ctx)

    assert captured == []
    assert stage._current_gain_db == 15.0


def test_cooldown_blocks_rapid_adjustments():
    ctx, _, captured = _make_context(rf_gain=29.7)
    stage = AGCStage(cooldown_s=10.0, gain_step_db=2.5)
    stage.on_config_change(ctx.device_context.config)

    stage.process(_batch(_clipping_iq()), ctx)
    stage.process(_batch(_clipping_iq()), ctx)

    assert len(captured) == 1
