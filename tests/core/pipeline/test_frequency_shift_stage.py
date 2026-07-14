"""FrequencyShiftStage: per-batch offset from the dial, sign pinned by a tone."""

from dataclasses import dataclass, field

import numpy as np

from tsdr.core.events.bus import EventBus
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.frequency_shift_stage import FrequencyShiftStage
from tsdr.core.sdr.samples_batch import SamplesBatch

_SR = 48_000.0
_N = 4096


@dataclass
class _FakeDeviceContext:
    config: DeviceConfig = field(default_factory=DeviceConfig)


def _ctx(tuned: float) -> PipelineContext:
    dc = _FakeDeviceContext(config=DeviceConfig(tuned_frequency=tuned, center_frequency=tuned))
    return PipelineContext(device_context=dc, event_bus=EventBus(), config=SDRConfig())  # type: ignore[arg-type]


def _tone_batch(tone_offset_hz: float, center: float) -> SamplesBatch:
    t = np.arange(_N) / _SR
    iq = np.exp(2j * np.pi * tone_offset_hz * t).astype(np.complex64)
    return SamplesBatch(iq_samples=iq, center_frequency=center, sample_rate=_SR)


def _peak_offset_hz(iq: np.ndarray) -> float:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq)))
    return (int(np.argmax(spectrum)) - _N // 2) * _SR / _N


def test_tone_at_dial_lands_on_baseband() -> None:
    # Dial 6 kHz above the capture center: the tone at the dial must end at DC.
    center, tuned = 100e6, 100e6 + 6000.0
    stage = FrequencyShiftStage()
    out = stage.process(_tone_batch(6000.0, center), _ctx(tuned))

    assert out is not None and out.iq_samples is not None
    assert out.center_frequency == tuned
    assert _peak_offset_hz(out.iq_samples) == 0.0


def test_stale_capture_center_is_compensated() -> None:
    # Hardware lagging a recenter: batch still carries the old capture
    # center, 6 kHz below the dial, so the dial's signal sits at +6 kHz.
    tuned = 100e6
    stage = FrequencyShiftStage()
    out = stage.process(_tone_batch(6000.0, center=tuned - 6000.0), _ctx(tuned))

    assert out is not None and out.iq_samples is not None
    assert out.center_frequency == tuned
    assert _peak_offset_hz(out.iq_samples) == 0.0


def test_zero_offset_passthrough() -> None:
    stage = FrequencyShiftStage()
    batch = _tone_batch(6000.0, center=100e6)
    assert stage.process(batch, _ctx(100e6)) is batch


def test_offset_change_resets_phase() -> None:
    stage = FrequencyShiftStage()
    stage.process(_tone_batch(6000.0, center=100e6), _ctx(100e6 + 6000.0))
    assert stage.phase_accumulator != 0.0
    stage.process(_tone_batch(5000.0, center=100e6), _ctx(100e6 + 5000.0))
    fresh = FrequencyShiftStage()
    fresh.process(_tone_batch(5000.0, center=100e6), _ctx(100e6 + 5000.0))
    assert stage.frequency_offset == -5000.0
    assert stage.phase_accumulator == fresh.phase_accumulator


def test_offset_past_nyquist_passes_unshifted() -> None:
    stage = FrequencyShiftStage()
    batch = _tone_batch(6000.0, center=100e6)
    assert stage.process(batch, _ctx(100e6 + _SR)) is batch


def test_phase_continuity_across_batches() -> None:
    # Two consecutive batches must equal one shift over the concatenation.
    center, tuned = 100e6, 100e6 + 6000.0
    t = np.arange(2 * _N) / _SR
    iq = np.exp(2j * np.pi * 6000.0 * t).astype(np.complex64)

    whole = FrequencyShiftStage().process(
        SamplesBatch(iq_samples=iq, center_frequency=center, sample_rate=_SR), _ctx(tuned)
    )
    stage = FrequencyShiftStage()
    first = stage.process(
        SamplesBatch(iq_samples=iq[:_N], center_frequency=center, sample_rate=_SR), _ctx(tuned)
    )
    second = stage.process(
        SamplesBatch(iq_samples=iq[_N:], center_frequency=center, sample_rate=_SR), _ctx(tuned)
    )

    assert whole is not None and first is not None and second is not None
    chunked = np.concatenate([first.iq_samples, second.iq_samples])
    np.testing.assert_allclose(chunked, whole.iq_samples, atol=1e-4)
