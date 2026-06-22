from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import RecordingFinishedEvent
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, SDRConfig, StageType
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.io import load_iq
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.record_stage import RecordStage
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.devices import MockParams


@dataclass
class _FakeDeviceContext:
    device_id: str = "test"


def _make_context() -> tuple[PipelineContext, EventBus, list[RecordingFinishedEvent]]:
    bus = EventBus()
    received: list[RecordingFinishedEvent] = []

    def handler(ev: RecordingFinishedEvent) -> None:
        received.append(ev)

    bus.subscribe(RecordingFinishedEvent, handler)
    ctx = PipelineContext(device_context=_FakeDeviceContext(), event_bus=bus, config=SDRConfig())  # type: ignore[arg-type]
    return ctx, bus, received


def _batch(iq: np.ndarray, sample_rate: float) -> SamplesBatch:
    return SamplesBatch(
        iq_samples=iq.astype(np.complex64),
        sample_rate=sample_rate,
        center_frequency=100e6,
        rf_gain=20.0,
    )


def test_passthrough_round_trip(tmp_path: Path) -> None:
    """No resample: cu8 round-trip preserves samples within quantization error."""
    rng = np.random.default_rng(0)
    n = 10_000
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.3
    out = tmp_path / "test.cu8.zst"

    stage = RecordStage(output_path=out)
    ctx, _, _ = _make_context()

    stage.process(_batch(iq[: n // 2], 250_000.0), ctx)
    stage.process(_batch(iq[n // 2 :], 250_000.0), ctx)
    stage.close()

    assert out.exists()
    loaded = load_iq(out)
    assert len(loaded) == n
    # cu8 has ~1/127.5 quantization on each component → RMS error ≲ 0.005 for unit-scale.
    assert np.mean(np.abs(loaded - iq)) < 0.01


def test_complex64_preserves_subcu8_lsb(tmp_path: Path) -> None:
    """cf32 format preserves low-amplitude signal that cu8 would crush to 127/128.

    Mirrors the Airspy HF+ / SpyServer case: a real but tiny-amplitude signal is
    below one cu8 LSB, so 8-bit recording flatlines it. float32 is lossless.
    """
    rng = np.random.default_rng(0)
    n = 8_000
    # Far below one cu8 LSB (1/255): every sample rounds to the 127/128 boundary.
    iq = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.0005).astype(np.complex64)

    cf32_out = tmp_path / "hi.cf32.zst"
    stage = RecordStage(output_path=cf32_out, sample_format=SampleFormat.COMPLEX64)
    ctx, _, _ = _make_context()
    stage.process(_batch(iq, 250_000.0), ctx)
    stage.close()
    loaded = load_iq(cf32_out)
    assert len(loaded) == n
    np.testing.assert_array_equal(loaded, iq)  # float32 -> float32: bit-exact

    # The same data through cu8 collapses to the two center codes (information lost).
    cu8_out = tmp_path / "lo.cu8.zst"
    stage2 = RecordStage(output_path=cu8_out, sample_format=SampleFormat.UINT8_IQ)
    stage2.process(_batch(iq, 250_000.0), ctx)
    stage2.close()
    crushed = load_iq(cu8_out)
    # Collapsed to the two center codes (127/128): only one bit of amplitude survives.
    assert len(np.unique(crushed.real)) <= 2
    assert len(np.unique(crushed.imag)) <= 2


def test_tone_preserved(tmp_path: Path) -> None:
    """A pure tone round-trips with its peak bin in the right place."""
    sr = 250_000.0
    n = 4096
    t = np.arange(n) / sr
    iq = 0.5 * np.exp(2j * np.pi * 12_500 * t).astype(np.complex64)
    out = tmp_path / "tone.cu8.zst"

    stage = RecordStage(output_path=out)
    ctx, _, _ = _make_context()
    stage.process(_batch(iq, sr), ctx)
    stage.close()

    loaded = load_iq(out)
    assert len(loaded) == n
    spectrum = np.abs(np.fft.fft(loaded))
    peak_bin = int(np.argmax(spectrum))
    expected_bin = int(round(12_500 * n / sr))
    assert abs(peak_bin - expected_bin) <= 1


def test_integer_decimation_fast_path(tmp_path: Path) -> None:
    """2.4 MHz → 240 kHz: (1, 10), must use StreamingDecimFilter and produce ~n/10 samples."""
    sr = 2_400_000.0
    decim = 10
    n = sr_int = int(sr)  # 1 s of data
    # DC + small offset tone at 10 kHz (well inside post-decim Nyquist of 120 kHz).
    t = np.arange(n) / sr
    iq = (0.3 * np.exp(2j * np.pi * 10_000 * t)).astype(np.complex64)
    out = tmp_path / "decim.cu8.zst"

    stage = RecordStage(output_path=out, resample=(1, decim))
    ctx, _, _ = _make_context()
    # Feed in ~40 ms chunks to exercise cross-chunk filter state.
    chunk = 96_000
    for start in range(0, n, chunk):
        stage.process(_batch(iq[start : start + chunk], sr), ctx)
    stage.close()

    loaded = load_iq(out)
    # Allow small slack for filter warm-up tail handling.
    assert abs(len(loaded) - sr_int // decim) <= decim
    # The 10 kHz tone at device rate stays at 10 kHz at decimated rate.
    post_sr = sr / decim
    spectrum = np.abs(np.fft.fft(loaded))
    peak_bin = int(np.argmax(spectrum))
    expected_bin = int(round(10_000 * len(loaded) / post_sr))
    assert abs(peak_bin - expected_bin) <= 2


def test_rational_resample(tmp_path: Path) -> None:
    """2.4 MHz → 64 kHz (up=2, down=75). Must produce ~n * 2/75 samples."""
    sr = 2_400_000.0
    up, down = 2, 75
    n = int(sr)  # 1 s
    t = np.arange(n) / sr
    iq = (0.3 * np.exp(2j * np.pi * 5_000 * t)).astype(np.complex64)
    out = tmp_path / "rational.cu8.zst"

    stage = RecordStage(output_path=out, resample=(up, down))
    ctx, _, _ = _make_context()
    stage.process(_batch(iq, sr), ctx)
    stage.close()

    loaded = load_iq(out)
    expected_n = n * up // down
    # Polyphase warm-up plus internal timing → small slack.
    assert abs(len(loaded) - expected_n) < 200


def test_max_samples_finalizes_and_publishes_event(tmp_path: Path) -> None:
    """Duration mode: stage closes once max_samples is reached and publishes an event."""
    sr = 250_000.0
    n = 50_000
    rng = np.random.default_rng(42)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.2
    out = tmp_path / "duration.cu8.zst"

    max_samples = 30_000
    stage = RecordStage(output_path=out, max_samples=max_samples)
    ctx, _, events = _make_context()

    # Feed more than max_samples worth of data in two chunks.
    stage.process(_batch(iq[:20_000], sr), ctx)
    stage.process(_batch(iq[20_000:], sr), ctx)

    loaded = load_iq(out)
    assert len(loaded) == max_samples
    assert len(events) == 1
    assert events[0].samples_written == max_samples
    assert events[0].path == str(out)


def test_max_samples_in_output_units_with_decimation(tmp_path: Path) -> None:
    """max_samples counts OUTPUT samples so the cap works under downsampling.

    Regression guard: earlier the command computed max_samples from the device
    rate while the stage counted post-resample samples - the cap was never hit
    and the file kept growing. max_samples must be in output units.
    """
    sr = 2_400_000.0
    decim = 10
    out_rate = sr / decim
    # Feed ~5 seconds of device-rate data; cap the stage at 2 seconds of OUTPUT.
    n = int(sr) * 5
    rng = np.random.default_rng(1)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.2
    out = tmp_path / "capped.cu8.zst"
    max_output_samples = int(out_rate * 2)  # 2 seconds of output

    stage = RecordStage(output_path=out, resample=(1, decim), max_samples=max_output_samples)
    ctx, _, events = _make_context()
    chunk = 96_000
    for start in range(0, n, chunk):
        stage.process(_batch(iq[start : start + chunk], sr), ctx)

    assert len(events) == 1
    loaded = load_iq(out)
    # Small slack for the cut-at-boundary step in the stage.
    assert abs(len(loaded) - max_output_samples) < decim


def test_close_is_idempotent(tmp_path: Path) -> None:
    """close() can be called twice without raising."""
    stage = RecordStage(output_path=tmp_path / "idem.cu8.zst")
    stage.close()  # never opened - should no-op
    stage.close()  # second close - must not raise


def test_no_iq_samples_is_noop(tmp_path: Path) -> None:
    """Batches without iq_samples pass through untouched and don't open the file."""
    out = tmp_path / "none.cu8.zst"
    stage = RecordStage(output_path=out)
    ctx, _, _ = _make_context()
    stage.process(SamplesBatch(sample_rate=250_000.0), ctx)
    stage.close()
    # File should not have been created at all.
    assert not out.exists()
    assert not out.with_suffix(out.suffix + ".partial").exists()


def test_engine_removes_pipeline_on_recording_finished_event(tmp_path: Path) -> None:
    """Layering check: the engine itself tears down the recording pipeline on auto-stop.

    Proves that pipeline lifecycle lives in core - no TUI needed. This is the
    scenario a headless/core-only caller relies on, and it would have been
    broken if the remove_pipeline call lived only in the TUI event handler.
    """
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.add_pipeline(
            "rec",
            "recording",
            PipelineConfig(
                stages=(StageType.RECORD,),
                record_path=str(tmp_path / "headless.cu8.zst"),
                record_max_samples=1000,
            ),
        )
        assert "recording" in engine.get_device("rec").config.pipelines

        engine.event_bus.publish(
            RecordingFinishedEvent(
                source_id="record_rec",
                device_id="rec",
                pipeline_name="recording",
                path=str(tmp_path / "headless.cu8.zst"),
                samples_written=1000,
            )
        )

        assert "recording" not in engine.get_device("rec").config.pipelines
    finally:
        engine.shutdown(timeout=2.0)
