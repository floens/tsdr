"""I/O-worker side of the SpectrumSource contract: device frames become
FFTUpdateEvents, and view changes reach the device exactly once per change."""

from unittest.mock import MagicMock

import numpy as np

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.workers.io_worker import IOWorker
from tsdr.core.workers import WorkerContext, WorkerLifecycle
from tsdr.devices.base import DeviceCapabilities, SpectrumFrame, SpectrumViewStatus

_CAPS = DeviceCapabilities(
    frequency_range=(0.0, 30e6),
    frequency_controllable=True,
    sample_rates=(12_000.0,),
    gain_supported=False,
    gain_range=(0.0, 0.0),
    gain_step=0.0,
    gain_unit="dB",
    bias_tee_supported=False,
    provides_spectrum=True,
)


class _SpectrumDevice:
    """Minimal SpectrumSource stand-in."""

    def __init__(self) -> None:
        self.frames: list[SpectrumFrame] = []
        self.view_calls: list[tuple[float, float]] = []

    @property
    def capabilities(self) -> DeviceCapabilities:
        return _CAPS

    def drain_spectrum_frames(self) -> list[SpectrumFrame]:
        frames = self.frames
        self.frames = []
        return frames

    def set_spectrum_view(self, center_hz: float, span_hz: float) -> None:
        self.view_calls.append((center_hz, span_hz))

    def spectrum_view_status(self) -> SpectrumViewStatus | None:
        return None


def _worker() -> tuple[IOWorker, _SpectrumDevice, WorkerContext, list[FFTUpdateEvent]]:
    device = _SpectrumDevice()
    device_context = MagicMock()
    device_context.device = device
    device_context.device_id = "kiwi"

    bus = EventBus()
    events: list[FFTUpdateEvent] = []
    bus.subscribe(FFTUpdateEvent, events.append)

    lifecycle = WorkerLifecycle()
    lifecycle.mark_running()
    ctx = WorkerContext(worker_id="test", event_bus=bus, lifecycle=lifecycle)
    return IOWorker(device_context), device, ctx, events


def _frame(seq: int, center: float = 15e6, span: float = 30e6) -> SpectrumFrame:
    return SpectrumFrame(
        db_bins=np.full(1024, -90.0, dtype=np.float32), center_hz=center, span_hz=span, seq=seq
    )


def test_drained_frames_become_fft_events():
    worker, device, ctx, events = _worker()
    device.frames = [_frame(1), _frame(2, center=7.1e6, span=117e3)]

    worker._drain_spectrum_frames(ctx)

    assert len(events) == 2
    assert events[0].center_frequency == 15e6
    assert events[0].sample_rate == 30e6
    assert events[1].center_frequency == 7.1e6
    assert events[1].sample_rate == 117e3
    assert events[1].frequencies is None  # widgets derive the axis from center/rate


def test_non_spectrum_device_emits_nothing():
    worker, _, ctx, events = _worker()
    worker.device_context.device = object()
    worker._drain_spectrum_frames(ctx)
    assert events == []


def test_view_pushed_once_per_change():
    worker, device, _, _ = _worker()
    config = DeviceConfig(
        tuned_frequency=7.1e6, center_frequency=7.1e6, sample_rate=12_000.0, spectrum_span=100e3
    )

    worker._maybe_apply_spectrum_view(config)
    worker._maybe_apply_spectrum_view(config)

    assert device.view_calls == [(7.1e6, 100e3)]


def test_dial_retune_moves_tracking_view():
    worker, device, _, _ = _worker()
    config = DeviceConfig(
        tuned_frequency=7.1e6, center_frequency=7.1e6, sample_rate=12_000.0, spectrum_span=100e3
    )
    worker._maybe_apply_spectrum_view(config)
    worker._maybe_apply_spectrum_view(config.with_changes(tuned_frequency=14.2e6))

    assert device.view_calls == [(7.1e6, 100e3), (14.2e6, 100e3)]


def test_full_band_view_uses_capability_range():
    worker, device, _, _ = _worker()
    worker._maybe_apply_spectrum_view(
        DeviceConfig(tuned_frequency=7.1e6, center_frequency=7.1e6, sample_rate=12_000.0)
    )
    assert device.view_calls == [(15e6, 30e6)]
