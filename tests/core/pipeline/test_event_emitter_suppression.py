"""Spectrum-providing devices publish wideband frames from the I/O worker;
the pipeline's narrowband IQ FFT event must stay silent there while stats
keep flowing from the IQ path."""

import queue
from dataclasses import dataclass, field

import numpy as np

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import FFTUpdateEvent, StatsUpdateEvent
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.event_emitter_stage import EventEmitterStage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.devices.base import DeviceCapabilities, SpectrumViewStatus


@dataclass
class _FakeDevice:
    capabilities: DeviceCapabilities


@dataclass
class _FakeDeviceContext:
    device: _FakeDevice
    device_id: str = "test"
    config: DeviceConfig = field(default_factory=DeviceConfig)
    demod_profile: None = None
    sample_queue: queue.Queue = field(default_factory=queue.Queue)
    total_samples_read: int = 0
    dropped_samples: int = 0
    active_mode: str = "OFF"
    stereo: bool = False


def _caps(provides_spectrum: bool) -> DeviceCapabilities:
    return DeviceCapabilities(
        frequency_range=None,
        frequency_controllable=True,
        sample_rates=None,
        gain_supported=False,
        gain_range=(0.0, 0.0),
        gain_step=0.0,
        gain_unit="dB",
        bias_tee_supported=False,
        provides_spectrum=provides_spectrum,
    )


def _run(provides_spectrum: bool) -> tuple[list[FFTUpdateEvent], list[StatsUpdateEvent]]:
    bus = EventBus()
    ffts: list[FFTUpdateEvent] = []
    stats: list[StatsUpdateEvent] = []
    bus.subscribe(FFTUpdateEvent, ffts.append)
    bus.subscribe(StatsUpdateEvent, stats.append)

    dc = _FakeDeviceContext(device=_FakeDevice(capabilities=_caps(provides_spectrum)))
    ctx = PipelineContext(device_context=dc, event_bus=bus, config=SDRConfig())  # type: ignore[arg-type]

    n = 1024
    batch = SamplesBatch(
        iq_samples=np.zeros(n, dtype=np.complex64),
        spectrum=np.full(n, -90.0, dtype=np.float32),
        frequencies=np.linspace(99e6, 101e6, n, dtype=np.float32),
        center_frequency=100e6,
        sample_rate=2.4e6,
        rf_gain=20.0,
    )
    EventEmitterStage(config=SDRConfig()).process(batch, ctx)
    return ffts, stats


def test_fft_suppressed_for_spectrum_device():
    ffts, stats = _run(provides_spectrum=True)
    assert ffts == []
    assert len(stats) == 1


def test_fft_flows_for_normal_device():
    ffts, stats = _run(provides_spectrum=False)
    assert len(ffts) == 1
    assert len(stats) == 1


class _FakeSpectrumDevice(_FakeDevice):
    """SpectrumSource-conforming fake; capabilities inherited."""

    def drain_spectrum_frames(self):
        return []

    def set_spectrum_view(self, center_hz: float, span_hz: float) -> None:
        pass

    def spectrum_view_status(self) -> SpectrumViewStatus:
        return SpectrumViewStatus(requested_zoom=8, requested_center_hz=7.1e6, zoom_cap=14)


def test_stats_carry_spectrum_view_status():
    bus = EventBus()
    stats: list[StatsUpdateEvent] = []
    bus.subscribe(StatsUpdateEvent, stats.append)

    dc = _FakeDeviceContext(device=_FakeSpectrumDevice(capabilities=_caps(True)))
    ctx = PipelineContext(device_context=dc, event_bus=bus, config=SDRConfig())  # type: ignore[arg-type]

    n = 1024
    batch = SamplesBatch(
        iq_samples=np.zeros(n, dtype=np.complex64),
        spectrum=np.full(n, -90.0, dtype=np.float32),
        frequencies=np.linspace(99e6, 101e6, n, dtype=np.float32),
        center_frequency=100e6,
        sample_rate=2.4e6,
        rf_gain=20.0,
    )
    EventEmitterStage(config=SDRConfig()).process(batch, ctx)

    assert len(stats) == 1
    sv = stats[0].spectrum_view
    assert sv is not None
    assert (sv.requested_zoom, sv.requested_center_hz, sv.zoom_cap) == (8, 7.1e6, 14)


def test_stats_spectrum_view_none_for_normal_device():
    _, stats = _run(provides_spectrum=False)
    assert stats[0].spectrum_view is None
