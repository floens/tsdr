"""Event definitions for worker communication.

Events are immutable dataclasses published on EventBus.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.devices.base import DeviceCapabilities


@dataclass(frozen=True, kw_only=True)
class Event:
    """Base event class for all domain events."""

    source_id: str = ""


# Device/Hardware Events


@dataclass(frozen=True)
class DeviceErrorEvent(Event):
    device_id: str
    error: str


@dataclass(frozen=True)
class SamplesDroppedEvent(Event):
    device_id: str
    count: int


@dataclass(frozen=True)
class JitterBufferUpdateEvent(Event):
    """Per-device jitter-buffer state, published by the I/O worker.

    Only emitted for devices that own a JitterBuffer (rtltcp, spyserver).
    Coalesced source-side so consumers don't see uninformative deltas.
    """

    device_id: str
    target_seconds: float
    fill_seconds: float
    fill_fraction: float  # 0.0–1.0; fill_seconds / target_seconds
    rebuffer_count: int
    rebuffering: bool


# Visualization Events


@dataclass(frozen=True)
class FFTUpdateEvent(Event):
    device_id: str
    spectrum: NDArray[np.float32]
    frequencies: NDArray[np.float32]
    center_frequency: float
    sample_rate: float


@dataclass(frozen=True)
class StatsUpdateEvent(Event):
    device_id: str
    center_frequency: float
    sample_rate: float
    rf_gain: float
    samples_processed: int
    samples_dropped: int
    queue_size: int
    queue_capacity: int
    peak_power: float
    average_power: float
    peak_frequency: float
    peak_bin: int
    noise_floor: float
    dynamic_range: float
    fft_size: int
    fft_window: str
    spectrum_bins: int
    demod_mode: str
    channel_snr: float | None = None
    stereo: bool | None = None
    iq_rms: float | None = None
    iq_peak: float | None = None
    iq_clip_pct: float | None = None
    update_rate_fps: int = 0
    performance_stats: dict[str, float] | None = None


# Pipeline Events


@dataclass(frozen=True)
class PipelineErrorEvent(Event):
    device_id: str
    pipeline_id: str
    stage_name: str
    error: str


# Audio Events


@dataclass(frozen=True)
class AudioOutputErrorEvent(Event):
    error: str


# Configuration Events


@dataclass(frozen=True)
class ConfigChangedEvent(Event):
    """Published when device config is updated.

    Allows UI widgets to refresh even when the device is stopped (no data
    events flowing).
    """

    device_id: str


@dataclass(frozen=True)
class DeviceStateChangedEvent(Event):
    """Emitted when a device transitions between RUNNING and STOPPED."""

    device_id: str
    running: bool


@dataclass(frozen=True)
class DeviceAddedEvent(Event):
    """Emitted when a device is added to the engine's device set."""

    device_id: str


@dataclass(frozen=True)
class DeviceRemovedEvent(Event):
    """Emitted when a device is removed from the engine's device set."""

    device_id: str


@dataclass(frozen=True)
class FocusChangedEvent(Event):
    """Emitted when the engine's focused device changes (including to/from None)."""

    focused_device_id: str | None


@dataclass(frozen=True)
class DeviceCapabilitiesChangedEvent(Event):
    device_id: str
    capabilities: DeviceCapabilities


# AGC Events


@dataclass(frozen=True)
class AGCGainChangeEvent(Event):
    """Client-side AGC requesting RF gain change."""

    device_id: str
    rf_gain: float


# Decoder Events


@dataclass(frozen=True)
class DecodedMessage:
    """A single decoded text message from a protocol decoder."""

    text: str
    timestamp: float
    data: object = None  # Optional typed payload (e.g. RDSData for RDS messages)


@dataclass(frozen=True)
class RecordingFinishedEvent(Event):
    """Published by RecordStage when a duration-mode recording reaches its sample target.

    Subscribed by the `record` command which calls engine.remove_pipeline to tear down
    the recording pipeline. Routing via the event bus keeps thread boundaries clean:
    the stage runs in the pipeline worker, the handler runs on whichever thread the
    bus delivers to (main thread for the command).
    """

    device_id: str
    pipeline_name: str
    path: str
    samples_written: int


@dataclass(frozen=True)
class PipelineChangedEvent(Event):
    """Pipeline added or removed from a device."""

    device_id: str
    pipeline_name: str
    active: bool
    mode: str = ""


@dataclass(frozen=True)
class DecoderOutputEvent(Event):
    """Text output from a protocol decoder."""

    device_id: str
    protocol: str
    messages: tuple[DecodedMessage, ...]


@dataclass(frozen=True)
class ConstellationUpdateEvent(Event):
    """Constellation scatter from a demod for visualization."""

    device_id: str
    points: NDArray[np.complex64]
    modulation: str  # e.g. "BPSK", "QPSK"


@dataclass(frozen=True)
class DemodStatusEvent(Event):
    """Live dynamic status from the active demodulator/decoder.

    Published when the DemodStatus reported by the demodulator changes
    (e.g. decoder updating its description with a newly decoded station name).
    """

    device_id: str
    demod_status: DemodStatus


# Memory Events


@dataclass(frozen=True)
class MemoriesChangedEvent(Event):
    """Emitted when the memory store changes (add/remove).

    Carries a full snapshot so subscribers don't need to query the store.
    """

    memories: tuple[object, ...]  # tuple[Memory, ...] - object to avoid circular import


# Bandplan Events


@dataclass(frozen=True)
class BandplanChangedEvent(Event):
    """Emitted when the active bandplan changes (loaded, cleared, swapped)."""

    bandplan: object | None = None  # Bandplan | None - object to avoid circular import


# Band-stack and tuning-state events


@dataclass(frozen=True)
class BandStackChangedEvent(Event):
    """Emitted when a band-stack register or current_idx changes."""

    band_stack: object | None = None  # BandStackStore - object to avoid circular import


@dataclass(frozen=True)
class TuningStateChangedEvent(Event):
    """Emitted when step, previous-tune-state, or current band key changes."""

    state: object | None = None  # TuningState - object to avoid circular import
