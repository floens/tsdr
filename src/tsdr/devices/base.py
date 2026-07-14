from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices._jitter_buffer import JitterBuffer

GainUnit = Literal["dB", "index"]


@dataclass(frozen=True)
class DeviceParams:
    """Base class for device-specific parameters."""

    def describe(self) -> str:
        return ""


@runtime_checkable
class NetworkDeviceParams(Protocol):
    """Optional capability: device params carry a remote `host:port` endpoint.

    rtltcp and spyserver params implement this implicitly via their
    `host: str` / `port: int` fields. Used to switch UI/commands on
    network-source devices without enumerating every concrete type.
    """

    host: str
    port: int


@dataclass(frozen=True)
class DeviceIdentity:
    type_label: str
    serial: str | None = None


@dataclass(frozen=True)
class DeviceCapabilities:
    frequency_range: tuple[float, float] | None
    # True when set_frequency can move the capture within `frequency_range`
    # (a shared SpyServer still tunes its own IQ sub-window inside the
    # controller's band). False = the capture is fixed (e.g. IQ file).
    frequency_controllable: bool
    sample_rates: tuple[float, ...] | None

    gain_supported: bool
    gain_range: tuple[float, float]
    gain_step: float
    gain_unit: GainUnit

    bias_tee_supported: bool

    # Device supplies pre-computed spectrum frames (SpectrumSource); the
    # engine-side IQ FFT is suppressed and widgets take the displayable band
    # from `frequency_range` instead of `center_frequency ± sample_rate/2`.
    provides_spectrum: bool = False

    # Populated when another client controls the hardware (shared SpyServer).
    controller_center_frequency: float | None = None
    controller_gain: int | None = None


class SDRDevice(Protocol):
    """Protocol implemented by every SDR device driver."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def interrupt(self) -> None:
        """Unblock any in-flight read_samples() without freeing resources.

        Safe to call from any thread, including while another thread is
        parked in read_samples(). Used by the stop sequence to break a
        worker out of a blocking network read so its lifecycle flag can
        take effect. Network-transport devices shutdown the socket;
        USB/file/mock devices are no-ops (their reads return on their own).

        Resource cleanup happens in close(), which is called only from
        the I/O worker's teardown.
        """
        ...

    def read_samples(self, count: int) -> bytes:
        """Read up to `count` bytes of raw IQ samples (format per driver)."""
        ...

    def set_frequency(self, freq: float) -> None: ...

    def set_sample_rate(self, rate: float) -> None: ...

    @property
    def actual_sample_rate(self) -> float:
        """Sample rate the device is actually delivering.

        Devices with discrete supported rates (e.g. RTL-SDR, SpyServer's
        decimation) may deliver a different rate than the one requested
        via `set_sample_rate`. The pipeline stamps `SamplesBatch.sample_rate`
        with this value so downstream stages compute frequency axes against
        what they actually received.
        """
        ...

    def set_gain(self, gain: float) -> None: ...

    def set_auto_gain(self, enable: bool) -> None: ...

    def get_sample_format(self) -> SampleFormat: ...

    @property
    def identity(self) -> DeviceIdentity: ...

    @property
    def capabilities(self) -> DeviceCapabilities: ...

    def set_bias_tee(self, enable: bool) -> None:
        """Enable or disable the antenna-port bias-T power supply.

        No-op on drivers that don't override it. Callers should check
        `capabilities.bias_tee_supported` before calling. Bias-T state is
        write-only on RTL-SDR hardware; there is no read-back.
        """
        ...

    def set_network_buffer_seconds(self, seconds: float) -> None:
        """Adjust the network jitter buffer pre-fill watermark.

        Implemented by network-transport devices (rtltcp, spyserver) to
        decouple bursty TCP arrivals from steady downstream consumption.
        No-op for USB/file/mock devices.
        """
        ...


@dataclass(frozen=True)
class SpectrumFrame:
    """One pre-computed spectrum line from a SpectrumSource device."""

    db_bins: NDArray[np.float32]  # dBm
    center_hz: float  # absolute (freq_offset included)
    span_hz: float
    seq: int


@dataclass(frozen=True)
class SpectrumViewStatus:
    """Debug snapshot of a SpectrumSource's view negotiation: what was last
    requested from the device/server vs. what the frames actually deliver."""

    requested_zoom: int
    requested_center_hz: float
    zoom_cap: int
    frame_zoom: int | None = None
    frame_center_hz: float | None = None
    frame_span_hz: float | None = None
    frame_bins: int | None = None
    # Frame rate: what the server advertises vs. what actually arrives
    # (shared servers timeslice and deliver well below nominal).
    expected_fps: float | None = None
    measured_fps: float | None = None


@runtime_checkable
class SpectrumSource(Protocol):
    """Optional capability: device supplies pre-computed spectrum frames.

    Pairs with `DeviceCapabilities.provides_spectrum`. The I/O worker drains
    frames into FFTUpdateEvents and forwards view changes to the device.
    """

    @property
    def capabilities(self) -> DeviceCapabilities: ...

    def drain_spectrum_frames(self) -> list[SpectrumFrame]:
        """All frames received since the last drain (non-blocking)."""
        ...

    def set_spectrum_view(self, center_hz: float, span_hz: float) -> None:
        """Request frames covering at least [center - span/2, center + span/2]."""
        ...

    def spectrum_view_status(self) -> SpectrumViewStatus | None:
        """Last requested view + last delivered frame geometry, for stats/debug."""
        ...


def as_spectrum_source(device: object) -> SpectrumSource | None:
    """The device as an *active* SpectrumSource, else None.

    Implementing the protocol is not enough: a device without a spectrum
    channel in its current build
    (wf_chans=0) has the methods but never delivers frames, and reports
    `provides_spectrum=False`. Both predicates must agree before callers
    drain frames, push views, or show view status.
    """
    if isinstance(device, SpectrumSource) and device.capabilities.provides_spectrum:
        return device
    return None


@runtime_checkable
class HasJitterBuffer(Protocol):
    """Optional capability: device exposes its jitter buffer for state reads.

    Implemented by network-transport devices. The I/O worker checks
    `isinstance(device, HasJitterBuffer)` to decide whether to publish
    JitterBufferUpdateEvent.
    """

    jitter: JitterBuffer

    @property
    def wire_bytes_per_sec(self) -> float:
        """Estimated bytes/sec on the wire; 0 before streaming starts."""
        ...
