from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices._jitter_buffer import JitterBuffer

GainUnit = Literal["dB", "index"]


@dataclass(frozen=True)
class DeviceParams:
    """Base class for device-specific parameters."""

    pass


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
    sample_rates: tuple[float, ...] | None

    gain_supported: bool
    gain_range: tuple[float, float]
    gain_step: float
    gain_unit: GainUnit

    bias_tee_supported: bool


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


@runtime_checkable
class HasJitterBuffer(Protocol):
    """Optional capability: device exposes its jitter buffer for state reads.

    Implemented by network-transport devices. The I/O worker checks
    `isinstance(device, HasJitterBuffer)` to decide whether to publish
    JitterBufferUpdateEvent.
    """

    jitter: JitterBuffer
