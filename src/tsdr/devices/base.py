from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices._jitter_buffer import JitterBuffer


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

    @property
    def frequency_range(self) -> tuple[float, float] | None:
        """Tunable frequency range in Hz as (min, max).

        Returns None if the device has no enforceable range (e.g. file
        playback, mock). The engine validates `set_frequency` arguments
        against this; the tuner widget clamps adjustments to it.
        """
        ...

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

    @property
    def gain_range(self) -> tuple[float, float]:
        """RF gain range in dB as (min, max).

        Used by client-side AGC to clamp step adjustments. Devices without
        controllable gain (file playback, mock) should return (0.0, 0.0).
        """
        ...

    def get_sample_format(self) -> SampleFormat: ...

    @property
    def supports_bias_tee(self) -> bool:
        """Whether this driver can drive an antenna-port bias-T.

        Default is False. Drivers that support bias-T control override
        this to return True (optionally after probing the underlying
        device at open time).
        """
        ...

    @property
    def supports_gain_control(self) -> bool:
        """Whether this client may set RF gain on the device.

        Most devices return True. SpyServer reports False when the server
        is configured to deny SET_SETTING from this client (read-only or
        another client holds control). UI grays the gain readout and
        commands refuse `--gain`/`--agc` when False.
        """
        ...

    def set_bias_tee(self, enable: bool) -> None:
        """Enable or disable the antenna-port bias-T power supply.

        No-op on drivers that don't override it. Callers should check
        `supports_bias_tee` before calling. Bias-T state is write-only
        on RTL-SDR hardware; there is no read-back.
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
