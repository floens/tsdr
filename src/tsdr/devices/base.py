from dataclasses import dataclass
from typing import Protocol

from tsdr.core.sdr.samples_batch import SampleFormat


@dataclass(frozen=True)
class DeviceParams:
    """Base class for device-specific parameters."""

    pass


class SDRDevice(Protocol):
    """Protocol implemented by every SDR device driver."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def read_samples(self, count: int) -> bytes:
        """Read up to `count` bytes of raw IQ samples (format per driver)."""
        ...

    def set_frequency(self, freq: float) -> None: ...

    def set_sample_rate(self, rate: float) -> None: ...

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

    def set_bias_tee(self, enable: bool) -> None:
        """Enable or disable the antenna-port bias-T power supply.

        No-op on drivers that don't override it. Callers should check
        `supports_bias_tee` before calling. Bias-T state is write-only
        on RTL-SDR hardware; there is no read-back.
        """
        ...
