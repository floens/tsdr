from dataclasses import dataclass

import numpy as np

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices.base import DeviceCapabilities, DeviceIdentity, DeviceParams

_MOCK_IDENTITY = DeviceIdentity(type_label="Mock", serial=None)
_MOCK_CAPABILITIES = DeviceCapabilities(
    frequency_range=None,
    sample_rates=None,
    gain_supported=True,
    gain_range=(0.0, 0.0),
    gain_step=1.0,
    gain_unit="dB",
    bias_tee_supported=False,
)


@dataclass(frozen=True)
class MockParams(DeviceParams):
    """Mock device parameters."""

    signal_freq_offset: float = 10e3
    noise_level: float = 0.1


class MockSDRDevice:
    """Mock SDR device that emits a complex sine plus noise.

    Generated signal:
        I(t) = cos(2π * signal_freq * t) + noise
        Q(t) = sin(2π * signal_freq * t) + noise
    """

    def __init__(
        self,
        center_freq: float = 100.0e6,
        sample_rate: float = 2.4e6,
        signal_freq_offset: float = 0.0,
        noise_level: float = 0.1,
    ):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.signal_freq_offset = signal_freq_offset
        self.noise_level = noise_level
        self._sample_index = 0
        self._is_open = False

    def open(self) -> None:
        self._is_open = True
        self._sample_index = 0

    def interrupt(self) -> None:
        pass

    def close(self) -> None:
        self._is_open = False

    def read_samples(self, count: int) -> bytes:
        if not self._is_open:
            raise DeviceError("Device not open")

        # complex64 = 8 bytes per sample
        num_samples = count // 8

        t = (self._sample_index + np.arange(num_samples)) / self.sample_rate
        self._sample_index += num_samples

        omega = 2.0 * np.pi * self.signal_freq_offset
        signal = np.exp(1j * omega * t).astype(np.complex64)

        if self.noise_level > 0:
            noise_i = np.random.normal(0, self.noise_level, num_samples)
            noise_q = np.random.normal(0, self.noise_level, num_samples)
            noise = (noise_i + 1j * noise_q).astype(np.complex64)
            signal += noise

        result: bytes = signal.tobytes()  # type: ignore[assignment]
        return result

    def set_frequency(self, freq: float) -> None:
        self.center_freq = freq

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = rate
        self._sample_index = 0  # reset phase

    @property
    def actual_sample_rate(self) -> float:
        return self.sample_rate

    def set_gain(self, gain: float) -> None:
        pass

    def set_auto_gain(self, enable: bool) -> None:
        pass

    def set_bias_tee(self, enable: bool) -> None:
        pass

    @property
    def identity(self) -> DeviceIdentity:
        return _MOCK_IDENTITY

    @property
    def capabilities(self) -> DeviceCapabilities:
        return _MOCK_CAPABILITIES

    def set_network_buffer_seconds(self, seconds: float) -> None:
        pass

    def get_sample_format(self) -> SampleFormat:
        return SampleFormat.COMPLEX64

    def __str__(self) -> str:
        status = "open" if self._is_open else "closed"
        return (
            f"MockSDRDevice(center={self.center_freq / 1e6:.1f}MHz, "
            f"rate={self.sample_rate / 1e6:.1f}MHz, {status})"
        )
