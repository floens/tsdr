"""SDR configuration management.

Two config layers:
- SDRConfig: Engine-global settings (display, processing, audio)
- DeviceConfig: Per-device settings (hardware, queues, pipelines)
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import TypedDict, Unpack


class StageType(Enum):
    """Pipeline stage types."""

    AGC = "agc"
    FFT = "fft"
    EVENT_EMITTER = "event_emitter"
    DEMODULATOR = "demodulator"
    FREQUENCY_SHIFT = "frequency_shift"
    RECORD = "record"


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline composition descriptor."""

    stages: tuple[StageType, ...]
    demod_mode: str | None = None
    frequency_offset: float = 0.0
    # RecordStage config: output path, optional (up, down) rational resample,
    # optional max sample count at which the stage self-terminates.
    record_path: str | None = None
    record_resample: tuple[int, int] | None = None
    record_max_samples: int | None = None
    squelch_enabled: bool = False
    squelch_threshold_db: float = -50.0
    squelch_hang_ms: float = 100.0
    fm_deviation_hz: float | None = None


DEFAULT_PIPELINES: MappingProxyType[str, PipelineConfig] = MappingProxyType(
    {
        "visualization": PipelineConfig(
            stages=(StageType.AGC, StageType.FFT, StageType.EVENT_EMITTER)
        ),
    }
)


class DeviceConfigChanges(TypedDict, total=False):
    """Optional fields for device config updates."""

    center_frequency: float
    sample_rate: float
    rf_gain: float
    auto_gain: bool
    enable_agc: bool
    bias_tee: bool
    buffer_samples: int | None
    target_fps: float
    queue_size: int
    channel_bandwidth: float | None
    calculate_constellation: bool
    pipelines: MappingProxyType[str, PipelineConfig]


class GlobalConfigChanges(TypedDict, total=False):
    """Optional fields for global config updates."""

    fft_size: int
    fft_window: str
    update_rate_fps: int
    spectrum_averaging: int
    dc_offset_correction: bool
    iq_imbalance_correction: bool
    audio_volume: float


@dataclass(frozen=True)
class DeviceConfig:
    """Immutable per-device configuration.

    Hardware and device-specific parameters. Each SDRDeviceContext
    owns its own DeviceConfig instance.
    """

    center_frequency: float = 100.0e6  # Hz
    sample_rate: float = 2.4e6  # Samples/sec
    rf_gain: float = 30.0  # dB
    auto_gain: bool = False  # Disable AGC for consistent gain control
    enable_agc: bool = False  # Client-side RF gain AGC
    bias_tee: bool = False  # Antenna-port bias-T power (driver-dependent)
    buffer_samples: int | None = None  # Samples per read, None = auto from sample_rate/target_fps
    target_fps: float = 20.0  # Target UI update rate for auto buffer sizing
    queue_size: int = 15  # Max batches in sample queue
    channel_bandwidth: float | None = None  # Hz, None = demodulator default
    calculate_constellation: bool = False  # Enable constellation point collection from decoders
    pipelines: MappingProxyType[str, PipelineConfig] = field(default=DEFAULT_PIPELINES)

    @property
    def effective_buffer_samples(self) -> int:
        """Sample count per device read, auto-calculated if not explicitly set."""
        if self.buffer_samples is not None:
            return self.buffer_samples
        return int(self.sample_rate / self.target_fps)

    def with_changes(self, **kwargs: Unpack[DeviceConfigChanges]) -> DeviceConfig:
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.center_frequency <= 0:
            raise ValueError(f"center_frequency must be positive, got {self.center_frequency}")
        if self.queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {self.queue_size}")
        if self.buffer_samples is not None and self.buffer_samples <= 0:
            raise ValueError(f"buffer_samples must be positive, got {self.buffer_samples}")
        if self.target_fps <= 0 or self.target_fps > 120:
            raise ValueError(f"target_fps must be between 1 and 120, got {self.target_fps}")


@dataclass(frozen=True)
class SDRConfig:
    """Immutable engine-global configuration.

    Processing, display, and audio parameters shared across all devices.
    The SDREngine owns one SDRConfig instance.
    """

    fft_size: int = 2**15  # FFT window size (power of 2)
    fft_window: str = "hanning"  # "hanning", "hamming", "blackman"
    audio_volume: float = 0.8  # 0.0 to 1.0
    update_rate_fps: int = 20  # UI updates per second
    spectrum_averaging: int = 3  # Number of FFTs to average
    dc_offset_correction: bool = True  # Remove DC offset
    iq_imbalance_correction: bool = True  # Correct IQ imbalance

    def with_changes(self, **kwargs: Unpack[GlobalConfigChanges]) -> SDRConfig:
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.fft_size & (self.fft_size - 1) != 0:
            raise ValueError(f"fft_size must be power of 2, got {self.fft_size}")
        if self.fft_size < 64 or self.fft_size > 65536:
            raise ValueError(f"fft_size must be between 64 and 65536, got {self.fft_size}")
        if self.update_rate_fps <= 0:
            raise ValueError(f"update_rate_fps must be positive, got {self.update_rate_fps}")
        valid_windows = ["hanning", "hamming", "blackman", "bartlett", "rectangular"]
        if self.fft_window not in valid_windows:
            raise ValueError(f"fft_window must be one of {valid_windows}, got {self.fft_window}")
