"""SDR configuration management.

Two config layers:
- SDRConfig: Engine-global settings (display, processing, audio)
- DeviceConfig: Per-device settings (hardware, queues, pipelines)
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Literal, TypedDict, Unpack, get_args

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.samples_batch import SampleFormat

TuningMode = Literal["center", "free"]
FFTWindow = Literal["hanning", "hamming", "blackman", "bartlett", "rectangular"]

# Floor for the spectrum view span, sized so the deepest server-provided
# frame (~1.8 kHz span) stretches at most 2x; for IQ devices it is purely a
# display crop.
MIN_SPECTRUM_SPAN_HZ = 1_000.0


class StageType(Enum):
    """Pipeline stage types."""

    AGC = "agc"
    FFT = "fft"
    EVENT_EMITTER = "event_emitter"
    DEMODULATOR = "demodulator"
    DENOISER = "denoiser"
    FREQUENCY_SHIFT = "frequency_shift"
    RECORD = "record"


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline composition descriptor."""

    stages: tuple[StageType, ...]
    audio_spec: DemodSpec | None = None
    # RecordStage config: output path, optional (up, down) rational resample,
    # optional max sample count at which the stage self-terminates.
    record_path: str | None = None
    record_resample: tuple[int, int] | None = None
    record_max_samples: int | None = None
    record_sample_format: SampleFormat = SampleFormat.UINT8_IQ
    squelch_enabled: bool = False
    squelch_threshold_db: float = -50.0
    squelch_hang_ms: float = 100.0


DEFAULT_PIPELINES: MappingProxyType[str, PipelineConfig] = MappingProxyType(
    {
        "visualization": PipelineConfig(
            stages=(StageType.AGC, StageType.FFT, StageType.EVENT_EMITTER)
        ),
    }
)


class DeviceConfigChanges(TypedDict, total=False):
    """Optional fields for device config updates."""

    tuned_frequency: float
    center_frequency: float
    tuning_mode: TuningMode
    sample_rate: float
    rf_gain: float
    auto_gain: bool
    enable_agc: bool
    bias_tee: bool
    buffer_samples: int | None
    target_fps: float
    queue_size: int
    channel_bandwidth: float | None
    fft_size: int
    fft_window: FFTWindow
    spectrum_center: float | None
    spectrum_span: float | None
    calculate_constellation: bool
    network_buffer_seconds: float
    pipelines: MappingProxyType[str, PipelineConfig]


class GlobalConfigChanges(TypedDict, total=False):
    """Optional fields for global config updates."""

    update_rate_fps: int
    spectrum_averaging: int
    dc_offset_correction: bool
    iq_imbalance_correction: bool
    audio_volume: float
    denoise: bool


@dataclass(frozen=True)
class DeviceConfig:
    """Immutable per-device configuration.

    Hardware and device-specific parameters. Each SDRDeviceContext
    owns its own DeviceConfig instance.
    """

    tuned_frequency: float = 100.0e6  # Hz, the dial / user intent
    center_frequency: float = 100.0e6  # Hz, hardware capture center (derived from the dial)
    # center: hardware follows every dial move; free: DSP offset until the band edge
    tuning_mode: TuningMode = "center"
    sample_rate: float = 2.4e6  # Samples/sec
    rf_gain: float = 30.0  # dB
    auto_gain: bool = False  # Disable AGC for consistent gain control
    enable_agc: bool = False  # Client-side RF gain AGC
    bias_tee: bool = False  # Antenna-port bias-T power (driver-dependent)
    buffer_samples: int | None = None  # Samples per read, None = auto from sample_rate/target_fps
    target_fps: float = 20.0  # Target UI update rate for auto buffer sizing
    queue_size: int = 15  # Max batches in sample queue
    channel_bandwidth: float | None = None  # Hz, None = demodulator default
    fft_size: int = 2**15  # Spectrum FFT window size (power of 2)
    fft_window: FFTWindow = "hanning"
    spectrum_center: float | None = (
        None  # View pan in Hz (free tuning mode only); None tracks the dial
    )
    spectrum_span: float | None = None  # View span in Hz; None = full band
    calculate_constellation: bool = False  # Enable constellation point collection from decoders
    # Pre-fill watermark for network-source jitter buffers (rtltcp, spyserver).
    # No-op for USB/file/mock devices. Lower → less latency, less jitter
    # tolerance; higher → more latency, more tolerance.
    network_buffer_seconds: float = 0.5
    pipelines: MappingProxyType[str, PipelineConfig] = field(default=DEFAULT_PIPELINES)

    def buffer_samples_for(self, delivered_rate: float) -> int:
        """Samples per read to hit `target_fps` at the device's delivered rate.

        Uses `delivered_rate` (the device's `actual_sample_rate`) rather than
        the configured `sample_rate` so reads stay sized to real throughput
        even before the engine has snapped `sample_rate` to a device-supported
        value. Without this a device that delivers far below the configured
        rate (e.g. a 12 kHz network channel against the 2.4 MHz default) would size its
        first read past its jitter-buffer capacity. Falls back to the
        configured rate when the device has not reported one yet (rate <= 0).
        """
        if self.buffer_samples is not None:
            return self.buffer_samples
        rate = delivered_rate if delivered_rate > 0 else self.sample_rate
        return int(rate / self.target_fps)

    @property
    def effective_buffer_samples(self) -> int:
        """Sample count per device read at the configured sample rate."""
        return self.buffer_samples_for(self.sample_rate)

    def with_changes(self, **kwargs: Unpack[DeviceConfigChanges]) -> DeviceConfig:
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.center_frequency <= 0:
            raise ValueError(f"center_frequency must be positive, got {self.center_frequency}")
        if self.tuned_frequency <= 0:
            raise ValueError(f"tuned_frequency must be positive, got {self.tuned_frequency}")
        if self.tuning_mode not in get_args(TuningMode):
            raise ValueError(f"tuning_mode must be 'center' or 'free', got {self.tuning_mode}")
        if self.queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {self.queue_size}")
        if self.buffer_samples is not None and self.buffer_samples <= 0:
            raise ValueError(f"buffer_samples must be positive, got {self.buffer_samples}")
        if self.target_fps <= 0 or self.target_fps > 120:
            raise ValueError(f"target_fps must be between 1 and 120, got {self.target_fps}")
        if not 0.05 <= self.network_buffer_seconds <= 5.0:
            raise ValueError(
                f"network_buffer_seconds must be 0.05–5.0, got {self.network_buffer_seconds}"
            )
        if self.fft_size & (self.fft_size - 1) != 0:
            raise ValueError(f"fft_size must be power of 2, got {self.fft_size}")
        if not 64 <= self.fft_size <= 65536:
            raise ValueError(f"fft_size must be between 64 and 65536, got {self.fft_size}")
        valid_windows = get_args(FFTWindow)
        if self.fft_window not in valid_windows:
            raise ValueError(f"fft_window must be one of {valid_windows}, got {self.fft_window}")
        if self.spectrum_span is not None and self.spectrum_span <= 0:
            raise ValueError(f"spectrum_span must be positive, got {self.spectrum_span}")
        if self.spectrum_center is not None and self.spectrum_center <= 0:
            raise ValueError(f"spectrum_center must be positive, got {self.spectrum_center}")


@dataclass(frozen=True)
class SDRConfig:
    """Immutable engine-global configuration.

    Processing, display, and audio parameters shared across all devices.
    The SDREngine owns one SDRConfig instance.
    """

    audio_volume: float = 0.8  # 0.0 to 1.0
    update_rate_fps: int = 20  # UI updates per second
    spectrum_averaging: int = 3  # Number of FFTs to average
    dc_offset_correction: bool = True  # Remove DC offset
    iq_imbalance_correction: bool = True  # Correct IQ imbalance
    denoise: bool = False

    def with_changes(self, **kwargs: Unpack[GlobalConfigChanges]) -> SDRConfig:
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.update_rate_fps <= 0:
            raise ValueError(f"update_rate_fps must be positive, got {self.update_rate_fps}")
