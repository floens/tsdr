from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AudioBatch:
    """Batch of audio samples for output.

    Immutable container for audio samples passed from demodulator
    to audio output thread.

    Attributes:
        samples: Float audio samples (-1.0 to 1.0)
        sample_rate: Sample rate in Hz (e.g., 48000)
        timestamp: Capture timestamp from original IQ data
    """

    samples: np.ndarray  # Float audio samples (-1.0 to 1.0)
    sample_rate: float  # Sample rate (e.g., 48000)
    timestamp: float  # Capture timestamp
    stereo: bool = False  # Whether audio is stereo
    prebuffer_seconds: float = 0.15  # Desired prebuffer before playback starts

    @property
    def duration_seconds(self) -> float:
        """Duration of audio batch in seconds."""
        if self.sample_rate > 0:
            return len(self.samples) / self.sample_rate
        return 0.0


@dataclass(frozen=True)
class SignalInfo:
    """Describes the signal a demodulator or decoder is processing.

    Attributes:
        label: Human-readable name (e.g. "Wideband FM", "DAB+ Mode I")
        channel_bandwidth: Channel bandwidth in Hz
        modulation: Modulation type (e.g. "FM", "OFDM-DQPSK")
        sample_rate: Required input sample rate in Hz, or None if any rate works
    """

    label: str
    channel_bandwidth: float
    modulation: str
    sample_rate: float | None = None
    has_audio: bool = False
    has_text: bool = False
    message_type: str | None = None  # None, "text", "rds" - UI picks widget
    quality_label: str | None = None  # e.g. "50% CRC", "98% BER"
    quality: float | None = None  # 0.0 (worst) to 1.0 (best)
    description: str | None = None  # Protocol-specific identifier for display
    squelch_open: bool | None = None  # None = no squelch, True/False = gate state
    squelch_threshold_db: float | None = None  # Threshold (dBFS) when squelch is configured
    # SSB-style asymmetric filter: channel passband sits entirely on one side
    # of the carrier ("upper" → [center, center+bw], "lower" → [center-bw, center]).
    # None means symmetric around the carrier.
    sideband: Literal["upper", "lower"] | None = None


@dataclass(frozen=True)
class SignalStatistics:
    """Signal-level statistics computed from spectrum.

    Attributes:
        peak_power: Maximum power (dB)
        average_power: Mean power (dB)
        peak_frequency: Frequency of peak power (Hz)
        peak_bin: Bin index of peak power
        noise_floor: Estimated noise floor (dB)
        dynamic_range: Dynamic range (peak - noise floor, dB)
        channel_snr: Channel SNR in dB (mean power in channel vs noise outside)
    """

    peak_power: float
    average_power: float
    peak_frequency: float
    peak_bin: int
    noise_floor: float
    dynamic_range: float
    channel_snr: float | None = None
