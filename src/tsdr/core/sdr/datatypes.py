from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AudioBatch:
    """Batch of audio samples for output.

    Immutable container for audio samples passed from demodulator
    to audio output thread.
    """

    samples: np.ndarray  # Float audio samples (-1.0 to 1.0)
    sample_rate: float  # Sample rate (e.g., 48000)
    stereo: bool = False  # Whether audio is stereo
    prebuffer_seconds: float = 0.15  # Desired prebuffer before playback starts

    @property
    def duration_seconds(self) -> float:
        """Duration of audio batch in seconds."""
        if self.sample_rate > 0:
            return len(self.samples) / self.sample_rate
        return 0.0


@dataclass(frozen=True)
class DemodProfile:
    """Structural (desired-state) description of a demodulator/decoder.

    Attributes:
        label: Human-readable name (e.g. "Wideband FM", "DAB+ Mode I")
        modulation: Modulation type (e.g. "FM", "OFDM-DQPSK")
        channel_bandwidth: Channel bandwidth in Hz
        sample_rate: Required input sample rate in Hz, or None if any rate works
        message_type: None, "text", "rds", … — UI picks the decoder widget
        sideband: SSB-style asymmetric filter — channel passband sits entirely
            on one side of the carrier ("upper" → [center, center+bw], "lower" →
            [center-bw, center]). None means symmetric around the carrier.
    """

    label: str
    modulation: str
    channel_bandwidth: float
    sample_rate: float | None = None
    has_audio: bool = False
    has_text: bool = False
    message_type: str | None = None
    sideband: Literal["upper", "lower"] | None = None


@dataclass(frozen=True)
class DemodStatus:
    """Dynamic (actual-state) status of a running demodulator/decoder.

    Attributes:
        quality_label: e.g. "Pilot 12 dB", "CRC 98%"
        quality: 0.0 (worst) to 1.0 (best)
        description: protocol-specific identifier / refinement for display
            (RDS station name, detected SSTV submode, DMR colour code, …)
        squelch_open: None = no squelch, True/False = gate state
        squelch_threshold_db: threshold (dBFS) when squelch is configured
    """

    quality_label: str | None = None
    quality: float | None = None
    description: str | None = None
    squelch_open: bool | None = None
    squelch_threshold_db: float | None = None


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
