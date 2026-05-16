from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from tsdr.radio.dsp._kernels import _sint8_iq_to_complex64, _uint8_iq_to_complex64

if TYPE_CHECKING:
    from tsdr.core.events.events import DecodedMessage
    from tsdr.core.sdr.datatypes import SignalInfo


class SampleFormat(Enum):
    """IQ sample data format.

    Defines how raw bytes should be interpreted as IQ samples.
    """

    UINT8_IQ = "uint8_iq"  # 2 bytes/sample: [I, Q] as uint8 (0..255)
    SINT8_IQ = "sint8_iq"  # 2 bytes/sample: [I, Q] as int8 (-128..127)
    COMPLEX64 = "complex64"  # 8 bytes/sample: complex64 (float32+float32)

    @property
    def bytes_per_sample(self) -> int:
        """Bytes per IQ sample for this wire format."""
        if self is SampleFormat.COMPLEX64:
            return 8
        return 2


@dataclass(frozen=True)
class SamplesBatch:
    """Batch of samples flowing through the system.

    This is the single data container used across all subsystems.
    Different subsystems populate different fields:
    - I/O worker: raw_samples, sample_format, RF context
    - Pipeline worker: iq_samples (converted from raw_samples)
    - FFT stage: spectrum
    - Demodulator stage: audio_samples, decoded_messages

    RF context (center_frequency, sample_rate, rf_gain) flows with the data
    so stages can compute frequency axes and statistics. FrequencyShiftStage
    modifies center_frequency to reflect the effective tuning.
    """

    # Raw input (populated by I/O worker, None after IQ conversion)
    raw_samples: bytes | None = None
    sample_format: SampleFormat | None = None

    # Processed data (populated by different stages)
    iq_samples: np.ndarray | None = None
    audio_samples: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    constellation_points: np.ndarray | None = None
    constellation_modulation: str = ""  # e.g. "BPSK", "QPSK"
    decoded_messages: tuple[DecodedMessage, ...] = field(default_factory=tuple)
    signal_info: SignalInfo | None = None

    # RF context at time of capture
    center_frequency: float = 0.0
    sample_rate: float = 0.0
    rf_gain: float = 0.0

    # Timing
    timestamp: float = 0.0

    # Processing hints
    stage_name: str = "unknown"

    @property
    def sample_count(self) -> int:
        """Number of IQ samples in this batch."""
        if self.iq_samples is not None:
            return len(self.iq_samples)
        if self.raw_samples is not None and self.sample_format is not None:
            return len(self.raw_samples) // self.sample_format.bytes_per_sample
        return 0

    @property
    def duration_seconds(self) -> float:
        """Duration of this batch in seconds."""
        if self.sample_rate == 0:
            return 0.0
        return self.sample_count / self.sample_rate

    def to_iq_array(self) -> np.ndarray:
        """Convert raw bytes to complex64 array.

        Returns:
            Complex numpy array (dtype=complex64)
        """
        if self.iq_samples is not None:
            return self.iq_samples

        if self.raw_samples is None:
            raise ValueError("No sample data available")

        if self.sample_format == SampleFormat.COMPLEX64:
            return np.frombuffer(self.raw_samples, dtype=np.complex64)

        elif self.sample_format == SampleFormat.UINT8_IQ:
            uint8_data = np.frombuffer(self.raw_samples, dtype=np.uint8)
            if len(uint8_data) % 2 != 0:
                raise ValueError(
                    f"Sample count must be even for IQ pairs, got {len(uint8_data)} bytes"
                )
            result: np.ndarray = _uint8_iq_to_complex64(uint8_data)
            return result

        elif self.sample_format == SampleFormat.SINT8_IQ:
            int8_data = np.frombuffer(self.raw_samples, dtype=np.int8)
            if len(int8_data) % 2 != 0:
                raise ValueError(
                    f"Sample count must be even for IQ pairs, got {len(int8_data)} bytes"
                )
            result = _sint8_iq_to_complex64(int8_data)
            return result

        else:
            raise ValueError(f"Unknown sample format: {self.sample_format}")

    def with_changes(self, **kwargs) -> SamplesBatch:
        """Create new SamplesBatch with specified changes."""
        return replace(self, **kwargs)

    def __str__(self) -> str:
        parts = []
        if self.raw_samples is not None:
            parts.append(f"raw={len(self.raw_samples)}b")
        if self.iq_samples is not None:
            parts.append(f"iq={len(self.iq_samples)}")
        if self.audio_samples is not None:
            parts.append(f"audio={len(self.audio_samples)}")
        if self.spectrum is not None:
            parts.append(f"spectrum={len(self.spectrum)}")
        if self.frequencies is not None:
            parts.append(f"freqs={len(self.frequencies)}")
        if self.decoded_messages:
            parts.append(f"messages={len(self.decoded_messages)}")

        data_str = ", ".join(parts) if parts else "no data"

        return f"SamplesBatch(stage={self.stage_name}, {data_str})"

    def __repr__(self) -> str:
        return self.__str__()
