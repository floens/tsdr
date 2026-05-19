from abc import ABC, abstractmethod

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, SignalInfo


class Demodulator(ABC):
    """Base class for all demodulators and decoders."""

    stereo_detected: bool = False

    def __init__(self) -> None:
        self._audio_batches: list[AudioBatch] = []

    @abstractmethod
    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None: ...

    @abstractmethod
    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread.

        Called from the UI thread via `SDRDeviceContext.active_demod_info`
        while the pipeline worker is concurrently mutating the demodulator's
        state. Implementations must not iterate mutable containers
        (list/dict/deque/set) without synchronization. Prefer cached scalar
        fields updated by the worker on writes.
        """
        ...

    def reset(self) -> None:
        self._audio_batches.clear()

    def set_channel_bandwidth(self, bandwidth: float) -> None:  # noqa: B027
        pass

    def set_sample_rate(self, rate: float) -> None:  # noqa: B027
        """Rebuild rate-dependent filter taps and helpers.

        Default no-op for protocol decoders that expect a spec-locked
        rate (DAB=2.048M, ADSB=2.4M, …) where a live switch is a user
        error.
        """
        pass

    def set_deviation(self, deviation: float) -> None:  # noqa: B027
        """Update FM peak deviation. NFM-only; default no-op elsewhere."""
        pass

    def set_squelch(  # noqa: B027
        self, enabled: bool, threshold_db: float, hang_ms: float
    ) -> None:
        """Configure the audio squelch gate. Default no-op for modes that don't support it."""
        pass

    def set_tuned_frequency(self, frequency_hz: int) -> None:  # noqa: B027
        """Notify the demodulator of the current tuned center frequency.

        Optional hook for decoders (e.g. TETRA) that need to compare a
        network-announced downlink frequency against the currently tuned
        frequency to determine whether the user is on an MCCH or a TCH.
        Default implementation is a no-op.
        """
        pass

    @property
    def audio_prebuffer_seconds(self) -> float:
        """Minimum prebuffer duration in seconds before audio playback starts."""
        return 0.15

    def _emit_audio(
        self,
        samples: np.ndarray,
        sample_rate: float,
        timestamp: float,
        *,
        stereo: bool = False,
    ) -> None:
        """Buffer one AudioBatch for later retrieval via `get_audio()`."""
        self._audio_batches.append(
            AudioBatch(
                samples=samples,
                sample_rate=sample_rate,
                timestamp=timestamp,
                stereo=stereo,
                prebuffer_seconds=self.audio_prebuffer_seconds,
            )
        )

    def get_audio(self) -> list[AudioBatch]:
        batches = self._audio_batches
        self._audio_batches = []
        return batches

    def get_messages(self) -> list[DecodedMessage]:
        return []

    def get_constellation(self) -> tuple[np.ndarray, str] | None:
        """Return (points, modulation_type) or None."""
        return None


__all__ = [
    "Demodulator",
    "SignalInfo",
]
