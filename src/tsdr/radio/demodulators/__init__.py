from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, DemodProfile, DemodStatus

# Fraction of Nyquist below which a post-decimation channel filter's cutoff
# must sit to keep the FIR transition band inside the passband.
NYQUIST_MARGIN = 0.95


class Demodulator(ABC):
    """Base class for all demodulators and decoders."""

    stereo_detected: bool = False
    DEFAULT_CHANNEL_BANDWIDTH: ClassVar[float] = 12_500
    # Post-decimation demods override with their decimated audio Nyquist cap;
    # the inf default applies to WFM (filters at native sample rate) and to
    # protocol decoders that ignore channel_bandwidth.
    MAX_CHANNEL_BANDWIDTH: ClassVar[float] = float("inf")

    LABEL: ClassVar[str] = ""
    MODULATION: ClassVar[str] = ""
    MESSAGE_TYPE: ClassVar[str | None] = None
    HAS_AUDIO: ClassVar[bool] = False
    HAS_TEXT: ClassVar[bool] = False
    SIDEBAND: ClassVar[Literal["upper", "lower"] | None] = None
    FIXED_CHANNEL_BANDWIDTH: ClassVar[float | None] = None
    FIXED_SAMPLE_RATE: ClassVar[float | None] = None

    def __init__(self) -> None:
        self._audio_batches: list[AudioBatch] = []

    @abstractmethod
    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None: ...

    @classmethod
    def profile(cls, *, mode: str, channel_bandwidth: float | None) -> DemodProfile:
        """Structural (desired-state) description for `mode` — no instance needed.

        The single source of the demod's fixed fields; `status()` carries the
        runtime ones. Derivable synchronously from (mode, config), so the UI can
        read it the instant config changes.
        """
        if cls.FIXED_CHANNEL_BANDWIDTH is not None:
            bw = cls.FIXED_CHANNEL_BANDWIDTH
        elif channel_bandwidth is not None:
            bw = channel_bandwidth
        else:
            bw = cls.DEFAULT_CHANNEL_BANDWIDTH
        return DemodProfile(
            label=cls._label_for(mode),
            modulation=cls.MODULATION,
            channel_bandwidth=bw,
            sample_rate=cls.FIXED_SAMPLE_RATE,
            has_audio=cls.HAS_AUDIO,
            has_text=cls.HAS_TEXT,
            message_type=cls.MESSAGE_TYPE,
            sideband=cls._sideband_for(mode),
        )

    @classmethod
    def _label_for(cls, mode: str) -> str:
        return cls.LABEL

    @classmethod
    def _sideband_for(cls, mode: str) -> Literal["upper", "lower"] | None:
        return cls.SIDEBAND

    def status(self) -> DemodStatus:
        """Dynamic (actual-state) status. Thread-safe: callable from any thread.

        Called from the UI thread via `SDRDeviceContext.demod_status` and from
        the pipeline worker while it mutates demodulator state. Implementations
        must not iterate mutable containers without synchronization; prefer
        cached scalar fields updated by the worker on writes. Default: no
        dynamic fields.
        """
        return DemodStatus()

    @classmethod
    def bandwidth_override_on_mode_switch(cls, current_bw: float | None) -> float | None:
        """Return ``DEFAULT_CHANNEL_BANDWIDTH`` if `current_bw` doesn't fit this demod, else None.

        Used by the engine when the audio demod is swapped, to detect when an
        inherited bandwidth from the previous mode would exceed this demod's
        ``MAX_CHANNEL_BANDWIDTH`` (e.g. 200 kHz inherited from WFM into AM).
        Returning None means "keep the user's current bandwidth."
        """
        if current_bw is None or current_bw <= cls.MAX_CHANNEL_BANDWIDTH:
            return None
        return cls.DEFAULT_CHANNEL_BANDWIDTH

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

    def set_sstv_mode(self, name: str | None) -> None:  # noqa: B027
        """Force a specific SSTV submode (or clear). SSTV-only; no-op elsewhere."""
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
        *,
        stereo: bool = False,
    ) -> None:
        """Buffer one AudioBatch for later retrieval via `get_audio()`."""
        self._audio_batches.append(
            AudioBatch(
                samples=samples,
                sample_rate=sample_rate,
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
    "DemodProfile",
    "DemodStatus",
    "Demodulator",
]
