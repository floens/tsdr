from __future__ import annotations

import numpy as np

from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.radio.demodulators import NYQUIST_MARGIN, Demodulator
from tsdr.radio.dsp import (
    FMDiscriminator,
    SquelchGate,
    StreamingDecimFilter,
    StreamingFilter,
    firwin,
    iq_power_db,
)


class NarrowbandFMDemodulator(Demodulator):
    """Narrowband FM demodulator.

    Demodulates narrowband FM signals (2-way radio, amateur radio, weather radio).

    NFM specifications:
    - Deviation: ±5 kHz (typical)
    - Audio bandwidth: 3 kHz
    - De-emphasis: 750 µs or none

    Example:
        >>> demod = NarrowbandFMDemodulator(
        ...     sample_rate=240e3,
        ...     audio_rate=48000
        ... )
        >>> demod.demodulate(iq_samples, 0.0)
        >>> batches = demod.get_audio()
    """

    has_audio = True

    # Default channel bandwidth for NFM (12.5 kHz standard channel spacing)
    DEFAULT_CHANNEL_BANDWIDTH = 12_500
    MAX_CHANNEL_BANDWIDTH = 48_000 * NYQUIST_MARGIN

    def __init__(
        self,
        sample_rate: float,
        audio_rate: float = 48000,
        channel_bandwidth: float | None = None,
        deviation: float | None = None,
        de_emphasis_tc: float = 750e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.channel_bandwidth = channel_bandwidth or self.DEFAULT_CHANNEL_BANDWIDTH
        self._deviation_override = deviation
        self.deviation = deviation if deviation is not None else self.channel_bandwidth / 2.0
        self.de_emphasis_tc = de_emphasis_tc

        self._setup_channel_filter()

        # ±deviation maps to ±1.0 audio.
        self._fm_discrim = FMDiscriminator(self.decimated_rate, self.deviation)

        if de_emphasis_tc > 0:
            self._setup_deemphasis_filter()
            self.use_deemphasis = True
        else:
            self.use_deemphasis = False

        self._squelch = SquelchGate(audio_rate=self.decimated_rate)

    def _setup_channel_filter(self) -> None:
        """Setup decimating anti-alias FIR + sharp post-decim channel filter.

        Stage 1 (StreamingDecimFilter): broad anti-alias just below audio
        Nyquist, decimates input rate -> ~audio_rate.
        Stage 2 (StreamingFilter): sharp 128-tap complex LPF at the decimated
        rate, where reasonable tap counts give a transition BW ~1.2 kHz.
        """
        self.channel_decimation = max(1, int(self.sample_rate // self.audio_rate))
        self.decimated_rate = self.sample_rate / self.channel_decimation
        self.channel_bandwidth = min(self.channel_bandwidth, self.decimated_rate * NYQUIST_MARGIN)

        aa_cutoff = self.audio_rate * 0.45
        self._decim = StreamingDecimFilter(
            firwin(64, aa_cutoff, fs=self.sample_rate, window=("kaiser", 6.0)),
            decimation=self.channel_decimation,
            dtype=np.complex64,
            expected_input_size=200_000,
        )
        self._channel = self._build_channel_filter(self.channel_bandwidth)

    def _build_channel_filter(self, bandwidth: float) -> StreamingFilter:
        # At narrow channel_bandwidth near or below Carson's BW
        # (deviation + max_audio), upper modulator sidebands get clipped on
        # fully-deviated signals -- intended trade for tighter selectivity.
        return StreamingFilter(
            firwin(128, bandwidth / 2, fs=self.decimated_rate),
            [1.0],
            dtype=np.complex64,
        )

    def set_channel_bandwidth(self, bandwidth: float) -> None:
        """Update channel bandwidth at runtime.

        Args:
            bandwidth: New channel bandwidth in Hz
        """
        self.channel_bandwidth = min(float(bandwidth), self.decimated_rate * NYQUIST_MARGIN)
        self._channel = self._build_channel_filter(self.channel_bandwidth)
        if self._deviation_override is None:
            self.deviation = self.channel_bandwidth / 2.0
            self._fm_discrim.set_deviation(self.decimated_rate, self.deviation)

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = float(rate)
        self._setup_channel_filter()
        self._fm_discrim = FMDiscriminator(self.decimated_rate, self.deviation)
        if self.use_deemphasis:
            self._setup_deemphasis_filter()
        self._squelch = SquelchGate(audio_rate=self.decimated_rate)

    def set_deviation(self, deviation: float) -> None:
        self.deviation = float(deviation)
        self._deviation_override = self.deviation
        self._fm_discrim.set_deviation(self.decimated_rate, self.deviation)

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread."""
        return SignalInfo(
            label="Narrowband FM",
            channel_bandwidth=self.channel_bandwidth,
            modulation="FM",
            has_audio=True,
            squelch_open=self._squelch.is_open if self._squelch.enabled else None,
            squelch_threshold_db=self._squelch.threshold_db,
        )

    def set_squelch(self, enabled: bool, threshold_db: float, hang_ms: float) -> None:
        self._squelch.configure(enabled=enabled, threshold_db=threshold_db, hang_ms=hang_ms)

    def _setup_deemphasis_filter(self) -> None:
        """Setup de-emphasis filter.

        Creates IIR filter for vectorized de-emphasis using lfilter.
        Filter equation: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        """
        dt = 1.0 / self.decimated_rate
        alpha = dt / (self.de_emphasis_tc + dt)

        # IIR filter: b = [alpha], a = [1, alpha - 1]
        self._deemph = StreamingFilter(
            np.array([alpha]),
            np.array([1.0, alpha - 1.0]),
            dtype=np.float64,
        )

    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None:
        """Demodulate IQ samples to audio.

        Args:
            iq_samples: Complex IQ samples
            timestamp: Capture timestamp
        """
        if len(iq_samples) == 0:
            return

        iq_lo = self._decim.process(iq_samples)
        iq_filtered = self._channel.process(iq_lo)
        power_db = iq_power_db(iq_filtered)

        audio_raw = self._fm_discrim.process(iq_filtered)

        # Clip
        audio_raw = np.clip(audio_raw, -1.0, 1.0)

        if self.use_deemphasis:
            audio_final = self._apply_deemphasis(audio_raw)
        else:
            audio_final = audio_raw

        audio_samples = audio_final.astype(np.float32)
        envelope = self._squelch.process(power_db, len(audio_samples))
        if envelope is not None:
            audio_samples *= envelope
        self._emit_audio(audio_samples, self.decimated_rate, timestamp)

    def _apply_deemphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply de-emphasis filter.

        Uses vectorized lfilter for 40-60x speedup over Python loop.

        Args:
            audio: Input audio samples

        Returns:
            De-emphasized audio samples
        """
        return np.asarray(self._deemph.process(audio), dtype=np.float32)

    def reset(self) -> None:
        """Reset demodulator state.

        Clears phase history and filter state to prevent artifacts
        when restarting or switching modes.
        """
        super().reset()
        self._fm_discrim.reset()
        self._decim.reset()
        self._channel.reset()
        if self.use_deemphasis:
            self._deemph.reset()
        self._squelch.reset()
