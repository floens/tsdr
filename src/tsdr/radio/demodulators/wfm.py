from __future__ import annotations

import math

import numba as nb
import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.core.tracing import span
from tsdr.radio.decoders.rds import RDSData, RDSDecoder
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import FMDiscriminator, StreamingDecimFilter, StreamingFilter, firwin


@nb.njit(cache=True, fastmath=True)
def _pll_stereo_carrier(
    pilot_filtered: np.ndarray,
    phase_in: float,
    freq_in: float,
    integrator_in: float,
    pll_alpha: float,
    pll_beta: float,
) -> tuple[np.ndarray, float, float]:
    """PLL loop: track 19 kHz pilot, generate coherent 38 kHz carrier.

    Returns (carrier_38k, phase_out, integrator_out).
    """
    n = pilot_filtered.shape[0]
    carrier_38k = np.empty(n, dtype=np.float32)
    phase = phase_in
    freq = freq_in
    integrator = integrator_in
    two_pi = 2.0 * math.pi

    for i in range(n):
        carrier_38k[i] = 2.0 * math.cos(2.0 * phase)
        error = pilot_filtered[i] * (-math.sin(phase))
        integrator += pll_beta * error
        freq_correction = pll_alpha * error + integrator
        phase += freq + freq_correction
        if phase > two_pi:
            phase -= two_pi
        elif phase < -two_pi:
            phase += two_pi

    return carrier_38k, phase, integrator


class WidebandFMDemodulator(Demodulator):
    """Wideband FM demodulator with stereo decoding.

    Demodulates broadcast FM signals (88-108 MHz). Uses phase difference
    method with de-emphasis filtering. Stereo decoding via PLL-locked
    19 kHz pilot recovery and 38 kHz subcarrier demodulation.

    FM broadcast specifications:
    - Deviation: ±75 kHz
    - Audio bandwidth: 15 kHz
    - De-emphasis: 75 µs (USA) or 50 µs (Europe)
    - Pilot tone: 19 kHz (stereo indicator)
    - Stereo subcarrier: 38 kHz (DSB-SC L-R signal)

    Always outputs 2-channel audio (mono is duplicated to both channels).

    Example:
        >>> demod = WidebandFMDemodulator(
        ...     sample_rate=2.4e6,
        ...     audio_rate=48000
        ... )
        >>> demod.demodulate(iq_samples, 0.0)
        >>> batches = demod.get_audio()
    """

    has_audio = True

    # Default channel bandwidth for WFM (200 kHz for stereo + RDS)
    DEFAULT_CHANNEL_BANDWIDTH = 200_000

    # Pilot detection hysteresis thresholds (SNR in dB vs adjacent noise band)
    # Real stereo stations show ~20+ dB pilot SNR; noise is ~0 dB
    _PILOT_LOCK_SNR = 8.0  # dB, lock when pilot SNR exceeds this
    _PILOT_UNLOCK_SNR = 3.0  # dB, unlock when pilot SNR drops below this

    def __init__(
        self,
        sample_rate: float,
        audio_rate: float = 48000,
        channel_bandwidth: float | None = None,
        de_emphasis_tc: float = 50e-6,  # 50 µs for Europe
        rds_enabled: bool = True,
    ):
        """Initialize WFM demodulator.

        Args:
            sample_rate: Input IQ sample rate (Hz)
            audio_rate: Output audio sample rate (Hz)
            channel_bandwidth: Channel filter bandwidth in Hz (default: 200 kHz)
            de_emphasis_tc: De-emphasis time constant (seconds)
                75e-6 for USA, 50e-6 for Europe
            rds_enabled: Whether to decode RDS data
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.channel_bandwidth = channel_bandwidth or self.DEFAULT_CHANNEL_BANDWIDTH
        self.de_emphasis_tc = de_emphasis_tc
        self.rds_enabled = rds_enabled

        self._setup_channel_filter()

        # ±75 kHz deviation maps to ±1.0 audio.
        self._fm_discrim = FMDiscriminator(self.intermediate_rate, 75000.0)

        self.audio_decimation_factor = max(1, round(self.intermediate_rate / audio_rate))
        self.output_sample_rate = self.intermediate_rate / self.audio_decimation_factor

        self._setup_deemphasis_filter()
        self._setup_audio_lpf()
        self._setup_pilot_filter()
        self._setup_stereo_filters()

        self._channel_decim_phase = 0
        self._rds_decoder: RDSDecoder | None = None
        self.stereo_detected: bool = False
        # Cached station name kept for SignalInfo while the worker mutates state on another thread.
        self._last_ps_name: str = ""
        self._messages: list[DecodedMessage] = []

    def _setup_channel_filter(self) -> None:
        """Setup channel filter for isolating the FM station.

        Decimates from input sample rate to ~250 kHz intermediate rate.
        Uses fused filter+decimate kernel for ~10x speedup over filter-then-slice.
        """
        target_intermediate = 250_000
        self.channel_decimation = max(1, round(self.sample_rate / target_intermediate))
        self.intermediate_rate = self.sample_rate / self.channel_decimation

        cutoff = self.channel_bandwidth / 2
        self._channel_filter = StreamingDecimFilter(
            firwin(64, cutoff, fs=self.sample_rate),
            decimation=self.channel_decimation,
            dtype=np.complex64,
            expected_input_size=200_000,
        )

    def set_channel_bandwidth(self, bandwidth: float) -> None:
        """Update channel bandwidth at runtime.

        Args:
            bandwidth: New channel bandwidth in Hz
        """
        self.channel_bandwidth = bandwidth
        cutoff = bandwidth / 2
        self._channel_filter = StreamingDecimFilter(
            firwin(64, cutoff, fs=self.sample_rate),
            decimation=self.channel_decimation,
            dtype=np.complex64,
            expected_input_size=200_000,
        )

    def set_sample_rate(self, rate: float) -> None:
        # Order mirrors __init__: channel filter sets intermediate_rate
        # before downstream helpers consume it.
        self.sample_rate = float(rate)
        self._setup_channel_filter()
        self._fm_discrim = FMDiscriminator(self.intermediate_rate, 75000.0)
        self.audio_decimation_factor = max(1, round(self.intermediate_rate / self.audio_rate))
        self.output_sample_rate = self.intermediate_rate / self.audio_decimation_factor
        self._setup_deemphasis_filter()
        self._setup_audio_lpf()
        self._setup_pilot_filter()
        self._setup_stereo_filters()
        self._channel_decim_phase = 0
        self._rds_decoder = None
        self.stereo_detected = False
        self._last_ps_name = ""
        self._messages.clear()

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread. Reads scalar fields only."""
        # Normalize pilot SNR: 3 dB (unlock) = 0.0, 20 dB (strong) = 1.0
        quality = max(0.0, min(1.0, (self.pilot_snr_ema - 3.0) / 17.0))
        quality_label = f"Pilot {self.pilot_snr_ema:.0f} dB"
        return SignalInfo(
            label="Wideband FM",
            channel_bandwidth=self.channel_bandwidth,
            modulation=("Stereo " if self.stereo_detected else "") + "FM",
            has_audio=True,
            has_text=True,
            message_type="rds",
            quality_label=quality_label,
            quality=quality,
            description=self._last_ps_name or None,
        )

    def _setup_deemphasis_filter(self) -> None:
        """Setup de-emphasis filters via bilinear transform.

        H(s) = 1 / (1 + s*tau)  ->  H(z) = (1-a) / (1 - a*z^-1)
        where a = exp(-1 / (fs * tau)).

        Runs at output_sample_rate (post-decimation) for efficiency.
        Two independent filter states: one for L+R, one for L-R.
        """
        alpha = np.exp(-1.0 / (self.output_sample_rate * self.de_emphasis_tc))
        deemph_b = np.array([1.0 - alpha])
        deemph_a = np.array([1.0, -alpha])
        self._deemph_lr = StreamingFilter(deemph_b, deemph_a, dtype=np.float32)
        self._deemph_lmr = StreamingFilter(deemph_b, deemph_a, dtype=np.float32)

    def _setup_audio_lpf(self) -> None:
        """Setup 15 kHz audio low-pass filter for L+R path at intermediate rate.

        Fuses filter + decimation to audio rate (computes only needed outputs).
        Removes pilot (19 kHz), stereo subcarrier (38 kHz), and RDS (57 kHz)
        to prevent aliasing.
        """
        audio_lpf_taps = firwin(128, 15000, fs=self.intermediate_rate)
        self._audio_lpf = StreamingDecimFilter(
            audio_lpf_taps, decimation=self.audio_decimation_factor
        )
        # Keep taps for stereo LPF (same taps, separate state)
        self._audio_lpf_taps = audio_lpf_taps

    def _setup_pilot_filter(self) -> None:
        """Setup 19 kHz pilot bandpass and adjacent noise reference filters.

        Pilot detection uses SNR: pilot band power vs adjacent noise band power.
        This distinguishes a real 19 kHz tone from broadband noise.
        """
        # 19 kHz pilot bandpass (18.5-19.5 kHz)
        # Non-decimating: StreamingFilter (DF-II-T) is faster than StreamingDecimFilter
        # here because DF-I's padded buffer copy overhead isn't amortized at m=1.
        self._pilot_bp = StreamingFilter(
            firwin(128, [18500, 19500], pass_zero=False, fs=self.intermediate_rate),
            [1.0],
            dtype=np.float32,
        )

        # Noise reference bandpass (16-17 kHz) - same bandwidth, adjacent to pilot
        self._noise_ref = StreamingFilter(
            firwin(128, [16000, 17000], pass_zero=False, fs=self.intermediate_rate),
            [1.0],
            dtype=np.float32,
        )

        # PLL state for pilot recovery
        self.pll_phase = 0.0  # Phase accumulator (radians)
        self.pll_freq = 2.0 * np.pi * 19000.0 / self.intermediate_rate  # Normalized frequency
        self.pll_integrator = 0.0  # Loop filter integrator

        # PLL loop filter coefficients (type 2, second-order loop)
        # Bandwidth ~50 Hz for stable lock
        wn = 2.0 * np.pi * 50.0 / self.intermediate_rate  # Natural frequency
        damping = 0.707
        self.pll_alpha = 2.0 * damping * wn  # Proportional gain
        self.pll_beta = wn * wn  # Integral gain

        # Pilot SNR EMA for lock detection
        # TC stored in seconds; per-chunk alpha computed at runtime based on chunk length
        self.pilot_snr_ema = 0.0  # dB
        self._pilot_ema_tc = 0.3  # 300ms time constant

    def _setup_stereo_filters(self) -> None:
        """Setup L-R path: decimating 15 kHz LPF with independent filter state."""
        self._stereo_lpf = StreamingDecimFilter(
            self._audio_lpf_taps, decimation=self.audio_decimation_factor
        )

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        if len(iq_samples) == 0:
            return

        # Channel filter + decimate to intermediate rate (~240 kHz)
        with span("channel_filter"):
            if self.channel_decimation > 1:
                iq_filtered = self._channel_filter.process(iq_samples)
            else:
                iq_filtered = iq_samples

        # FM discriminator: conjugate-product method (single-pass Numba kernel)
        with span("phase_calc"):
            audio_raw = self._fm_discrim.process(iq_filtered)

        # RDS decoding (at intermediate rate, needs intact 57 kHz content)
        if self.rds_enabled:
            with span("rds_decode"):
                rds_data = self._decode_rds(audio_raw)
                if rds_data is not None:
                    if rds_data.ps_name:
                        self._last_ps_name = rds_data.ps_name
                    summary_parts = []
                    if rds_data.ps_name:
                        summary_parts.append(rds_data.ps_name)
                    if rds_data.radio_text:
                        summary_parts.append(rds_data.radio_text)
                    summary = " | ".join(summary_parts) if summary_parts else "RDS"
                    self._messages.append(
                        DecodedMessage(text=summary, timestamp=capture_utc_s, data=rds_data)
                    )

        # Clip for audio output (after RDS extraction, in-place)
        np.clip(audio_raw, -1.0, 1.0, out=audio_raw)

        # Stereo decoding: pilot detection + L-R extraction
        # Audio/stereo LPFs fuse filter+decimate, output is at audio rate
        with span("stereo_decode"):
            lr_sum, lr_diff = self._decode_stereo(audio_raw)

        # De-emphasis at audio rate (post-decimation)
        with span("deemphasis"):
            lr_sum_deemph = self._deemph_lr.process(lr_sum)

        # Combine to L/R channels: L = (L+R + L-R) / 2, R = (L+R - L-R) / 2
        if lr_diff is not None:
            lr_diff_deemph = self._deemph_lmr.process(lr_diff)
            left = np.float32(0.5) * (lr_sum_deemph + lr_diff_deemph)
            right = np.float32(0.5) * (lr_sum_deemph - lr_diff_deemph)
        else:
            left = lr_sum_deemph
            right = lr_sum_deemph

        audio_samples = np.column_stack([left, right])

        self._emit_audio(
            audio_samples,
            self.output_sample_rate,
            stereo=self.stereo_detected,
        )

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._messages
        self._messages = []
        return messages

    def get_constellation(self) -> tuple[np.ndarray, str] | None:
        if self._rds_decoder is None:
            return None
        points = self._rds_decoder.get_constellation()
        if points is None:
            return None
        return points, "BPSK"

    def _decode_stereo(self, audio_clipped: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract L+R and L-R from FM baseband using PLL-locked pilot.

        1. Bandpass filter to extract 19 kHz pilot
        2. PLL locks to pilot, generates coherent 38 kHz carrier
        3. L+R: 15 kHz LPF on baseband
        4. L-R: baseband × 38 kHz carrier -> 15 kHz LPF

        Returns:
            (lr_sum, lr_diff): L+R and L-R signals at intermediate rate
        """
        n_samples = len(audio_clipped)

        # Extract pilot tone via bandpass filter
        pilot_filtered = self._pilot_bp.process(audio_clipped)

        # Extract noise reference (adjacent band)
        noise_ref = self._noise_ref.process(audio_clipped)

        # SNR-based pilot detection: compare pilot band power to noise band power
        # np.dot(x,x)/n avoids allocating a temporary squared array
        n_filt = len(pilot_filtered)
        pilot_power = np.dot(pilot_filtered, pilot_filtered) / n_filt
        noise_power = np.dot(noise_ref, noise_ref) / n_filt
        # Avoid log(0); floor noise at -60 dB relative to pilot
        snr_db = 10.0 * np.log10(pilot_power / max(noise_power, pilot_power * 1e-6, 1e-20))

        # EMA smoothing of SNR (per-chunk alpha from chunk duration)
        chunk_duration = n_samples / self.intermediate_rate
        ema_alpha = 1.0 - np.exp(-chunk_duration / self._pilot_ema_tc)
        self.pilot_snr_ema += ema_alpha * (snr_db - self.pilot_snr_ema)

        # Hysteresis-based stereo detection on SNR
        if self.stereo_detected:
            if self.pilot_snr_ema < self._PILOT_UNLOCK_SNR:
                self.stereo_detected = False
        else:
            if self.pilot_snr_ema > self._PILOT_LOCK_SNR:
                self.stereo_detected = True

        # PLL: track pilot phase sample-by-sample, generate 38 kHz carrier
        carrier_38k, phase, integrator = _pll_stereo_carrier(
            pilot_filtered,
            self.pll_phase,
            self.pll_freq,
            self.pll_integrator,
            self.pll_alpha,
            self.pll_beta,
        )
        self.pll_phase = phase
        self.pll_freq = 2.0 * np.pi * 19000.0 / self.intermediate_rate
        self.pll_integrator = integrator

        # L+R path: decimating 15 kHz LPF
        lr_sum = self._audio_lpf.process(audio_clipped)

        # L-R path: demodulate 38 kHz subcarrier then decimating 15 kHz LPF
        # (carrier already scaled by 2.0 in _pll_stereo_carrier)
        if self.stereo_detected:
            lr_diff_raw = audio_clipped * carrier_38k
            lr_diff = self._stereo_lpf.process(lr_diff_raw)
        else:
            # Sync decimation phase without running the filter
            self._stereo_lpf._phase = self._audio_lpf._phase
            lr_diff = None

        return lr_sum, lr_diff

    def _decode_rds(self, audio_raw: np.ndarray) -> RDSData | None:
        """Decode RDS from FM-demodulated audio.

        Args:
            audio_raw: FM-demodulated audio at intermediate rate (normalized ±1.0)

        Returns:
            RDSData snapshot or None if decoder not available
        """
        if self._rds_decoder is None:
            self._rds_decoder = RDSDecoder(self.intermediate_rate)

        return self._rds_decoder.process(audio_raw)

    def reset(self) -> None:
        """Reset demodulator state.

        Clears phase history, filter states, PLL state, and stereo detection
        to prevent artifacts when restarting or switching modes.
        """
        super().reset()
        self._fm_discrim.reset()
        self._channel_filter.reset()
        self._deemph_lr.reset()
        self._deemph_lmr.reset()
        self._audio_lpf.reset()

        # Reset stereo state
        self._pilot_bp.reset()
        self._noise_ref.reset()
        self._stereo_lpf.reset()
        self.pll_phase = 0.0
        self.pll_freq = 2.0 * np.pi * 19000.0 / self.intermediate_rate
        self.pll_integrator = 0.0
        self.pilot_snr_ema = 0.0
        self.stereo_detected = False

        if self._rds_decoder is not None:
            self._rds_decoder.reset()

        self._messages.clear()
        self._last_ps_name = ""
