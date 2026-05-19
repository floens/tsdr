"""SSB (Single Sideband) demodulator for USB/LSB amateur radio.

Pipeline (mirrors SDR++ Brown's frequency-shift approach):
    IQ -> decimating anti-alias FIR
       -> sideband shift (USB: -bw/2, LSB: +bw/2; centers desired sideband at DC)
       -> sharp complex LPF at bw/2 (now selects desired sideband, rejects opposite)
       -> sideband unshift (back to original frequencies)
       -> real projection
       -> DC blocker
       -> AGC
       -> clip
       -> squelch

A symmetric complex LPF cannot reject the opposite sideband because real-tap
filters have a magnitude response that's mirrored around DC. The shift+LPF
trick effectively constructs an asymmetric complex bandpass at IF.

``channel_bandwidth`` represents the audio passband upper cutoff (matching
SDRangel's ``m_rfBandwidth``).
"""

from __future__ import annotations

import numpy as np

from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import (
    AGC,
    DCBlocker,
    SquelchGate,
    StreamingDecimFilter,
    StreamingFilter,
    firwin,
    iq_power_db,
)
from tsdr.radio.dsp._kernels import apply_freq_shift_c64


class SSBDemodulator(Demodulator):
    """SSB (Single Sideband) demodulator for USB/LSB amateur-radio voice."""

    # Audio passband upper cutoff in Hz; matches SDRangel default.
    DEFAULT_CHANNEL_BANDWIDTH = 3_000

    # DC blocker / sub-audio HPF cutoff in Hz.
    DC_BLOCKER_CUTOFF = 100.0

    def __init__(
        self,
        mode: str,
        sample_rate: float,
        audio_rate: float = 48000,
        channel_bandwidth: float | None = None,
    ):
        super().__init__()
        self.mode = mode.upper()
        if self.mode not in ("USB", "LSB"):
            raise ValueError(f"Invalid SSB mode: {mode}. Must be 'USB' or 'LSB'")

        self.sample_rate = float(sample_rate)
        self.audio_rate = float(audio_rate)
        self.channel_bandwidth = float(channel_bandwidth or self.DEFAULT_CHANNEL_BANDWIDTH)

        self._fwd_phase = 0.0
        self._back_phase = 0.0

        self._setup_channel_filter()
        self._dc_blocker = DCBlocker(self.decimated_rate, cutoff_hz=self.DC_BLOCKER_CUTOFF)
        self._agc = AGC(
            self.decimated_rate,
            attack_ms=5.0,
            decay_ms=200.0,
            setpoint=0.5,
        )
        self._squelch = SquelchGate(audio_rate=self.decimated_rate)

    def _setup_channel_filter(self) -> None:
        self.channel_decimation = max(1, int(self.sample_rate // self.audio_rate))
        self.decimated_rate = self.sample_rate / self.channel_decimation

        aa_cutoff = self.audio_rate * 0.45
        self._decim = StreamingDecimFilter(
            firwin(64, aa_cutoff, fs=self.sample_rate, window=("kaiser", 6.0)),
            decimation=self.channel_decimation,
            dtype=np.complex64,
            expected_input_size=200_000,
        )
        self._channel = self._build_channel_filter(self.channel_bandwidth)

    def _build_channel_filter(self, bandwidth: float) -> StreamingFilter:
        # After the sideband shift the desired sideband occupies [-bw/2, +bw/2]
        # around DC; LPF at bw/2 captures it.
        return StreamingFilter(
            firwin(128, bandwidth / 2, fs=self.decimated_rate),
            [1.0],
            dtype=np.complex64,
        )

    @property
    def _fwd_offset_hz(self) -> float:
        # USB shifts the input down by bw/2; LSB shifts up by bw/2. The back
        # shift is just the inverse offset.
        sign = +1.0 if self.mode == "USB" else -1.0
        return sign * (self.channel_bandwidth / 2.0)

    def set_channel_bandwidth(self, bandwidth: float) -> None:
        """Update channel bandwidth at runtime.

        Args:
            bandwidth: New audio passband upper cutoff in Hz.
        """
        self.channel_bandwidth = float(bandwidth)
        self._channel = self._build_channel_filter(self.channel_bandwidth)

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = float(rate)
        self._fwd_phase = 0.0
        self._back_phase = 0.0
        self._setup_channel_filter()
        self._dc_blocker = DCBlocker(self.decimated_rate, cutoff_hz=self.DC_BLOCKER_CUTOFF)
        self._agc = AGC(
            self.decimated_rate,
            attack_ms=5.0,
            decay_ms=200.0,
            setpoint=0.5,
        )
        self._squelch = SquelchGate(audio_rate=self.decimated_rate)

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread."""
        return SignalInfo(
            label=self.mode,
            channel_bandwidth=self.channel_bandwidth,
            modulation="SSB",
            has_audio=True,
            squelch_open=self._squelch.is_open if self._squelch.enabled else None,
            squelch_threshold_db=self._squelch.threshold_db,
            sideband="upper" if self.mode == "USB" else "lower",
        )

    def set_squelch(self, enabled: bool, threshold_db: float, hang_ms: float) -> None:
        self._squelch.configure(enabled=enabled, threshold_db=threshold_db, hang_ms=hang_ms)

    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None:
        if len(iq_samples) == 0:
            return

        iq_lo = self._decim.process(iq_samples)
        if len(iq_lo) == 0:
            return

        offset = self._fwd_offset_hz
        iq_shifted, self._fwd_phase = apply_freq_shift_c64(
            iq_lo, offset, self.decimated_rate, self._fwd_phase
        )
        iq_filt = self._channel.process(iq_shifted)
        power_db = iq_power_db(iq_filt)
        iq_back, self._back_phase = apply_freq_shift_c64(
            iq_filt, -offset, self.decimated_rate, self._back_phase
        )

        audio = iq_back.real.astype(np.float32, copy=False)
        audio = self._dc_blocker.process(audio)
        audio = self._agc.process(audio)
        np.clip(audio, -1.0, 1.0, out=audio)

        gate = self._squelch.process(power_db, len(audio))
        if gate is not None:
            audio *= gate

        self._emit_audio(audio, self.decimated_rate, timestamp)

    def reset(self) -> None:
        """Reset demodulator state."""
        super().reset()
        self._fwd_phase = 0.0
        self._back_phase = 0.0
        self._decim.reset()
        self._channel.reset()
        self._dc_blocker.reset()
        self._agc.reset()
        self._squelch.reset()
