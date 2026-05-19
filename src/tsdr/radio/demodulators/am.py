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


class AMDemodulator(Demodulator):
    """AM envelope demodulator.

    Pipeline:
        IQ -> decimating anti-alias FIR -> sharp complex channel-select FIR
           -> envelope (|.|) -> DC blocker -> AGC -> audio LPF -> squelch.

    The decimating FIR brings the signal down to ``audio_rate`` so the
    subsequent sharp channel filter only needs reasonable tap counts to be
    selective (a sharp narrow FIR at the input rate is mathematically
    impractical -- transition bandwidth scales with fs/numtaps).
    """

    DEFAULT_CHANNEL_BANDWIDTH = 10_000

    def __init__(
        self,
        sample_rate: float,
        audio_rate: float = 48000,
        channel_bandwidth: float | None = None,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.audio_rate = float(audio_rate)
        self.channel_bandwidth = float(channel_bandwidth or self.DEFAULT_CHANNEL_BANDWIDTH)

        self._build_filters()
        self._dc = DCBlocker(self.decimated_rate, cutoff_hz=16.0)
        self._agc = AGC(
            self.decimated_rate,
            attack_ms=5.0,
            decay_ms=200.0,
            setpoint=0.5,
        )
        self._squelch = SquelchGate(audio_rate=self.decimated_rate)

    def _build_filters(self) -> None:
        self.channel_decimation = max(1, int(self.sample_rate // self.audio_rate))
        self.decimated_rate = self.sample_rate / self.channel_decimation

        # Anti-alias decimator: input rate -> ~audio_rate. Wide cutoff just
        # below audio Nyquist; this is not the channel filter.
        aa_cutoff = self.audio_rate * 0.45
        self._decim = StreamingDecimFilter(
            firwin(64, aa_cutoff, fs=self.sample_rate, window=("kaiser", 6.0)),
            decimation=self.channel_decimation,
            dtype=np.complex64,
        )
        self._channel = self._build_channel_filter(self.channel_bandwidth)
        self._audio_lpf = self._build_audio_lpf(self.channel_bandwidth)

    def _build_channel_filter(self, bandwidth: float) -> StreamingFilter:
        return StreamingFilter(
            firwin(128, bandwidth / 2, fs=self.decimated_rate),
            [1.0],
            dtype=np.complex64,
        )

    def _build_audio_lpf(self, bandwidth: float) -> StreamingFilter:
        return StreamingFilter(
            firwin(64, bandwidth / 2, fs=self.decimated_rate),
            [1.0],
            dtype=np.float32,
        )

    def set_channel_bandwidth(self, bandwidth: float) -> None:
        self.channel_bandwidth = float(bandwidth)
        # Rebuild only bandwidth-dependent filters; keep DC/AGC envelope state.
        self._channel = self._build_channel_filter(self.channel_bandwidth)
        self._audio_lpf = self._build_audio_lpf(self.channel_bandwidth)

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = float(rate)
        self._build_filters()
        self._dc = DCBlocker(self.decimated_rate, cutoff_hz=16.0)
        self._agc = AGC(
            self.decimated_rate,
            attack_ms=5.0,
            decay_ms=200.0,
            setpoint=0.5,
        )
        self._squelch = SquelchGate(audio_rate=self.decimated_rate)

    def info(self) -> SignalInfo:
        return SignalInfo(
            label="AM",
            channel_bandwidth=self.channel_bandwidth,
            modulation="AM",
            has_audio=True,
            squelch_open=self._squelch.is_open if self._squelch.enabled else None,
            squelch_threshold_db=self._squelch.threshold_db,
        )

    def set_squelch(self, enabled: bool, threshold_db: float, hang_ms: float) -> None:
        self._squelch.configure(enabled=enabled, threshold_db=threshold_db, hang_ms=hang_ms)

    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None:
        if len(iq_samples) == 0:
            return

        iq_lo = self._decim.process(iq_samples)
        iq_ch = self._channel.process(iq_lo)
        # np.abs(complex64) returns float32 directly -- no astype needed.
        env = np.abs(iq_ch)
        env = self._dc.process(env)
        env = self._agc.process(env)
        audio = self._audio_lpf.process(env)
        np.clip(audio, -1.0, 1.0, out=audio)

        power_db = iq_power_db(iq_ch)
        gate = self._squelch.process(power_db, len(audio))
        if gate is not None:
            audio *= gate

        self._emit_audio(audio, self.decimated_rate, timestamp)

    def reset(self) -> None:
        super().reset()
        self._decim.reset()
        self._channel.reset()
        self._audio_lpf.reset()
        self._dc.reset()
        self._agc.reset()
        self._squelch.reset()
