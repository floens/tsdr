"""CW (Morse code) demodulator.

Uses an explicit Beat Frequency Oscillator (BFO) to convert the on/off-keyed
carrier into an audible tone at a fixed pitch (default 700 Hz).

Pipeline:
    IQ -> decimating anti-alias FIR
       -> sharp 1024-tap complex LPF at channel_bandwidth/2 (channel selection)
       -> BFO frequency shift (multiply by exp(j*2pi*tone_hz*t))
       -> real projection
       -> AGC (slow attack/decay -- doesn't pump on keying)
       -> clip
       -> squelch

The operator must tune the receiver to put the carrier at IF=0; the BFO then
converts that DC-centred carrier into an audible tone at ``tone_hz``.

Image-frequency note: after the BFO shift and real projection, IF content at
+/-2*tone_hz both fold onto +tone_hz audio. The 1024-tap LPF (transition BW
~155 Hz at 48 kHz) attenuates +/-1400 Hz by 50+ dB, so the image is not
audible at default settings.

Narrow-CW caveat: for ``channel_bandwidth < 100 Hz`` the LPF transition
(155 Hz at 48 kHz) starts to dominate the cutoff. The filter still
narrows the passband but is no longer brickwall. A future v2 may add a
two-stage decimation to a low IF rate (e.g. 4 kHz) to enable sharper
narrow filters.
"""

from __future__ import annotations

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.radio.decoders.morse import MorseDecoder
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import (
    AGC,
    SquelchGate,
    StreamingDecimFilter,
    StreamingFilter,
    firwin,
    iq_power_db,
)
from tsdr.radio.dsp._kernels import apply_freq_shift_c64


class CWDemodulator(Demodulator):
    """CW (Morse code) demodulator with explicit BFO."""

    DEFAULT_CHANNEL_BANDWIDTH = 200.0
    DEFAULT_TONE_HZ = 700.0

    def __init__(
        self,
        sample_rate: float,
        audio_rate: float = 48000,
        channel_bandwidth: float | None = None,
        tone_hz: float | None = None,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.audio_rate = float(audio_rate)
        self.channel_bandwidth = float(channel_bandwidth or self.DEFAULT_CHANNEL_BANDWIDTH)
        self.tone_hz = float(tone_hz or self.DEFAULT_TONE_HZ)

        self._bfo_phase = 0.0

        self._setup_channel_filter()
        # Slow AGC: dit onsets pass uncompressed (attack=100 ms), and gain
        # holds steady through inter-element silences (decay=500 ms) so noise
        # doesn't pump up between dits/dahs.
        self._agc = AGC(
            self.decimated_rate,
            attack_ms=100.0,
            decay_ms=500.0,
            setpoint=0.5,
        )
        self._squelch = SquelchGate(audio_rate=self.decimated_rate)
        self._morse = MorseDecoder(self.decimated_rate)

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
        # 1024 taps for ~155 Hz transition at 48 kHz fs. Sufficient for default
        # 200 Hz BW; soft for bw < 100 Hz where transition starts to dominate.
        return StreamingFilter(
            firwin(1024, bandwidth / 2, fs=self.decimated_rate),
            [1.0],
            dtype=np.complex64,
        )

    def set_channel_bandwidth(self, bandwidth: float) -> None:
        self.channel_bandwidth = float(bandwidth)
        self._channel = self._build_channel_filter(self.channel_bandwidth)

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread."""
        return SignalInfo(
            label=f"CW {int(self.tone_hz)} Hz",
            channel_bandwidth=self.channel_bandwidth,
            modulation="CW",
            has_audio=True,
            has_text=True,
            message_type="text",
            squelch_open=self._squelch.is_open if self._squelch.enabled else None,
            squelch_threshold_db=self._squelch.threshold_db,
        )

    def set_squelch(self, enabled: bool, threshold_db: float, hang_ms: float) -> None:
        self._squelch.configure(enabled=enabled, threshold_db=threshold_db, hang_ms=hang_ms)

    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None:
        if len(iq_samples) == 0:
            return

        iq_lo = self._decim.process(iq_samples)
        iq_filt = self._channel.process(iq_lo)
        if len(iq_filt) == 0:
            return

        power_db = iq_power_db(iq_filt)

        # Morse text decoder consumes the keying envelope (post-LPF magnitude).
        env = np.abs(iq_filt).astype(np.float32, copy=False)
        self._morse.process(env, timestamp)

        iq_shifted, self._bfo_phase = apply_freq_shift_c64(
            iq_filt, -self.tone_hz, self.decimated_rate, self._bfo_phase
        )

        audio = iq_shifted.real.astype(np.float32, copy=False)
        audio = self._agc.process(audio)
        np.clip(audio, -1.0, 1.0, out=audio)

        gate = self._squelch.process(power_db, len(audio))
        if gate is not None:
            audio *= gate

        self._emit_audio(audio, self.decimated_rate, timestamp)

    def get_messages(self) -> list[DecodedMessage]:
        return self._morse.get_messages()

    def reset(self) -> None:
        """Reset demodulator state."""
        super().reset()
        self._bfo_phase = 0.0
        self._decim.reset()
        self._channel.reset()
        self._agc.reset()
        self._squelch.reset()
        self._morse.reset()
