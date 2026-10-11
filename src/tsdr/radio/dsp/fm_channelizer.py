"""NBFM channelizer: decimate + FM discriminate -> real audio.

Shared front-end for narrowband-FM data decoders (FLEX, APRS, DMR). Brings the
IQ stream down to ~``target_rate`` through a multistage decimator, which keeps
the band free of folded signals and only computes the samples it keeps, selects
the channel with an FIR at that rate, then FM-demodulates it, holding the
filter and discriminator state across chunks.
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.dsp._kernels import StreamingFilter
from tsdr.radio.dsp.decimator import MultistageDecimator, smooth_decimation
from tsdr.radio.dsp.filters import firwin
from tsdr.radio.dsp.fm_discriminator import FMDiscriminator

# Channel-selection FIR at the decimated rate: Kaiser for ~70 dB stopband.
_CHANNEL_TAPS = 101
_CHANNEL_KAISER_BETA = 6.8


class FMChannelizer:
    def __init__(
        self,
        sample_rate: float,
        deviation: float,
        *,
        target_rate: float,
        channel_cutoff: float | None = None,
    ) -> None:
        self._decimation = smooth_decimation(sample_rate / target_rate)
        self.audio_rate = sample_rate / self._decimation
        # Default channel half-width is 2x deviation; pass `channel_cutoff` for a
        # wider channel (e.g. APRS, to tolerate a few kHz of PPM tuning offset).
        cutoff = min(self.audio_rate * 0.45, channel_cutoff or deviation * 2)
        # The decimator only keeps aliases out of the band, up to its output
        # Nyquist; the FIR at the output rate then cuts at the channel edge.
        self._decimator = MultistageDecimator(
            sample_rate, self._decimation, self.audio_rate * 0.45, dtype=np.complex64
        )
        taps = firwin(
            _CHANNEL_TAPS, cutoff, fs=self.audio_rate, window=("kaiser", _CHANNEL_KAISER_BETA)
        )
        self._channel = StreamingFilter(taps, [1.0], dtype=np.complex64)
        self._fm = FMDiscriminator(self.audio_rate, deviation)

    def _select(self, iq: np.ndarray) -> np.ndarray:
        """The channel alone, at the decimated rate."""
        return self._channel.process(self._decimator.process(iq))

    def process(self, iq: np.ndarray) -> np.ndarray:
        return self._fm.process(self._select(iq))

    def reset(self) -> None:
        self._decimator.reset()
        self._channel.reset()
        self._fm.reset()
