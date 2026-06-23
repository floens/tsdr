"""NBFM channelizer: anti-alias + decimate + FM discriminate -> real audio.

Shared front-end for narrowband-FM data decoders (FLEX, APRS, ...). Brings the
IQ stream down to ~``target_rate`` and FM-demodulates it, holding the streaming
filter / decimation-phase / discriminator state across chunks. This is the
front-end the decoders used to inline; centralizing it keeps one decimation-phase
implementation instead of a copy per decoder.
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.dsp._kernels import StreamingFilter
from tsdr.radio.dsp.filters import firwin
from tsdr.radio.dsp.fm_discriminator import FMDiscriminator


class FMChannelizer:
    def __init__(
        self,
        sample_rate: float,
        deviation: float,
        *,
        target_rate: float,
        channel_cutoff: float | None = None,
        taps: int = 101,
    ) -> None:
        self._decimation = max(1, round(sample_rate / target_rate))
        self.audio_rate = sample_rate / self._decimation
        # Default channel half-width is 2x deviation; pass `channel_cutoff` for a
        # wider channel (e.g. APRS, to tolerate a few kHz of PPM tuning offset).
        cutoff = min(self.audio_rate * 0.45, channel_cutoff or deviation * 2)
        self._antialias = StreamingFilter(
            firwin(taps, cutoff, fs=sample_rate), [1.0], dtype=np.complex64
        )
        self._decim_phase = 0
        self._fm = FMDiscriminator(self.audio_rate, deviation)

    def process(self, iq: np.ndarray) -> np.ndarray:
        filtered = self._antialias.process(iq)
        decimated = filtered[self._decim_phase :: self._decimation]
        # Samples to skip at the start of the next chunk so the decimation grid
        # stays continuous across chunk boundaries.
        n_used = len(filtered) - self._decim_phase
        self._decim_phase = (self._decimation - n_used % self._decimation) % self._decimation
        return self._fm.process(decimated)

    def reset(self) -> None:
        self._antialias.reset()
        self._decim_phase = 0
        self._fm.reset()
