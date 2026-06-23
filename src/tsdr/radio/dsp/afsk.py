"""Bell 202 AFSK1200 non-coherent tone demodulator.

Quadrature-mixes the (FM-discriminated) audio against the mark and space tones,
low-passes each to an envelope, and returns ``|mark| - |space|`` as an NRZ
decision signal at the input rate. Per-tone phase accumulators keep the mixers
continuous across chunks. A reusable primitive (any AFSK1200 link); the bit
timing (`MuellerMuller`) and framing live above it.
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.dsp._kernels import StreamingFilter
from tsdr.radio.dsp.filters import firwin

_TWO_PI = 2.0 * np.pi


class AFSK1200Demod:
    def __init__(
        self,
        sample_rate: float,
        *,
        mark: float = 1200.0,
        space: float = 2200.0,
        lpf_cutoff: float = 1200.0,
        lpf_taps: int | None = None,
    ) -> None:
        self._w_mark = _TWO_PI * mark / sample_rate
        self._w_space = _TWO_PI * space / sample_rate
        self._ph_mark = 0.0
        self._ph_space = 0.0
        ntaps = lpf_taps if lpf_taps is not None else (int(round(sample_rate / 1200.0)) | 1)
        taps = firwin(ntaps, lpf_cutoff, fs=sample_rate)
        self._lp_mark = StreamingFilter(taps, [1.0], dtype=np.complex64)
        self._lp_space = StreamingFilter(taps, [1.0], dtype=np.complex64)

    def process(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        if n == 0:
            return np.array([], dtype=np.float32)
        a = np.asarray(audio, dtype=np.float32)
        idx = np.arange(n)
        mix_mark = np.exp(-1j * (self._ph_mark + self._w_mark * idx)).astype(np.complex64)
        mix_space = np.exp(-1j * (self._ph_space + self._w_space * idx)).astype(np.complex64)
        self._ph_mark = (self._ph_mark + self._w_mark * n) % _TWO_PI
        self._ph_space = (self._ph_space + self._w_space * n) % _TWO_PI
        m = np.abs(self._lp_mark.process(a * mix_mark))
        s = np.abs(self._lp_space.process(a * mix_space))
        decision: np.ndarray = (m - s).astype(np.float32)
        return decision

    def reset(self) -> None:
        self._ph_mark = 0.0
        self._ph_space = 0.0
        self._lp_mark.reset()
        self._lp_space.reset()
