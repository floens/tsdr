"""Non-coherent 2-FSK front-end with adaptive threshold and gated bit-sync.

Recovers one soft bit per baud (sign = mark(+)/space(-)) from a narrow-shift FSK
signal and keeps its state across chunks, so it streams. A reusable primitive for
async 2-FSK teleprinter modes (RTTY, NAVTEX/SITOR-B, ...); framing and the code
alphabet live above it. Works down to ~0 dB in-band SNR.
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.dsp._kernels import StreamingFilter, _fsk_gated_bitsync, apply_freq_shift_c64
from tsdr.radio.dsp.filters import firwin


class FSKFrontEnd:
    def __init__(
        self,
        sample_rate: float,
        baud: float,
        shift_hz: float,
        *,
        center_hz: float = 0.0,
    ) -> None:
        deviation = shift_hz / 2.0
        self._sample_rate = sample_rate
        self._mark_hz = center_hz + deviation
        self._space_hz = center_hz - deviation
        self._ph_mark = 0.0
        self._ph_space = 0.0
        self._bsc = sample_rate / baud
        taps = firwin(int(self._bsc * 2) | 1, 0.7 * baud, fs=sample_rate)
        self._lp_mark = StreamingFilter(taps, [1.0], dtype=np.complex64)
        self._lp_space = StreamingFilter(taps, [1.0], dtype=np.complex64)
        self._env = np.zeros(4, dtype=np.float64)
        self._acc = np.zeros(3, dtype=np.float64)
        self._avg = np.zeros(3, dtype=np.float64)
        self._evt = np.array([0.0, self._bsc / 5.0, self._bsc * 2.0 / 5.0, 0.0])
        self._out = np.empty(1024, dtype=np.float32)
        self._mark_a = np.empty(1024, dtype=np.float32)
        self._space_a = np.empty(1024, dtype=np.float32)

    def process(self, iq: np.ndarray) -> np.ndarray:
        n = len(iq)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        z = np.ascontiguousarray(iq, dtype=np.complex64)
        mark_iq, self._ph_mark = apply_freq_shift_c64(
            z, self._mark_hz, self._sample_rate, self._ph_mark
        )
        space_iq, self._ph_space = apply_freq_shift_c64(
            z, self._space_hz, self._sample_rate, self._ph_space
        )
        if len(self._mark_a) < n:
            self._mark_a = np.empty(n, dtype=np.float32)
            self._space_a = np.empty(n, dtype=np.float32)
        mark_a = np.abs(self._lp_mark.process(mark_iq), out=self._mark_a[:n])
        space_a = np.abs(self._lp_space.process(space_iq), out=self._space_a[:n])
        cap = int(n / self._bsc) + 4
        if len(self._out) < cap:
            self._out = np.empty(cap, dtype=np.float32)
        count = _fsk_gated_bitsync(
            mark_a, space_a, self._bsc, self._env, self._acc, self._avg, self._evt, self._out
        )
        return self._out[:count].copy()

    def reset(self) -> None:
        self._ph_mark = 0.0
        self._ph_space = 0.0
        self._lp_mark.reset()
        self._lp_space.reset()
        self._env[:] = 0.0
        self._acc[:] = 0.0
        self._avg[:] = 0.0
        self._evt[:] = (0.0, self._bsc / 5.0, self._bsc * 2.0 / 5.0, 0.0)


def estimate_fsk_shift(
    iq: np.ndarray, sample_rate: float, *, nominal_hz: float = 170.0
) -> tuple[float, float]:
    """Estimate (center_hz, shift_hz) of a 2-FSK signal from its two tone peaks.

    Averages a narrowband PSD, picks the strongest tone and the strongest other
    tone at least 50 Hz away; their midpoint is the center and their spacing the
    shift. Falls back to ``nominal_hz`` when the estimate is implausible.
    """
    z = np.ascontiguousarray(iq, dtype=np.complex64)
    if len(z) < 256:
        return 0.0, nominal_hz
    n_fft = min(4096, 1 << int(np.log2(len(z))))  # largest power of 2 that fits
    win = np.hanning(n_fft)
    acc = np.zeros(n_fft)
    count = 0
    for i in range(0, len(z) - n_fft + 1, n_fft // 2):
        acc += np.abs(np.fft.fftshift(np.fft.fft(z[i : i + n_fft] * win))) ** 2
        count += 1
    acc /= count
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / sample_rate))
    band = np.abs(freqs) < min(sample_rate * 0.45, 1000.0)
    fb, pb = freqs[band], acc[band]
    order = np.argsort(pb)[::-1]
    peak = fb[order[0]]
    other = [fb[k] for k in order if abs(fb[k] - peak) > 50.0]
    if not other:
        return 0.0, nominal_hz
    partner = other[0]
    center = (peak + partner) / 2.0
    shift = abs(peak - partner)
    if not 50.0 <= shift <= 1000.0:
        return center, nominal_hz
    return center, shift
