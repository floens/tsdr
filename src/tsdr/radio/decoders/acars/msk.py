"""Coherent MSK demodulator for ACARS.

ACARS is 2400-baud MSK carried as AM. This runs on the real AM-envelope audio at
12 kHz (5 samples/symbol): a carrier NCO mixes the audio to complex baseband in a
short circular buffer; a bit clock derived from the carrier (threshold
`2*pi*center/baud`) fires one symbol per period; a half-sine matched filter over
the 2-bit window is correlated and normalised; the MSK offset-QPSK arms are read
alternately (even=I, odd=Q) with a 2-symbol sign flip; and a PI-controller PLL
tracks carrier/timing jointly. Output is one soft bit per symbol (sign =
mark/space); per-burst polarity is the framer's concern.

Kernel-local to this decoder (like `tetra/_kernels.py`, `wsjt/sync.py`): the
carrier-derived clock assumes the MSK relationship baud = 2*(center-space), which
holds for ACARS, so this is not a general-purpose IQ MSK/GMSK demod.
"""

from __future__ import annotations

import numba as nb
import numpy as np

_MFLTOVER = 240  # matched-filter fractional-timing oversample factor


@nb.njit(cache=True)
def _msk_kernel(x, h, base, thr, ki, kp, p, dphi, df, clk, msks, idx, inb):
    n = x.shape[0]
    bitlen = inb.shape[0]
    out = np.empty(n // 5 + 16, dtype=np.float32)
    n_out = 0
    two_pi = 2.0 * np.pi
    for k in range(n):
        s = base + dphi
        clk += s
        if clk > thr:
            clk -= thr
            o = int(_MFLTOVER * (clk / s))
            if o > _MFLTOVER:
                o = _MFLTOVER
            v = 0.0 + 0.0j
            oo = o
            for j in range(bitlen):
                v += h[oo] * inb[(j + idx) % bitlen]
                oo += _MFLTOVER
            v = v / (abs(v) + 1e-8)
            if msks & 1:
                vo = v.imag
                perr = -v.real if vo >= 0.0 else v.real
            else:
                vo = v.real
                perr = v.imag if vo >= 0.0 else -v.imag
            out[n_out] = -vo if (msks & 2) else vo
            n_out += 1
            msks += 1
            df += ki * perr
            dphi = df + kp * perr
        p += s
        if p >= two_pi:
            p -= two_pi
        inb[idx] = x[k] * (np.cos(p) - 1j * np.sin(p))
        idx = (idx + 1) % bitlen
    return out[:n_out], p, dphi, df, clk, msks, idx


class MSKDemod:
    def __init__(
        self,
        sample_rate: float = 12_000.0,
        baud: float = 2_400.0,
        *,
        center_hz: float = 1_800.0,
        space_hz: float = 1_200.0,
    ) -> None:
        self._bitlen = int(np.ceil(sample_rate / space_hz))  # matched-filter window
        n = self._bitlen * _MFLTOVER + 1
        self._h = np.sin(np.pi * space_hz * np.arange(n) / sample_rate / _MFLTOVER).astype(
            np.float64
        )
        self._base = 2.0 * np.pi * center_hz / sample_rate
        self._thr = 2.0 * np.pi * center_hz / baud
        # PLL gains scale with the window length.
        self._ki = 71e-7 / self._bitlen
        self._kp = 60e-3 / self._bitlen
        self.reset()

    def reset(self) -> None:
        self._p = 0.0
        self._dphi = 0.0
        self._df = 0.0
        self._clk = 0.0
        self._msks = 0
        self._idx = 0
        self._inb = np.zeros(self._bitlen, dtype=np.complex128)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Real AM-envelope audio -> float32 soft bits (one per symbol)."""
        if len(audio) == 0:
            return np.empty(0, dtype=np.float32)
        x = np.ascontiguousarray(audio)
        out, self._p, self._dphi, self._df, self._clk, self._msks, self._idx = _msk_kernel(
            x,
            self._h,
            self._base,
            self._thr,
            self._ki,
            self._kp,
            self._p,
            self._dphi,
            self._df,
            self._clk,
            self._msks,
            self._idx,
            self._inb,
        )
        soft: np.ndarray = out
        return soft
