"""Digital-PLL bit-clock recovery for NRZ soft-decision signals.

Recovers one hard bit per symbol period from a baseband NRZ decision (e.g. the
AFSK1200 ``|mark| - |space|`` signal): samples at the eye centre and nudges the
phase so transitions land mid-period. Unlike `MuellerMuller` (whose timing-error
detector is tuned for shaped PSK symbols and can false-lock on the AFSK decision
shape) it locks reliably regardless of the initial bit phase, which is what
async HDLC packets need. State persists across `process()` calls.
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.dsp._kernels import _dpll_bitsync


class DPLLBitSync:
    def __init__(self, sps: float, k: float = 0.1) -> None:
        self._inc = 1.0 / sps
        self._k = k
        self._phase = 0.0
        self._prev = 0.0

    def process(self, decision: np.ndarray) -> np.ndarray:
        if len(decision) == 0:
            return np.empty(0, dtype=np.uint8)
        bits, self._phase, self._prev = _dpll_bitsync(
            np.ascontiguousarray(decision, dtype=np.float32),
            self._inc,
            self._k,
            self._phase,
            self._prev,
        )
        result: np.ndarray = bits
        return result

    def reset(self) -> None:
        self._phase = 0.0
        self._prev = 0.0
