from __future__ import annotations

import math

import numpy as np

from tsdr.radio.dsp._kernels import _dc_blocker_f32


class DCBlocker:
    """Single-pole IIR DC blocker.

    y[n] = x[n] - offset
    offset += y[n] * rate

    The cutoff is approximately ``rate * fs / (2*pi)``. State (the running DC
    offset estimate) persists across ``process()`` calls.
    """

    def __init__(self, sample_rate: float, cutoff_hz: float = 16.0):
        self._rate = float(2.0 * math.pi * cutoff_hz / sample_rate)
        self._state = np.zeros(1, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = _dc_blocker_f32(
            np.ascontiguousarray(x, dtype=np.float32),
            self._state,
            self._rate,
        )
        return result

    def reset(self) -> None:
        self._state[0] = 0.0
