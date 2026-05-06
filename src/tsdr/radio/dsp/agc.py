from __future__ import annotations

import math

import numpy as np

from tsdr.radio.dsp._kernels import _agc_f32


class AGC:
    """Sample-by-sample AGC with attack/decay smoothing.

    Tracks ``|x|`` with an asymmetric one-pole envelope follower; gain is
    ``min(setpoint / amp, max_gain)``. Time constants are specified as the
    1/e settling time in milliseconds. State (the running amplitude estimate)
    persists across ``process()`` calls.
    """

    def __init__(
        self,
        sample_rate: float,
        attack_ms: float = 5.0,
        decay_ms: float = 200.0,
        setpoint: float = 0.5,
        max_gain: float = 1000.0,
    ):
        self._setpoint = float(setpoint)
        self._max_gain = float(max_gain)
        self._attack = float(1.0 - math.exp(-1.0 / (attack_ms * 1e-3 * sample_rate)))
        self._decay = float(1.0 - math.exp(-1.0 / (decay_ms * 1e-3 * sample_rate)))
        # Initialise amp at the setpoint so startup gain is ~unity.
        self._state = np.array([self._setpoint], dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        result: np.ndarray = _agc_f32(
            np.ascontiguousarray(x, dtype=np.float32),
            self._state,
            self._attack,
            self._decay,
            self._setpoint,
            self._max_gain,
        )
        return result

    def reset(self) -> None:
        self._state[0] = self._setpoint
