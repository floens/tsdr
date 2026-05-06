"""MBE model parameters carried across decoder stages.

Indices 1..L are the valid harmonic range; index 0 is a sentinel used
by the log2_ml interpolation in ``pyambe.params_decode._reconstruct_log2ml``.
Arrays are float32 for single-precision math.

Array size: we allocate 58 slots so the
``prev_mp.log2_ml[intkl[l] + 1]`` read in the Part 1 interpolation
stays in-bounds when ``intkl[l] == 56``. A 57-slot array would read
past the end in that case; the extra zero slot makes the lookup safe
and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32
PARAMS_LEN = 58


def _zeros_f32() -> np.ndarray:
    return np.zeros(PARAMS_LEN, dtype=F32)


def _zeros_i32() -> np.ndarray:
    return np.zeros(PARAMS_LEN, dtype=np.int32)


def _psi_init() -> np.ndarray:
    return np.full(PARAMS_LEN, np.pi / 2, dtype=F32)


@dataclass
class MbeParams:
    """Per-frame MBE model parameters.

    Defaults match ``mbe_initMbeParms``: w0 ≈ 0.094 rad/sample (the
    startup fundamental before any voice arrives), L = 30 harmonics,
    all unvoiced, PSIl pre-loaded with π/2 so the first synthesized
    frame has a reasonable starting phase.
    """

    w0: float = 0.09378
    L: int = 30
    gamma: float = 0.0
    repeat: int = 0
    Vl: np.ndarray = field(default_factory=_zeros_i32)
    Ml: np.ndarray = field(default_factory=_zeros_f32)
    log2_ml: np.ndarray = field(default_factory=_zeros_f32)
    PHIl: np.ndarray = field(default_factory=_zeros_f32)
    PSIl: np.ndarray = field(default_factory=_psi_init)

    def copy_from(self, other: MbeParams) -> None:
        self.w0 = other.w0
        self.L = other.L
        self.gamma = other.gamma
        self.repeat = other.repeat
        self.Vl[:] = other.Vl
        self.Ml[:] = other.Ml
        self.log2_ml[:] = other.log2_ml
        self.PHIl[:] = other.PHIl
        self.PSIl[:] = other.PSIl

    def use_last(self, prev: MbeParams) -> None:
        """Overwrite self with prev. Called on high-errs2 frames to
        reuse the last good parameter set instead of the freshly
        (but error-prone) decoded values."""
        self.copy_from(prev)

    def reset(self) -> None:
        self.w0 = 0.09378
        self.L = 30
        self.gamma = 0.0
        self.repeat = 0
        self.Vl.fill(0)
        self.Ml.fill(0.0)
        self.log2_ml.fill(0.0)
        self.PHIl.fill(0.0)
        self.PSIl.fill(np.pi / 2)
