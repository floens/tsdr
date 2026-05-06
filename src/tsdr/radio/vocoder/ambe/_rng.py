"""Xorshift32 RNG for deterministic unvoiced synthesis.

The sequence is::

    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;

``rand()`` returns ``(x & 0x7fffffff) / 0x7fffffff`` as a float32.
The division is float32, so we cast through ``np.float32`` to match.

Used for unvoiced phase noise in synthesis and seeded per-frame in
tests for reproducible output.
"""

from __future__ import annotations

import numpy as np


class Xorshift32:
    """Seedable xorshift32 matching the patched libmbe."""

    __slots__ = ("state",)

    def __init__(self, seed: int = 1) -> None:
        self.state = seed & 0xFFFFFFFF if seed else 1

    def seed(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF if seed else 1

    def u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x

    def rand(self) -> float:
        """Uniform float in [0, 1]. Cast through float32 to match the
        C `(float)(x & 0x7fffffff) / (float)0x7fffffff` exactly."""
        x = self.u32() & 0x7FFFFFFF
        return float(np.float32(x) / np.float32(0x7FFFFFFF))

    def rand_phase(self) -> float:
        """Uniform float in [-π, π). Matches `mbe_rand_phase` in patched libmbe."""
        two_pi = np.float32(np.pi * 2.0)
        pi = np.float32(np.pi)
        return float(np.float32(self.rand()) * two_pi - pi)
