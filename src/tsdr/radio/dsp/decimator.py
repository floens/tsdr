"""Multistage decimation: a chain of short decimating FIRs.

One FIR that decimates 10 MS/s to 50 kS/s needs a transition band a fraction
of a percent of its input rate, i.e. thousands of taps. In a cascade each
stage only has to stop what would fold into the final passband, so the early
stages run at a high rate with a wide transition band and few taps, and the
sharp edge is cut by the last stage at a low rate, where taps are cheap. Each
stage computes only the samples it keeps (StreamingDecimFilter).
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np

from tsdr.radio.dsp._kernels import StreamingDecimFilter
from tsdr.radio.dsp.filters import firwin

# Design rejection of everything that would fold into the passband. Kaiser's
# length estimate runs a little short, so the cascade's measured worst case
# is 2-3 dB under this: at least 81 dB from 1 to 20 MS/s.
STOPBAND_DB = 84.0

# Largest prime factor allowed in a chosen decimation: a large prime factor
# forces one stage with a sharp edge at a high rate.
_MAX_PRIME = 7


def smooth_decimation(ratio: float) -> int:
    """Largest integer <= `ratio` (at least 1) whose prime factors are all <= 7."""
    for d in range(max(1, int(ratio)), 1, -1):
        rest = d
        for p in (2, 3, 5, 7):
            while rest % p == 0:
                rest //= p
        if rest == 1:
            return d
    return 1


def _kaiser_taps(fs: float, transition_hz: float, atten_db: float) -> int:
    """Kaiser's estimate of the FIR length for a transition band at rate `fs`."""
    return math.ceil((atten_db - 7.95) / (14.36 * transition_hz / fs)) + 1


def _kaiser_beta(atten_db: float) -> float:
    return 0.1102 * (atten_db - 8.7)


def plan_stages(
    sample_rate: float, decimation: int, passband_hz: float, atten_db: float = STOPBAND_DB
) -> tuple[int, ...]:
    """Stage factors (in order) that decimate by `decimation` with the fewest
    multiply-accumulates per second, keeping `passband_hz` free of aliases.

    A stage from rate fs_in to fs_out must pass the passband and stop from
    fs_out - passband up, where signals would fold onto it; what lies between
    is removed by the later stages.
    """
    out_rate = sample_rate / decimation
    if passband_hz >= out_rate / 2:
        raise ValueError(f"passband {passband_hz} Hz does not fit {out_rate} S/s")

    @cache
    def best(remaining: int) -> tuple[float, tuple[int, ...]]:
        if remaining == 1:
            return 0.0, ()
        fs_in = out_rate * remaining
        options = []
        for d in range(2, remaining + 1):
            if remaining % d:
                continue
            fs_out = fs_in / d
            taps = _kaiser_taps(fs_in, fs_out - 2 * passband_hz, atten_db)
            rest_cost, rest = best(remaining // d)
            options.append((taps * fs_out + rest_cost, (d, *rest)))
        return min(options)

    return best(decimation)[1]


class MultistageDecimator:
    """Decimate by `decimation` through the cascade `plan_stages` picks.

    `passband_hz` is the band (either side of 0 for complex input) that must
    stay free of folded signals. Same contract as StreamingDecimFilter:
    `process` returns an array owned by the decimator, valid until the next
    call.
    """

    def __init__(
        self,
        sample_rate: float,
        decimation: int,
        passband_hz: float,
        *,
        dtype: type = np.complex64,
        expected_input_size: int = 200_000,
        atten_db: float = STOPBAND_DB,
    ):
        self.decimation = decimation
        self.stages = plan_stages(sample_rate, decimation, passband_hz, atten_db)
        self.taps: list[int] = []
        self._filters: list[StreamingDecimFilter] = []
        fs_in = float(sample_rate)
        size = expected_input_size
        for d in self.stages:
            fs_out = fs_in / d
            stop = fs_out - passband_hz
            n = _kaiser_taps(fs_in, stop - passband_hz, atten_db)
            taps = firwin(
                n,
                (passband_hz + stop) / 2,
                fs=fs_in,
                window=("kaiser", _kaiser_beta(atten_db)),
            )
            self._filters.append(
                StreamingDecimFilter(taps, decimation=d, dtype=dtype, expected_input_size=size)
            )
            self.taps.append(n)
            fs_in = fs_out
            size = size // d + 1

    def process(self, x: np.ndarray) -> np.ndarray:
        for f in self._filters:
            x = f.process(x)
        return x

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
