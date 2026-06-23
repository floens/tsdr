from __future__ import annotations

import numpy as np

from tsdr.radio.dsp import DPLLBitSync

_SPS = 25


def _nrz(bits: np.ndarray, pad: int) -> np.ndarray:
    sig = np.repeat(bits.astype(np.float32) * 2 - 1, _SPS)
    return np.concatenate([np.zeros(pad, np.float32), sig])


def _contains(haystack: np.ndarray, needle: np.ndarray) -> bool:
    h = "".join(map(str, haystack.tolist()))
    n = "".join(map(str, needle.tolist()))
    return n in h


def test_locks_regardless_of_initial_phase() -> None:
    # The MM bug we fixed: timing must lock for any bit phase within the period,
    # not just when the eye happens to sit near the loop's starting phase.
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 200).astype(np.uint8)
    for pad in (0, 6, 13, 19, 24):
        rec = DPLLBitSync(float(_SPS)).process(_nrz(bits, pad))
        assert _contains(rec, bits[2:-2]), f"phase pad={pad}: bit pattern not recovered"


def test_chunk_invariance() -> None:
    rng = np.random.default_rng(1)
    sig = _nrz(rng.integers(0, 2, 300).astype(np.uint8), pad=7)
    whole = DPLLBitSync(float(_SPS)).process(sig)
    d = DPLLBitSync(float(_SPS))
    streamed = np.concatenate([d.process(sig[i : i + 333]) for i in range(0, len(sig), 333)])
    assert np.array_equal(whole, streamed)
