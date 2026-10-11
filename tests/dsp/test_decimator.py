"""Multistage decimation: stage planning, alias rejection, streaming, kernels."""

import numpy as np
import pytest

from tsdr.radio.dsp._kernels import fir_decim_c64_into, fir_decim_c64_split_into
from tsdr.radio.dsp.decimator import MultistageDecimator, plan_stages, smooth_decimation


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(41.67, 40), (208.3, 200), (50.0, 50), (13.3, 12), (833.3, 810), (0.5, 1), (1.0, 1)],
)
def test_smooth_decimation(ratio: float, expected: int) -> None:
    assert smooth_decimation(ratio) == expected


def test_plan_multiplies_out_and_splits_large_factors() -> None:
    stages = plan_stages(10e6, 200, 22.5e3)
    assert int(np.prod(stages)) == 200
    assert len(stages) >= 2


def test_plan_rejects_a_passband_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        plan_stages(2e6, 40, 30e3)  # 50 kS/s out: passband must stay under 25 kHz


def _gain_db(dec: MultistageDecimator, fs: float, f: np.ndarray) -> np.ndarray:
    """Tone gain through the cascade: each stage sees the tone at its folded frequency."""
    gain = np.ones_like(f)
    rate = fs
    ff = f.copy()
    for stage in dec._filters:
        taps = stage._flipped[::-1].astype(np.float64)
        n = np.arange(len(taps))
        gain *= np.abs(np.exp(-2j * np.pi * np.outer(ff, n) / rate) @ taps)
        rate /= stage._decimation
        ff = (ff + rate / 2) % rate - rate / 2
    return 20 * np.log10(gain + 1e-15), ff


@pytest.mark.parametrize(("fs", "decimation"), [(2e6, 40), (2.4e6, 50), (10e6, 200), (20e6, 405)])
def test_folded_signals_are_80_db_down_and_passband_flat(fs: float, decimation: int) -> None:
    out_rate = fs / decimation
    passband = 0.45 * out_rate
    dec = MultistageDecimator(fs, decimation, passband)
    f = np.linspace(-fs / 2, fs / 2, 200_001)
    db, folded = _gain_db(dec, fs, f)
    lands_in_passband = np.abs(folded) <= passband
    in_passband = np.abs(f) <= passband
    assert db[lands_in_passband & ~in_passband].max() < -80.0
    assert np.ptp(db[in_passband]) < 0.1


def test_streaming_matches_one_shot() -> None:
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(123_457) + 1j * rng.standard_normal(123_457)).astype(np.complex64)
    whole = MultistageDecimator(10e6, 200, 22.5e3).process(x).copy()
    chunked = MultistageDecimator(10e6, 200, 22.5e3)
    parts = []
    for a, b in ((0, 1_000), (1_000, 1_017), (1_017, 60_000), (60_000, 123_457)):
        parts.append(chunked.process(x[a:b]).copy())
    np.testing.assert_allclose(np.concatenate(parts), whole, rtol=1e-5, atol=1e-6)


def test_split_kernel_matches_interleaved_kernel() -> None:
    rng = np.random.default_rng(4)
    taps = rng.standard_normal(47).astype(np.float32)
    flipped = np.ascontiguousarray(taps[::-1])
    m = 5
    hist_a = np.zeros(46, np.complex64)
    hist_b = np.zeros(46, np.complex64)
    phase_a = phase_b = 0
    for n in (1_000, 7, 3_333):
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        pad = np.empty(n + 46, np.complex64)
        pad_re = np.empty(n + 46, np.float32)
        pad_im = np.empty(n + 46, np.float32)
        y_a = np.empty(n // m + 2, np.complex64)
        y_b = np.empty(n // m + 2, np.complex64)
        n_a, phase_a = fir_decim_c64_into(x, flipped, m, hist_a, phase_a, pad, y_a)
        n_b, phase_b = fir_decim_c64_split_into(x, flipped, m, hist_b, phase_b, pad_re, pad_im, y_b)
        assert (n_a, phase_a) == (n_b, phase_b)
        np.testing.assert_allclose(y_b[:n_b], y_a[:n_a], rtol=1e-4, atol=1e-5)
