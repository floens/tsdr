from __future__ import annotations

import numpy as np

from tsdr.radio.dsp import FMChannelizer, FMDiscriminator, StreamingFilter, firwin

_FS = 240_000.0
_DEV = 4800.0
_TARGET = 32_000.0


def _iq(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.3).astype(np.complex64)


def test_matches_manual_frontend() -> None:
    """FMChannelizer is exactly anti-alias + strided decimate + discriminate -- the
    front-end FLEX inlined before migration (guards that the migration is a no-op)."""
    iq = _iq(20_000, 0)
    out = FMChannelizer(_FS, _DEV, target_rate=_TARGET).process(iq)

    decim = max(1, round(_FS / _TARGET))
    rate = _FS / decim
    cutoff = min(rate * 0.45, _DEV * 2)
    aa = StreamingFilter(firwin(101, cutoff, fs=_FS), [1.0], dtype=np.complex64)
    fm = FMDiscriminator(rate, _DEV)
    ref = fm.process(aa.process(iq)[::decim])

    assert np.array_equal(out, ref)


def test_chunk_invariance() -> None:
    iq = _iq(50_000, 1)
    whole = FMChannelizer(_FS, _DEV, target_rate=_TARGET).process(iq)
    ch = FMChannelizer(_FS, _DEV, target_rate=_TARGET)
    streamed = np.concatenate([ch.process(iq[i : i + 997]) for i in range(0, len(iq), 997)])
    n = min(len(whole), len(streamed))
    assert abs(len(whole) - len(streamed)) <= 1
    assert np.allclose(whole[:n], streamed[:n], atol=1e-4)


def test_recovers_tone_frequency() -> None:
    # A constant-frequency offset tone discriminates to a constant level whose
    # sign follows the offset direction.
    fs, dev, target = 240_000.0, 4800.0, 32_000.0
    t = np.arange(40_000) / fs
    iq = np.exp(1j * 2 * np.pi * 2000.0 * t).astype(np.complex64)
    out = FMChannelizer(fs, dev, target_rate=target).process(iq)
    settled = out[200:]
    assert np.mean(settled) > 0  # positive offset -> positive discriminator output
    assert np.std(settled) < 0.05  # constant tone -> flat output
