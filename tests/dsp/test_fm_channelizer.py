from __future__ import annotations

import numpy as np

from tsdr.radio.dsp import FMChannelizer

_FS = 240_000.0
_DEV = 4800.0
_TARGET = 32_000.0


def _iq(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.3).astype(np.complex64)


def test_rejects_signals_that_would_fold_onto_the_channel() -> None:
    """A tone one decimated rate away used to fold onto the channel through the
    single full-rate 101-tap filter; the multistage decimator stops it."""
    fs = 2.4e6
    ch = FMChannelizer(fs, _DEV, target_rate=_TARGET)
    t = np.arange(240_000) / fs

    def power(offset: float) -> float:
        dec = FMChannelizer(fs, _DEV, target_rate=_TARGET)._decimator
        y = dec.process(np.exp(2j * np.pi * offset * t).astype(np.complex64))
        return float(np.mean(np.abs(y[len(y) // 2 :]) ** 2))

    assert 10 * np.log10(power(ch.audio_rate + 1_000) / power(1_000)) < -80.0


def test_selects_the_channel_inside_the_decimated_band() -> None:
    """The decimator passes everything up to its output Nyquist. A station
    15 kHz from a DMR channel (5832 Hz half-width, 48 kS/s, decimation 1)
    reached the discriminator at full strength (66 dB down before)."""
    fs = 48_000.0
    t = np.arange(48_000) / fs

    def power(offset: float) -> float:
        ch = FMChannelizer(fs, 1944.0, target_rate=48_000.0, channel_cutoff=5832.0)
        y = ch._select(np.exp(2j * np.pi * offset * t).astype(np.complex64))
        return float(np.mean(np.abs(y[len(y) // 2 :]) ** 2))

    assert 10 * np.log10(power(15_000) / power(1_000)) < -60.0


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
