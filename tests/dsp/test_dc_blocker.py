import math

import numpy as np

from tsdr.radio.dsp import DCBlocker


def test_dc_input_decays_to_zero():
    fs = 48000.0
    cutoff = 16.0
    blocker = DCBlocker(sample_rate=fs, cutoff_hz=cutoff)
    x = np.ones(int(fs * 1.0), dtype=np.float32)  # 1 s of DC
    y = blocker.process(x)
    # After many time constants the output should be close to 0.
    assert abs(float(y[-1])) < 0.01


def test_decay_time_constant_matches_cutoff():
    fs = 48000.0
    cutoff = 16.0
    blocker = DCBlocker(sample_rate=fs, cutoff_hz=cutoff)
    x = np.ones(int(fs * 0.5), dtype=np.float32)
    y = blocker.process(x)

    # Discrete one-pole rate; theoretical 1/e settling time is 1/rate samples.
    rate = 2.0 * math.pi * cutoff / fs
    n_one_tc = int(round(1.0 / rate))
    # At one TC the unit-step response of the HPF is ~1/e.
    assert abs(float(y[n_one_tc]) - math.exp(-1.0)) < 0.05


def test_passes_high_frequency():
    fs = 48000.0
    cutoff = 16.0
    blocker = DCBlocker(sample_rate=fs, cutoff_hz=cutoff)
    f = 1000.0  # well above cutoff
    n = int(fs * 0.5)
    t = np.arange(n, dtype=np.float64) / fs
    x = np.sin(2.0 * np.pi * f * t).astype(np.float32)
    y = blocker.process(x)

    # Skip the startup transient.
    skip = n // 4
    rms_in = float(np.sqrt(np.mean(np.square(x[skip:]))))
    rms_out = float(np.sqrt(np.mean(np.square(y[skip:]))))
    # < 0.5 dB attenuation at 1 kHz with a 16 Hz HPF.
    assert 20.0 * math.log10(rms_out / rms_in) > -0.5


def test_chunk_continuity_bit_exact():
    """Splitting the input must not change the output (proves stateful continuity)."""
    fs = 48000.0
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(10_000).astype(np.float32) + 0.5).astype(np.float32)

    full = DCBlocker(sample_rate=fs)
    split = DCBlocker(sample_rate=fs)

    y_full = full.process(x)

    chunks = np.array_split(x, 7)
    y_split = np.concatenate([split.process(np.ascontiguousarray(c)) for c in chunks])

    np.testing.assert_array_equal(y_full, y_split)


def test_reset_zeros_state():
    blocker = DCBlocker(sample_rate=48000.0)
    x = np.ones(2000, dtype=np.float32)
    blocker.process(x)
    assert blocker._state[0] != 0.0
    blocker.reset()
    assert blocker._state[0] == 0.0
