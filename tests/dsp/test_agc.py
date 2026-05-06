import math

import numpy as np

from tsdr.radio.dsp import AGC


def test_converges_to_setpoint():
    fs = 48000
    setpoint = 0.5
    agc = AGC(sample_rate=fs, attack_ms=5.0, decay_ms=200.0, setpoint=setpoint)
    x = np.full(fs, 2.0, dtype=np.float32)  # 1 s of constant level above setpoint
    y = agc.process(x)
    # After many attack time constants the output magnitude tracks setpoint.
    assert abs(abs(y[-1]) - setpoint) < 0.01


def test_attack_speed():
    fs = 48000
    attack_ms = 5.0
    agc = AGC(sample_rate=fs, attack_ms=attack_ms, decay_ms=200.0, setpoint=0.5)
    n_quiet = int(0.05 * fs)
    n_loud = int(0.1 * fs)
    x = np.concatenate(
        [
            np.full(n_quiet, 0.5, dtype=np.float32),
            np.full(n_loud, 5.0, dtype=np.float32),
        ]
    )
    y = agc.process(x)
    # After roughly 5 attack TCs, the loud section should be near setpoint.
    one_tc = max(1, int(attack_ms * 1e-3 * fs))
    settled = abs(y[n_quiet + 5 * one_tc])
    assert abs(settled - 0.5) < 0.05


def test_decay_is_slower_than_attack():
    fs = 48000
    attack_ms = 5.0
    decay_ms = 200.0
    agc = AGC(sample_rate=fs, attack_ms=attack_ms, decay_ms=decay_ms, setpoint=0.5)
    # Loud burst then silence; with slow decay the gain should rise gradually,
    # not snap instantly. We probe by feeding a small probe tone after silence
    # and checking that gain has not yet fully recovered.
    n_loud = int(0.1 * fs)
    n_silent = max(1, int(decay_ms * 1e-3 * fs / 4))  # ~1/4 of decay TC
    x = np.concatenate(
        [
            np.full(n_loud, 5.0, dtype=np.float32),
            np.full(n_silent, 0.001, dtype=np.float32),
        ]
    )
    y = agc.process(x)
    # During silence right after a loud burst, gain should still be ~unity-ish
    # (because the envelope estimate decays slowly). So output for tiny input
    # remains tiny.
    silent_section = y[n_loud : n_loud + n_silent]
    assert float(np.max(np.abs(silent_section))) < 0.5


def test_zero_input_no_nan_or_inf():
    agc = AGC(sample_rate=48000)
    x = np.zeros(1000, dtype=np.float32)
    y = agc.process(x)
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))
    # Zero input -> zero output regardless of gain ceiling.
    assert float(np.max(np.abs(y))) == 0.0


def test_chunk_continuity_bit_exact():
    fs = 48000
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(10_000).astype(np.float32) * 0.3).astype(np.float32)

    full = AGC(sample_rate=fs)
    split = AGC(sample_rate=fs)

    y_full = full.process(x)

    chunks = np.array_split(x, 7)
    y_split = np.concatenate([split.process(np.ascontiguousarray(c)) for c in chunks])

    np.testing.assert_array_equal(y_full, y_split)


def test_reset_restores_initial_amp():
    setpoint = 0.5
    agc = AGC(sample_rate=48000, setpoint=setpoint)
    initial = float(agc._state[0])
    assert math.isclose(initial, setpoint)
    x = np.full(1000, 5.0, dtype=np.float32)
    agc.process(x)
    assert agc._state[0] != initial
    agc.reset()
    assert math.isclose(float(agc._state[0]), initial)
