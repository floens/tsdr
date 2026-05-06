import numpy as np

from tsdr.radio.dsp import SquelchGate
from tsdr.radio.dsp.squelch import _build_envelope

AUDIO_RATE = 48_000.0


def _gate(
    threshold_db: float = -50.0,
    hang_ms: float = 500.0,
    enabled: bool = True,
    ramp_ms: float = 10.0,
) -> SquelchGate:
    gate = SquelchGate(audio_rate=AUDIO_RATE, ramp_ms=ramp_ms)
    gate.configure(enabled=enabled, threshold_db=threshold_db, hang_ms=hang_ms)
    return gate


def _pump(gate: SquelchGate, power_db: float, chunks: int, n_per_chunk: int = 480) -> None:
    for _ in range(chunks):
        gate.process(power_db, n_per_chunk)


def test_disabled_returns_none():
    gate = _gate(enabled=False)
    assert gate.process(-10.0, 480) is None
    assert not gate.is_open


def test_opens_on_strong_signal():
    gate = _gate(threshold_db=-50.0)
    _pump(gate, power_db=-20.0, chunks=30)
    assert gate.is_open


def test_stays_closed_on_weak_signal():
    gate = _gate(threshold_db=-50.0)
    _pump(gate, power_db=-80.0, chunks=30)
    assert not gate.is_open


def test_hysteresis_prevents_chatter_near_threshold():
    """Signal at the exact threshold should not open the gate (needs +3 dB)."""
    gate = _gate(threshold_db=-50.0)
    _pump(gate, power_db=-50.0, chunks=80)
    assert not gate.is_open


def test_hysteresis_needs_3db_above_to_open():
    gate = _gate(threshold_db=-50.0)
    _pump(gate, power_db=-46.0, chunks=80)
    assert gate.is_open


def test_hang_delays_closing():
    gate = _gate(threshold_db=-50.0, hang_ms=200.0)
    _pump(gate, power_db=-20.0, chunks=50)
    assert gate.is_open

    n_per_chunk = 480
    chunks_before_close = 0
    for _ in range(400):
        gate.process(-120.0, n_per_chunk)
        chunks_before_close += 1
        if not gate.is_open:
            break

    elapsed_ms = chunks_before_close * n_per_chunk / AUDIO_RATE * 1000.0
    assert 150.0 < elapsed_ms < 800.0, f"closed after {elapsed_ms:.1f} ms"


def test_opening_ramps_up_smoothly():
    """When the gate opens, gain should rise to 1.0, not snap."""
    gate = _gate(threshold_db=-50.0, ramp_ms=10.0, hang_ms=500.0)
    _pump(gate, power_db=-120.0, chunks=10, n_per_chunk=480)
    assert gate._gain == 0.0

    ramped_smoothly = False
    reached_one = False
    for _ in range(40):
        envelope = gate.process(-10.0, 480)
        assert envelope is not None
        if not reached_one and gate.is_open and 0.0 < envelope[0] < 1.0:
            ramped_smoothly = True
        if envelope[-1] == 1.0:
            reached_one = True
            break

    assert ramped_smoothly
    assert reached_one


def test_closing_ramps_down_smoothly():
    gate = _gate(threshold_db=-50.0, hang_ms=0.0, ramp_ms=10.0)
    _pump(gate, power_db=-10.0, chunks=30)
    assert gate.is_open
    assert gate._gain == 1.0

    saw_partial_gain = False
    reached_zero = False
    for _ in range(100):
        envelope = gate.process(-120.0, 480)
        assert envelope is not None
        if 0.0 < envelope[-1] < 1.0 or 0.0 < envelope[0] < 1.0:
            saw_partial_gain = True
        if envelope[-1] == 0.0 and envelope[0] < 0.1:
            reached_zero = True
            break

    assert saw_partial_gain
    assert reached_zero


def test_reset_clears_state():
    gate = _gate(threshold_db=-50.0)
    _pump(gate, power_db=-10.0, chunks=30)
    assert gate.is_open

    gate.reset()
    assert not gate.is_open
    assert gate.power_ema_db == -120.0


def test_threshold_change_midstream():
    gate = _gate(threshold_db=-30.0)
    _pump(gate, power_db=-40.0, chunks=80)
    assert not gate.is_open

    gate.configure(enabled=True, threshold_db=-60.0, hang_ms=500.0)
    _pump(gate, power_db=-40.0, chunks=30)
    assert gate.is_open


def test_envelope_dtype_and_length():
    gate = _gate()
    envelope = gate.process(-10.0, 1024)
    assert envelope is not None
    assert envelope.dtype == np.float32
    assert len(envelope) == 1024


# _build_envelope pure-function tests


def test_envelope_flat_when_start_equals_target():
    env = _build_envelope(start=1.0, target=1.0, n=100, ramp_samples=480)
    assert np.all(env == 1.0)

    env = _build_envelope(start=0.0, target=0.0, n=100, ramp_samples=480)
    assert np.all(env == 0.0)


def test_envelope_linear_ramp_up():
    env = _build_envelope(start=0.0, target=1.0, n=480, ramp_samples=480)
    assert env[0] > 0.0
    assert env[-1] == 1.0
    diffs = np.diff(env[:-1])
    assert np.allclose(diffs, diffs[0], atol=1e-5)


def test_envelope_linear_ramp_down():
    env = _build_envelope(start=1.0, target=0.0, n=480, ramp_samples=480)
    assert env[0] < 1.0
    assert env[-1] == 0.0


def test_envelope_clamps_at_target():
    env = _build_envelope(start=0.0, target=1.0, n=1000, ramp_samples=480)
    assert env[-1] == 1.0
    assert env[999] == 1.0
    assert env[480] == 1.0


def test_envelope_partial_ramp_across_chunks():
    """Step rate stays 1/ramp_samples per sample regardless of chunk size."""
    env1 = _build_envelope(start=0.0, target=1.0, n=200, ramp_samples=960)
    assert 0.0 < env1[-1] < 1.0
    env2 = _build_envelope(start=float(env1[-1]), target=1.0, n=200, ramp_samples=960)
    assert env2[0] > env1[-1]
    assert env2[-1] > env1[-1]
    # Constant step rate: 1/960 per sample, so after 400 total samples gain ≈ 400/960.
    assert abs(float(env2[-1]) - 400.0 / 960.0) < 1e-4
