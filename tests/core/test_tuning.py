from __future__ import annotations

from tsdr.core.tuning import (
    STEP_LADDER,
    bandwidth_step,
    resolve_auto_step,
    snap_step_value,
    snap_to_grid,
)


def test_mode_steps() -> None:
    assert resolve_auto_step("CW", 14_000_000) == 50
    assert resolve_auto_step("USB", 14_000_000) == 100
    assert resolve_auto_step("LSB", 14_000_000) == 100
    assert resolve_auto_step("AM", 14_000_000) == 1_000
    assert resolve_auto_step("NFM", 145_500_000) == 12_500
    assert resolve_auto_step("WFM", 95_500_000) == 100_000


def test_raw_table() -> None:
    # LF
    assert resolve_auto_step("RAW", 200_000) == 100
    # MW/SW
    assert resolve_auto_step("RAW", 1_500_000) == 1_000
    assert resolve_auto_step("RAW", 14_175_000) == 1_000
    # VHF low
    assert resolve_auto_step("RAW", 60_000_000) == 12_500
    # FM broadcast
    assert resolve_auto_step("RAW", 95_500_000) == 100_000
    # Air band
    assert resolve_auto_step("RAW", 120_000_000) == 25_000
    # VHF/UHF NFM
    assert resolve_auto_step("RAW", 145_500_000) == 12_500
    assert resolve_auto_step("RAW", 433_000_000) == 12_500
    # L/S band
    assert resolve_auto_step("RAW", 1_500_000_000) == 100_000
    # Microwave
    assert resolve_auto_step("RAW", 10_000_000_000) == 1_000_000


def test_raw_table_boundary() -> None:
    # The boundary values themselves fall into the *next* bucket (strict <).
    assert resolve_auto_step("RAW", 500_000) == 1_000
    assert resolve_auto_step("RAW", 88_000_000) == 100_000


def test_snap_to_grid_on_grid() -> None:
    # Already on grid: one step forward = +step, backward = -step.
    assert snap_to_grid(14_175_000, 100, 1) == 14_175_100
    assert snap_to_grid(14_175_000, 100, -1) == 14_174_900


def test_snap_to_grid_off_grid() -> None:
    # Off-grid current snaps onto nearest grid then moves one step in direction.
    # 14_175_123 rounded to nearest 100 = 14_175_100; +1 step = 14_175_200.
    assert snap_to_grid(14_175_123, 100, 1) == 14_175_200
    assert snap_to_grid(14_175_123, 100, -1) == 14_175_000


def test_snap_step_value_powers() -> None:
    # Largest 1/2/5×10^n that fits.
    assert snap_step_value(300) == 200
    assert snap_step_value(1_250) == 1_000
    assert snap_step_value(20_000) == 20_000
    assert snap_step_value(1.0) == 1


def test_snap_step_value_floor() -> None:
    assert snap_step_value(0) == 1
    assert snap_step_value(-5) == 1


def test_bandwidth_step_proportional() -> None:
    # ~20% of current, snapped to 1/2/5×10ⁿ.
    assert bandwidth_step(3000) == 500  # 600 -> snap to 500
    assert bandwidth_step(12_500) == 2_000  # 2500 -> 2000
    assert bandwidth_step(200_000) == 20_000  # 40_000 -> 20_000
    assert bandwidth_step(10_000) == 2_000  # 2000 exact -> 2000


def test_bandwidth_step_floor() -> None:
    # Tiny current values floor at 10 Hz target → 10.
    assert bandwidth_step(50) == 10
    assert bandwidth_step(0) == 10


def test_step_ladder_has_auto_first() -> None:
    assert STEP_LADDER[0] is None
    assert all(step is None or step > 0 for step in STEP_LADDER)
