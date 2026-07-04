from __future__ import annotations

import pytest

from tsdr.core.units import axis_si_prefix, parse_hz


def _label(interval: float, value: float, ref: float) -> str:
    divisor, suffix, decimals = axis_si_prefix(interval, ref)
    return f"{value / divisor:.{decimals}f}{suffix}"


def test_axis_hf_is_integer_khz() -> None:
    # 6.9 MHz HF with 10 kHz ticks reads as integer kHz, not 6.94M.
    assert _label(10e3, 6.94e6, 6.965e6) == "6940k"


def test_axis_hf_5k_step_integer_khz() -> None:
    assert _label(5e3, 7.035e6, 7.04e6) == "7035k"


def test_axis_fm_stays_mhz() -> None:
    # VHF stays in conventional MHz with decimals.
    assert _label(50e3, 100.05e6, 100.3e6) == "100.05M"


def test_axis_2m_stays_mhz() -> None:
    assert _label(50e3, 145.45e6, 145.6e6) == "145.45M"


def test_axis_wide_span_stays_mhz() -> None:
    assert _label(500e3, 97e6, 100e6) == "97.0M"


def test_axis_ghz() -> None:
    assert _label(5e6, 1.2e9, 1.22e9) == "1200M"


def test_axis_prefix_shared_across_ten_mhz_reference() -> None:
    # A single axis picks one prefix from its top magnitude, so a range that
    # crosses 1 MHz stays in kHz for every tick instead of mixing k and M.
    ref = 1.08e6
    assert _label(20e3, 0.96e6, ref) == "960k"
    assert _label(20e3, 1.0e6, ref) == "1000k"
    assert _label(20e3, 1.08e6, ref) == "1080k"


def test_axis_prefix_flips_to_mhz_at_10mhz() -> None:
    # 10000 kHz is the cutoff: at/above it the whole axis uses MHz.
    assert axis_si_prefix(20e3, 9.9e6)[1] == "k"
    assert axis_si_prefix(20e3, 10.1e6)[1] == "M"


def test_plain_integer() -> None:
    assert parse_hz("100100000") == 100_100_000


def test_underscore_digits() -> None:
    assert parse_hz("100_100_000") == 100_100_000


def test_comma_digits() -> None:
    assert parse_hz("100,100,000") == 100_100_000


def test_float_plain() -> None:
    assert parse_hz("1000.5") == 1000


def test_m_suffix() -> None:
    assert parse_hz("100M") == 100_000_000
    assert parse_hz("100m") == 100_000_000


def test_mhz_suffix() -> None:
    assert parse_hz("100MHz") == 100_000_000
    assert parse_hz("100.1MHz") == 100_100_000


def test_fractional_m() -> None:
    assert parse_hz("100.5M") == 100_500_000


def test_k_suffix() -> None:
    assert parse_hz("250k") == 250_000
    assert parse_hz("250kHz") == 250_000


def test_g_suffix() -> None:
    assert parse_hz("1.5G") == 1_500_000_000
    assert parse_hz("1GHz") == 1_000_000_000


def test_hz_suffix_alone() -> None:
    assert parse_hz("500Hz") == 500


def test_whitespace_trimmed() -> None:
    assert parse_hz("  100M  ") == 100_000_000


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_hz("")


def test_suffix_only_raises() -> None:
    with pytest.raises(ValueError):
        parse_hz("M")


def test_non_numeric_raises() -> None:
    with pytest.raises(ValueError):
        parse_hz("abc")


def test_trailing_garbage_raises() -> None:
    with pytest.raises(ValueError):
        parse_hz("100MXYZ")
