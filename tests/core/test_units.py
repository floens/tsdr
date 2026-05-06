from __future__ import annotations

import pytest

from tsdr.core.units import parse_hz


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
