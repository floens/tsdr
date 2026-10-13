"""Batch capture metadata: times backdated past the batch and past what the
device still holds after it (a native reader's ring after a stall), and the
center frequency the device says the samples were taken at."""

import pytest

from tsdr.core.sdr.workers.io_worker import capture_center, first_sample_utc


class _Holding:
    read_backlog_samples = 500
    read_center_frequency: float | None = None


class _BeforeRetune(_Holding):
    read_center_frequency = 162.5e6


def test_backdates_past_the_batch() -> None:
    assert first_sample_utc(object(), 1000, 1000.0, 100.0) == pytest.approx(99.0)


def test_backdates_past_what_the_device_still_holds() -> None:
    assert first_sample_utc(_Holding(), 1000, 1000.0, 100.0) == pytest.approx(98.5)


def test_unknown_rate_keeps_now() -> None:
    assert first_sample_utc(_Holding(), 1000, 0.0, 100.0) == 100.0


def test_batch_center_is_where_the_device_says_it_captured() -> None:
    assert capture_center(_BeforeRetune(), 163e6) == 162.5e6


def test_batch_center_falls_back_to_the_configured_one() -> None:
    assert capture_center(_Holding(), 163e6) == 163e6
    assert capture_center(object(), 163e6) == 163e6
