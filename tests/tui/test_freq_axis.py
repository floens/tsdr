from __future__ import annotations

from typing import Any, cast

from tsdr.tui.widgets.spectrum_widget import SpectrumWidget


def _labels(width: int, freq_min: float, freq_max: float) -> list[str]:
    # _compute_freq_labels touches no instance state, so call it unbound.
    pairs = SpectrumWidget._compute_freq_labels(cast(Any, None), width, freq_min, freq_max)
    return [label for _, label in pairs]


def _suffix(label: str) -> str:
    return next((c for c in label if c in "kMG"), "")


def test_axis_uses_single_unit() -> None:
    # Every case: all labels on one axis share one SI suffix (no k next to M).
    for fmin, fmax in [
        (6.905e6, 6.965e6),
        (99.9e6, 100.3e6),
        (0.9e6, 1.1e6),
        (145.4e6, 145.6e6),
        (1.18e9, 1.22e9),
    ]:
        labels = _labels(80, fmin, fmax)
        assert labels
        suffixes = {_suffix(x) for x in labels}
        assert len(suffixes) == 1, (fmin, fmax, labels)


def test_axis_hf_reads_integer_khz() -> None:
    labels = _labels(80, 6.905e6, 6.965e6)
    assert all(x.endswith("k") for x in labels)
    assert "6940k" in labels


def test_axis_boundary_crossing_stays_khz() -> None:
    # An axis straddling 1 MHz must not mix 980k with 1.00M.
    labels = _labels(80, 0.9e6, 1.1e6)
    assert all(x.endswith("k") for x in labels)
    assert "1000k" in labels


def test_axis_fm_band_stays_mhz() -> None:
    labels = _labels(80, 96e6, 100e6)
    assert all(x.endswith("M") for x in labels)
