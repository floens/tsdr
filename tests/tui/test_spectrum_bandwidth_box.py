"""Bandwidth-box column clamps: the box tracks the dial while the view can
pan away from it entirely, so both columns must stay inside [0, width]."""

from tsdr.tui.widgets.spectrum_widget import SpectrumWidget


def _widget(tuned: float, bw: float | None, sideband: str | None = None) -> SpectrumWidget:
    w = SpectrumWidget()
    w._tuned_frequency = tuned
    w._channel_bandwidth = bw
    w._sideband = sideband
    return w


def test_box_inside_view() -> None:
    w = _widget(tuned=100e6, bw=20e3)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) == (40, 60)


def test_box_right_of_view_is_none() -> None:
    w = _widget(tuned=120e6, bw=12.5e3)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) is None


def test_box_left_of_view_is_none() -> None:
    w = _widget(tuned=90e6, bw=12.5e3)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) is None


def test_box_partially_off_view_clamps() -> None:
    w = _widget(tuned=100.05e6, bw=20e3)
    low, high = w._compute_bandwidth_range(100, 99.95e6, 100.05e6)
    assert (low, high) == (90, 100)


def test_no_bandwidth_is_none() -> None:
    w = _widget(tuned=100e6, bw=None)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) is None
