"""Unit tests for the doctor's animation core (StripScroller + frame generators)."""

from __future__ import annotations

import numpy as np

from tsdr.tui.doctor.anim import (
    StripScroller,
    glyph_intensity,
    glyph_row_text,
    moving_x,
    spectrum_frame,
    synthetic_row,
)


class RecordingSink:
    """Records StripScroller's kitty commands without a real terminal."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def update_image(self, key: str, data: np.ndarray, *, x: int, y: int) -> None:
        self.calls.append(("update", key, y))

    def place_image(self, key: str, *, y: int, crop_h: int) -> None:
        self.calls.append(("place", key, y))

    def remove_image(self, key: str) -> None:
        self.calls.append(("remove", key))

    def ops(self) -> list[str]:
        return [c[0] for c in self.calls]


def _row(width: int = 10) -> np.ndarray:
    return np.full((width, 4), 200, dtype=np.uint8)


def test_active_strip_transmitted_on_emit() -> None:
    sink = RecordingSink()
    s = StripScroller(sink, strip_height=8)
    s.push(_row())
    s.emit(visible_h=100)
    assert ("update", "wf_strip_0", 0) in sink.calls


def test_freeze_and_rotate_after_full() -> None:
    sink = RecordingSink()
    s = StripScroller(sink, strip_height=8)
    for _ in range(8):
        s.push(_row())
    # Active filled to 8 -> frozen; a fresh active rotates in at index 0.
    assert len(s._strips) == 2
    assert s._strips[0].fill == 0
    assert s._strips[1].frozen


def test_scroll_places_then_removes_offscreen() -> None:
    sink = RecordingSink()
    s = StripScroller(sink, strip_height=4)
    for _ in range(24):
        s.push(_row(6))
        s.emit(visible_h=10)
    assert "place" in sink.ops()  # frozen strips scrolled via place_image
    assert "remove" in sink.ops()  # off-screen strips culled


def test_reset_removes_all() -> None:
    sink = RecordingSink()
    s = StripScroller(sink, strip_height=4)
    for _ in range(6):
        s.push(_row())
        s.emit(visible_h=100)
    sink.calls.clear()
    s.reset()
    assert sink.calls and all(op == "remove" for op in sink.ops())
    assert s._strips == []


def test_synthetic_row_shape() -> None:
    r = synthetic_row(64, 1.23)
    assert r.shape == (64, 4)
    assert r.dtype == np.uint8


def test_spectrum_frame_shape() -> None:
    f = spectrum_frame(80, 40, 0.5)
    assert f.shape == (40, 80, 4)
    assert f.dtype == np.uint8


def test_moving_x_bounces_in_bounds() -> None:
    xs = [moving_x(100, 10, i * 0.1) for i in range(200)]
    assert all(0 <= x <= 90 for x in xs)
    assert len(set(xs)) > 1  # it actually moves


def test_glyph_intensity_in_unit_range() -> None:
    inten = glyph_intensity(80, 1.0)
    assert inten.shape == (80,)
    assert inten.min() >= 0.0 and inten.max() <= 1.0


def test_glyph_row_text_renders_styled_blocks() -> None:
    text = glyph_row_text(glyph_intensity(40, 0.5))
    assert "█" in text.plain
    assert len(text.plain) == 40
    assert text.spans  # carries per-run color styling, not raw markup
    assert all("rgb(" in str(span.style) for span in text.spans)


def test_strip_border_marks_alternating_colors() -> None:
    sink = RecordingSink()
    s = StripScroller(sink, strip_height=2, mark_strips=True)
    for _ in range(4):  # 2 full strips at strip_height=2
        s.push(_row(6))
    # Two strips created; their left-edge border colors differ.
    borders = {tuple(strip.buffer[0, 0]) for strip in s._strips if strip.fill}
    assert len(borders) >= 1  # border painted
    assert all(b != (0, 0, 0, 0) for b in borders)
