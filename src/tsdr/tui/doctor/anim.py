"""Animation core for the doctor's Live tab.

Self-contained replicas of TSDR's two hot kitty-graphics paths so a terminal's
real-time performance can be judged by eye:

- ``StripScroller`` mirrors ``WaterfallWidget``'s image-mode strip technique
  (fixed-height strips, new rows at the top, freeze + rotate, scroll by
  re-placing, remove off-screen). It is sink-agnostic so it unit-tests against a
  recording stub without a real terminal.
- ``spectrum_frame`` builds a full-width RGBA frame each tick with a moving
  ball, mirroring the spectrum widget's per-frame full redraw.
- ``glyph_*`` render a no-kitty, glyph-only waterfall for text-throughput.

Pure helpers take a float ``t`` (seconds) so motion is deterministic and testable.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rich.style import Style
from rich.text import Text

STRIP_HEIGHT = 64


def _build_colormap() -> np.ndarray:
    """256-entry RGBA LUT: navy -> blue -> cyan -> white (compact, self-contained)."""
    stops = np.array([(0, 0, 32), (0, 0, 144), (30, 144, 255), (255, 255, 255)], dtype=np.float64)
    lut = np.empty((256, 4), dtype=np.uint8)
    lut[:, 3] = 255
    for i in range(256):
        t = i / 255.0 * (len(stops) - 1)
        idx = min(int(t), len(stops) - 2)
        frac = t - idx
        lut[i, :3] = np.clip(stops[idx] * (1 - frac) + stops[idx + 1] * frac + 0.5, 0, 255)
    return lut


_LUT = _build_colormap()


def synthetic_row(width: int, t: float) -> np.ndarray:
    """One waterfall row (width, 4) RGBA: a Gaussian peak sweeping left/right."""
    x = np.arange(width, dtype=np.float64)
    center = (0.5 + 0.4 * np.sin(t * 1.7)) * width
    sigma = max(width / 24.0, 1.0)
    intensity = np.exp(-(((x - center) / sigma) ** 2))
    intensity += 0.15 * (0.5 + 0.5 * np.sin(x * 0.20 + t * 3.0))  # moving ripple floor
    idx = np.clip(intensity * 255.0, 0, 255).astype(np.intp)
    row: np.ndarray = _LUT[idx]
    return row


def moving_x(width: int, sprite_w: int, t: float) -> int:
    """Sweeping horizontal position (triangle wave) within [0, width - sprite_w]."""
    span = max(width - sprite_w, 1)
    phase = (t * 0.5) % 2.0  # one full there-and-back every ~4 s
    pos = phase if phase <= 1.0 else 2.0 - phase
    return int(pos * span)


def spectrum_frame(width: int, height: int, t: float) -> np.ndarray:
    """Full RGBA frame (height, width, 4): animated baseline + a moving ball."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, 3] = 255

    x = np.arange(width)
    curve = (0.5 + 0.45 * np.sin(x * 0.05 + t * 2.0)) * (height - 1)
    rows = curve.astype(np.intp)
    img[rows, x, :3] = (30, 144, 255)

    r = max(min(width, height) // 16, 4)
    cx = moving_x(width, 2 * r, t) + r
    cy = int((0.35 + 0.1 * np.sin(t * 2.3)) * height)
    yy, xx = np.ogrid[:height, :width]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[disc] = (255, 220, 0, 255)
    return img


def glyph_intensity(width: int, t: float) -> np.ndarray:
    """Per-column intensity (0..1) for the glyph-only waterfall - a sweeping peak."""
    x = np.arange(width, dtype=np.float64)
    center = (0.5 + 0.4 * np.sin(t * 1.7)) * width
    sigma = max(width / 16.0, 1.0)
    inten = np.exp(-(((x - center) / sigma) ** 2))
    inten += 0.1 * (0.5 + 0.5 * np.sin(x * 0.3 + t * 3.0))
    clamped: np.ndarray = np.clip(inten, 0.0, 1.0)
    return clamped


def glyph_row_text(intensity: np.ndarray) -> Text:
    """Render a column-intensity row as a Rich ``Text`` of run-length-encoded
    colored block glyphs.

    Returns a pre-styled ``Text`` (not a markup string) so callers can recycle
    already-built rows without re-parsing console markup every frame.
    """
    idx = np.clip(intensity * 255.0, 0, 255).astype(np.intp)
    text = Text(no_wrap=True)
    i = 0
    n = len(idx)
    while i < n:
        j = i + 1
        while j < n and idx[j] == idx[i]:
            j += 1
        r, g, b = (int(c) for c in _LUT[idx[i], :3])
        text.append("█" * (j - i), style=Style(color=f"rgb({r},{g},{b})"))
        i = j
    return text


class _Sink(Protocol):
    def update_image(self, key: str, data: np.ndarray, *, x: int, y: int) -> None: ...
    def place_image(self, key: str, *, y: int, crop_h: int) -> None: ...
    def remove_image(self, key: str) -> None: ...


# Per-strip left-edge marker so the eye can tell the waterfall is built from
# discrete kitty image strips (adjacent strips alternate colour).
_BORDER_W = 20
_BORDER_COLORS = ((255, 80, 80, 255), (80, 200, 255, 255))


@dataclass
class _Strip:
    key: str
    buffer: np.ndarray  # (STRIP_HEIGHT, W, 4) RGBA
    border: tuple[int, int, int, int]
    fill: int = 0
    frozen: bool = False
    transmitted_fill: int = 0


class StripScroller:
    """Scrolling waterfall built from fixed-height kitty image strips.

    Mirrors ``WaterfallWidget`` image mode: the active strip (index 0) takes a
    new row at the top each frame, freezes when full, and a fresh active strip
    rotates in. ``emit`` re-transmits the (changing) active strip and re-places
    the (immutable) frozen strips at growing y-offsets, removing off-screen ones.
    """

    def __init__(
        self, sink: _Sink, *, strip_height: int = STRIP_HEIGHT, mark_strips: bool = False
    ) -> None:
        self._sink = sink
        self._strip_height = strip_height
        self._mark_strips = mark_strips
        self._strips: list[_Strip] = []
        self._counter = 0

    def _new_strip(self, width: int) -> None:
        buf = np.zeros((self._strip_height, width, 4), dtype=np.uint8)
        border = _BORDER_COLORS[self._counter % len(_BORDER_COLORS)]
        self._strips.insert(0, _Strip(key=f"wf_strip_{self._counter}", buffer=buf, border=border))
        self._counter += 1

    def push(self, row: np.ndarray) -> None:
        """Insert one RGBA row (W, 4) at the top of the active strip."""
        width = row.shape[0]
        if not self._strips or self._strips[0].buffer.shape[1] != width:
            self.reset()
            self._new_strip(width)

        active = self._strips[0]
        if active.fill > 0:
            limit = min(active.fill, self._strip_height - 1)
            active.buffer[1 : 1 + limit] = active.buffer[:limit]
        active.buffer[0] = row
        if self._mark_strips:
            active.buffer[0, :_BORDER_W] = active.border
        active.fill = min(active.fill + 1, self._strip_height)

        if active.fill >= self._strip_height:
            active.frozen = True
            self._new_strip(width)

    def emit(self, visible_h: int) -> None:
        """Transmit the active strip and scroll/cull the frozen strips."""
        if not self._strips:
            return

        active = self._strips[0]
        if active.fill > 0:
            self._sink.update_image(active.key, active.buffer[: active.fill], x=0, y=0)
            active.transmitted_fill = active.fill

        y = active.fill
        removed: list[str] = []
        for strip in self._strips[1:]:
            if y >= visible_h:
                if strip.transmitted_fill > 0:
                    self._sink.remove_image(strip.key)
                removed.append(strip.key)
                continue
            crop_h = visible_h - y if y + strip.fill > visible_h else 0
            if strip.fill != strip.transmitted_fill:
                self._sink.update_image(strip.key, strip.buffer[: strip.fill], x=0, y=y)
                strip.transmitted_fill = strip.fill
            else:
                self._sink.place_image(strip.key, y=y, crop_h=crop_h)
            y += strip.fill

        if removed:
            self._strips = [s for s in self._strips if s.key not in removed]

    def reset(self) -> None:
        """Remove all strips from the sink and clear state."""
        for strip in self._strips:
            if strip.transmitted_fill > 0:
                self._sink.remove_image(strip.key)
        self._strips.clear()
        self._counter = 0
