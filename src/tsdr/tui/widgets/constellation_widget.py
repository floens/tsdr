import math

import numpy as np
from numba import njit
from numpy.typing import NDArray
from textual.geometry import Size
from textual.strip import Strip
from textual.widget import Widget

from tsdr.core.events.events import ConstellationUpdateEvent
from tsdr.core.tracing import span
from tsdr.tui.widgets.kitty_image import KittyImageWidget

# Alpha decay factor per frame (persistence trail effect)
_DECAY = np.float32(0.85)

# Auto-scale: range adapts to signal amplitude
_RANGE_MARGIN = 1.3  # 30% margin around 95th percentile
_RANGE_MIN = 1e-4  # floor to prevent division by zero
_RANGE_DECAY = 0.95  # slow shrink per frame (fast attack, slow decay)

# Max points to plot per frame (subsample if more)
_MAX_POINTS = 2048

_DOT_COLOR = np.array([0x1E, 0x90, 0xFF, 0xFF], dtype=np.uint8)  # dodger blue
_AXIS_COLOR = np.array([40, 40, 40, 255], dtype=np.uint8)


@njit(cache=True)
def _decay_alpha(alpha: np.ndarray, factor: np.float32) -> None:
    """In-place alpha decay - avoids float32 temp array allocation."""
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            alpha[i, j] = np.uint8(alpha[i, j] * factor)


@njit(cache=True)
def _plot(buf: np.ndarray, px: np.ndarray, py: np.ndarray, color: np.ndarray) -> None:
    """Plot points into RGBA buffer."""
    size = buf.shape[0]
    for i in range(len(px)):
        x = px[i]
        y = py[i]
        if 0 <= x < size and 0 <= y < size:
            for c in range(4):
                buf[y, x, c] = color[c]


class ConstellationWidget(Widget):
    """Constellation diagram rendered via Kitty image protocol.

    Only active when image mode is on. Renders a scatter plot of
    complex symbols with persistence decay.
    """

    def __init__(self) -> None:
        super().__init__()
        self._kitty: KittyImageWidget | None = None
        self._buffer: NDArray[np.uint8] | None = None  # persistent RGBA frame
        self._buf_size = 0
        self._image_mode = False
        self._range = 1.5  # display range ±_range, auto-scaled

    # Default cell pixel sizes (matches KittyImageWidget defaults)
    _cell_width_px = 8
    _cell_height_px = 16

    def on_mount(self) -> None:
        self._kitty = KittyImageWidget()
        self.mount(self._kitty)
        # Layout may have cached height=0 from before mount; force re-query
        self.clear_cached_dimensions()

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        cw = self._cell_width_px
        ch = self._cell_height_px
        if self._kitty is not None:
            cw = self._kitty.cell_width_px
            ch = self._kitty.cell_height_px
        if ch == 0:
            return 0
        return math.ceil(width * cw / ch)

    def on_hide(self) -> None:
        self._clear_image()

    def set_image_mode(self, enabled: bool) -> None:
        self._image_mode = enabled
        if not enabled:
            self._clear_image()

    def _clear_image(self) -> None:
        if self._kitty is not None:
            self._kitty.remove_image("constellation")
        self._buffer = None
        self._buf_size = 0
        self._range = 1.5

    def update_constellation(self, event: ConstellationUpdateEvent) -> None:
        if self._kitty is None or not self._image_mode:
            return

        size = self._pixel_size()
        if size <= 0:
            return

        # Allocate or resize persistent buffer
        if self._buffer is None or self._buf_size != size:
            self._buffer = np.zeros((size, size, 4), dtype=np.uint8)
            self._buf_size = size
            self._draw_grid(self._buffer)

        buf = self._buffer

        with span("constellation.decay"):
            _decay_alpha(buf[:, :, 3], _DECAY)

        points = event.points
        if points is not None and len(points) > 0:
            # Subsample if too many points
            if len(points) > _MAX_POINTS:
                step = len(points) // _MAX_POINTS
                points = points[::step]

            with span("constellation.autoscale"):
                mag = np.abs(points)
                mag.sort()
                mag_95 = float(mag[int(len(mag) * 0.95)])
                target = max(mag_95 * _RANGE_MARGIN, _RANGE_MIN)
                if target > self._range:
                    self._range = target  # fast attack
                elif target < self._range * 0.4:
                    # Regime change (e.g. sync acquired) - snap and clear stale pixels
                    self._range = target
                    buf[:] = 0
                else:
                    self._range = self._range * _RANGE_DECAY + target * (1 - _RANGE_DECAY)

            with span("constellation.plot"):
                r = self._range
                scale = 0.5 * (size - 1) / r
                mid = (size - 1) * 0.5
                real = points.real.astype(np.float32)
                imag = points.imag.astype(np.float32)
                px = (real * scale + mid).astype(np.intp)
                py = (mid - imag * scale).astype(np.intp)
                _plot(buf, px, py, _DOT_COLOR)

        self._draw_grid(buf)

        with span("constellation.transmit"):
            self._kitty.update_image("constellation", buf)

    def _pixel_size(self) -> int:
        """Square size in pixels that fits the widget width."""
        if self._kitty is None:
            return 0
        return int(self.size.width * self._kitty.cell_width_px)

    def _draw_grid(self, buf: NDArray[np.uint8]) -> None:
        """Draw I/Q axis crosshairs."""
        size = buf.shape[0]
        center = size // 2
        buf[center, :] = _AXIS_COLOR
        buf[:, center] = _AXIS_COLOR

    def render_line(self, y: int) -> Strip:
        return Strip.blank(self.size.width, self.rich_style)
