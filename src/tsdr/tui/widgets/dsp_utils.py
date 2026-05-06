import numpy as np
from numba import njit
from numpy.typing import NDArray
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from tsdr.tui.state import UIState

_STYLE_NONE = Style()


@njit
def render_spectrum_to_buf(
    buf: np.ndarray,
    y_vals: np.ndarray,
    fill_color: np.ndarray,
    trace_color: np.ndarray,
    bw_low: int,
    bw_high: int,
    bw_color: np.ndarray,
    center_x: int,
    center_color: np.ndarray,
) -> None:
    """Render spectrum trace, fill, bandwidth box, and center line into RGBA buffer."""
    plot_h, w, _ = buf.shape
    for x in range(w):
        y = y_vals[x]
        # Bandwidth box (above trace)
        if bw_low <= x < bw_high:
            for row in range(y):
                for c in range(4):
                    buf[row, x, c] = bw_color[c]
        # Fill below trace
        for row in range(y, plot_h):
            for c in range(4):
                buf[row, x, c] = fill_color[c]
        # Trace line (overwrites fill at trace position)
        if x < w - 1:
            y1 = y_vals[x + 1]
            ymin = min(y, y1)
            ymax = max(y, y1)
            for row in range(ymin, ymax + 1):
                for c in range(4):
                    buf[row, x, c] = trace_color[c]
        else:
            for c in range(4):
                buf[y, x, c] = trace_color[c]
        # Center frequency line (overwrites everything)
        if x == center_x:
            for row in range(plot_h):
                for c in range(4):
                    buf[row, x, c] = center_color[c]


def decimate_spectrum(spectrum: NDArray[np.float32], target: int) -> NDArray[np.float32]:
    """Resample spectrum to exactly *target* bins.

    Downsampling: average into blocks, then interpolate the remainder.
    Upsampling: linear interpolation.
    """
    n = len(spectrum)
    if n == target:
        return spectrum
    if n < target:
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target)
        return np.interp(x_new, x_old, spectrum).astype(np.float32)
    # Downsample: average into coarse bins, then interpolate to exact target
    factor = n // target
    coarse_n = n // factor  # >= target
    used = coarse_n * factor
    # Use all bins from the start; the tail remainder (n - used) is at most (factor-1) bins
    coarse: NDArray[np.float32] = spectrum[:used].reshape(coarse_n, factor).mean(axis=1)
    if coarse_n == target:
        return coarse
    # Interpolate coarse bins to exact target (handles the remainder)
    x_old = np.linspace(0, 1, coarse_n)
    x_new = np.linspace(0, 1, target)
    return np.interp(x_new, x_old, coarse).astype(np.float32)


def zoom_spectrum(spectrum: NDArray[np.float32], zoom: float) -> NDArray[np.float32]:
    """Extract center portion of spectrum based on zoom level."""
    if zoom <= 1.0:
        return spectrum
    n = len(spectrum)
    visible = max(int(n / zoom), 1)
    start = (n - visible) // 2
    return spectrum[start : start + visible]


def normalize_spectrum(
    spectrum: NDArray[np.float32], db_min: float, db_max: float
) -> NDArray[np.float32]:
    """Normalize spectrum to 0-1 range using dB bounds."""
    result: NDArray[np.float32] = np.clip((spectrum - db_min) / (db_max - db_min), 0, 1)
    return result


def status_strip(width: int, ui_state: UIState, style: Style | None = None) -> Strip:
    """Build status line showing zoom and dB levels.

    `style` should be the owning widget's `rich_style` so blank cells inherit
    its CSS background; segments with `Style()` don't pick up the widget bg.
    """
    text = f"Zoom: {ui_state.zoom:.1f}x | Min: {ui_state.db_min:.0f} dB | Max: {ui_state.db_max:.0f} dB"
    return Strip([Segment(text.ljust(width)[:width], style or _STYLE_NONE)], width)
