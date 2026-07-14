from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from tsdr.core.units import format_hz

if TYPE_CHECKING:
    from tsdr.core.events.events import FFTUpdateEvent

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


def project_spectrum(
    spectrum: NDArray[np.float32],
    e_fmin: float,
    e_fmax: float,
    v_fmin: float,
    v_fmax: float,
    target: int,
) -> NDArray[np.float32]:
    """Map `spectrum` (covering [e_fmin, e_fmax]) into `target` bins over [v_fmin, v_fmax].

    Bins outside the event's freq range are filled with -inf so they clip to 0
    after normalization: bars disappear where we have no captured data.
    """
    if e_fmin == v_fmin and e_fmax == v_fmax:
        return decimate_spectrum(spectrum, target)

    out = np.full(target, -np.inf, dtype=np.float32)
    e_span = e_fmax - e_fmin
    v_span = v_fmax - v_fmin
    n = len(spectrum)
    if e_span <= 0 or v_span <= 0 or n == 0:
        return out

    v_freqs = v_fmin + (np.arange(target) + 0.5) / target * v_span
    src = ((v_freqs - e_fmin) / e_span * n).astype(np.intp)
    mask = (src >= 0) & (src < n)
    out[mask] = spectrum[src[mask]]
    return out


def transient_view_shift(
    view: tuple[float, float], event_center_hz: float, capture_center_hz: float
) -> tuple[float, float]:
    """Anchor the view to the capture during a retune transient.

    While the hardware lags the dial, the latest event still covers the old
    capture; shifting the projection window by the stale delta keeps bars at
    their old positions under the already-moved axis — zoomed and full-band
    alike — until data catches up (the delta is 0 in steady state).
    Not for spectrum-providing devices: their frames aren't tied to the IQ
    capture center.
    """
    delta = event_center_hz - capture_center_hz
    return view[0] + delta, view[1] + delta


def normalize_spectrum(
    spectrum: NDArray[np.float32], db_min: float, db_max: float
) -> NDArray[np.float32]:
    """Normalize spectrum to 0-1 range using dB bounds."""
    result: NDArray[np.float32] = np.clip((spectrum - db_min) / (db_max - db_min), 0, 1)
    return result


_TRACE_IIR_RATE_PER_Z = 6.0
_TRACE_IIR_MIN_RATE = 3.0


def iir_trace_filter(
    avg: NDArray[np.float32] | None, z: NDArray[np.float32], dt: float
) -> NDArray[np.float32]:
    """Advance the per-column trace average by one frame (`dt` seconds), in place.

    The smoothing rate follows the *incoming* value: strong columns update
    fast while the noise floor crawls at the minimum rate — peaks appear
    immediately, the floor stays calm. Input is the aperture-normalized 0..1
    trace; a None or shape-mismatched `avg` reseeds from `z`.
    """
    if avg is None or avg.shape != z.shape:
        return z.astype(np.float32, copy=True)
    rate = np.maximum(_TRACE_IIR_RATE_PER_Z * z, _TRACE_IIR_MIN_RATE)
    gain = 1.0 - np.exp(-rate * dt)
    avg += gain * (z - avg)
    return avg


def fmt_hz(hz: float) -> str:
    """SI-suffixed frequency for status readouts; em dash for 0/unknown."""
    if hz <= 0:
        return "—"
    return format_hz(hz, decimals=3, long_suffix=True)


def span_rbw(view: tuple[float, float] | None, event: FFTUpdateEvent | None) -> tuple[float, float]:
    """Status-line inputs: view span (config-led) and event RBW (data-led)."""
    span = view[1] - view[0] if view is not None else 0.0
    rbw = (
        event.sample_rate / len(event.spectrum)
        if event is not None and len(event.spectrum)
        else 0.0
    )
    return span, rbw


def status_text(span_hz: float, rbw_hz: float, db_min: float, db_max: float) -> str:
    """Status line: view span, resolution bandwidth, and the dB window."""
    return (
        f"Span: {fmt_hz(span_hz)} | RBW: {fmt_hz(rbw_hz)} | "
        f"Min: {db_min:.0f} dB | Max: {db_max:.0f} dB"
    )


def status_strip(
    width: int,
    span_hz: float,
    rbw_hz: float,
    db_min: float,
    db_max: float,
    style: Style | None = None,
) -> Strip:
    """Build the status line strip.

    `style` should be the owning widget's `rich_style` so blank cells inherit
    its CSS background; segments with `Style()` don't pick up the widget bg.
    """
    text = status_text(span_hz, rbw_hz, db_min, db_max)
    return Strip([Segment(text.ljust(width)[:width], style or _STYLE_NONE)], width)
