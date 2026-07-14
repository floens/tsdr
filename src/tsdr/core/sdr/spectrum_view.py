"""Spectrum view resolution.

The view (center, span) is client state on DeviceConfig that changes
instantly; widgets crop the latest FFT event into it and data sources
asynchronously supply the best bins they can for it. All clamping happens
here so gestures, rendering, and device view requests agree.
"""

from tsdr.core.sdr.config import MIN_SPECTRUM_SPAN_HZ, DeviceConfig
from tsdr.devices.base import DeviceCapabilities

_SPAN_STEP = 1.5


def full_view_range(config: DeviceConfig, caps: DeviceCapabilities | None) -> tuple[float, float]:
    """Widest displayable frequency range for a device.

    For spectrum-providing devices that is the device's band from
    capabilities: their `sample_rate` is the narrowband audio channel, not
    the displayable span.
    """
    if caps is not None and caps.provides_spectrum and caps.frequency_range is not None:
        return caps.frequency_range
    half = config.sample_rate / 2
    return (config.center_frequency - half, config.center_frequency + half)


def resolve_view(config: DeviceConfig, caps: DeviceCapabilities | None) -> tuple[float, float]:
    """Return the clamped (center, span) view for a device config.

    `spectrum_center` (view panning) only applies in free tuning mode; in
    center mode the view is always centered on the dial — a pinned view
    would hide every tune.
    """
    lo, hi = full_view_range(config, caps)
    full = hi - lo
    span = min(config.spectrum_span, full) if config.spectrum_span is not None else full
    center = (
        config.spectrum_center
        if config.spectrum_center is not None and config.tuning_mode == "free"
        else config.tuned_frequency
    )
    center = min(max(center, lo + span / 2), hi - span / 2)
    return center, span


def view_range(config: DeviceConfig, caps: DeviceCapabilities | None) -> tuple[float, float]:
    """The resolved view as a (freq_min, freq_max) window."""
    center, span = resolve_view(config, caps)
    return center - span / 2, center + span / 2


def adjusted_span(
    config: DeviceConfig, caps: DeviceCapabilities | None, direction: int
) -> float | None:
    """New spectrum_span for a ±1 zoom step (×/÷1.5). None = full band."""
    lo, hi = full_view_range(config, caps)
    full = hi - lo
    _, current = resolve_view(config, caps)
    new = current / _SPAN_STEP if direction > 0 else current * _SPAN_STEP
    if new >= full:
        return None
    return max(new, MIN_SPECTRUM_SPAN_HZ)
