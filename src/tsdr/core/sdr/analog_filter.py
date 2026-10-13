"""Width of the radio's analog (anti-alias) filter.

The filter ahead of the ADC keeps strong signals outside the captured band
from using up the converter's headroom (blocking, intermod) and from folding
into the band. Some drivers retune it with the sample rate; others leave it
where it is (the Pluto sits at 18 MHz whatever the rate), so tsdr tracks the
rate for those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsdr.core.sdr.config import AnalogFilter
    from tsdr.devices.base import DeviceCapabilities


def effective_mode(setting: AnalogFilter, caps: DeviceCapabilities) -> AnalogFilter:
    """`setting` with auto resolved: track where the driver leaves the filter
    alone, off where it manages the filter itself."""
    if setting == "auto":
        return "track" if caps.static_analog_filter else "off"
    return setting


def target_bandwidth(
    *,
    setting: AnalogFilter,
    width_hz: float | None,
    fraction: float,
    sample_rate: float,
    caps: DeviceCapabilities,
) -> float | None:
    """Filter width to request from the driver, or None to leave its own.

    The driver clamps and rounds the request; callers read back what it set.
    """
    if not caps.analog_filter_supported:
        return None
    mode = effective_mode(setting, caps)
    if mode == "off":
        return None
    if mode == "fixed":
        return width_hz
    return fraction * sample_rate
