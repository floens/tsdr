"""Hardware-center policy for VFO tuning.

The dial (`DeviceConfig.tuned_frequency`) moves freely; how the hardware
capture center (`center_frequency`) follows depends on `tuning_mode`:
"center" retunes on every dial move, "free" uses a DSP offset until the
tuned channel no longer fits in the captured band. Devices that provide
their own spectrum tune a server-side channel instead, so their center
always follows the dial.

While a demodulator runs, the LO offset keeps the hardware center beside
the channel instead of on it: a zero-IF receiver's LO leakage, DC offset,
1/f noise and second-order intermod all land at the capture center.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsdr.core.sdr.config import LoOffset, TuningMode
    from tsdr.devices.base import DeviceCapabilities

# Keep the channel off the anti-alias rolloff at the capture edges.
_EDGE_MARGIN_FRACTION = 0.05

# Smallest automatic LO offset; narrow channels need their LO well clear of
# the near-DC noise and intermod, not just outside the channel filter.
MIN_LO_OFFSET_HZ = 50e3


def effective_lo_side(setting: LoOffset, caps: DeviceCapabilities) -> LoOffset:
    """`setting` with auto resolved: below on zero-IF radios, off otherwise."""
    if setting == "auto":
        return "below" if caps.zero_if else "off"
    return setting


def lo_offset_for(
    *,
    setting: LoOffset,
    size_hz: float | None,
    channel_bandwidth: float | None,
    sample_rate: float,
    caps: DeviceCapabilities,
) -> float:
    """Signed hardware-center offset from the dial (center - tuned).

    `channel_bandwidth` is None when no demodulator runs: nothing to protect,
    so the LO sits on the dial. The automatic size is one channel width
    (at least MIN_LO_OFFSET_HZ), which leaves the LO half a channel outside
    the channel filter. A channel that cannot sit that far from the LO and
    still clear the edge margin (DAB, ADS-B fill the capture) stays centered.
    """
    if channel_bandwidth is None:
        return 0.0
    side = effective_lo_side(setting, caps)
    if side == "off":
        return 0.0
    size = size_hz if size_hz is not None else max(MIN_LO_OFFSET_HZ, channel_bandwidth)
    usable_half = sample_rate / 2 - _EDGE_MARGIN_FRACTION * sample_rate
    if size + channel_bandwidth / 2 > usable_half:
        return 0.0
    return -size if side == "below" else size


def _offset_center(tuned: float, lo_offset: float, caps: DeviceCapabilities) -> float:
    """`tuned + lo_offset`, mirrored to the dial's other side when that one
    would leave the device's range (a dial at the bottom of the band)."""
    if lo_offset == 0.0 or caps.frequency_range is None:
        return tuned + lo_offset
    lo, hi = caps.frequency_range
    for center in (tuned + lo_offset, tuned - lo_offset):
        if lo <= center <= hi:
            return center
    return tuned


def derive_center_frequency(
    *,
    tuned: float,
    center: float,
    sample_rate: float,
    channel_bandwidth: float,
    caps: DeviceCapabilities,
    running: bool,
    mode: TuningMode,
    lo_offset: float = 0.0,
    recenter: bool = False,
) -> float:
    """Hardware center for the dial. `recenter` makes free mode place the LO
    afresh (an LO offset change) instead of keeping a center that still fits."""
    if not caps.frequency_controllable:
        return center
    if caps.provides_spectrum:
        return tuned
    target = _offset_center(tuned, lo_offset, caps)
    if mode == "center" or not running or recenter:
        return target
    margin = _EDGE_MARGIN_FRACTION * sample_rate
    if abs(tuned - center) + channel_bandwidth / 2 <= sample_rate / 2 - margin:
        return center
    return target
