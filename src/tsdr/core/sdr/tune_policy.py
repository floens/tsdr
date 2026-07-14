"""Hardware-center policy for VFO tuning.

The dial (`DeviceConfig.tuned_frequency`) moves freely; how the hardware
capture center (`center_frequency`) follows depends on `tuning_mode`:
"center" retunes on every dial move, "free" uses a DSP offset until the
tuned channel no longer fits in the captured band. Devices that provide
their own spectrum tune a server-side channel instead, so their center
always follows the dial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsdr.core.sdr.config import TuningMode
    from tsdr.devices.base import DeviceCapabilities

# Keep the channel off the anti-alias rolloff at the capture edges.
_EDGE_MARGIN_FRACTION = 0.05


def derive_center_frequency(
    *,
    tuned: float,
    center: float,
    sample_rate: float,
    channel_bandwidth: float,
    caps: DeviceCapabilities,
    running: bool,
    mode: TuningMode,
) -> float:
    if not caps.frequency_controllable:
        return center
    if caps.provides_spectrum:
        return tuned
    if mode == "center":
        return tuned
    if not running:
        return tuned
    margin = _EDGE_MARGIN_FRACTION * sample_rate
    if abs(tuned - center) + channel_bandwidth / 2 <= sample_rate / 2 - margin:
        return center
    return tuned
