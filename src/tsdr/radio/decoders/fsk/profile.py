"""Declarative parameters for an FSK teleprinter mode.

One `FSKProfile` describes a mode for the shared `FSKFrontEnd` + framer stack:
physical layer (baud, shift), framing, and code alphabet. Adding a mode
(AMTOR-FEC, DSC, other RTTY bauds/shifts) is a new profile, not new DSP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FRAMINGS = ("sitor_b", "start_stop")  # canonical set for validation/completion


@dataclass(frozen=True)
class FSKProfile:
    baud: float
    shift_hz: float
    framing: Literal["sitor_b", "start_stop"]
    alphabet: Literal["ccir476", "ita2"]
    data_bits: int = 5  # start_stop word length
    polarity: Literal["normal", "reverse"] = "normal"  # reverse for LSB reception
    internal_rate: float = 6000.0  # front-end processing rate target


NAVTEX_PROFILE = FSKProfile(baud=100.0, shift_hz=170.0, framing="sitor_b", alphabet="ccir476")
RTTY_PROFILE = FSKProfile(baud=45.45, shift_hz=170.0, framing="start_stop", alphabet="ita2")
GENERIC_PROFILE = RTTY_PROFILE  # default for the generic `fsk` mode; overridable per axis

# Standard teleprinter bauds tried during auto-acquisition (start_stop framing).
STANDARD_BAUDS = [45.45, 50.0, 75.0, 100.0]

# Standard FSK shifts; an auto-detected shift within tolerance snaps to one of these.
STANDARD_SHIFTS = [170.0, 425.0, 450.0, 850.0]

# Named presets loadable via `demod ... --preset NAME`.
PROFILES: dict[str, FSKProfile] = {
    "ham": RTTY_PROFILE,
    "dwd": FSKProfile(
        baud=50.0, shift_hz=450.0, framing="start_stop", alphabet="ita2", polarity="reverse"
    ),
    "weather75": FSKProfile(baud=75.0, shift_hz=850.0, framing="start_stop", alphabet="ita2"),
}
