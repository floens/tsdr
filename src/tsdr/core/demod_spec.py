"""Value objects for the active demodulator or protocol decoder.

`DemodSpec` bundles the user-facing knobs that select and configure a demod
(mode name and mode-specific settings: FM deviation, SSTV submode, FSK
baud/shift/polarity/alphabet/framing). It is the single argument to
``SDREngine.set_audio_demod`` and the unit of persistence in memories, band
registers, device prefs, and the A/B swap.

Future per-demod settings (CW BFO, NFM emphasis, ...) extend the spec only —
the plumbing through engine/persistence/recall stays the same shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DemodSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    fm_deviation_hz: float | None = None
    sstv_mode: str | None = None
    # FSK-family (rtty/fsk) overrides; None on any axis = auto-acquire it.
    fsk_baud: float | None = None
    fsk_shift_hz: float | None = None
    fsk_reverse: bool | None = None
    fsk_alphabet: str | None = None
    fsk_framing: str | None = None


class PreviousTuneState(BaseModel):
    """Snapshot of the previously-tuned state for A/B swap."""

    model_config = ConfigDict(frozen=True)

    frequency_hz: float
    bandwidth_hz: float
    spec: DemodSpec
