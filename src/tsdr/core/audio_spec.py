"""Value objects for the active audio demodulator.

`AudioDemodSpec` bundles the user-facing knobs that select and configure an
audio demodulator (mode name, frequency offset, FM deviation, SSTV submode).
It is the single argument to ``SDREngine.set_audio_demod`` and the unit of
persistence in memories, band registers, device prefs, and the A/B swap.

Future per-demod settings (CW BFO, NFM emphasis, ...) extend the spec only —
the plumbing through engine/persistence/recall stays the same shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AudioDemodSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    frequency_offset: float = 0.0
    fm_deviation_hz: float | None = None
    sstv_mode: str | None = None


class PreviousTuneState(BaseModel):
    """Snapshot of the previously-tuned state for A/B swap."""

    model_config = ConfigDict(frozen=True)

    frequency_hz: float
    bandwidth_hz: float
    spec: AudioDemodSpec
