from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TuningState:
    step: float | None = None  # None = auto
    previous: tuple[float, str, float] | None = None  # (freq_hz, mode, bandwidth_hz)
    current_band_key: int | None = None  # 1..9 — None when not on a band


_state: TuningState | None = None


def get_tuning_state() -> TuningState:
    if _state is None:
        raise RuntimeError("TuningState not initialized")
    return _state


def init_tuning_state() -> TuningState:
    global _state
    _state = TuningState()
    return _state
