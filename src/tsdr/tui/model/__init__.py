"""Reactive UI model — the single source of truth for what widgets exist and their structural props.

UIModel is frozen; mutations go through UIStore. Stream data (FFT frames,
signal info, memories, decoder messages, configs) is intentionally NOT held
here — it flows via events through EventRouter, or is read from existing stores
by widgets in their update_* methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tsdr.tui.commands.registry import MenuItem

DecoderKind = Literal["rds", "dab", "adsb", "tetra", "dmr", "text", "sstv"]
ActivePanel = Literal["stats", "performance"]


@dataclass(frozen=True)
class DeviceUIState:
    """Per-device structural state."""

    device_id: str
    has_audio_pipeline: bool = False
    active_decoder_kind: DecoderKind | None = None


@dataclass(frozen=True)
class ConsoleUIState:
    """Console autocomplete popup state."""

    autocomplete_visible: bool = False
    autocomplete_items: tuple[MenuItem, ...] = ()
    selected_index: int = -1


@dataclass(frozen=True)
class UIModel:
    """Frozen UI model. Build new models via dataclasses.replace; use UIStore for mutations."""

    zoom: float = 1.0
    db_min: float = -100.0
    db_max: float = -30.0
    image_mode: bool = False
    active_panel: ActivePanel | None = None
    clock_visible: bool = True
    timezone: str | None = None
    ntp_server: str | None = None

    devices: tuple[DeviceUIState, ...] = ()
    focused_device_id: str | None = None

    console: ConsoleUIState = field(default_factory=ConsoleUIState)

    @classmethod
    def initial(cls, prefs: dict[str, Any]) -> UIModel:
        """Build the starting model from load_preferences() output."""
        ui = prefs.get("ui", {})
        defaults = cls()
        return cls(
            zoom=float(ui["zoom"]) if "zoom" in ui else defaults.zoom,
            db_min=float(ui["db_min"]) if "db_min" in ui else defaults.db_min,
            db_max=float(ui["db_max"]) if "db_max" in ui else defaults.db_max,
            image_mode=bool(ui["image_mode"]) if "image_mode" in ui else defaults.image_mode,
            active_panel=_coerce_panel(ui.get("active_panel")),
            clock_visible=(
                bool(ui["clock_visible"]) if "clock_visible" in ui else defaults.clock_visible
            ),
            timezone=_coerce_str(ui.get("timezone")),
            ntp_server=_coerce_str(ui.get("ntp_server")),
        )


def _coerce_panel(value: object) -> ActivePanel | None:
    if value == "stats" or value == "performance":
        return value  # type: ignore[return-value]
    return None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


# Helpers for adjusting prefs-style fields. Used by keyboard handlers and the
# spectrum widget's mouse-scroll zoom — they mutate UIStore via these functions
# rather than computing in line.


def adjusted_zoom(current: float, direction: int) -> float:
    """Zoom by ±1.5×, clamped to [1.0, 512.0]."""
    if direction > 0:
        return round(min(512.0, current * 1.5), 1)
    return round(max(1.0, current / 1.5), 1)


# dB-window (dBFS) step and bounds. Floor reaches below -90 dBFS for
# high-dynamic-range receivers whose noise floor sits there.
_DB_STEP = 5.0
_DB_FLOOR = -200.0  # absolute lower bound for db_min
_DB_CEIL = 0.0  # full scale; nothing exceeds 0 dBFS
_DB_MIN_GAP = 10.0  # keep the window at least this wide


def adjusted_db_min(current: float, db_max: float, direction: int) -> float:
    """Adjust min dB by ±_DB_STEP, clamped to [_DB_FLOOR, db_max - _DB_MIN_GAP]."""
    new_min = current + direction * _DB_STEP
    return max(_DB_FLOOR, min(new_min, db_max - _DB_MIN_GAP))


def adjusted_db_max(current: float, db_min: float, direction: int) -> float:
    """Adjust max dB by ±_DB_STEP, clamped to [db_min + _DB_MIN_GAP, _DB_CEIL]."""
    new_max = current + direction * _DB_STEP
    return min(_DB_CEIL, max(new_max, db_min + _DB_MIN_GAP))
