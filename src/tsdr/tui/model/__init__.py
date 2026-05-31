"""Reactive UI model — the single source of truth for what widgets exist and their structural props.

UIModel is frozen; mutations go through UIStore. Stream data (FFT frames,
signal info, memories, decoder messages, configs) is intentionally NOT held
here — it flows via events through EventRouter, or is read from existing stores
by widgets in their update_* methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from tsdr.tui.commands.registry import MenuItem

DecoderKind = Literal["rds", "dab", "adsb", "tetra", "dmr", "text", "sstv"]
Edge = Literal["left", "right", "bottom"]


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
class EdgePanels:
    panels: tuple[str, ...] = ()
    active: str | None = None


@dataclass(frozen=True)
class UILayout:
    left: EdgePanels = field(default_factory=EdgePanels)
    right: EdgePanels = field(default_factory=EdgePanels)
    bottom: EdgePanels = field(default_factory=EdgePanels)
    hotkeys: tuple[tuple[int, str], ...] = ()
    strips_visible: bool = True


DEFAULT_LAYOUT = UILayout(
    left=EdgePanels(panels=("decoder-output",)),
    right=EdgePanels(panels=("stats", "performance")),
    bottom=EdgePanels(panels=("demod",)),
    hotkeys=(
        (1, "decoder-output"),
        (2, "demod"),
        (3, "stats"),
        (4, "performance"),
    ),
)


@dataclass(frozen=True)
class UIModel:
    """Frozen UI model. Build new models via dataclasses.replace; use UIStore for mutations."""

    zoom: float = 1.0
    db_min: float = -100.0
    db_max: float = -30.0
    image_mode: bool = False
    layout: UILayout = DEFAULT_LAYOUT
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
            layout=_coerce_layout(ui.get("layout")),
            clock_visible=(
                bool(ui["clock_visible"]) if "clock_visible" in ui else defaults.clock_visible
            ),
            timezone=_coerce_str(ui.get("timezone")),
            ntp_server=_coerce_str(ui.get("ntp_server")),
        )


def focused_device(model: UIModel) -> DeviceUIState | None:
    """The focused device's UIState, or None when nothing is focused."""
    return next((d for d in model.devices if d.device_id == model.focused_device_id), None)


def _coerce_layout(value: object) -> UILayout:
    if not isinstance(value, dict):
        return DEFAULT_LAYOUT
    try:
        strips_raw = value.get("strips_visible", DEFAULT_LAYOUT.strips_visible)
        layout = UILayout(
            left=_coerce_edge(value.get("left")),
            right=_coerce_edge(value.get("right")),
            bottom=_coerce_edge(value.get("bottom")),
            # Hotkeys aren't user-editable in v1, so they're a code-level default
            # rather than persisted state — always take the current default so a
            # renumbering applies to everyone, not just fresh installs.
            hotkeys=DEFAULT_LAYOUT.hotkeys,
            strips_visible=bool(strips_raw),
        )
    except (TypeError, ValueError, KeyError):
        return DEFAULT_LAYOUT
    return _augment_with_missing_panels(layout)


def _augment_with_missing_panels(layout: UILayout) -> UILayout:
    """Append any PANELS id absent from the saved layout to the edge it lives on
    in DEFAULT_LAYOUT. Lets us add panels (e.g. the new demod multiplexer
    replacing the per-decoder panels) without users losing access — their
    custom edge placements for existing panels are preserved."""
    from tsdr.tui.view.panels import PANELS  # noqa: PLC0415

    docked = set(layout.left.panels + layout.right.panels + layout.bottom.panels)
    missing = [pid for pid in PANELS if pid not in docked]
    if not missing:
        return layout
    left, right, bottom = layout.left, layout.right, layout.bottom
    for pid in missing:
        if pid in DEFAULT_LAYOUT.left.panels:
            left = replace(left, panels=left.panels + (pid,))
        elif pid in DEFAULT_LAYOUT.right.panels:
            right = replace(right, panels=right.panels + (pid,))
        else:
            bottom = replace(bottom, panels=bottom.panels + (pid,))
    return replace(layout, left=left, right=right, bottom=bottom)


def _coerce_edge(value: object) -> EdgePanels:
    # Import inside to avoid a tui.model → tui.view import edge.
    from tsdr.tui.view.panels import PANELS  # noqa: PLC0415

    if not isinstance(value, dict):
        return EdgePanels()
    panels_raw = value.get("panels", ())
    if isinstance(panels_raw, (list, tuple)):
        panels = tuple(p for p in panels_raw if isinstance(p, str) and p in PANELS)
    else:
        panels = ()
    active_raw = value.get("active")
    active = (
        active_raw if isinstance(active_raw, str) and active_raw and active_raw in panels else None
    )
    return EdgePanels(panels=panels, active=active)


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
