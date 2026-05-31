"""PrefsSync — persists UI-prefs fields on change, debounced.

Subscribes to UIStore and persists when any prefs-relevant field changes;
coalesces rapid changes (e.g. holding `k` to zoom) into one write ~250ms after
the burst settles. `flush_prefs(model)` is exported so headless mode can
subscribe synchronously (no Textual timer available there).
"""

from __future__ import annotations

import logging

from textual.app import App
from textual.timer import Timer

from tsdr.core import storage
from tsdr.core.preferences import PREFERENCES_FILE
from tsdr.tui.model import UILayout, UIModel
from tsdr.tui.model.store import UIStore

logger = logging.getLogger(__name__)

PREFS_FIELDS = frozenset(
    {
        "zoom",
        "db_min",
        "db_max",
        "image_mode",
        "layout",
        "clock_visible",
        "timezone",
        "ntp_server",
    }
)

_DEBOUNCE_SECONDS = 0.25


def flush_prefs(model: UIModel) -> None:
    """Write the prefs-relevant fields of `model` to the prefs file."""
    prefs = storage.load_toml(PREFERENCES_FILE)
    prefs["ui"] = {
        "zoom": model.zoom,
        "db_min": model.db_min,
        "db_max": model.db_max,
        "image_mode": model.image_mode,
        "layout": _serialize_layout(model.layout),
        "timezone": model.timezone or "",
        "clock_visible": model.clock_visible,
        "ntp_server": model.ntp_server or "",
    }
    storage.save_toml(PREFERENCES_FILE, prefs)
    logger.debug("prefs_sync_flushed")


def _serialize_layout(layout: UILayout) -> dict[str, object]:
    return {
        "left": {"panels": list(layout.left.panels), "active": layout.left.active or ""},
        "right": {"panels": list(layout.right.panels), "active": layout.right.active or ""},
        "bottom": {"panels": list(layout.bottom.panels), "active": layout.bottom.active or ""},
        # Hotkeys are a code-level default (not user-editable in v1), so they are
        # intentionally not persisted — _coerce_layout always uses DEFAULT_LAYOUT's.
        "strips_visible": layout.strips_visible,
    }


class PrefsSync:
    def __init__(self, store: UIStore, app: App) -> None:
        self._store = store
        self._app = app
        self._timer: Timer | None = None
        self._unsub = store.subscribe(self._on_change)

    def close(self) -> None:
        self._unsub()
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
            flush_prefs(self._store.model)

    def _on_change(self, old: UIModel, new: UIModel) -> None:
        if not any(getattr(old, f) != getattr(new, f) for f in PREFS_FIELDS):
            return
        if self._timer is not None:
            return
        self._timer = self._app.set_timer(_DEBOUNCE_SECONDS, self._flush)

    def _flush(self) -> None:
        self._timer = None
        flush_prefs(self._store.model)
