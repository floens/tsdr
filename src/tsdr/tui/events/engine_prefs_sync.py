"""EnginePrefsSync — persists engine + device config on engine events, debounced.

EventRouter calls `mark_dirty()` from its @on handlers for the events that
change persisted state (ConfigChanged, PipelineChanged, DeviceAdded,
DeviceRemoved, FocusChanged); this class throttles them to one
`save_device(engine)` + `save_engine_config(engine)` write per 250 ms window
(the timer starts at the first event and is not reset by later ones, so a
long burst like continuous dial scrolling writes ~4x/s, each flush capturing
the then-current state).
"""

from __future__ import annotations

import logging

from textual.app import App
from textual.timer import Timer

from tsdr.core.preferences import save_device, save_engine_config
from tsdr.core.sdr.engine import SDREngine

logger = logging.getLogger(__name__)

_DEBOUNCE_SECONDS = 0.25


class EnginePrefsSync:
    def __init__(self, engine: SDREngine, app: App) -> None:
        self._engine = engine
        self._app = app
        self._timer: Timer | None = None

    def mark_dirty(self) -> None:
        """Schedule a throttled write: at most one per 250 ms window."""
        if self._timer is not None:
            return
        self._timer = self._app.set_timer(_DEBOUNCE_SECONDS, self._flush)

    def close(self) -> None:
        """Stop the timer and flush any pending change synchronously."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
            self._write()

    def _flush(self) -> None:
        self._timer = None
        self._write()

    def _write(self) -> None:
        save_device(self._engine)
        save_engine_config(self._engine)
        logger.debug("engine_prefs_sync_flushed")
