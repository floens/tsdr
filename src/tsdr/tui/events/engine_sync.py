"""EngineSync — pushes derived engine config when UIModel changes.

Derives:
  * update_rate_fps: 60 when image_mode, else 20 (engine-global).
  * calculate_constellation: True when image_mode AND the stats panel is
    active on any edge (the constellation widget mounts alongside stats
    wherever the user pinned it), applied to the focused device — and
    explicitly disabled on the previously-focused device when focus moves.

Dedupes per-device for calculate_constellation so an unrelated model change
doesn't re-push, but a focus change still propagates correctly.
"""

from __future__ import annotations

import logging

from tsdr.core.sdr.engine import SDREngine
from tsdr.tui.model import UIModel
from tsdr.tui.model.store import UIStore

logger = logging.getLogger(__name__)


class EngineSync:
    def __init__(self, store: UIStore, engine: SDREngine) -> None:
        self._engine = engine
        self._last_rate_fps: int | None = None
        # Per-device cache: device_id -> last value we pushed. A global bool
        # would mis-skip after focus change (new device never gets True, old
        # device never gets False).
        self._calc_constellation_by_device: dict[str, bool] = {}
        self._unsub = store.subscribe(self._on_change)
        # Push initial state — subscribers only fire on change, so otherwise
        # image_mode=True from prefs would leave the engine at default FPS.
        self._on_change(store.model, store.model)

    def close(self) -> None:
        self._unsub()

    def _on_change(self, _old: UIModel, new: UIModel) -> None:
        rate = 60 if new.image_mode else 20
        if rate != self._last_rate_fps:
            self._engine.update_global_config(update_rate_fps=rate)
            self._last_rate_fps = rate
            logger.debug("engine_sync_rate device_global rate_fps=%d", rate)

        # Read focus from the model, not the engine — the model is the
        # source of truth the rest of the UI agrees with, and the engine
        # may transiently disagree during a multi-event sequence.
        focused_id = new.focused_device_id
        stats_active = (
            new.layout.left.active == "stats"
            or new.layout.right.active == "stats"
            or new.layout.bottom.active == "stats"
        )
        want_calc = new.image_mode and stats_active

        # Disable on any device we previously enabled that isn't the focused
        # one — covers focus changes and device removal.
        for dev_id, last in list(self._calc_constellation_by_device.items()):
            if last and dev_id != focused_id and dev_id in self._engine.devices:
                self._engine.update_device_config(dev_id, calculate_constellation=False)
                self._calc_constellation_by_device[dev_id] = False
                logger.debug(
                    "engine_sync_constellation device=%s calculate=False reason=focus_changed",
                    dev_id,
                )

        if focused_id is not None and focused_id in self._engine.devices:
            last = self._calc_constellation_by_device.get(focused_id, False)
            if want_calc != last:
                self._engine.update_device_config(focused_id, calculate_constellation=want_calc)
                self._calc_constellation_by_device[focused_id] = want_calc
                logger.debug(
                    "engine_sync_constellation device=%s calculate=%s",
                    focused_id,
                    want_calc,
                )
