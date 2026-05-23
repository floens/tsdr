"""UIStore — owns the live UIModel and notifies subscribers on change.

Every mutation computes a new immutable model via dataclasses.replace,
short-circuits if new == old, appends a Mutation entry to a ring buffer,
and fires subscribers in registration order with (old, new).

Main-thread only — not thread-safe by design.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from tsdr.tui.model import DeviceUIState, UIModel

logger = logging.getLogger(__name__)

_MUTATION_LOG_SIZE = 200

Subscriber = Callable[[UIModel, UIModel], None]


@dataclass(frozen=True, eq=False)
class Mutation:
    op: str
    args: dict[str, Any]
    ts: float


class UIStore:
    def __init__(self, initial: UIModel) -> None:
        self._model = initial
        self._subscribers: list[Subscriber] = []
        self._mutations: deque[Mutation] = deque(maxlen=_MUTATION_LOG_SIZE)

    @property
    def model(self) -> UIModel:
        return self._model

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe function."""
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(fn)
            except ValueError:
                pass

        return unsubscribe

    def recent_mutations(self) -> tuple[Mutation, ...]:
        return tuple(self._mutations)

    def update(self, **changes: Any) -> None:
        """Update top-level UIModel fields."""
        self._commit("update", changes, replace(self._model, **changes))

    def update_console(self, **changes: Any) -> None:
        """Update fields inside the ConsoleUIState subtree."""
        new_console = replace(self._model.console, **changes)
        self._commit("update_console", changes, replace(self._model, console=new_console))

    def set_devices(self, devices: tuple[DeviceUIState, ...]) -> None:
        """Replace the entire devices tuple."""
        self._commit(
            "set_devices",
            {"devices": devices},
            replace(self._model, devices=devices),
        )

    def update_device(self, device_id: str, **changes: Any) -> None:
        """Replace one device in the devices tuple. No-op if device_id absent."""
        devices = self._model.devices
        for i, d in enumerate(devices):
            if d.device_id == device_id:
                new_d = replace(d, **changes)
                new_devices = devices[:i] + (new_d,) + devices[i + 1 :]
                self._commit(
                    "update_device",
                    {"device_id": device_id, **changes},
                    replace(self._model, devices=new_devices),
                )
                return

    def upsert_device(self, device_id: str, **changes: Any) -> None:
        """Add device if absent, else update in place."""
        devices = self._model.devices
        for i, d in enumerate(devices):
            if d.device_id == device_id:
                new_d = replace(d, **changes)
                new_devices = devices[:i] + (new_d,) + devices[i + 1 :]
                self._commit(
                    "upsert_device",
                    {"device_id": device_id, **changes},
                    replace(self._model, devices=new_devices),
                )
                return
        new_d = DeviceUIState(device_id=device_id, **changes)
        new_devices = devices + (new_d,)
        self._commit(
            "upsert_device",
            {"device_id": device_id, **changes},
            replace(self._model, devices=new_devices),
        )

    def remove_device(self, device_id: str) -> None:
        """Remove a device by id. No-op if absent. Clears focused_device_id if it matched."""
        devices = self._model.devices
        new_devices = tuple(d for d in devices if d.device_id != device_id)
        if new_devices == devices:
            return
        new_focused = (
            None if self._model.focused_device_id == device_id else self._model.focused_device_id
        )
        self._commit(
            "remove_device",
            {"device_id": device_id},
            replace(self._model, devices=new_devices, focused_device_id=new_focused),
        )

    def _commit(self, op: str, args: dict[str, Any], new_model: UIModel) -> None:
        if new_model == self._model:
            return
        old, self._model = self._model, new_model
        self._mutations.append(Mutation(op=op, args=args, ts=time.monotonic()))
        logger.debug("ui_store_mutation op=%s args=%r", op, args)
        for sub in list(self._subscribers):
            try:
                sub(old, new_model)
            except Exception as e:  # noqa: BLE001 — isolate one buggy subscriber from the rest
                logger.error(
                    "ui_store_subscriber_error op=%s subscriber=%r error=%r",
                    op,
                    sub,
                    e,
                    exc_info=True,
                )


_store: UIStore | None = None


def init_ui_store(initial: UIModel | None = None) -> UIStore:
    global _store
    _store = UIStore(initial if initial is not None else UIModel())
    return _store


def get_ui_store() -> UIStore:
    if _store is None:
        raise RuntimeError("UIStore not initialized")
    return _store
