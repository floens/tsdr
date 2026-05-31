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

from tsdr.tui.model import DeviceUIState, Edge, EdgePanels, UILayout, UIModel

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

    def toggle_panel(self, panel_id: str) -> None:
        layout = self._model.layout
        edge_name = _find_panel_edge(layout, panel_id)
        if edge_name is None:
            logger.warning("panel_toggle_unknown panel=%s", panel_id)
            return
        edge = _get_edge(layout, edge_name)
        new_active = None if edge.active == panel_id else panel_id
        new_edge = replace(edge, active=new_active)
        new_layout = _replace_edge(layout, edge_name, new_edge)
        self._commit(
            "toggle_panel",
            {"panel_id": panel_id, "edge": edge_name, "active": new_active},
            replace(self._model, layout=new_layout),
        )

    def set_panel_active(self, edge: Edge, panel_id: str | None) -> None:
        layout = self._model.layout
        edge_panels = _get_edge(layout, edge)
        if panel_id is not None and panel_id not in edge_panels.panels:
            logger.warning("panel_set_active_invalid edge=%s panel=%s", edge, panel_id)
            return
        new_edge = replace(edge_panels, active=panel_id)
        new_layout = _replace_edge(layout, edge, new_edge)
        self._commit(
            "set_panel_active",
            {"edge": edge, "panel_id": panel_id},
            replace(self._model, layout=new_layout),
        )

    def set_strips_visible(self, visible: bool) -> None:
        layout = self._model.layout
        if layout.strips_visible == visible:
            return
        new_layout = replace(layout, strips_visible=visible)
        self._commit(
            "set_strips_visible",
            {"visible": visible},
            replace(self._model, layout=new_layout),
        )

    def move_panel(self, panel_id: str, target_edge: Edge, *, index: int | None = None) -> None:
        layout = self._model.layout
        src_edge = _find_panel_edge(layout, panel_id)

        if src_edge == target_edge:
            edge = _get_edge(layout, src_edge)
            without = tuple(p for p in edge.panels if p != panel_id)
            new_panels = _insert_at(without, panel_id, index)
            new_active = edge.active if edge.active in new_panels else None
            new_layout = _replace_edge(
                layout, src_edge, EdgePanels(panels=new_panels, active=new_active)
            )
        else:
            if src_edge is not None:
                src = _get_edge(layout, src_edge)
                new_src_panels = tuple(p for p in src.panels if p != panel_id)
                new_src_active = src.active if src.active != panel_id else None
                layout = _replace_edge(
                    layout, src_edge, EdgePanels(panels=new_src_panels, active=new_src_active)
                )
            tgt = _get_edge(layout, target_edge)
            existing = tuple(p for p in tgt.panels if p != panel_id)
            new_tgt_panels = _insert_at(existing, panel_id, index)
            new_tgt_active = tgt.active if tgt.active in new_tgt_panels else None
            new_layout = _replace_edge(
                layout, target_edge, EdgePanels(panels=new_tgt_panels, active=new_tgt_active)
            )
        self._commit(
            "move_panel",
            {"panel_id": panel_id, "target_edge": target_edge, "index": index},
            replace(self._model, layout=new_layout),
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


def _find_panel_edge(layout: UILayout, panel_id: str) -> Edge | None:
    if panel_id in layout.left.panels:
        return "left"
    if panel_id in layout.right.panels:
        return "right"
    if panel_id in layout.bottom.panels:
        return "bottom"
    return None


def _get_edge(layout: UILayout, edge: Edge) -> EdgePanels:
    if edge == "left":
        return layout.left
    if edge == "right":
        return layout.right
    return layout.bottom


def _replace_edge(layout: UILayout, edge: Edge, new_edge: EdgePanels) -> UILayout:
    if edge == "left":
        return replace(layout, left=new_edge)
    if edge == "right":
        return replace(layout, right=new_edge)
    return replace(layout, bottom=new_edge)


def _insert_at(items: tuple[str, ...], value: str, index: int | None) -> tuple[str, ...]:
    if index is None or index >= len(items):
        return items + (value,)
    i = max(0, index)
    return items[:i] + (value,) + items[i:]


_store: UIStore | None = None


def init_ui_store(initial: UIModel | None = None) -> UIStore:
    global _store
    _store = UIStore(initial if initial is not None else UIModel())
    return _store


def get_ui_store() -> UIStore:
    if _store is None:
        raise RuntimeError("UIStore not initialized")
    return _store
