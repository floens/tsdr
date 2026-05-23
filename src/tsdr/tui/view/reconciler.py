"""Keyed reconciler — diffs a desired WidgetSpec tree against the live widget tree.

Maintains a `key -> Widget` map so EventRouter can do
``reconciler.get(key).update_xxx(event)`` for stream events.

``schedule(root)`` coalesces to at most one reconcile per frame via
``call_after_refresh``; the reconcile runs inside ``app.batch_update()`` so
mount/remove/setattr happen atomically. Keying preserves widget identity across
renders so stateful widgets (FFT history, waterfall buffer, decoder accumulators)
keep their state.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from textual.app import App
from textual.widget import Widget

from tsdr.tui.view.spec import WidgetSpec

logger = logging.getLogger(__name__)


class Reconciler:
    def __init__(self, app: App, factory: Mapping[str, Callable[[], Widget]]) -> None:
        self._app = app
        self._factory = factory
        self._by_key: dict[str, Widget] = {}
        self._target: WidgetSpec | None = None
        self._scheduled = False
        self._running = False

    def get(self, key: str) -> Widget | None:
        """Return the live widget for `key`, or None if unmounted.

        Callers must check `if w is not None` — between unmount and the next
        reconcile, stream events may target a widget that no longer exists.
        """
        return self._by_key.get(key)

    def schedule(self, root: WidgetSpec) -> None:
        """Queue a reconcile against `root` for the next frame.

        Multiple calls within the same frame coalesce — only the most recent
        spec is reconciled.
        """
        self._target = root
        if self._scheduled:
            return
        self._scheduled = True
        self._app.call_after_refresh(self._run)

    async def run_initial(self, root: WidgetSpec) -> None:
        """Awaited first reconcile from on_mount, so always-present widgets
        (#command-input, status bar) exist before any worker can post events."""
        with self._app.batch_update():
            await self._reconcile_root(root)

    async def _run(self) -> None:
        # Re-entrance guard: a watcher fired during setattr may mutate the
        # store, queueing another _run. Without this guard, two _run coroutines
        # could interleave at mount/remove await points and race on
        # parent.children / _by_key. With it, the queued _run early-returns
        # and the in-flight pass picks up the new _target via the while-loop.
        if self._running:
            return
        self._running = True
        try:
            while self._target is not None:
                spec = self._target
                self._target = None
                self._scheduled = False
                with self._app.batch_update():
                    await self._reconcile_root(spec)
        finally:
            self._running = False

    async def _reconcile_root(self, root: WidgetSpec) -> None:
        """The root spec's children become the screen's children."""
        await self._reconcile_children(self._app.screen, root.children)

    async def _reconcile_children(self, parent: Widget, specs: tuple[WidgetSpec, ...]) -> None:
        desired_keys = {s.key for s in specs}

        # Phase 1 — remove children whose key disappeared
        removed: list[str] = []
        for w in list(parent.children):
            key = self._key_of(w)
            if key is not None and key not in desired_keys:
                removed.append(key)
                self._forget_subtree(w)
                await w.remove()

        # Phase 2 — walk specs in order; mount missing, update existing
        added: list[str] = []
        anchor: Widget | None = None
        for spec in specs:
            existing = self._by_key.get(spec.key)
            if existing is None:
                w = self._factory[spec.kind]()
                w.id = _safe_id(spec.key)
                self._by_key[spec.key] = w
                if anchor is None:
                    if parent.children:
                        await parent.mount(w, before=parent.children[0])
                    else:
                        await parent.mount(w)
                else:
                    await parent.mount(w, after=anchor)
                added.append(spec.key)
            else:
                w = existing
            self._apply_props(w, spec.props)
            if spec.children:
                await self._reconcile_children(w, spec.children)
            anchor = w

        if added or removed:
            parent_id = getattr(parent, "id", None) or type(parent).__name__
            logger.debug("reconcile_diff parent=%s added=%r removed=%r", parent_id, added, removed)

    def _key_of(self, w: Widget) -> str | None:
        for k, ww in self._by_key.items():
            if ww is w:
                return k
        return None

    def _forget_subtree(self, root: Widget) -> None:
        """Drop every `_by_key` entry whose widget is `root` or a descendant.

        Captures descendants while the widget is still attached, so the caller
        can `await root.remove()` immediately after.
        """
        descendants = _collect_descendants(root)
        keys_to_forget = [k for k, ww in self._by_key.items() if ww in descendants]
        for k in keys_to_forget:
            del self._by_key[k]

    def _apply_props(self, w: Widget, props: Mapping[str, Any]) -> None:
        """Set every prop as a reactive attribute. Textual handles equality and refresh.

        Each setattr is isolated so one bad prop (e.g. a typo'd `id` field, or
        a watcher that raises) doesn't abort the rest of the frame's reconcile.
        """
        for name, value in props.items():
            try:
                setattr(w, name, value)
            except Exception as e:  # noqa: BLE001 — isolate one prop from siblings
                logger.error(
                    "reconciler_apply_prop_failed widget=%s prop=%s error=%r",
                    type(w).__name__,
                    name,
                    e,
                    exc_info=True,
                )


def _collect_descendants(root: Widget) -> set[Widget]:
    seen: set[Widget] = {root}
    stack = list(root.children)
    while stack:
        w = stack.pop()
        seen.add(w)
        stack.extend(w.children)
    return seen


def _safe_id(key: str) -> str:
    """CSS-safe widget id from a WidgetSpec key.

    Textual widget ids match `[a-zA-Z][a-zA-Z0-9_-]*`; colons in our key
    namespace (e.g. ``decoder:rtl0:rds``) are replaced with `--`.
    """
    return key.replace(":", "--")
