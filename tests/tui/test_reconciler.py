"""Reconciler tests with stub Widget/App — no Textual app loop needed."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any

from tsdr.tui.view.reconciler import Reconciler, _safe_id
from tsdr.tui.view.spec import WidgetSpec


class FakeWidget:
    """Stand-in for textual.widget.Widget with the interface the reconciler uses."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.children: list[FakeWidget] = []
        self.parent: FakeWidget | None = None
        self.removed = False
        self.props_log: list[tuple[str, Any]] = []

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D401
        # Record prop assignments after construction; lets tests verify
        # the reconciler called setattr the right way.
        if "_initialized" in self.__dict__ and name not in {
            "id",
            "children",
            "parent",
            "removed",
            "props_log",
        }:
            self.props_log.append((name, value))
        object.__setattr__(self, name, value)
        if name == "props_log":
            object.__setattr__(self, "_initialized", True)

    async def mount(
        self,
        *widgets: FakeWidget,
        before: FakeWidget | int | None = None,
        after: FakeWidget | int | None = None,
    ) -> None:
        for w in widgets:
            w.parent = self
            if before is not None:
                idx = before if isinstance(before, int) else self.children.index(before)
                self.children.insert(idx, w)
            elif after is not None:
                idx = after if isinstance(after, int) else self.children.index(after)
                self.children.insert(idx + 1, w)
            else:
                self.children.append(w)

    async def remove(self) -> None:
        self.removed = True
        if self.parent:
            self.parent.children.remove(self)
            self.parent = None


class FakeApp:
    def __init__(self) -> None:
        self.screen = FakeWidget()
        self.screen.id = "screen"
        self.scheduled: list[Any] = []
        self.batch_calls = 0

    @contextmanager
    def batch_update(self):
        self.batch_calls += 1
        yield

    def call_after_refresh(self, callback: Any) -> bool:
        self.scheduled.append(callback)
        return True


def make_factory(*kinds: str) -> dict[str, Any]:
    return dict.fromkeys(kinds, FakeWidget)


def run(coro):
    return asyncio.run(coro)


# ----------------------------- mount/remove ---------------------------------


def test_initial_mount_creates_widgets() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a", "b"))
    spec = WidgetSpec(
        "root",
        "root",
        {},
        (
            WidgetSpec("a", "a"),
            WidgetSpec("b", "b"),
        ),
    )
    run(r.run_initial(spec))
    assert [c.id for c in app.screen.children] == ["a", "b"]
    assert app.batch_calls == 1


def test_get_returns_mounted_widget() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a"))
    run(r.run_initial(WidgetSpec("root", "root", {}, (WidgetSpec("a", "a"),))))
    w = r.get("a")
    assert w is not None
    assert w.id == "a"


def test_get_returns_none_for_unknown_key() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory())
    assert r.get("nope") is None


def test_unchanged_spec_does_not_remount() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a"))
    spec = WidgetSpec("root", "root", {}, (WidgetSpec("a", "a"),))
    run(r.run_initial(spec))
    first = app.screen.children[0]
    run(r.run_initial(spec))
    assert app.screen.children[0] is first
    assert not first.removed


def test_added_child_appends() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a", "b", "c"))
    run(
        r.run_initial(
            WidgetSpec(
                "root",
                "root",
                {},
                (
                    WidgetSpec("a", "a"),
                    WidgetSpec("b", "b"),
                ),
            )
        )
    )
    a, b = app.screen.children
    run(
        r.run_initial(
            WidgetSpec(
                "root",
                "root",
                {},
                (
                    WidgetSpec("a", "a"),
                    WidgetSpec("b", "b"),
                    WidgetSpec("c", "c"),
                ),
            )
        )
    )
    assert [c.id for c in app.screen.children] == ["a", "b", "c"]
    assert app.screen.children[0] is a
    assert app.screen.children[1] is b


def test_inserted_child_at_start() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a", "b"))
    run(r.run_initial(WidgetSpec("root", "root", {}, (WidgetSpec("b", "b"),))))
    b = app.screen.children[0]
    run(
        r.run_initial(
            WidgetSpec(
                "root",
                "root",
                {},
                (
                    WidgetSpec("a", "a"),
                    WidgetSpec("b", "b"),
                ),
            )
        )
    )
    assert [c.id for c in app.screen.children] == ["a", "b"]
    assert app.screen.children[1] is b


def test_inserted_child_in_middle() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a", "b", "c"))
    run(
        r.run_initial(
            WidgetSpec(
                "root",
                "root",
                {},
                (
                    WidgetSpec("a", "a"),
                    WidgetSpec("c", "c"),
                ),
            )
        )
    )
    a, c = app.screen.children
    run(
        r.run_initial(
            WidgetSpec(
                "root",
                "root",
                {},
                (
                    WidgetSpec("a", "a"),
                    WidgetSpec("b", "b"),
                    WidgetSpec("c", "c"),
                ),
            )
        )
    )
    assert [c.id for c in app.screen.children] == ["a", "b", "c"]
    assert app.screen.children[0] is a
    assert app.screen.children[2] is c


def test_removed_child_unmounts() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a", "b"))
    run(
        r.run_initial(
            WidgetSpec(
                "root",
                "root",
                {},
                (
                    WidgetSpec("a", "a"),
                    WidgetSpec("b", "b"),
                ),
            )
        )
    )
    a, b = app.screen.children
    run(r.run_initial(WidgetSpec("root", "root", {}, (WidgetSpec("a", "a"),))))
    assert [c.id for c in app.screen.children] == ["a"]
    assert b.removed is True
    assert r.get("b") is None
    assert r.get("a") is a


def test_removed_subtree_clears_descendants_from_map() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("parent", "child"))
    run(
        r.run_initial(
            WidgetSpec(
                "root",
                "root",
                {},
                (WidgetSpec("parent", "parent", {}, (WidgetSpec("child", "child"),)),),
            )
        )
    )
    assert r.get("parent") is not None
    assert r.get("child") is not None
    run(r.run_initial(WidgetSpec("root", "root", {}, ())))
    assert r.get("parent") is None
    assert r.get("child") is None


# ----------------------------- props ----------------------------------------


def test_props_applied_via_setattr() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a"))
    run(
        r.run_initial(
            WidgetSpec(
                "root", "root", {}, (WidgetSpec("a", "a", {"zoom": 2.0, "image_mode": True}),)
            )
        )
    )
    w = r.get("a")
    assert ("zoom", 2.0) in w.props_log
    assert ("image_mode", True) in w.props_log


def test_props_re_applied_on_each_reconcile() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a"))
    run(r.run_initial(WidgetSpec("root", "root", {}, (WidgetSpec("a", "a", {"zoom": 1.0}),))))
    w = r.get("a")
    w.props_log.clear()
    run(r.run_initial(WidgetSpec("root", "root", {}, (WidgetSpec("a", "a", {"zoom": 5.0}),))))
    assert ("zoom", 5.0) in w.props_log


# ----------------------------- scheduling -----------------------------------


def test_schedule_coalesces_within_a_frame() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a"))
    spec1 = WidgetSpec("root", "root", {}, (WidgetSpec("a", "a", {"zoom": 1.0}),))
    spec2 = WidgetSpec("root", "root", {}, (WidgetSpec("a", "a", {"zoom": 2.0}),))
    spec3 = WidgetSpec("root", "root", {}, (WidgetSpec("a", "a", {"zoom": 3.0}),))
    r.schedule(spec1)
    r.schedule(spec2)
    r.schedule(spec3)
    assert len(app.scheduled) == 1
    run(app.scheduled[0]())
    assert ("zoom", 3.0) in r.get("a").props_log


def test_schedule_after_run_re_schedules() -> None:
    app = FakeApp()
    r = Reconciler(app, make_factory("a"))
    r.schedule(WidgetSpec("root", "root", {}, (WidgetSpec("a", "a"),)))
    run(app.scheduled[0]())
    app.scheduled.clear()
    r.schedule(WidgetSpec("root", "root", {}, ()))
    assert len(app.scheduled) == 1
    run(app.scheduled[0]())
    assert r.get("a") is None


# ----------------------------- safe id --------------------------------------


def test_safe_id_replaces_colons() -> None:
    assert _safe_id("decoder:rtl0:rds") == "decoder--rtl0--rds"


def test_safe_id_passes_through_simple() -> None:
    assert _safe_id("spectrum") == "spectrum"
