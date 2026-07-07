"""Smoke + layout-geometry tests for the live app.

Catches CSS / widget initialization bugs that pass unit tests but blow up in
the compositor, and asserts the dock layout has the right shape after mount:
the single bottom panel-bar spans full width and clears the overlay zone, and
active side panels sit inside their dock above that zone.
"""

from __future__ import annotations

import asyncio

from textual.css.query import NoMatches

from tsdr.tui.app import TSDRApp
from tsdr.tui.model.store import get_ui_store

SCREEN_W = 120
SCREEN_H = 40
# Bottom 5 rows of the screen are reserved for the overlay layer:
# 3 (collapsed ConsoleWidget) + 1 (StatusBar) + 1 (StatusBar margin) = 5.
OVERLAY_ROWS = 5


def _run(coro_factory) -> None:
    asyncio.run(coro_factory())


def test_app_mounts_and_renders_on_resize() -> None:
    """Mount + resize must not crash the compositor."""

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(100, 32)
            await pilot.pause()

    _run(go)


def test_panel_bar_full_width_above_overlay() -> None:
    """The single panel-bar spans the full screen width and ends exactly
    OVERLAY_ROWS rows above the bottom edge (just above the console)."""

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause()
            get_ui_store().set_strips_visible(True)
            await pilot.pause()
            bar = app.query_one("#panel-bar")
            assert bar.region.width == SCREEN_W, (
                f"panel-bar width {bar.region.width}, expected {SCREEN_W}"
            )
            bottom = bar.region.y + bar.region.height
            assert bottom == SCREEN_H - OVERLAY_ROWS, (
                f"panel-bar ends at row {bottom}, expected {SCREEN_H - OVERLAY_ROWS}"
            )

    _run(go)


def test_panel_bar_stays_full_width_with_side_panel_open() -> None:
    """Opening a right-dock panel must not narrow the bar: it sits below the
    docks row and spans under the side docks."""

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause()
            get_ui_store().set_panel_active("right", "stats")
            await pilot.pause()
            bar = app.query_one("#panel-bar")
            assert bar.region.width == SCREEN_W
            # The dock itself must stay above the overlay zone.
            dock = app.query_one("#dock--right")
            assert dock.region.y + dock.region.height <= SCREEN_H - OVERLAY_ROWS

    _run(go)


def test_active_side_panel_mounts_content() -> None:
    """Activating a left/right panel mounts its panel-content wrapper inside the
    dock; no separate strip exists on the edge anymore."""

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause()
            get_ui_store().set_panel_active("left", "decoder-output")
            await pilot.pause()
            wrapper = app.query_one("#panel-content--left")
            assert wrapper.parent.id == "dock--left"
            # No per-edge strip widgets remain.
            for selector in ("#dock--left--strip", "#dock--right--strip", "#dock--bottom--strip"):
                try:
                    app.query_one(selector)
                except NoMatches:
                    continue
                raise AssertionError(f"{selector} should not exist")

    _run(go)


def test_closing_panel_unmounts_panel_content() -> None:
    """Regression: toggling a panel off must remove the panel-content wrapper from
    the live tree. The reconciler must run its children pass even when a dock
    transitions from 1 child to 0 (empty children tuple)."""

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause()
            store = get_ui_store()
            store.toggle_panel("stats")
            await pilot.pause()
            app.query_one("#panel-content--right")  # mounted while active
            store.toggle_panel("stats")
            await pilot.pause()
            try:
                app.query_one("#panel-content--right")
            except NoMatches:
                pass
            else:
                raise AssertionError("#panel-content--right still mounted after panel closed")
            assert len(app.query_one("#dock--right").children) == 0

    _run(go)


def test_move_active_panel_to_earlier_dock_keeps_widget() -> None:
    """Regression: moving a mounted panel from the right dock to the left dock
    must remount it under the new dock. docks-row reconciles left before right,
    so the key is still mapped to the old (not-yet-removed) widget when the left
    wrapper is built; the reconciler must mount a fresh instance rather than
    reuse the stale one in place and drop it."""

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause()
            store = get_ui_store()
            store.set_panel_active("right", "stats")
            await pilot.pause()
            app.query_one("#panel-content--right")  # mounted on the right

            store.move_panel("stats", "left")
            store.set_panel_active("left", "stats")
            await pilot.pause()

            wrapper = app.query_one("#panel-content--left")
            assert [w.__class__.__name__ for w in wrapper.children] == ["StatsWidget"]

    _run(go)
