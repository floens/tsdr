"""Smoke test: the interactive doctor screen must compose without crashing.

Exercises compose + resize under ansi_color=True with the KittyImageWidget and
OOB-escape plumbing (headless driver, so no real graphics are emitted).
"""

from __future__ import annotations

import asyncio

import pytest
from textual.css.query import NoMatches
from textual.events import Resize
from textual.geometry import Size
from textual.widgets import Static, TabbedContent, TabPane

from tsdr.tui.doctor import app as doctor_app
from tsdr.tui.doctor.app import DoctorApp
from tsdr.tui.doctor.checks import CheckResult, Status
from tsdr.tui.widgets.kitty_image import KittyImageWidget

_TABS = ("tab-render", "tab-image", "tab-system", "tab-live-image", "tab-live-text")


def _results() -> list[CheckResult]:
    return [
        CheckResult("truecolor", Status.OK, "24-bit", True, "render"),
        CheckResult("kitty_graphics", Status.UNKNOWN, "not a tty", True, "render"),
        CheckResult("window_size", Status.OK, "120x40 cells, 960x640 px", False, "render"),
        CheckResult("pixel_size", Status.OK, "8x16 px/cell (source: stub)", False, "render"),
        CheckResult("terminal_size", Status.OK, "120x40", False, "protocol"),
        CheckResult("synchronized_output", Status.WARN, "n/a", False, "protocol"),
        CheckResult("ssh_session", Status.OK, "local", False, "session"),
        CheckResult("python_version", Status.OK, "3.13", True, "runtime"),
        CheckResult("audio_backend", Status.OK, "soundcard", True, "deps"),
        CheckResult("cpu", Status.OK, "8 cores", False, "system"),
        CheckResult("memory", Status.OK, "32 GiB", False, "system"),
    ]


def test_doctor_app_mounts_and_switches_tabs() -> None:
    async def go() -> None:
        app = DoctorApp(_results())
        app._kitty_ok = True  # mount the kitty demo widgets/tabs this test navigates
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            for pane in _TABS:
                tabs.active = pane
                await pilot.pause()
            await pilot.resize_terminal(100, 30)
            await pilot.pause()

    asyncio.run(go())


def test_live_image_tab_animates_and_hides_on_switch() -> None:
    async def go() -> None:
        app = DoctorApp(_results())
        app._kitty_ok = True  # the Live (image) tab + kitty widgets are gated on this
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-live-image"
            await pilot.pause()

            for _ in range(5):
                app._animate_live()
            await pilot.pause()

            wf = app.query_one("#wf-image", KittyImageWidget)
            sp = app.query_one("#spectrum-image", KittyImageWidget)
            assert len(wf._images) >= 1  # waterfall strips transmitted
            assert "spectrum" in sp._images  # full-frame spectrum drawn
            assert wf._visible and sp._visible

            tabs.active = "tab-render"
            await pilot.pause()
            # The widgets hide themselves on tab switch (data kept for cheap re-show).
            assert not wf._visible
            assert not sp._visible

            tabs.active = "tab-live-image"
            await pilot.pause()
            assert wf._visible and sp._visible  # re-shown on return

    asyncio.run(go())


def test_live_text_tab_animates_glyph_waterfall() -> None:
    async def go() -> None:
        app = DoctorApp(_results())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-live-text"
            await pilot.pause()
            for _ in range(5):
                app._animate_live()
            await pilot.pause()
            glyph = app.query_one("#glyph-wf")
            assert glyph._strips  # glyph-only waterfall accumulated cached strips

    asyncio.run(go())


def test_checker_in_image_tab_hidden_and_reshown() -> None:
    async def go() -> None:
        app = DoctorApp(_results())
        app._kitty_ok = True  # the Image-tab checkerboard is gated on this
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            doc = app.query_one("#doctor-image", KittyImageWidget)

            tabs.active = "tab-image"
            await pilot.pause()
            assert "doctor" in doc._images  # checkerboard drawn on its own resize
            assert doc._visible

            tabs.active = "tab-render"
            await pilot.pause()
            assert not doc._visible  # widget hides its image when its tab hides

            tabs.active = "tab-image"
            await pilot.pause()
            assert doc._visible  # re-shown on return

    asyncio.run(go())


def test_kitty_widgets_absent_when_unsupported() -> None:
    """When kitty graphics aren't detected, the demo widgets/tab are not mounted."""

    async def go() -> None:
        app = DoctorApp(_results())
        app._kitty_ok = False
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            pane_ids = {p.id for p in tabs.query(TabPane)}
            assert "tab-live-image" not in pane_ids
            assert "tab-image" not in pane_ids
            for missing in ("#doctor-image", "#spectrum-image", "#wf-image"):
                with pytest.raises(NoMatches):
                    app.query_one(missing)

    asyncio.run(go())


def test_resize_refreshes_stale_geometry_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor_app,
        "check_window_size",
        lambda spec=None: CheckResult("window_size", Status.OK, "RESIZED window", False, "render"),
    )
    monkeypatch.setattr(
        doctor_app,
        "check_pixel_size",
        lambda spec=None: CheckResult("pixel_size", Status.OK, "RESIZED px/cell", False, "render"),
    )
    monkeypatch.setattr(
        doctor_app,
        "check_terminal_size",
        lambda: CheckResult("terminal_size", Status.OK, "RESIZED cols", False, "protocol"),
    )

    async def go() -> None:
        app = DoctorApp(_results())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.resize_terminal(90, 30)
            await pilot.pause()
            by_name = {r.name: r for r in app._results}
            assert by_name["window_size"].summary == "RESIZED window"
            assert by_name["pixel_size"].summary == "RESIZED px/cell"
            assert by_name["terminal_size"].summary == "RESIZED cols"
            content = str(app.query_one("#res-kitty", Static).content)
            assert "RESIZED window" in content and "RESIZED px/cell" in content
            assert "RESIZED cols" in str(app.query_one("#res-protocol", Static).content)

    asyncio.run(go())


def test_resize_reports_in_band_pixel_source() -> None:
    """A Resize carrying pixels drives the pixel_size check live, naming the source."""

    async def go() -> None:
        app = DoctorApp(_results())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # 960x640 px over a 120x40 cell grid -> 8x16 px/cell, sourced from the
            # in-band resize pixels rather than the CSI snapshot.
            app.on_resize(Resize(Size(120, 40), Size(120, 40), pixel_size=Size(960, 640)))
            await pilot.pause()
            summary = {r.name: r for r in app._results}["pixel_size"].summary
            assert "8x16 px/cell" in summary
            assert "live Resize event" in summary
            assert "8x16 px/cell" in str(app.query_one("#res-kitty", Static).content)

    asyncio.run(go())
