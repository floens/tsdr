"""Binding-table integrity and table-driven dispatch."""

from __future__ import annotations

import asyncio

from tsdr.tui.app import TSDRApp
from tsdr.tui.console.terminal_input import TerminalInput
from tsdr.tui.keybindings import BINDINGS, Action, Context, build_lookup, format_keys
from tsdr.tui.model.store import get_ui_store


def test_lookup_builds_without_duplicates_or_gaps():
    assert build_lookup()


def test_new_view_keys_are_bound():
    lookup = build_lookup()
    assert lookup[(Context.GLOBAL, "c")] is Action.TUNE_MODE_TOGGLE
    assert lookup[(Context.GLOBAL, "C")] is Action.DISPLAY_CENTER_ON_DIAL


def test_every_binding_has_display_label():
    for binding in BINDINGS:
        assert format_keys(binding), binding.description


def test_dispatch_smoke():
    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            store = get_ui_store()
            cmd_input = app.query_one("#command-input", TerminalInput)

            before = store.model.image_mode
            await pilot.press("i")
            await pilot.pause()
            assert store.model.image_mode is not before

            assert app._console_shift_tab() is False

            await pilot.press("grave_accent")
            await pilot.pause()
            assert cmd_input.active

            await pilot.press("i")
            await pilot.pause()
            assert store.model.image_mode is not before
            assert cmd_input.value == "i"

            await pilot.press("escape")
            await pilot.pause()
            assert not cmd_input.active

    asyncio.run(go())
