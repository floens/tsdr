"""The console takes the keyboard when it opens, whatever was clicked before."""

from __future__ import annotations

import asyncio

from tsdr.tui.app import TSDRApp
from tsdr.tui.console import TerminalInput
from tsdr.tui.model.store import get_ui_store
from tsdr.tui.widgets.section_panel import SectionPanel


def test_typing_reaches_the_console_after_a_panel_was_clicked() -> None:
    # A click on a panel (Stats, Performance, a decoder's) gave it Textual's focus;
    # ` then opened the console, but every key went to the panel and was dropped.
    typed: list[str] = []

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            get_ui_store().toggle_panel("stats")
            await pilot.pause(0.2)
            panel = next(iter(app.query(SectionPanel)))
            await pilot.click(panel)
            await pilot.press("grave_accent", "h", "e", "l", "p")
            await pilot.pause()
            typed.append(app.query_one("#command-input", TerminalInput).value)

    asyncio.run(go())
    assert typed == ["help"]
