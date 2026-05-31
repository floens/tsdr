"""Render tests for the single bottom panel bar (EdgeStrip).

The bar's per-fragment styling comes from app.tcss component classes, so the
widget must be mounted for `get_component_rich_style` to resolve — these render
the live `#panel-bar` inside a running app.
"""

from __future__ import annotations

import asyncio

from rich.text import Text

from tsdr.tui.app import TSDRApp
from tsdr.tui.widgets.edge_strip import EdgeStrip


def _rendered(glyphs) -> tuple[Text, EdgeStrip]:
    """Mount the app, set the panel-bar's glyphs, return (rendered text, widget)."""
    captured: dict[str, object] = {}

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one("#panel-bar", EdgeStrip)
            bar.glyphs = glyphs
            captured["text"] = bar.render()
            captured["bar"] = bar

    asyncio.run(go())
    return captured["text"], captured["bar"]  # type: ignore[return-value]


def test_empty_glyphs_render_blank() -> None:
    text, _ = _rendered(())
    assert text.plain == ""


def test_button_format_is_digit_then_title() -> None:
    """Each button is `<digit> <title>`, buttons padded apart by two spaces."""
    text, _ = _rendered((("1", "Decoder", False), ("2", "Demod", False)))
    assert text.plain == "1 Decoder  2 Demod"


def test_blank_digit_keeps_alignment() -> None:
    """A hotkey-less panel renders a space where the digit would be."""
    text, _ = _rendered((("", "Demod", False),))
    assert text.plain == "  Demod"


def test_active_and_inactive_titles_use_their_component_styles() -> None:
    """Active title uses the active component style, inactive uses the title one,
    and the two differ (amber/bold vs dim) — all sourced from app.tcss."""
    text, bar = _rendered((("1", "Decoder", False), ("2", "Demod", True)))
    active_style = bar.get_component_rich_style("edge-strip--active")
    title_style = bar.get_component_rich_style("edge-strip--title")
    style_by_text = {text.plain[s.start : s.end]: s.style for s in text.spans}
    assert style_by_text["Demod"] == active_style
    assert style_by_text["Decoder"] == title_style
    assert active_style != title_style
