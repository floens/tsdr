"""DecoderOutputWidget is a single RichLog whose last line is 'live': streaming
partials redraw it in place, identical lines fold into `… ×N`, and it seals into
permanent history when a distinct line supersedes it. Orientation classes track
the dock edge; a partial flood is debounced."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from textual.app import App, ComposeResult

from tsdr.core.events.events import DecodedMessage, DecoderOutputEvent
from tsdr.tui.widgets.decoder_output_widget import _TAIL_REDRAW_INTERVAL, DecoderOutputWidget

_SETTLE = _TAIL_REDRAW_INTERVAL + 0.05  # long enough for the debounce timer to fire


class _Harness(App):
    CSS = "DecoderOutputWidget { height: 12; }"

    def compose(self) -> ComposeResult:
        yield DecoderOutputWidget()


def _event(text: str, *, partial: bool = False) -> DecoderOutputEvent:
    return DecoderOutputEvent(
        device_id="dev",
        protocol="TEST",
        messages=(DecodedMessage(text=text, timestamp=0.0, partial=partial),),
    )


def _run(go: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(go())


def _lines(w: DecoderOutputWidget) -> list[str]:
    return [line.text for line in w.lines]


def test_dock_edge_toggles_orientation_classes() -> None:
    async def go() -> None:
        app = _Harness()
        async with app.run_test(size=(80, 24)) as pilot:
            w = app.query_one(DecoderOutputWidget)
            w.dock_edge = "left"
            await pilot.pause()
            assert w.has_class("panel-tall") and not w.has_class("panel-wide")
            w.dock_edge = "bottom"
            await pilot.pause()
            assert w.has_class("panel-wide") and not w.has_class("panel-tall")

    _run(go)


def test_streaming_redraws_the_last_line_in_place() -> None:
    async def go() -> None:
        app = _Harness()
        async with app.run_test(size=(80, 24)) as pilot:
            w = app.query_one(DecoderOutputWidget)
            await pilot.pause()

            w.update_decoder(_event("ab", partial=True))
            await pilot.pause(_SETTLE)
            w.update_decoder(_event("abcd", partial=True))
            await pilot.pause(_SETTLE)
            # Redrawn in place: still a single line, now showing the longer text.
            assert len(w.lines) == 1
            assert "abcd" in w.lines[-1].text

            w.update_decoder(_event("abcd"))  # seal (same content)
            await pilot.pause(_SETTLE)
            assert len(w.lines) == 1  # sealed line is still the live last line

            w.update_decoder(_event("next"))  # distinct line supersedes it
            await pilot.pause(_SETTLE)
            assert len(w.lines) == 2
            assert "abcd" in w.lines[0].text and "next" in w.lines[-1].text

    _run(go)


def test_repeat_collapse_folds_in_place_then_commits() -> None:
    async def go() -> None:
        app = _Harness()
        async with app.run_test(size=(80, 24)) as pilot:
            w = app.query_one(DecoderOutputWidget)
            await pilot.pause()

            for _ in range(3):
                w.update_decoder(_event("RYRY"))
            await pilot.pause(_SETTLE)
            assert len(w.lines) == 1  # collapsed, not three lines
            assert "RYRY" in w.lines[-1].text and "×3" in w.lines[-1].text

            w.update_decoder(_event("END"))
            await pilot.pause(_SETTLE)
            assert len(w.lines) == 2
            assert "×3" in w.lines[0].text and "END" in w.lines[-1].text

    _run(go)


def test_partial_flood_is_debounced() -> None:
    async def go() -> None:
        app = _Harness()
        async with app.run_test(size=(80, 24)) as pilot:
            w = app.query_one(DecoderOutputWidget)
            await pilot.pause()

            # First redraw is immediate; a rapid burst after it coalesces behind one
            # trailing timer rather than rewriting the line per message.
            w.update_decoder(_event("a", partial=True))
            w.update_decoder(_event("ab", partial=True))
            w.update_decoder(_event("abc", partial=True))
            assert w._timer is not None
            assert w._stream_line.endswith("abc")  # state is current despite deferral

            await pilot.pause(_SETTLE)
            assert w._timer is None
            assert len(w.lines) == 1 and "abc" in w.lines[-1].text

    _run(go)


def test_follow_lock_does_not_yank_when_scrolled_up() -> None:
    async def go() -> None:
        app = _Harness()
        async with app.run_test(size=(80, 24)) as pilot:
            w = app.query_one(DecoderOutputWidget)
            await pilot.pause()

            for i in range(30):
                w.update_decoder(_event(f"line-{i}"))
                await pilot.pause(_SETTLE)
            assert w.is_vertical_scroll_end  # followed to the bottom

            w.scroll_home(animate=False)
            await pilot.pause()
            assert not w.is_vertical_scroll_end
            top = w.scroll_offset.y

            w.update_decoder(_event("line-new"))
            await pilot.pause(_SETTLE)
            assert w.scroll_offset.y == top  # did not yank down
            assert not w.is_vertical_scroll_end

    _run(go)
