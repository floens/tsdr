"""TextualEventAdapter: engine events re-posted to the Textual app."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from textual.app import App
from textual.message import Message

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import ConfigChangedEvent
from tsdr.tui.messages import ConfigChanged
from tsdr.tui.textual_adapter import TextualEventAdapter


def test_config_changes_from_a_worker_thread_reach_the_ui(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The pipeline worker publishes ConfigChangedEvent when it changes the config
    # (an AGC gain step, a SEARCH jump). A second one within the refresh interval
    # used to start a Textual timer from that thread, which raised: the UI kept
    # the old config.
    posted: list[ConfigChanged] = []

    async def go() -> None:
        app = App()
        async with app.run_test() as pilot:
            bus = EventBus()
            adapter = TextualEventAdapter(app, bus)
            post = app.post_message

            def capture(message: Message) -> bool:
                if isinstance(message, ConfigChanged):
                    posted.append(message)
                    return True
                return post(message)

            app.post_message = capture  # type: ignore[method-assign]
            adapter.start()
            worker = threading.Thread(
                target=lambda: [bus.publish(ConfigChangedEvent(device_id="d")) for _ in range(3)]
            )
            worker.start()
            worker.join()
            await pilot.pause(0.2)
            adapter.stop()

    with caplog.at_level(logging.ERROR):
        asyncio.run(go())
    assert "event_handler_failed" not in caplog.text
    assert len([m for m in posted if isinstance(m, ConfigChanged)]) == 3
