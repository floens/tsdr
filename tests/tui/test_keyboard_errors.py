"""Engine errors raised by a keypress are reported, not fatal.

Textual exits the app on any exception that escapes a key handler, so an
out-of-range tune from a shortcut used to close TSDR. Commands and mouse
handlers already show SDRException in the status bar; keys must too.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from tsdr.core.demod_spec import DemodSpec, PreviousTuneState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.tuning_state import get_tuning_state
from tsdr.devices import mock as mock_module
from tsdr.tui.app import TSDRApp


@pytest.fixture
def airspy_like_band(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = replace(mock_module._MOCK_CAPABILITIES, frequency_range=(24e6, 1.8e9))
    monkeypatch.setattr(mock_module, "_MOCK_CAPABILITIES", caps)


def _capture_errors(app: TSDRApp) -> list[str]:
    errors: list[str] = []
    app._show_error = errors.append  # type: ignore[method-assign]
    return errors


def test_ab_swap_to_out_of_range_frequency_reports_error(airspy_like_band: None) -> None:
    async def go() -> None:
        app = TSDRApp(startup_commands=["add m --type mock"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            errors = _capture_errors(app)
            # A/B state saved while another radio was on 2.4 GHz.
            saved = PreviousTuneState(
                frequency_hz=2.4e9, bandwidth_hz=12_500.0, spec=DemodSpec(mode="NFM")
            )
            get_tuning_state().previous = saved
            before = get_engine().devices["m"].config.tuned_frequency

            await pilot.press("ctrl+0")
            await pilot.pause()

            assert app.is_running
            assert get_engine().devices["m"].config.tuned_frequency == before
            assert errors and "out of device range" in errors[-1]
            # A refused swap leaves the saved A/B state alone.
            assert get_tuning_state().previous == saved

    asyncio.run(go())


def test_engine_error_from_any_shortcut_is_shown_not_fatal() -> None:
    async def go() -> None:
        app = TSDRApp(startup_commands=["add m --type mock"])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            errors = _capture_errors(app)

            def boom(*args, **kwargs) -> None:
                raise SDRException("radio said no")

            app._tune = boom  # type: ignore[method-assign]
            app._recall_band = boom  # type: ignore[method-assign]

            await pilot.press("right")
            await pilot.press("ctrl+1")
            await pilot.pause()

            assert app.is_running
            assert errors == ["radio said no", "radio said no"]

    asyncio.run(go())
