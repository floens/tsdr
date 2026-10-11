"""A device whose hardware fails to open must not be reported as running.

The I/O worker opens the device on its own thread, after `start_device`
returns. When the open fails (unreachable network radio, unplugged USB
device, missing IQ file), the device must end in ERROR, the UI must be told
it is no longer running, and start/stop/remove must work without first
issuing a manual `stop`.
"""

from __future__ import annotations

import time

import pytest

from tsdr.core.events.events import DeviceErrorEvent, DeviceStateChangedEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.exceptions import DeviceError
from tsdr.devices import MockParams
from tsdr.devices.mock import MockSDRDevice


class _OpenSwitch:
    """Makes MockSDRDevice.open fail until `healthy` is set."""

    def __init__(self) -> None:
        self.healthy = False
        self._original = MockSDRDevice.open

    def open(self, device: MockSDRDevice) -> None:
        if not self.healthy:
            raise DeviceError("radio unreachable")
        self._original(device)


@pytest.fixture
def switch(monkeypatch) -> _OpenSwitch:
    sw = _OpenSwitch()
    monkeypatch.setattr(MockSDRDevice, "open", lambda self: sw.open(self))
    return sw


@pytest.fixture
def engine():
    eng = SDREngine()
    eng.add_device("m", "mock", MockParams(), DeviceConfig())
    yield eng
    eng.shutdown(timeout=2.0)


def _wait_for_state(engine: SDREngine, state: DeviceState, timeout: float = 3.0) -> DeviceState:
    ctx = engine.devices["m"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and ctx.state != state:
        time.sleep(0.01)
    return ctx.state


def test_failed_open_ends_in_error_state(engine: SDREngine, switch: _OpenSwitch) -> None:
    states: list[bool] = []
    errors: list[str] = []
    engine.event_bus.subscribe(DeviceStateChangedEvent, lambda e: states.append(e.running))  # type: ignore[attr-defined]
    engine.event_bus.subscribe(DeviceErrorEvent, lambda e: errors.append(e.error))  # type: ignore[attr-defined]

    engine.start_device("m")

    assert _wait_for_state(engine, DeviceState.ERROR) == DeviceState.ERROR
    assert engine.devices["m"].error_message == "radio unreachable"
    assert errors == ["radio unreachable"]
    assert states and states[-1] is False


def test_immediate_failure_is_not_overwritten_by_start(
    engine: SDREngine, switch: _OpenSwitch
) -> None:
    """The worker can fail before start() finishes; start must not mark it RUNNING."""
    for _ in range(20):
        engine.start_device("m")
        assert _wait_for_state(engine, DeviceState.ERROR) == DeviceState.ERROR
        engine.stop_device("m")
        assert engine.devices["m"].state == DeviceState.STOPPED


def test_start_retries_after_failure_without_manual_stop(
    engine: SDREngine, switch: _OpenSwitch
) -> None:
    engine.start_device("m")
    assert _wait_for_state(engine, DeviceState.ERROR) == DeviceState.ERROR

    switch.healthy = True
    engine.start_device("m")

    ctx = engine.devices["m"]
    assert ctx.state == DeviceState.RUNNING
    assert ctx.error_message is None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and ctx.total_samples_read == 0:
        time.sleep(0.01)
    assert ctx.total_samples_read > 0


def test_failed_device_can_be_removed(engine: SDREngine, switch: _OpenSwitch) -> None:
    engine.start_device("m")
    assert _wait_for_state(engine, DeviceState.ERROR) == DeviceState.ERROR

    engine.remove_device("m")

    assert "m" not in engine.devices


def test_failed_device_leaves_no_workers_after_stop(engine: SDREngine, switch: _OpenSwitch) -> None:
    engine.start_device("m")
    assert _wait_for_state(engine, DeviceState.ERROR) == DeviceState.ERROR

    engine.stop_device("m")

    assert engine.devices["m"].state == DeviceState.STOPPED
    assert engine.worker_runner._workers == {}
