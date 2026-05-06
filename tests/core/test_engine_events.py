from __future__ import annotations

from tsdr.core.events.events import ConfigChangedEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams


def test_add_device_publishes_config_changed() -> None:
    """Without this event, UI widgets stay blank until the user pokes something.

    The tuner widget mounts before the device is restored (restore is scheduled
    via call_after_refresh from on_mount). Its on_mount sees no focused device
    and bails. The only way it can populate afterwards is by receiving a
    ConfigChangedEvent once the device exists.
    """
    engine = SDREngine()
    received: list[ConfigChangedEvent] = []

    def handler(event: object) -> None:
        assert isinstance(event, ConfigChangedEvent)
        received.append(event)

    engine.event_bus.subscribe(ConfigChangedEvent, handler)

    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())

    assert len(received) == 1
    assert received[0].device_id == "rtl0"


def test_add_device_sets_focused_before_event() -> None:
    """Handlers read engine.get_focused_device(); it must be set by the time
    they run, otherwise they bail and stay blank."""
    engine = SDREngine()
    observed_focused: list[str | None] = []

    def handler(event: object) -> None:
        assert isinstance(event, ConfigChangedEvent)
        observed_focused.append(engine.focused_device)

    engine.event_bus.subscribe(ConfigChangedEvent, handler)

    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())

    assert observed_focused == ["rtl0"]
