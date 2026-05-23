from __future__ import annotations

from tsdr.core.events.events import (
    DeviceAddedEvent,
    DeviceCapabilitiesChangedEvent,
    DeviceRemovedEvent,
    FocusChangedEvent,
)
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams
from tsdr.devices.base import DeviceCapabilities


def test_add_device_publishes_device_added() -> None:
    engine = SDREngine()
    received: list[DeviceAddedEvent] = []

    def handler(event: object) -> None:
        assert isinstance(event, DeviceAddedEvent)
        received.append(event)

    engine.event_bus.subscribe(DeviceAddedEvent, handler)

    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())

    assert len(received) == 1
    assert received[0].device_id == "rtl0"


def test_first_add_publishes_focus_changed() -> None:
    """Adding the first device implicitly focuses it; a FocusChangedEvent
    fires so the UI can re-seed without waiting for the next pipeline event."""
    engine = SDREngine()
    received: list[FocusChangedEvent] = []
    engine.event_bus.subscribe(FocusChangedEvent, lambda e: received.append(e))  # type: ignore[arg-type]

    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())

    assert [e.focused_device_id for e in received] == ["rtl0"]


def test_second_add_does_not_publish_focus_changed() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    received: list[FocusChangedEvent] = []
    engine.event_bus.subscribe(FocusChangedEvent, lambda e: received.append(e))  # type: ignore[arg-type]

    engine.add_device("rtl1", "mock", MockParams(), DeviceConfig())

    assert received == []


def test_add_device_sets_focused_before_event() -> None:
    """Handlers depend on engine.focused_device being set by the time the
    DeviceAdded event reaches them."""
    engine = SDREngine()
    observed_focused: list[str | None] = []

    def handler(event: object) -> None:
        assert isinstance(event, DeviceAddedEvent)
        observed_focused.append(engine.focused_device)

    engine.event_bus.subscribe(DeviceAddedEvent, handler)

    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())

    assert observed_focused == ["rtl0"]


def test_set_focused_device_publishes_focus_changed() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    engine.add_device("rtl1", "mock", MockParams(), DeviceConfig())
    received: list[FocusChangedEvent] = []
    engine.event_bus.subscribe(FocusChangedEvent, lambda e: received.append(e))  # type: ignore[arg-type]

    engine.set_focused_device("rtl1")

    assert [e.focused_device_id for e in received] == ["rtl1"]


def test_remove_device_publishes_device_removed_and_focus_changed() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    removed: list[DeviceRemovedEvent] = []
    focus: list[FocusChangedEvent] = []
    engine.event_bus.subscribe(DeviceRemovedEvent, lambda e: removed.append(e))  # type: ignore[arg-type]
    engine.event_bus.subscribe(FocusChangedEvent, lambda e: focus.append(e))  # type: ignore[arg-type]

    engine.remove_device("rtl0")

    assert [e.device_id for e in removed] == ["rtl0"]
    assert [e.focused_device_id for e in focus] == [None]


def test_remove_unfocused_device_does_not_publish_focus_changed() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    engine.add_device("rtl1", "mock", MockParams(), DeviceConfig())
    focus: list[FocusChangedEvent] = []
    engine.event_bus.subscribe(FocusChangedEvent, lambda e: focus.append(e))  # type: ignore[arg-type]

    engine.remove_device("rtl1")  # rtl0 is focused

    assert focus == []


def _caps(
    *,
    gain_supported: bool = True,
    gain_range: tuple[float, float] = (0.0, 49.6),
    freq_range: tuple[float, float] | None = None,
    sample_rates: tuple[float, ...] | None = None,
    bias_tee_supported: bool = True,
) -> DeviceCapabilities:
    return DeviceCapabilities(
        frequency_range=freq_range,
        frequency_controllable=True,
        sample_rates=sample_rates,
        gain_supported=gain_supported,
        gain_range=gain_range,
        gain_step=1.0,
        gain_unit="dB",
        bias_tee_supported=bias_tee_supported,
    )


def _publish_caps(engine: SDREngine, **caps_kwargs) -> None:
    engine.event_bus.publish(
        DeviceCapabilitiesChangedEvent(
            device_id="rtl0",
            capabilities=_caps(**caps_kwargs),
            source_id="test",
        )
    )


def test_capabilities_change_clamps_out_of_range_gain() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(rf_gain=36.0))
    _publish_caps(engine, gain_range=(0.0, 8.0))
    assert engine.devices["rtl0"].config.rf_gain == 8.0


def test_capabilities_change_clamps_gain_to_degenerate_range() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(rf_gain=30.0))
    _publish_caps(engine, gain_range=(0.0, 0.0))
    assert engine.devices["rtl0"].config.rf_gain == 0.0


def test_capabilities_change_leaves_in_range_gain_alone() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(rf_gain=20.0))
    _publish_caps(engine, gain_range=(0.0, 49.6))
    assert engine.devices["rtl0"].config.rf_gain == 20.0


def test_capabilities_change_clamps_out_of_range_frequency() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(center_frequency=2.0e9))
    _publish_caps(engine, freq_range=(24e6, 1766e6))
    assert engine.devices["rtl0"].config.center_frequency == 1766e6


def test_capabilities_change_clears_bias_tee_when_unsupported() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(bias_tee=True))
    _publish_caps(engine, bias_tee_supported=False)
    assert engine.devices["rtl0"].config.bias_tee is False


def test_capabilities_change_disables_agc_when_gain_unsupported() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(enable_agc=True))
    _publish_caps(engine, gain_supported=False, gain_range=(0.0, 0.0))
    assert engine.devices["rtl0"].config.enable_agc is False


def test_capabilities_change_picks_nearest_sample_rate() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(sample_rate=2.5e6))
    _publish_caps(engine, sample_rates=(2.4e6, 1.2e6, 600e3))
    assert engine.devices["rtl0"].config.sample_rate == 2.4e6
