from __future__ import annotations

from tsdr.core.events.events import (
    ConfigChangedEvent,
    DeviceCapabilitiesChangedEvent,
)
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams
from tsdr.devices.base import DeviceCapabilities


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
