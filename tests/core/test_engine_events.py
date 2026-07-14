from __future__ import annotations

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import (
    DeviceAddedEvent,
    DeviceCapabilitiesChangedEvent,
    DeviceRemovedEvent,
    FocusChangedEvent,
)
from tsdr.core.sdr.config import DeviceConfig, StageType
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.exceptions import SDRException
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


def test_remove_focused_refocuses_most_recent() -> None:
    engine = SDREngine()
    engine.add_device("a", "mock", MockParams(), DeviceConfig())
    engine.add_device("b", "mock", MockParams(), DeviceConfig())
    engine.set_focused_device("b")
    engine.set_focused_device("a")  # focus recency: b then a

    engine.remove_device("a")

    assert engine.focused_device == "b"


def test_remove_focused_falls_back_to_never_focused_survivor() -> None:
    engine = SDREngine()
    engine.add_device("a", "mock", MockParams(), DeviceConfig())
    engine.add_device("b", "mock", MockParams(), DeviceConfig())  # b never focused

    engine.remove_device("a")

    assert engine.focused_device == "b"


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
    engine.add_device(
        "rtl0",
        "mock",
        MockParams(),
        DeviceConfig(tuned_frequency=2.0e9, center_frequency=2.0e9),
    )
    _publish_caps(engine, freq_range=(24e6, 1766e6))
    assert engine.devices["rtl0"].config.tuned_frequency == 1766e6
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


def test_tuned_frequency_update_recenters_stopped_device() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    engine.update_device_config("rtl0", tuned_frequency=105e6)
    config = engine.devices["rtl0"].config
    assert config.tuned_frequency == 105e6
    assert config.center_frequency == 105e6


def test_explicit_center_update_leaves_tuned_alone() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    engine.update_device_config("rtl0", center_frequency=105e6)
    config = engine.devices["rtl0"].config
    assert config.tuned_frequency == 100e6
    assert config.center_frequency == 105e6


def test_set_audio_demod_always_prepends_frequency_shift() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    engine.set_audio_demod("rtl0", DemodSpec(mode="AM"))
    stages = engine.devices["rtl0"].config.pipelines["audio"].stages
    assert stages == (
        StageType.FREQUENCY_SHIFT,
        StageType.DEMODULATOR,
        StageType.DENOISER,
        StageType.EVENT_EMITTER,
    )


def test_legacy_demod_spec_with_frequency_offset_still_parses() -> None:
    spec = DemodSpec.model_validate({"mode": "AM", "frequency_offset": 25e3})
    assert spec.mode == "AM"
    assert not hasattr(spec, "frequency_offset")


def test_capabilities_change_rederives_center_from_tuned() -> None:
    # A dial tune can race the device handshake (KiwiSDR): center stays put
    # under stale caps, then the real capabilities must pull it to the dial.
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    engine.update_device_config("rtl0", center_frequency=105e6)
    assert engine.devices["rtl0"].config.center_frequency == 105e6

    _publish_caps(engine)

    config = engine.devices["rtl0"].config
    assert config.center_frequency == config.tuned_frequency == 100e6


def test_default_tuning_mode_is_center() -> None:
    assert DeviceConfig().tuning_mode == "center"


def test_enabling_center_mode_recenters() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(tuning_mode="free"))
    engine.update_device_config("rtl0", center_frequency=105e6)
    assert engine.devices["rtl0"].config.center_frequency == 105e6

    engine.update_device_config("rtl0", tuning_mode="center")

    config = engine.devices["rtl0"].config
    assert config.tuning_mode == "center"
    assert config.center_frequency == config.tuned_frequency == 100e6


def test_config_tuning_center_clears_view_pan() -> None:
    engine = SDREngine()
    engine.add_device(
        "rtl0",
        "mock",
        MockParams(),
        DeviceConfig(tuning_mode="free", spectrum_center=101e6, spectrum_span=200e3),
    )
    engine.update_device_config("rtl0", tuning_mode="center")
    assert engine.devices["rtl0"].config.spectrum_center is None


def test_sample_rate_change_rederives_center() -> None:
    # Shrinking the capture is a fit-test input; the derivation must re-run.
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(tuning_mode="free"))
    engine.update_device_config("rtl0", center_frequency=105e6)
    engine.update_device_config("rtl0", sample_rate=1.024e6)
    config = engine.devices["rtl0"].config
    assert config.center_frequency == config.tuned_frequency


def test_channel_bandwidth_change_rederives_center() -> None:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig(tuning_mode="free"))
    engine.update_device_config("rtl0", center_frequency=105e6)
    engine.update_device_config("rtl0", channel_bandwidth=200e3)
    config = engine.devices["rtl0"].config
    assert config.center_frequency == config.tuned_frequency


def test_zero_frequency_raises_sdr_exception() -> None:
    # Bands can start at 0 Hz (KiwiSDR); 0 must be a catchable SDRException,
    # not DeviceConfig.validate's ValueError.
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    with pytest.raises(SDRException):
        engine.update_device_config("rtl0", tuned_frequency=0.0)
