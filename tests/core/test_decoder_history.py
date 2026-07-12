from __future__ import annotations

from tsdr.core.events.events import DecodedMessage, DecoderOutputEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DECODER_HISTORY_MAX
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams


def _engine_with_device() -> SDREngine:
    engine = SDREngine()
    engine.add_device("rtl0", "mock", MockParams(), DeviceConfig())
    return engine


def _emit(engine: SDREngine, protocol: str, *messages: DecodedMessage) -> None:
    engine.event_bus.publish(
        DecoderOutputEvent(source_id="test", device_id="rtl0", protocol=protocol, messages=messages)
    )


def test_sealed_messages_retained_partials_dropped() -> None:
    engine = _engine_with_device()
    _emit(
        engine,
        "NAVTEX",
        DecodedMessage("line1", 0.0),
        DecodedMessage("in-progress", 0.0, partial=True),
        DecodedMessage("line2", 0.0),
    )

    protocol, messages = engine.devices["rtl0"].snapshot_decoder_history()
    assert protocol == "NAVTEX"
    assert [m.text for m in messages] == ["line1", "line2"]


def test_heavy_data_payload_is_stripped() -> None:
    engine = _engine_with_device()
    _emit(engine, "ADSB", DecodedMessage("ICAO abc123", 0.0, data=object()))

    _, messages = engine.devices["rtl0"].snapshot_decoder_history()
    assert messages[0].text == "ICAO abc123"
    assert messages[0].data is None


def test_protocol_switch_resets_history() -> None:
    engine = _engine_with_device()
    _emit(engine, "NAVTEX", DecodedMessage("navtex line", 0.0))
    _emit(engine, "RTTY", DecodedMessage("rtty line", 0.0))

    protocol, messages = engine.devices["rtl0"].snapshot_decoder_history()
    assert protocol == "RTTY"
    assert [m.text for m in messages] == ["rtty line"]


def test_maxlen_evicts_oldest() -> None:
    engine = _engine_with_device()
    overflow = DECODER_HISTORY_MAX + 5
    _emit(engine, "NAVTEX", *(DecodedMessage(f"m{i}", 0.0) for i in range(overflow)))

    _, messages = engine.devices["rtl0"].snapshot_decoder_history()
    assert len(messages) == DECODER_HISTORY_MAX
    assert messages[0].text == f"m{overflow - DECODER_HISTORY_MAX}"
    assert messages[-1].text == f"m{overflow - 1}"


def test_snapshot_is_a_stable_copy() -> None:
    engine = _engine_with_device()
    _emit(engine, "NAVTEX", DecodedMessage("first", 0.0))
    _, snapshot = engine.devices["rtl0"].snapshot_decoder_history()

    _emit(engine, "NAVTEX", DecodedMessage("second", 0.0))
    assert [m.text for m in snapshot] == ["first"]


def test_unknown_device_is_ignored() -> None:
    engine = _engine_with_device()
    engine.event_bus.publish(
        DecoderOutputEvent(
            source_id="test",
            device_id="does-not-exist",
            protocol="NAVTEX",
            messages=(DecodedMessage("x", 0.0),),
        )
    )

    _, messages = engine.devices["rtl0"].snapshot_decoder_history()
    assert messages == ()
