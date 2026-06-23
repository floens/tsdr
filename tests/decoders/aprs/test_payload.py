from __future__ import annotations

from types import SimpleNamespace

import pytest

from tsdr.radio.decoders.aprs import ax25, payload


def _parse(synth: SimpleNamespace, src: str, dest: str, digis: tuple[str, ...], info: str):
    frame = ax25.parse_frame(synth.build_ax25(src, dest, digis, info))
    assert frame is not None
    return payload.parse(frame)


def test_uncompressed_position(synth: SimpleNamespace) -> None:
    p = _parse(synth, "DL1ABC", "APRS", (), "!4807.00N/01133.00E-Hello")
    assert p.info_type == "position"
    assert p.latitude == pytest.approx(48.1167, abs=1e-3)
    assert p.longitude == pytest.approx(11.55, abs=1e-3)
    assert p.symbol == "/-"
    assert p.comment == "Hello"


def test_position_with_timestamp_and_course_speed(synth: SimpleNamespace) -> None:
    p = _parse(synth, "DL1ABC-9", "APRS", (), "@092345z4807.00N/01133.00E>088/036Going")
    assert p.info_type == "position"
    assert p.timestamp == "092345z"
    assert p.course == 88
    assert p.speed == pytest.approx(36.0)
    assert p.symbol == "/>"
    assert p.comment == "Going"


def test_position_south_west(synth: SimpleNamespace) -> None:
    p = _parse(synth, "VK1XYZ", "APRS", (), "!3357.00S/15112.00E-")
    assert p.latitude == pytest.approx(-33.95, abs=1e-3)
    assert p.longitude == pytest.approx(151.20, abs=1e-3)


def test_message(synth: SimpleNamespace) -> None:
    p = _parse(synth, "DL1ABC", "APRS", (), ":DL1IGW-10:hello there{42")
    assert p.info_type == "message"
    assert p.addressee == "DL1IGW-10"
    assert p.message_text == "hello there{42"


def test_status(synth: SimpleNamespace) -> None:
    p = _parse(synth, "DL1ABC", "APRS", (), ">Weather station online")
    assert p.info_type == "status"
    assert p.status == "Weather station online"


def test_third_party_unwrap(synth: SimpleNamespace) -> None:
    # iGate-gated third-party shape: "}SRC>TOCALL,TCPIP,igate*:<payload>".
    info = "}DL2XYZ-8>APRS,TCPIP,DL1IGW-10*:=4807.00N/01133.00E-"
    p = _parse(synth, "DL1IGW-10", "APRS", ("WIDE1-1",), info)
    assert p.info_type == "third_party"
    inner = p.third_party
    assert inner is not None
    assert inner.source == "DL2XYZ-8"
    assert inner.dest == "APRS"
    assert inner.info_type == "position"
    assert inner.latitude == pytest.approx(48.1167, abs=1e-3)
    assert inner.longitude == pytest.approx(11.55, abs=1e-3)


def test_unknown_type_keeps_raw(synth: SimpleNamespace) -> None:
    p = _parse(synth, "DL1ABC", "APRS", (), ";object   *...")
    assert p.info_type == "other"
    assert p.raw_info == ";object   *..."
