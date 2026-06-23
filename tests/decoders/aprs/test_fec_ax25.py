from __future__ import annotations

from types import SimpleNamespace

from tsdr.radio.decoders.aprs import ax25
from tsdr.radio.decoders.aprs.fec import crc16_x25, fcs_ok


def test_crc16_x25_check_value() -> None:
    # Documented CRC-16/X.25 check value for "123456789".
    assert crc16_x25(b"123456789") == 0x906E


def test_fcs_ok_roundtrip(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC", "APRS", info="!4807.00N/01133.00E-test")
    assert fcs_ok(frame)
    assert not fcs_ok(frame[:-2] + b"\x00\x00")


def test_parse_frame_roundtrip(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC-9", "APRS", ("WIDE1-1",), "!4807.00N/01133.00E-test")
    f = ax25.parse_frame(frame)
    assert f is not None
    assert str(f.src) == "DL1ABC-9"
    assert str(f.dest) == "APRS"
    assert [str(d) for d in f.digis] == ["WIDE1-1"]
    assert f.control == 0x03 and f.pid == 0xF0
    assert f.info == b"!4807.00N/01133.00E-test"


def test_parse_frame_rejects_bad_fcs(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC", "APRS", info="x")
    corrupt = frame[:-3] + bytes([frame[-3] ^ 0xFF]) + frame[-2:]
    assert ax25.parse_frame(corrupt) is None


def test_path_str_marks_repeated_digi(synth: SimpleNamespace) -> None:
    frame = synth.build_ax25("DL1ABC-9", "APRS", ("DL2ABC-1*", "WIDE2-1"), "x")
    f = ax25.parse_frame(frame)
    assert f is not None
    assert f.path_str() == "DL1ABC-9>APRS,DL2ABC-1*,WIDE2-1"


def test_validate_strict(synth: SimpleNamespace) -> None:
    good = ax25.parse_frame(synth.build_ax25("DL1ABC", "APRS", info="!pos"))
    assert good is not None and ax25.validate_strict(good)

    non_ui = ax25.parse_frame(synth.build_ax25("DL1ABC", "APRS", info="x", control=0x00))
    assert non_ui is not None and not ax25.validate_strict(non_ui)

    bad_pid = ax25.parse_frame(synth.build_ax25("DL1ABC", "APRS", info="x", pid=0x00))
    assert bad_pid is not None and not ax25.validate_strict(bad_pid)
