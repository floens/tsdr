"""Message unpacker regression tests.

Each vector binds a known input string to the 10-byte FT8 payload encoding
and the text it decodes back. Tests run our unpacker against the same
payloads.
"""

import pytest

from tsdr.radio.decoders.wsjt.message import MessageType, decode_message, get_message_type

from . import _wsjt_msg_vectors as v

STANDARD_CASES = [
    ("std_cq", v.PAYLOAD_std_cq, v.EXPECTED_std_cq, v.MSGTYPE_std_cq),
    ("std_qso", v.PAYLOAD_std_qso, v.EXPECTED_std_qso, v.MSGTYPE_std_qso),
    ("std_report", v.PAYLOAD_std_report, v.EXPECTED_std_report, v.MSGTYPE_std_report),
    ("std_r_report", v.PAYLOAD_std_r_report, v.EXPECTED_std_r_report, v.MSGTYPE_std_r_report),
    ("std_rrr", v.PAYLOAD_std_rrr, v.EXPECTED_std_rrr, v.MSGTYPE_std_rrr),
    ("std_rr73", v.PAYLOAD_std_rr73, v.EXPECTED_std_rr73, v.MSGTYPE_std_rr73),
    ("std_73", v.PAYLOAD_std_73, v.EXPECTED_std_73, v.MSGTYPE_std_73),
    ("std_cq_region", v.PAYLOAD_std_cq_region, v.EXPECTED_std_cq_region, v.MSGTYPE_std_cq_region),
    ("std_grid_far", v.PAYLOAD_std_grid_far, v.EXPECTED_std_grid_far, v.MSGTYPE_std_grid_far),
]

FREE_TEXT_CASES = [
    ("free_short", v.PAYLOAD_free_short, v.EXPECTED_free_short, v.MSGTYPE_free_short),
    ("free_numeric", v.PAYLOAD_free_numeric, v.EXPECTED_free_numeric, v.MSGTYPE_free_numeric),
    ("free_punct", v.PAYLOAD_free_punct, v.EXPECTED_free_punct, v.MSGTYPE_free_punct),
    ("free_signoff", v.PAYLOAD_free_signoff, v.EXPECTED_free_signoff, v.MSGTYPE_free_signoff),
]


@pytest.mark.parametrize("name,payload,expected,mtype", STANDARD_CASES)
def test_decode_standard(name: str, payload: bytes, expected: str, mtype: int) -> None:
    result = decode_message(payload)
    assert result.text == expected, f"{name}: got {result.text!r}, want {expected!r}"
    assert int(result.msg_type) == mtype


@pytest.mark.parametrize("name,payload,expected,mtype", FREE_TEXT_CASES)
def test_decode_free_text(name: str, payload: bytes, expected: str, mtype: int) -> None:
    result = decode_message(payload)
    assert result.text == expected, f"{name}: got {result.text!r}, want {expected!r}"
    assert int(result.msg_type) == mtype


def test_get_message_type_standard_i3_1() -> None:
    # std_qso has i3 in {1,2}: assert STANDARD
    assert get_message_type(v.PAYLOAD_std_qso) == MessageType.STANDARD


def test_get_message_type_free_text() -> None:
    assert get_message_type(v.PAYLOAD_free_short) == MessageType.FREE_TEXT


def test_nonstd_compound_returns_placeholder_without_hash_table() -> None:
    result = decode_message(v.PAYLOAD_nonstd_compound)
    assert result.text == v.EXPECTED_nonstd_compound


def test_decode_message_rejects_short_payload() -> None:
    with pytest.raises(ValueError):
        decode_message(b"\x00" * 5)
