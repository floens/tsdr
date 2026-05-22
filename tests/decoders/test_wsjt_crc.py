"""CRC-14 regression vectors for the WSJT-X CRC-14 (polynomial 0x2757)."""

import numpy as np
import pytest

from tsdr.radio.decoders.wsjt.crc import add_crc, compute_crc, extract_crc, verify_crc


def _hx(s: str) -> bytes:
    return bytes.fromhex(s)


def test_compute_crc_raw_known_vectors() -> None:
    msg = _hx("deadbeef0123456789abcdef")
    assert compute_crc(msg, 82) == 0x15FF
    assert compute_crc(msg, 91) == 0x3917
    assert compute_crc(msg, 77) == 0x2F83


def test_add_crc_all_zero_payload() -> None:
    a91 = add_crc(bytes(10))
    assert a91 == _hx("000000000000000000000000")
    assert extract_crc(a91) == 0x0000


def test_add_crc_alt_aa55() -> None:
    payload = _hx("aa55aa55aa55aa55aa50")
    a91 = add_crc(payload)
    assert a91 == _hx("aa55aa55aa55aa55aa5605c0")
    assert extract_crc(a91) == 0x302E


def test_add_crc_all_ones() -> None:
    payload = _hx("fffffffffffffffffff8")
    a91 = add_crc(payload)
    assert a91 == _hx("fffffffffffffffffff8f620")
    assert extract_crc(a91) == 0x07B1


def test_add_crc_ascending() -> None:
    payload = _hx("00010203040506070808")
    a91 = add_crc(payload)
    assert a91 == _hx("00010203040506070808ce40")
    assert extract_crc(a91) == 0x0672


def test_add_crc_arbitrary_pattern() -> None:
    payload = _hx("123456789abcdef01358")
    a91 = add_crc(payload)
    assert a91 == _hx("123456789abcdef013593200")
    assert extract_crc(a91) == 0x0990


def test_verify_crc_roundtrip() -> None:
    for payload_hex in (
        "00000000000000000000",
        "aa55aa55aa55aa55aa50",
        "fffffffffffffffffff8",
        "00010203040506070808",
        "123456789abcdef01358",
    ):
        a91 = add_crc(_hx(payload_hex))
        ok, _ = verify_crc(a91)
        assert ok, f"verify failed for {payload_hex}"


def test_verify_crc_rejects_corrupted() -> None:
    a91 = bytearray(add_crc(_hx("aa55aa55aa55aa55aa50")))
    a91[0] ^= 0x80  # flip MSB of payload
    ok, _ = verify_crc(bytes(a91))
    assert not ok


def test_compute_crc_accepts_ndarray() -> None:
    msg = np.array(list(_hx("deadbeef0123456789abcdef")), dtype=np.uint8)
    assert compute_crc(msg, 82) == 0x15FF


def test_add_crc_rejects_short_payload() -> None:
    with pytest.raises(ValueError):
        add_crc(bytes(5))
