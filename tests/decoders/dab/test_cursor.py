import pytest

from tsdr.radio.decoders.dab.cursor import Cursor, CursorTruncated, bits


def test_u8_advances():
    cur = Cursor(bytes([0x12, 0x34, 0x56]))
    assert cur.u8() == 0x12
    assert cur.u8() == 0x34
    assert cur.remaining == 1


def test_u16_big_endian():
    cur = Cursor(bytes([0xDE, 0xAD, 0xBE, 0xEF]))
    assert cur.u16() == 0xDEAD
    assert cur.u16() == 0xBEEF


def test_u32_big_endian():
    cur = Cursor(bytes([0xDE, 0xAD, 0xBE, 0xEF]))
    assert cur.u32() == 0xDEADBEEF
    assert cur.remaining == 0


def test_bytes_returns_slice_and_advances():
    cur = Cursor(b"hello world")
    assert cur.bytes(5) == b"hello"
    assert cur.u8() == ord(" ")
    assert cur.bytes(5) == b"world"
    assert cur.remaining == 0


def test_skip_advances_without_reading():
    cur = Cursor(bytes([0x01, 0x02, 0x03, 0x04]))
    cur.skip(2)
    assert cur.u8() == 0x03


def test_remaining_and_has():
    cur = Cursor(bytes(8))
    assert cur.remaining == 8
    assert cur.has(8)
    assert not cur.has(9)
    cur.skip(3)
    assert cur.remaining == 5
    assert cur.has(5)
    assert not cur.has(6)


@pytest.mark.parametrize(
    "op",
    [
        lambda c: c.u8(),
        lambda c: c.u16(),
        lambda c: c.u32(),
        lambda c: c.bytes(1),
        lambda c: c.skip(1),
    ],
)
def test_read_past_end_raises(op):
    cur = Cursor(b"")
    with pytest.raises(CursorTruncated):
        op(cur)


def test_accepts_memoryview():
    mv = memoryview(bytes([0xAA, 0xBB]))
    cur = Cursor(mv)
    assert cur.u16() == 0xAABB


def test_bits_extracts_high_low():
    # 0b1011_0110 = 0xB6
    assert bits(0xB6, 7, 7) == 0b1
    assert bits(0xB6, 7, 6) == 0b10
    assert bits(0xB6, 7, 4) == 0b1011
    assert bits(0xB6, 5, 4) == 0b11
    assert bits(0xB6, 3, 0) == 0b0110
    assert bits(0xB6, 7, 0) == 0xB6
    assert bits(0xB6, 0, 0) == 0
