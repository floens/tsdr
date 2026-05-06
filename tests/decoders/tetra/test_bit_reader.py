import numpy as np

from tsdr.radio.decoders.tetra.bit_reader import BitReader


def _bits(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.uint8)


def test_read_advances_cursor():
    # 8 bits = 0b1010_1100 = 0xAC
    r = BitReader(_bits(1, 0, 1, 0, 1, 1, 0, 0))
    assert r.u(4) == 0b1010
    assert r.pos == 4
    assert r.u(4) == 0b1100
    assert r.pos == 8


def test_single_bit_reads_match_raw_bits():
    raw = [1, 0, 1, 1, 0, 0, 1, 0]
    r = BitReader(_bits(*raw))
    for bit in raw:
        assert r.u(1) == bit


def test_peek_does_not_advance():
    r = BitReader(_bits(1, 1, 0, 1, 0, 1, 0, 1))
    assert r.peek(3) == 0b110
    assert r.pos == 0
    assert r.peek(3) == 0b110  # same value on a repeat peek
    assert r.pos == 0
    # Subsequent read should still see the same bits from position 0.
    assert r.u(3) == 0b110
    assert r.pos == 3


def test_skip_advances_without_reading():
    r = BitReader(_bits(1, 0, 1, 0, 1, 1, 0, 0))
    r.skip(2)
    assert r.pos == 2
    assert r.u(2) == 0b10
    assert r.pos == 4
    r.skip(2)
    assert r.u(2) == 0b00


def test_remaining_tracks_position():
    bits = _bits(*([1] * 16))
    r = BitReader(bits)
    assert r.remaining == 16
    r.u(5)
    assert r.remaining == 11
    r.skip(3)
    assert r.remaining == 8
    r.u(8)
    assert r.remaining == 0


def test_custom_start_position():
    r = BitReader(_bits(1, 0, 1, 0, 1, 1, 0, 0), pos=4)
    assert r.pos == 4
    assert r.remaining == 4
    assert r.u(4) == 0b1100


def test_fourteen_bit_read_matches_kernel_semantics():
    # 14-bit value 0x3BC = 956: 00 0011 1011 1100.
    raw = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
    r = BitReader(_bits(*raw))
    assert r.u(14) == 956
