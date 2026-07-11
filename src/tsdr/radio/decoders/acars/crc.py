"""ACARS block CRC: reflected CRC-16/CCITT (poly 0x8408, init 0).

A frame is valid when the running CRC over the block bytes plus the two
received CRC bytes is 0. Table generated from the polynomial at import, not
pasted.
"""

from __future__ import annotations

POLY = 0x8408  # reflected CRC-16/CCITT (0x1021 bit-reversed)


def _make_table() -> list[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ POLY if c & 1 else c >> 1
        table.append(c)
    return table


_TABLE = _make_table()


def update(crc: int, byte: int) -> int:
    return (crc >> 8) ^ _TABLE[(crc ^ byte) & 0xFF]


def crc16(data: bytes | bytearray | list[int]) -> int:
    crc = 0
    for b in data:
        crc = (crc >> 8) ^ _TABLE[(crc ^ b) & 0xFF]
    return crc


def valid(block: bytes | bytearray, crc0: int, crc1: int) -> bool:
    """True if `block` + the two received CRC bytes form a valid codeword."""
    crc = crc16(block)
    crc = update(crc, crc0)
    crc = update(crc, crc1)
    return crc == 0
