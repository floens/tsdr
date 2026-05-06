"""Byte cursor for FIG payload parsing.

Each FIG-extension parser receives a `Cursor` sliced to the FIG's advertised
length and lets reads past end-of-buffer raise `CursorTruncated`. The parent
dispatch catches that and moves on to the next FIG, so a wrong-width read
inside one extension cannot drift into the next.

DAB FIG fields are byte-aligned at FIG boundaries with a few sub-byte fields
inside, so a byte-oriented cursor with `bits()` for sub-byte extraction is
the right granularity (lighter than a full bit-array reader like
`tetra/bit_reader.py`).
"""

from __future__ import annotations


class CursorTruncated(Exception):
    """Raised when a Cursor read goes past end-of-buffer."""


class Cursor:
    """Position-tracking byte reader over a memoryview."""

    __slots__ = ("_buf", "_pos")

    def __init__(self, data: bytes | memoryview) -> None:
        self._buf = memoryview(data) if not isinstance(data, memoryview) else data
        self._pos = 0

    @property
    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def has(self, n: int) -> bool:
        return self.remaining >= n

    def _need(self, n: int) -> None:
        if self.remaining < n:
            raise CursorTruncated(
                f"need {n} bytes, have {self.remaining} (pos={self._pos}, len={len(self._buf)})"
            )

    def u8(self) -> int:
        self._need(1)
        v = self._buf[self._pos]
        self._pos += 1
        return v

    def u16(self) -> int:
        self._need(2)
        b = self._buf
        p = self._pos
        v = (b[p] << 8) | b[p + 1]
        self._pos += 2
        return v

    def u32(self) -> int:
        self._need(4)
        b = self._buf
        p = self._pos
        v = (b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]
        self._pos += 4
        return v

    def bytes(self, n: int) -> bytes:
        self._need(n)
        out = bytes(self._buf[self._pos : self._pos + n])
        self._pos += n
        return out

    def skip(self, n: int) -> None:
        self._need(n)
        self._pos += n


def bits(byte: int, hi: int, lo: int) -> int:
    """Extract bits[hi:lo] from a byte (MSB=7, LSB=0, both inclusive)."""
    width = hi - lo + 1
    return (byte >> lo) & ((1 << width) - 1)
