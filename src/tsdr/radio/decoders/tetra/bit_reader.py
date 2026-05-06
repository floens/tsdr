from __future__ import annotations

import numpy as np

from tsdr.radio.decoders.tetra._kernels import bits_to_uint


class BitReader:
    """Positional reader over a uint8 bit array (MSB-first per field).

    Typical usage mirrors an ETSI table line-by-line:

        r = BitReader(type1)
        r.skip(2)              # PDU type already matched by caller
        fill_bit = r.u(1)
        encryption = r.u(2)
        ...

    Bounds checking is the caller's responsibility: `r.remaining` tells
    you how many bits are left before a read would go off the end.
    """

    __slots__ = ("bits", "pos")

    def __init__(self, bits: np.ndarray, pos: int = 0) -> None:
        self.bits = bits
        self.pos = pos

    def u(self, n: int) -> int:
        """Read `n` bits as unsigned int, advance the cursor."""
        v = int(bits_to_uint(self.bits, self.pos, n))
        self.pos += n
        return v

    def skip(self, n: int) -> None:
        """Advance the cursor by `n` bits without reading."""
        self.pos += n

    def peek(self, n: int) -> int:
        """Read `n` bits as unsigned int without advancing."""
        return int(bits_to_uint(self.bits, self.pos, n))

    @property
    def remaining(self) -> int:
        """Number of bits left in the buffer from the current position."""
        return len(self.bits) - self.pos
