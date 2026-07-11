"""ACARS syndrome-based forward error correction.

Two correctors, dispatched by `frame.finalize`:
  - `fix_parity_errors`: correct up to `MAX_PARITY_ERRORS` bytes flagged by odd
    parity (single-bit-per-byte errors).
  - `fix_double_error`: correct a 2-bit-in-one-byte error that parity can't see
    (even number of flips), used when the CRC is bad but no parity error showed.

Both work off the CRC *syndrome*: the residual a single flipped bit leaves in the
block CRC. Because the CRC (init 0, no final XOR) is linear, the syndrome of
flipping bit `i` of the byte at distance `m` from the codeword end (the last CRC
byte is m=0) is just `crc16([1<<i] + m zero bytes)`, computed on demand rather
than from a precomputed 2 kB table.
"""

from __future__ import annotations

from functools import cache

from tsdr.radio.decoders.acars.crc import update


@cache
def _syndrome(m: int, i: int) -> int:
    crc = update(0, 1 << i)
    for _ in range(m):
        crc = update(crc, 0)
    return crc


def _crc_byte_error(crc: int) -> bool:
    """True if `crc` is the syndrome of a single-bit error inside the 2 CRC bytes."""
    return any(_syndrome(m, i) == crc for m in (0, 1) for i in range(8))


def fix_parity_errors(raw: bytearray, base_crc: int, pr: list[int]) -> bool:
    """Correct the parity-flagged bytes at indices `pr` (<= 3). Mutates `raw`."""
    txtlen = len(raw)

    def rec(crc: int, k: int) -> bool:
        if k == len(pr):
            return crc == 0 or _crc_byte_error(crc)
        m = txtlen - pr[k] + 1
        for i in range(8):
            if rec(crc ^ _syndrome(m, i), k + 1):
                raw[pr[k]] ^= 1 << i
                return True
        return False

    return rec(base_crc, 0)


def fix_double_error(raw: bytearray, base_crc: int) -> bool:
    """Correct a 2-bit error within one byte. Mutates `raw`."""
    if _crc_byte_error(base_crc):
        return True
    txtlen = len(raw)
    for k in range(txtlen):
        m = txtlen - k + 1
        for i in range(8):
            si = _syndrome(m, i)
            for j in range(i + 1, 8):
                if base_crc ^ si ^ _syndrome(m, j) == 0:
                    raw[k] ^= (1 << i) | (1 << j)
                    return True
    return False
