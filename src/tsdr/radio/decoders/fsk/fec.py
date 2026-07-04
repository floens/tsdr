"""SITOR-B / CCIR-476 forward error correction.

CCIR-476 characters carry a constant 4-mark / 3-space ratio, so a single bit
error is detectable. FEC mode B (NAVTEX) also transmits every character twice:
the repeat ("rep") arrives 35 bits (5 characters) before its "alpha" copy. When
the alpha copy fails the 4-of-7 check, `resolve_char` falls back to the rep, then
to the bitwise-summed average, then to single-bit-flip permutations.
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.decoders.fsk.tables import CCIR_REP

FEC_DELAY = 35  # bits from a rep character to its alpha copy (5 chars * 7 bits)


def check_bits(code: int) -> bool:
    """A valid CCIR-476 code has exactly four mark bits set."""
    return bin(code).count("1") == 4


def bytes_to_code(bits: np.ndarray, pos: int) -> int:
    """Pack seven soft bits (LSB-first, >0 = mark) starting at *pos* into a 7-bit code."""
    code = 0
    for i in range(7):
        if bits[pos + i] > 0.0:
            code |= 1 << i
    return code


def flip_smallest_bit(soft: np.ndarray) -> None:
    """Flip the least-confident bit of a 7-bit soft slice in place.

    Turns a 5-mark or 4-space slice (one bit off) into a valid 4-of-7 code.
    """
    min_zero, min_one = -np.inf, np.inf
    min_zero_pos, min_one_pos = -1, -1
    count_zero, count_one = 0, 0
    for i in range(7):
        v = soft[i]
        if v < 0.0:
            count_zero += 1
            if v > min_zero:
                min_zero, min_zero_pos = v, i
        else:
            count_one += 1
            if v < min_one:
                min_one, min_one_pos = v, i
    if count_zero == 4:
        soft[min_zero_pos] = -soft[min_zero_pos]
    elif count_one == 5:
        soft[min_one_pos] = -soft[min_one_pos]


def resolve_char(bits: np.ndarray, cursor: int, rep_cursor: int) -> tuple[int | None, int]:
    """Resolve the character at *cursor*, using the rep copy at *rep_cursor*.

    Returns ``(code, status)`` where status is +1 alpha valid, 0 rep replacement
    (or rep-marker skip -> code None), -1 FEC reconstruction, -2 hard failure.
    """
    code = bytes_to_code(bits, cursor)
    if check_bits(code):
        return code, 1
    if rep_cursor < 0:
        return None, -1

    rep = bytes_to_code(bits, rep_cursor)
    if check_bits(rep):
        if rep == CCIR_REP:
            return None, 0  # phase-aligned rep marker; skip without decoding
        return rep, 0

    avg = bits[cursor : cursor + 7] + bits[rep_cursor : rep_cursor + 7]
    calc = bytes_to_code(avg, 0)
    if check_bits(calc):
        return calc, -1

    flip_smallest_bit(bits[cursor : cursor + 7])
    calc = bytes_to_code(bits, cursor)
    if check_bits(calc):
        return calc, -1

    flip_smallest_bit(bits[rep_cursor : rep_cursor + 7])
    calc = bytes_to_code(bits, rep_cursor)
    if check_bits(calc):
        return calc, -1

    avg = bits[cursor : cursor + 7] + bits[rep_cursor : rep_cursor + 7]
    flip_smallest_bit(avg)
    calc = bytes_to_code(avg, 0)
    if check_bits(calc):
        return calc, -1

    return None, -2
