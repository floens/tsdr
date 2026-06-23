"""AX.25 frame check sequence: CRC-16/X.25.

Reflected polynomial 0x8408 (reverse of 0x1021), init 0xFFFF, xorout 0xFFFF,
refin/refout. The 2-byte FCS is transmitted low byte first. Cross-checked
against direwolf `fcs_calc.c` (same table, same init).
"""

from __future__ import annotations

_POLY = 0x8408
_INIT = 0xFFFF


def crc16_x25(data: bytes) -> int:
    """FCS value over ``data`` (the FCS that should follow it on the wire)."""
    crc = _INIT
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ _POLY if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def fcs_ok(frame: bytes) -> bool:
    """True if the trailing 2 bytes of ``frame`` are its valid FCS (low byte first)."""
    if len(frame) < 3:
        return False
    return crc16_x25(frame[:-2]) == (frame[-2] | (frame[-1] << 8))
