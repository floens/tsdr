"""FT8 / FT4 CRC-14 (polynomial 0x2757).

The CRC is computed MSB-first over the packed message bits; for
verification the receiver zero-extends the 77-bit payload to 82 bits and
recomputes.
"""

from __future__ import annotations

import numpy as np

CRC_WIDTH = 14
CRC_POLY = 0x2757
_TOPBIT = 1 << (CRC_WIDTH - 1)
_MASK = (1 << CRC_WIDTH) - 1


def compute_crc(message: bytes | np.ndarray, num_bits: int) -> int:
    """Compute 14-bit CRC over the first ``num_bits`` of ``message`` (MSB-first)."""
    if isinstance(message, np.ndarray):
        message = bytes(message)
    remainder = 0
    idx_byte = 0
    for idx_bit in range(num_bits):
        if idx_bit % 8 == 0:
            remainder ^= message[idx_byte] << (CRC_WIDTH - 8)
            idx_byte += 1
        if remainder & _TOPBIT:
            remainder = ((remainder << 1) ^ CRC_POLY) & ((_TOPBIT << 1) - 1)
        else:
            remainder = (remainder << 1) & ((_TOPBIT << 1) - 1)
    return remainder & _MASK


def extract_crc(a91: bytes | np.ndarray) -> int:
    """Return the 14-bit CRC stored in bits 77..90 of a 91-bit message (12-byte buffer)."""
    if isinstance(a91, np.ndarray):
        a91 = bytes(a91)
    return ((a91[9] & 0x07) << 11) | (a91[10] << 3) | (a91[11] >> 5)


def add_crc(payload: bytes | np.ndarray) -> bytes:
    """Return a 12-byte buffer containing the 77-bit payload + 14-bit CRC.

    The input ``payload`` must already be packed MSB-first into 10 bytes; the
    low 3 bits of byte 9 (i.e. payload bits 77..79) are ignored and overwritten
    with the top of the CRC.
    """
    if isinstance(payload, np.ndarray):
        payload = bytes(payload)
    if len(payload) < 10:
        raise ValueError(f"payload must be at least 10 bytes, got {len(payload)}")
    a91 = bytearray(12)
    a91[:10] = payload[:10]
    a91[9] &= 0xF8
    a91[10] = 0
    a91[11] = 0
    checksum = compute_crc(bytes(a91), 96 - CRC_WIDTH)  # 82 bits
    a91[9] |= checksum >> 11
    a91[10] = (checksum >> 3) & 0xFF
    a91[11] = (checksum << 5) & 0xFF
    return bytes(a91)


def verify_crc(a91: bytes | np.ndarray) -> tuple[bool, int]:
    """Return ``(ok, calculated)``: the recomputed CRC over the payload
    zero-extended from 77 to 82 bits, and whether it matches the stored CRC.
    """
    if isinstance(a91, np.ndarray):
        a91 = bytes(a91)
    if len(a91) < 12:
        raise ValueError(f"a91 must be at least 12 bytes, got {len(a91)}")
    extracted = extract_crc(a91)
    cleared = bytearray(a91[:12])
    cleared[9] &= 0xF8
    cleared[10] = 0
    calculated = compute_crc(bytes(cleared), 96 - CRC_WIDTH)
    return extracted == calculated, calculated
