"""Forward error correction primitives used by AMBE+2.

The codec uses two short block codes:

    Golay(23,12)      -- on C0 and C1 codewords (3-bit error correction)
    Hamming(15,11)    -- not used by AMBE 3600x2450 directly but is shared
                        with the IMBE 7200x4400 path; we include it for
                        completeness and possible reuse.

Given the same input codeword, the decoders return the same corrected
codeword and the same error count. Tests verify against fixture vectors
-- see tests/test_fec.py.
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.vocoder.ambe._constants import (
    GOLAY_GENERATOR,
    GOLAY_MATRIX,
    HAMMING_GENERATOR,
    HAMMING_MATRIX,
)


def golay_23_12(bits_in: np.ndarray) -> tuple[np.ndarray, int]:
    """Decode a Golay(23,12) codeword.

    `bits_in` is a length-23 array of 0/1. Bit ordering:
    bits_in[0] is the LSB, bits_in[22] is the MSB. The top 12 bits are
    the data and the low 11 bits are the ECC.

    Returns ``(bits_out, errs)`` where ``bits_out`` is the corrected
    codeword (same layout as input) and ``errs`` counts the number of
    data bits (bits 11..22) that changed during correction.
    """
    if bits_in.shape != (23,):
        raise ValueError(f"expected length-23 input, got shape {bits_in.shape}")

    # Pack into an integer, MSB-first. Matches: `block = (block << 1) + in[i]`
    # iterated from i=22 down to i=0.
    block = 0
    for i in range(22, -1, -1):
        block = (block << 1) | int(bits_in[i])

    # Compute expected ECC from the 12 data bits (top 12) by XOR-accumulating
    # rows of the Golay generator selected by each data bit.
    ecc_expected = 0
    mask = 0x400000  # bit 22
    for i in range(12):
        if block & mask:
            ecc_expected ^= int(GOLAY_GENERATOR[i])
        mask >>= 1
    ecc_bits = block & 0x7FF  # low 11 bits
    syndrome = ecc_expected ^ ecc_bits

    data_bits = (block >> 11) ^ int(GOLAY_MATRIX[syndrome])

    # Unpack corrected data back into bits_out[11..22]; bits 0..10 (ECC) are
    # copied unchanged.
    bits_out = np.empty(23, dtype=bits_in.dtype)
    tmp = data_bits
    for i in range(11, 23):
        bits_out[i] = tmp & 1
        tmp >>= 1
    bits_out[:11] = bits_in[:11]

    errs = int(np.count_nonzero(bits_out[11:23] != bits_in[11:23]))
    return bits_out, errs


def hamming_15_11(bits_in: np.ndarray) -> tuple[np.ndarray, int]:
    """Decode a Hamming(15,11) codeword.

    `bits_in` is length 15 LSB-first. Returns ``(bits_out, errs)`` where
    ``errs`` is 0 or 1 (the Hamming code corrects at most 1 bit).
    """
    if bits_in.shape != (15,):
        raise ValueError(f"expected length-15 input, got shape {bits_in.shape}")

    block = 0
    for i in range(14, -1, -1):
        block = (block << 1) | int(bits_in[i])

    syndrome = 0
    for i in range(4):
        syndrome <<= 1
        stmp = block & int(HAMMING_GENERATOR[i])
        parity = stmp & 1
        for _ in range(14):
            stmp >>= 1
            parity ^= stmp & 1
        syndrome |= parity

    errs = 0
    if syndrome > 0:
        errs = 1
        block ^= int(HAMMING_MATRIX[syndrome])

    bits_out = np.empty(15, dtype=bits_in.dtype)
    for i in range(15):
        bits_out[i] = (block >> i) & 1
    return bits_out, errs
