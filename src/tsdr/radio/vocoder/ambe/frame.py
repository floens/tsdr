"""Frame-level operations: PN descrambling and bit packing.

An AMBE+2 2450 frame from DMR arrives as four codewords (C0..C3) totalling
72 bits per 20 ms voice frame:

    C0: 24 bits (12 data + 12 Golay parity, plus 1 parity bit for C0 itself)
    C1: 23 bits (12 data + 11 Golay parity), PN-scrambled using seed from C0
    C2: 11 bits (raw data)
    C3: 14 bits (raw data)

Order of processing:

    1. Golay-correct C0 (see `pyambe.fec.golay_23_12`)
    2. Generate PN sequence from top 12 bits of (corrected) C0 and XOR
       onto C1
    3. Golay-correct C1
    4. Pack 49 voice bits into `ambe_d` in the layout expected by the
       parameter decoder
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.vocoder.ambe.fec import golay_23_12


def demodulate_c1(ambe_fr: np.ndarray) -> None:
    """In-place PN descramble of C1 using the LCG seeded from C0.

    `ambe_fr` is a (4, 24) array of 0/1 bits; only row 1 (C1) is modified.

    The PRNG is a linear congruential generator with multiplier 173,
    increment 13849, modulus 65536. It is seeded from the 12 high bits
    of row 0 (C0) shifted left by 4.
    """
    if ambe_fr.shape != (4, 24):
        raise ValueError(f"expected (4, 24) frame, got {ambe_fr.shape}")

    # Seed: row 0 bits 12..23 packed MSB-first, then shifted left by 4.
    seed = 0
    for i in range(23, 11, -1):
        seed = (seed << 1) | int(ambe_fr[0, i])
    pr = np.zeros(24, dtype=np.int32)
    pr[0] = (16 * seed) & 0xFFFF
    for i in range(1, 24):
        pr[i] = ((173 * int(pr[i - 1])) + 13849) % 65536
    # Only take the MSB of each 16-bit value (value / 32768 → 0 or 1).
    pr_bits = (pr // 32768).astype(ambe_fr.dtype)

    # XOR pr[1..23] onto ambe_fr[1, 22..0]
    k = 1
    for j in range(22, -1, -1):
        ambe_fr[1, j] ^= pr_bits[k]
        k += 1


def pack_ambe_d(ambe_fr: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply C1 Golay ECC and pack the 49 voice data bits into `ambe_d`.

    Layout of `ambe_d`:

        ambe_d[0..11]   = C0[23..12]   (already Golay-corrected)
        ambe_d[12..23]  = C1[22..11]   (Golay-corrected here)
        ambe_d[24..34]  = C2[10..0]
        ambe_d[35..48]  = C3[13..0]

    Returns ``(ambe_d, errs)`` where ``errs`` is the number of Golay
    correctable errors found in C1.
    """
    if ambe_fr.shape != (4, 24):
        raise ValueError(f"expected (4, 24) frame, got {ambe_fr.shape}")

    ambe_d = np.empty(49, dtype=ambe_fr.dtype)
    pos = 0

    # C0: copy bits 23..12
    for j in range(23, 11, -1):
        ambe_d[pos] = ambe_fr[0, j]
        pos += 1

    # C1: golay-correct rows 0..22, then copy corrected bits 22..11
    gin = ambe_fr[1, :23].copy()
    gout, errs = golay_23_12(gin)
    for j in range(22, 10, -1):
        ambe_d[pos] = gout[j]
        pos += 1

    # C2: copy bits 10..0
    for j in range(10, -1, -1):
        ambe_d[pos] = ambe_fr[2, j]
        pos += 1

    # C3: copy bits 13..0
    for j in range(13, -1, -1):
        ambe_d[pos] = ambe_fr[3, j]
        pos += 1

    assert pos == 49
    return ambe_d, errs


def ecc_c0(ambe_fr: np.ndarray) -> int:
    """Apply Golay(23,12) correction to C0 bits 1..23 (in-place).

    ``ambe_fr[0, 0]`` is the C0 Golay24 parity bit which is not checked.
    """
    gin = ambe_fr[0, 1:24].copy()
    gout, errs = golay_23_12(gin)
    ambe_fr[0, 1:24] = gout
    return errs
