"""Numba kernels for TETRA-specific DSP and channel coding hot paths.

`fastmath=True` is only enabled where downstream logic tolerates FP reordering
(freq shift, FIR); CRC / parity / sign-sensitive paths stay precise.
"""

import numba as nb
import numpy as np

from tsdr.radio.dsp._kernels import fir_decim_c64_into

# Rate 2/3 puncturing pattern (ETSI EN 300 392-2 Section 8.2.3.1.3):
#   P(1..3) = {0, 1, 2, 5}, T = 3, period = 8.
_P_RATE_2_3 = np.array([0, 1, 2, 5], dtype=np.int32)
_T_RATE_2_3 = 3
_PERIOD_RATE_2_3 = 8


@nb.njit(cache=True)
def deinterleave(soft_bits: np.ndarray, k: int, a: int) -> np.ndarray:
    """Deinterleave type-4 -> type-3 soft bits using permutation pi(i) = 1 + (a*i) mod K."""
    out = np.empty_like(soft_bits)
    for i in range(1, k + 1):
        pi = 1 + (a * i) % k
        out[i - 1] = soft_bits[pi - 1]
    return out


@nb.njit(cache=True)
def depuncture_2_3(type345: np.ndarray, mother_len: int) -> np.ndarray:
    """Depuncture rate 2/3: insert erasures (0.0) at missing positions."""
    mother = np.zeros(mother_len, dtype=np.float32)
    n = type345.shape[0]
    period = _PERIOD_RATE_2_3
    t = _T_RATE_2_3
    for j in range(1, n + 1):
        i = j  # i_func = identity for rate 2/3
        block = (i - 1) // t
        k = period * block + _P_RATE_2_3[i - t * block]
        mother[k - 1] = type345[j - 1]
    return mother


@nb.njit(cache=True)
def crc16_ccitt(bits: np.ndarray) -> int:
    """CRC-16-CCITT, poly=0x1021, init=0xFFFF."""
    crc = np.uint32(0xFFFF)
    n = bits.shape[0]
    for i in range(n):
        bit = np.uint32(bits[i] & 1)
        if ((crc >> np.uint32(15)) & np.uint32(1)) ^ bit:
            crc = ((crc << np.uint32(1)) ^ np.uint32(0x1021)) & np.uint32(0xFFFF)
        else:
            crc = (crc << np.uint32(1)) & np.uint32(0xFFFF)
    return int(crc)


@nb.njit(cache=True)
def bits_to_uint(bits: np.ndarray, start: int, length: int) -> int:
    """Read `length` bits as MSB-first unsigned integer."""
    val = np.uint64(0)
    for i in range(length):
        val = (val << np.uint64(1)) | np.uint64(bits[start + i] & 1)
    return int(val)


@nb.njit(cache=True, inline="always")
def _lfsr_step(lfsr: np.uint32) -> tuple[np.uint32, np.uint32]:
    """One step of the TETRA scrambler LFSR.

    Polynomial taps (ETSI EN 300 392-2 Section 8.2.5):
    {32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 4, 2, 1}.
    """
    one = np.uint32(1)
    bit = (
        (lfsr & one)
        ^ ((lfsr >> np.uint32(6)) & one)
        ^ ((lfsr >> np.uint32(9)) & one)
        ^ ((lfsr >> np.uint32(10)) & one)
        ^ ((lfsr >> np.uint32(16)) & one)
        ^ ((lfsr >> np.uint32(20)) & one)
        ^ ((lfsr >> np.uint32(21)) & one)
        ^ ((lfsr >> np.uint32(22)) & one)
        ^ ((lfsr >> np.uint32(24)) & one)
        ^ ((lfsr >> np.uint32(25)) & one)
        ^ ((lfsr >> np.uint32(27)) & one)
        ^ ((lfsr >> np.uint32(28)) & one)
        ^ ((lfsr >> np.uint32(30)) & one)
        ^ ((lfsr >> np.uint32(31)) & one)
    )
    new_lfsr = (lfsr >> np.uint32(1)) | (bit << np.uint32(31))
    return bit, new_lfsr


@nb.njit(cache=True)
def generate_scramble_bits(init: np.uint32, length: int) -> np.ndarray:
    """TETRA 32-bit Fibonacci LFSR scrambler."""
    lfsr = np.uint32(init)
    out = np.empty(length, dtype=np.uint8)
    for i in range(length):
        bit, lfsr = _lfsr_step(lfsr)
        out[i] = np.uint8(bit)
    return out


@nb.njit(cache=True)
def descramble_soft(soft_bits: np.ndarray, init: np.uint32) -> np.ndarray:
    """Fused LFSR + sign-flip: multiply soft bits by +1 (bit=0) or -1 (bit=1)."""
    n = soft_bits.shape[0]
    lfsr = np.uint32(init)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        bit, lfsr = _lfsr_step(lfsr)
        if bit == 1:
            out[i] = -soft_bits[i]
        else:
            out[i] = soft_bits[i]
    return out


@nb.njit(cache=True, fastmath=True)
def rm3014_decode_kernel(codebook: np.ndarray, soft_30: np.ndarray) -> np.ndarray:
    """Soft ML RM(30,14) decode: argmax of `codebook @ soft_30`, then bit-unpack.

    `codebook` is the precomputed 16384 x 30 float32 table of ±1 codewords
    (identity rows || parity rows). The inner tap loop is 30 float32 MACs
    per row with no loop-carried dependencies, so LLVM can auto-vectorize
    it under `fastmath`.
    """
    n = codebook.shape[0]
    k = codebook.shape[1]
    best_idx = 0
    best_score = np.float32(-1e30)
    for row in range(n):
        acc = np.float32(0.0)
        for col in range(k):
            acc += codebook[row, col] * soft_30[col]
        if acc > best_score:
            best_score = acc
            best_idx = row
    out = np.empty(14, dtype=np.uint8)
    for i in range(14):
        out[i] = np.uint8((best_idx >> (13 - i)) & 1)
    return out


def fir_decim_c64(
    x: np.ndarray,
    taps: np.ndarray,
    m: int,
    history: np.ndarray,
    phase_in: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Test/one-shot wrapper for `fir_decim_c64_into`."""
    k = taps.shape[0]
    flipped = np.ascontiguousarray(taps[::-1]).astype(np.float32)
    history_out = history.astype(np.complex64, copy=True)
    padded = np.empty(len(x) + k - 1, dtype=np.complex64)
    y_out = np.empty((len(x) + m - 1) // m + 2, dtype=np.complex64)
    n_out, phase_out = fir_decim_c64_into(x, flipped, m, history_out, phase_in, padded, y_out)
    return y_out[:n_out].copy(), history_out, phase_out


def fir_filter_c64(
    x: np.ndarray,
    taps: np.ndarray,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-decimating FIR (`m=1`)."""
    y, new_history, _ = fir_decim_c64(x, taps, 1, history, 0)
    return y, new_history
