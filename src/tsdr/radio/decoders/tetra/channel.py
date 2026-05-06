"""TETRA channel decoding: Type-5 -> Type-1 pipeline.

Descramble -> deinterleave -> depuncture -> Viterbi -> CRC check.
Plus RM(30,14) soft ML decoder for broadcast blocks.
"""

import numpy as np

from tsdr.radio.decoders.tetra._kernels import (
    crc16_ccitt as _kernel_crc16_ccitt,
)
from tsdr.radio.decoders.tetra._kernels import (
    deinterleave as _kernel_deinterleave,
)
from tsdr.radio.decoders.tetra._kernels import (
    depuncture_2_3 as _kernel_depuncture_2_3,
)
from tsdr.radio.decoders.tetra._kernels import (
    rm3014_decode_kernel as _kernel_rm3014_decode,
)
from tsdr.radio.decoders.tetra.scramble import descramble_soft, scramble_hard
from tsdr.radio.dsp.viterbi import ViterbiDecoder

TETRA_K = 5
TETRA_GENERATORS = [0o31, 0o27, 0o35, 0o33]

# Rate 2/3 puncturing parameters
_P_RATE_2_3 = [0, 1, 2, 5]
_T_RATE_2_3 = 3
_PERIOD_RATE_2_3 = 8

# Block parameters: (type345_bits, type2_bits, type1_bits, interleave_a)
BLOCK_PARAMS = {
    "SB1": (120, 80, 60, 11),
    "SB2": (216, 144, 124, 101),
    "NDB": (216, 144, 124, 101),
    "SCH_HU": (168, 112, 92, 13),
    "SCH_F": (432, 288, 268, 103),
}

# RM(30,14) generator matrix rows
_RM_GEN_PARITY = np.array(
    [
        [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    ],
    dtype=np.uint8,
)


def _conv_encode_tetra(bits: np.ndarray) -> np.ndarray:
    """Convolutional encode with TETRA K=5 rate-1/4 parameters."""
    state = 0
    out = []
    for bit in bits:
        reg = (int(bit) << (TETRA_K - 1)) | state
        state = reg >> 1
        for gen in TETRA_GENERATORS:
            out.append(bin(reg & gen).count("1") % 2)
    return np.array(out, dtype=np.uint8)


# Lazy-initialized Viterbi decoder and RM codebook
_viterbi: ViterbiDecoder | None = None
_rm_codebook: np.ndarray | None = None


def _get_viterbi() -> ViterbiDecoder:
    global _viterbi
    if _viterbi is None:
        _viterbi = ViterbiDecoder(TETRA_K, TETRA_GENERATORS)
    return _viterbi


def _get_rm_codebook() -> np.ndarray:
    """Precompute all 2^14 RM(30,14) codewords as ±1 arrays."""
    global _rm_codebook
    if _rm_codebook is not None:
        return _rm_codebook

    # Build full 30-bit generator rows: [14-bit identity | 16-bit parity]
    rows_30 = np.zeros((14, 30), dtype=np.uint8)
    for i in range(14):
        rows_30[i, i] = 1  # identity part (MSB first)
        rows_30[i, 14:] = _RM_GEN_PARITY[i]

    # Generate all 16384 codewords
    n_codewords = 1 << 14
    codebook = np.zeros((n_codewords, 30), dtype=np.float32)
    for val in range(n_codewords):
        codeword = np.zeros(30, dtype=np.uint8)
        for i in range(14):
            if (val >> (13 - i)) & 1:
                codeword ^= rows_30[i]
        # Convert 0->-1, 1->+1 (matches Viterbi soft bit convention)
        codebook[val] = 2.0 * codeword.astype(np.float32) - 1.0

    _rm_codebook = codebook
    return codebook


def deinterleave(soft_bits: np.ndarray, k: int, a: int) -> np.ndarray:
    """Deinterleave type-4 -> type-3 soft bits using permutation pi(i) = 1 + (a*i) mod K."""
    result: np.ndarray = _kernel_deinterleave(soft_bits, k, a)
    return result


def interleave(soft_bits: np.ndarray, k: int, a: int) -> np.ndarray:
    """Interleave type-3 -> type-4 soft bits (inverse of deinterleave)."""
    out = np.empty_like(soft_bits)
    for i in range(1, k + 1):
        pi = 1 + (a * i) % k
        out[pi - 1] = soft_bits[i - 1]
    return out


def depuncture_2_3(type345: np.ndarray, mother_len: int) -> np.ndarray:
    """Depuncture rate 2/3: insert erasures (0.0) at missing positions."""
    result: np.ndarray = _kernel_depuncture_2_3(type345, mother_len)
    return result


def puncture_2_3(mother: np.ndarray, type345_len: int) -> np.ndarray:
    """Puncture mother code to rate 2/3 (forward direction for testing)."""
    out = np.empty(type345_len, dtype=mother.dtype)
    for j in range(1, type345_len + 1):
        i = j
        k = (
            _PERIOD_RATE_2_3 * ((i - 1) // _T_RATE_2_3)
            + _P_RATE_2_3[i - _T_RATE_2_3 * ((i - 1) // _T_RATE_2_3)]
        )
        out[j - 1] = mother[k - 1]
    return out


def crc16_ccitt(bits: np.ndarray) -> int:
    """Compute CRC-16-CCITT over bit array. Poly=0x1021, init=0xFFFF."""
    result: int = _kernel_crc16_ccitt(bits)
    return result


def append_crc(type1: np.ndarray) -> np.ndarray:
    """Append 16-bit CRC to type1 bits, producing type2 bits."""
    crc = crc16_ccitt(type1) ^ 0xFFFF  # Final inversion per CRC-CCITT
    crc_bits = np.array([(crc >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)
    return np.concatenate([type1, crc_bits])


def check_crc(type2: np.ndarray) -> bool:
    """Check CRC-16 over type2 bits. Valid when residual = 0x1D0F."""
    return crc16_ccitt(type2) == 0x1D0F


def rm3014_encode(info_14: np.ndarray) -> np.ndarray:
    """Encode 14 info bits to 30-bit RM codeword."""
    rows_30 = np.zeros((14, 30), dtype=np.uint8)
    for i in range(14):
        rows_30[i, i] = 1
        rows_30[i, 14:] = _RM_GEN_PARITY[i]

    codeword = np.zeros(30, dtype=np.uint8)
    for i in range(14):
        if info_14[i]:
            codeword ^= rows_30[i]
    return codeword


def rm3014_decode(soft_30: np.ndarray) -> np.ndarray:
    """Soft ML decode 30 soft bits -> 14 info bits."""
    codebook = _get_rm_codebook()
    soft_f32 = np.ascontiguousarray(soft_30, dtype=np.float32)
    result: np.ndarray = _kernel_rm3014_decode(codebook, soft_f32)
    return result


def encode_block(type1: np.ndarray, block_type: str, scramble_init: int) -> np.ndarray:
    """Full forward chain: type1 -> type5 (for testing)."""
    type345, type2, _, a = BLOCK_PARAMS[block_type]

    # CRC
    type2_bits = append_crc(type1)
    # Append K-1 tail bits to flush encoder
    tail_bits = type2 - len(type2_bits)
    if tail_bits > 0:
        type2_bits = np.concatenate([type2_bits, np.zeros(tail_bits, dtype=np.uint8)])
    assert len(type2_bits) == type2

    # Convolutional encode
    mother = _conv_encode_tetra(type2_bits)
    assert len(mother) == type2 * 4

    # Puncture
    punctured = puncture_2_3(mother, type345)

    # Interleave
    interleaved = interleave(punctured, type345, a)

    # Scramble
    return scramble_hard(interleaved, scramble_init)


def decode_block(
    type5_soft: np.ndarray, block_type: str, scramble_init: int
) -> tuple[np.ndarray, bool]:
    """Full reverse chain: type5 soft -> (type1 hard bits, crc_ok)."""
    type345, type2, type1_len, a = BLOCK_PARAMS[block_type]

    # Descramble
    type4 = descramble_soft(type5_soft, scramble_init)

    # Deinterleave
    type3 = deinterleave(type4, type345, a)

    # Depuncture
    mother_len = type2 * 4
    mother = depuncture_2_3(type3, mother_len)

    # Viterbi decode
    dec = _get_viterbi()
    type2_bits = dec.decode(mother)
    assert len(type2_bits) == type2

    # CRC check (over type1 + CRC, excluding tail bits)
    crc_len = type1_len + 16
    crc_ok = check_crc(type2_bits[:crc_len])

    # Extract type1
    type1_bits = type2_bits[:type1_len]

    return type1_bits, crc_ok
