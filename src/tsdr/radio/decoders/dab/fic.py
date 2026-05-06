import numpy as np

from .constants import FIB_BITS, FIC_BLOCK_BITS, N_FIBS_PER_BLOCK, N_FIC_BLOCKS, N_FIC_SYMBOLS
from .fec import _PRBS_768, _check_fib_crc
from .ofdm import OFDMState, _dqpsk_to_soft_bits, _ofdm_demod_frame
from .viterbi import _viterbi_decode


def _build_depuncture_index() -> np.ndarray:
    """Precompute mapping from input positions to depunctured output positions."""
    indices = []  # output index for each input bit
    di = 0

    # PI_16: {1,1,1,0} × 8, repeated for 21×4 sub-blocks
    for _ in range(21 * 4):
        for _ in range(8):
            indices.append(di)
            di += 1
            indices.append(di)
            di += 1
            indices.append(di)
            di += 1
            di += 1  # skip punctured

    # PI_15: {1,1,1,0}×7 + {1,1,0,0}, for 3×4 sub-blocks
    for _ in range(3 * 4):
        for _ in range(7):
            indices.append(di)
            di += 1
            indices.append(di)
            di += 1
            indices.append(di)
            di += 1
            di += 1
        indices.append(di)
        di += 1
        indices.append(di)
        di += 1
        di += 2

    # Tail: 6 × {1,1,0,0}
    for _ in range(6):
        indices.append(di)
        di += 1
        indices.append(di)
        di += 1
        di += 2

    assert len(indices) == 2304
    assert di == 3096
    return np.array(indices, dtype=np.int32)


_DEPUNCTURE_INDEX = _build_depuncture_index()


def _fic_depuncture(soft_bits: np.ndarray) -> np.ndarray:
    """Depuncture FIC block: 2304 soft bits -> 3096 soft bits (vectorized).

    Inserts zero-confidence values at punctured positions using precomputed indices.
    """
    out = np.zeros(3096, dtype=np.float32)
    out[_DEPUNCTURE_INDEX] = soft_bits
    return out


def _demod_frame_fic(frame_iq: np.ndarray, state: OFDMState | None = None) -> np.ndarray:
    """Demodulate OFDM frame and extract FIC soft bits.

    Returns:
        Soft bits array of shape (9216,) - all FIC soft bits ready for
        splitting into 4 blocks of 2304.
    """
    fft_syms = _ofdm_demod_frame(frame_iq, state)
    if fft_syms is None:
        return np.zeros(9216, dtype=np.float32)
    # FIC = symbols 1-3 (DQPSK against PRS and each other), start_sym=0 gives PRS as reference
    return _dqpsk_to_soft_bits(fft_syms, 0, N_FIC_SYMBOLS)


def _decode_fic(fic_soft: np.ndarray) -> list[tuple[bytes, bool]]:
    """Decode FIC from 9216 soft bits into FIBs.

    The 9216 soft bits (3 symbols × 3072) are split into 4 blocks of 2304.
    Each block is depunctured, Viterbi-decoded, descrambled, and CRC-checked
    to produce 3 FIBs.

    Returns:
        List of (fib_bytes, crc_ok) tuples. Up to 12 FIBs per frame.
    """
    fibs = []

    for block_idx in range(N_FIC_BLOCKS):
        block_soft = fic_soft[block_idx * FIC_BLOCK_BITS : (block_idx + 1) * FIC_BLOCK_BITS]

        # Depuncture: 2304 -> 3096
        depunctured = _fic_depuncture(block_soft)

        # Viterbi decode: 3096 at rate 1/4 -> 774 bits (768 data + 6 tail)
        decoded = _viterbi_decode(depunctured)

        # Discard tail bits
        decoded = decoded[:768]

        # Energy dispersal: XOR with PRBS
        decoded = decoded ^ _PRBS_768

        # Split into 3 FIBs of 256 bits (32 bytes)
        for fib_idx in range(N_FIBS_PER_BLOCK):
            fib_bits = decoded[fib_idx * FIB_BITS : (fib_idx + 1) * FIB_BITS]
            fib_bytes = bytes(np.packbits(fib_bits))
            crc_ok = _check_fib_crc(fib_bytes)
            fibs.append((fib_bytes, crc_ok))

    return fibs
