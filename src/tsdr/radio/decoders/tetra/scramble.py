"""TETRA scrambler/descrambler using 32-bit Fibonacci LFSR.

Taps: {32,26,23,22,16,12,11,10,8,7,5,4,2,1} (from ETSI EN 300 392-2 Section 8.2.5).

The LFSR + sign-flip hot paths are numba kernels in `_kernels.py`. This file
holds the pure-Python public API + a small helper that the decoder callers
use directly.
"""

import numpy as np

from tsdr.radio.decoders.tetra._kernels import (
    descramble_soft as _descramble_soft_kernel,
)
from tsdr.radio.decoders.tetra._kernels import (
    generate_scramble_bits as _generate_scramble_bits_kernel,
)

SCRAMB_INIT = 3  # Fixed init for SB1


def scramble_init(mcc: int, mnc: int, colour_code: int) -> int:
    """Compute scramble init from network parameters."""
    return (
        ((colour_code & 0x3F) | ((mnc & 0x3FFF) << 6) | ((mcc & 0x3FF) << 20)) << 2
    ) | SCRAMB_INIT


def generate_scramble_bits(init: int, length: int) -> np.ndarray:
    """Generate scrambling bit sequence from LFSR (thin wrapper over numba kernel)."""
    result: np.ndarray = _generate_scramble_bits_kernel(np.uint32(init & 0xFFFFFFFF), length)
    return result


def descramble_soft(soft_bits: np.ndarray, init: int) -> np.ndarray:
    """Descramble soft bits: multiply by +1 (scramble=0) or -1 (scramble=1).

    Fused LFSR + sign-flip in a single numba kernel pass.
    """
    result: np.ndarray = _descramble_soft_kernel(soft_bits, np.uint32(init & 0xFFFFFFFF))
    return result


def scramble_hard(bits: np.ndarray, init: int) -> np.ndarray:
    """Scramble hard bits: XOR with scrambling sequence."""
    scramble = generate_scramble_bits(init, len(bits))
    result: np.ndarray = bits ^ scramble
    return result
