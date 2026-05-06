import numpy as np

from .ofdm import _dqpsk_to_soft_bits

# MSC: symbols 4-75 (72 symbols), DQPSK reference is symbol 3
N_MSC_SYMBOLS = 72
N_CIF_PER_FRAME = 4  # Mode I: 4 CIFs per frame
N_SYMBOLS_PER_CIF = 18  # 72/4
CIF_BITS = 55296  # 864 CUs × 64 bits/CU
BITS_PER_CU = 64


def _demod_frame_msc(fft_syms: np.ndarray) -> np.ndarray:
    """Demodulate MSC symbols (4-75) to soft bits.

    Args:
        fft_syms: Shape (76, 2048) from _ofdm_demod_frame.

    Returns:
        Soft bits array of shape (221184,) = 72 × 3072.
    """
    # Symbols 4-75 in the frame = indices 4..75 in fft_syms (0=PRS, 1-3=FIC)
    # DQPSK reference for symbol 4 is symbol 3 -> start_sym=3, n_syms=72
    return _dqpsk_to_soft_bits(fft_syms, 3, N_MSC_SYMBOLS)


def _msc_to_cifs(msc_soft: np.ndarray) -> list[np.ndarray]:
    """Split 221184 MSC soft bits into 4 CIFs of 55296 bits each.

    Each CIF = 18 symbols × 3072 bits = 55296 bits.
    """
    return [msc_soft[i * CIF_BITS : (i + 1) * CIF_BITS] for i in range(N_CIF_PER_FRAME)]


# Time De-interleaving
# ETSI EN 300 401 section 12.3: 16-frame convolutional interleaver


class _TimeDeinterleaver:
    """Time de-interleaver for one CIF position.

    Read-before-write order per ETSI convolutional interleaver spec.
    Map values: 15 - ETSI_delay.
    """

    # Interleave map (= 15 - ETSI_delay)
    _INTERLEAVE_MAP = np.array(
        [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15], dtype=np.int32
    )
    # Per-bit map value (same for all instances)
    _BIT_MAP = _INTERLEAVE_MAP[np.arange(CIF_BITS, dtype=np.int32) % 16]

    def __init__(self):
        self._buffer = [np.zeros(CIF_BITS, dtype=np.float32) for _ in range(16)]
        self._index = 0
        self._count = 0

    def push(self, cif_soft: np.ndarray) -> np.ndarray | None:
        """Push a new CIF and return de-interleaved output (None if not ready)."""
        self._count += 1

        # Read before write
        output = np.zeros(CIF_BITS, dtype=np.float32)
        for m in range(16):
            mask = m == self._BIT_MAP
            buf_idx = (self._index + m) % 16
            output[mask] = self._buffer[buf_idx][mask]

        # Write new data
        self._buffer[self._index] = cif_soft.copy()

        # Increment
        self._index = (self._index + 1) % 16

        if self._count <= 16:
            return None

        return output


def _extract_subchannel(cif_soft: np.ndarray, start_address: int, size: int) -> np.ndarray:
    """Extract subchannel's CUs from a de-interleaved CIF."""
    start_bit = start_address * BITS_PER_CU
    end_bit = start_bit + size * BITS_PER_CU
    return cif_soft[start_bit:end_bit]


# Puncturing patterns from ETSI EN 300 401 table 31
# Each pattern is 32 bits: 1=transmitted, 0=punctured
# Encoded as bit-packed uint32: MSB first, e.g. PI_1 = 0xC8888888
_PI_PACKED: dict[int, int] = {
    1: 0xC8888888,
    2: 0xC888C888,
    3: 0xC8C8C888,
    4: 0xC8C8C8C8,
    5: 0xCCC8C8C8,
    6: 0xCCC8CCC8,
    7: 0xCCCCCCC8,
    8: 0xCCCCCCCC,
    9: 0xECCCCCCC,
    10: 0xECCCECCC,
    11: 0xECECECCC,
    12: 0xECECECEC,
    13: 0xEEECECEC,
    14: 0xEEECEEEC,
    15: 0xEEEEEEEC,
    16: 0xEEEEEEEE,
    17: 0xFEEEEEEE,
    18: 0xFEEEFEEE,
    19: 0xFEFEFEEE,
    20: 0xFEFEFEFE,
    21: 0xFFFEFEFE,
    22: 0xFFFEFFFE,
    23: 0xFFFFFFFE,
    24: 0xFFFFFFFF,
}

_PI_PATTERNS: dict[int, list[int]] = {}
for _k, _v in _PI_PACKED.items():
    _PI_PATTERNS[_k] = [(_v >> (31 - i)) & 1 for i in range(32)]
del _k, _v


def _build_eep_depuncture_index(size: int, protection: int, option: int) -> tuple[np.ndarray, int]:
    """Build depuncture index for EEP subchannel.

    Args:
        size: Subchannel size in CUs
        protection: Protection level (0-3, maps to EEP 1-A..4-A or 1-B..4-B)
        option: 0=EEP-A, 1=EEP-B

    Returns:
        (depuncture_index, depunctured_length) - index maps input positions to
        output positions (same convention as _DEPUNCTURE_INDEX for FIC).

    Each L block = 128 coded bits = 4 repetitions of the 32-element PI pattern.
    n = subchannel_size / cu_factor (n is the bitrate parameter, not CU size).
    Tail bits: all 24 coded bits transmitted (PI_24 = all-ones).
    """
    if option == 0:
        # EEP-A: n = size / cu_factor, cu_factors: [12, 8, 6, 4]
        cu_factors = [12, 8, 6, 4]
        n = size // cu_factors[protection]
        eep_a = {
            0: (6 * n - 3, 24, 3, 23),  # EEP 1-A
            1: (2 * n - 3, 14, 4 * n + 3, 13),  # EEP 2-A
            2: (6 * n - 3, 8, 3, 7),  # EEP 3-A
            3: (4 * n - 3, 3, 2 * n + 3, 2),  # EEP 4-A
        }
        l1, pi1_idx, l2, pi2_idx = eep_a[protection]
    else:
        # EEP-B: n = size / cu_factor, cu_factors: [27, 21, 18, 15]
        cu_factors = [27, 21, 18, 15]
        n = size // cu_factors[protection]
        eep_b = {
            0: (24 * n - 3, 10, 3, 9),  # EEP 1-B
            1: (24 * n - 3, 6, 3, 5),  # EEP 2-B
            2: (24 * n - 3, 4, 3, 3),  # EEP 3-B
            3: (24 * n - 3, 2, 3, 1),  # EEP 4-B
        }
        l1, pi1_idx, l2, pi2_idx = eep_b[protection]

    pi1 = _PI_PATTERNS[pi1_idx]
    pi2 = _PI_PATTERNS[pi2_idx]

    # Build the depuncture index
    # Each L block = 128 coded bits, PI pattern (32 elements) applied 4 times
    indices = []
    di = 0  # output (depunctured) position

    for _ in range(l1):
        for _rep in range(4):
            for bit in pi1:
                if bit:
                    indices.append(di)
                di += 1

    for _ in range(l2):
        for _rep in range(4):
            for bit in pi2:
                if bit:
                    indices.append(di)
                di += 1

    # Tail: 6 groups × {1,1,0,0} = 24 coded bits, 12 transmitted
    for _ in range(6):
        indices.append(di)
        di += 1
        indices.append(di)
        di += 1
        di += 2  # punctured

    return np.array(indices, dtype=np.int32), di


def _eep_depuncture(soft_bits: np.ndarray, index: np.ndarray, out_len: int) -> np.ndarray:
    """Depuncture subchannel soft bits using precomputed index."""
    out = np.zeros(out_len, dtype=np.float32)
    out[index] = soft_bits[: len(index)]
    return out
