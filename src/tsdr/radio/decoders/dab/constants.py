import numpy as np

# DAB Mode I parameters
T_NULL = 2656  # Null symbol duration (samples)
T_S = 2552  # Total symbol duration (guard + useful)
T_U = 2048  # Useful symbol duration / FFT size
T_G = 504  # Guard interval (cyclic prefix)
N_SYMBOLS = 76  # Symbols per frame (after null)
N_CARRIERS = 1536  # Active subcarriers
T_FRAME = T_NULL + N_SYMBOLS * T_S  # 196608 samples per frame

# FIC structure (Mode I)
N_FIC_SYMBOLS = 3  # Symbols 1-3 carry FIC
N_FIC_BLOCKS = 4  # 9216 bits split into 4 blocks of 2304
FIC_BLOCK_BITS = 2304  # Transmitted bits per FIC block
N_FIBS_PER_BLOCK = 3  # Each block decodes to 3 FIBs
FIB_BITS = 256  # 32 bytes per FIB
FIB_DATA_BYTES = 30  # 30 data bytes + 2 CRC bytes

# Viterbi: convolutional code rate 1/4, K=7
VITERBI_K = 7
VITERBI_GENERATORS = [0o133, 0o171, 0o145, 0o133]  # G1-G4 (ETSI standard, input at MSB)
VITERBI_STATES = 1 << (VITERBI_K - 1)  # 64

# CRC-16-CCITT for FIB
CRC16_POLY = 0x1021
CRC16_INIT = 0xFFFF
CRC16_GOOD = 0x1D0F  # Residual when CRC is correct

# PRBS for energy dispersal: x^9 + x^5 + 1
PRBS_INIT = 0x1FF  # all ones (9 bits)


# Frequency De-interleaving Table
# ETSI EN 300 401 section 14.6
# Permutation: pi(0)=0, pi(n) = (13*pi(n-1)+511) % 2048
# Values are treated as centered carrier indices (subtract 1024).
# Active carriers: -768..+768 excluding 0.


def _compute_freq_deinterleave_table() -> np.ndarray:
    """Compute frequency de-interleaving table for Mode I.

    Maps logical bit position (0..1535) to carrier array index.
    Carrier ordering: -768..-1 -> indices 0..767, +1..+768 -> indices 768..1535.
    """
    perm = [0] * 2048
    for i in range(1, 2048):
        perm[i] = (13 * perm[i - 1] + 511) % 2048

    table = np.zeros(N_CARRIERS, dtype=np.int32)
    j = 0
    for i in range(2048):
        k = perm[i] - 1024  # centered carrier index
        if k == 0 or k < -768 or k > 768:
            continue
        # Map to array index: -768->0, -1->767, +1->768, +768->1535
        if k < 0:
            table[j] = k + 768
        else:
            table[j] = k + 767
        j += 1

    assert j == N_CARRIERS
    return table


FREQ_DEINTERLEAVE_TABLE = _compute_freq_deinterleave_table()

# Precompute FFT bin indices for all 1536 carriers (physical order)
# Carrier array: 0..767 = FFT bins 1280..2047, 768..1535 = FFT bins 1..768
_FFT_BIN_TABLE = np.where(
    np.arange(N_CARRIERS) < 768,
    np.arange(N_CARRIERS) + 1280,
    np.arange(N_CARRIERS) - 767,
).astype(np.int32)


# PRS Phase Reference Table (ETSI EN 300 401 section 14.3.2)

# h0-h3 lookup arrays (32 entries each, periodic)
# fmt: off
_H0 = np.array([0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1,
                 0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1], dtype=np.int8)
_H1 = np.array([0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0,
                 0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0], dtype=np.int8)
_H2 = np.array([0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3,
                 0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3], dtype=np.int8)
_H3 = np.array([0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2,
                 0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2], dtype=np.int8)
# fmt: on
_H_TABLES = [_H0, _H1, _H2, _H3]

# Mode I phase table: (kmin, kmax, i, n) for each carrier group
_MODE_I_TABLE = [
    (-768, -737, 0, 1),
    (-736, -705, 1, 2),
    (-704, -673, 2, 0),
    (-672, -641, 3, 1),
    (-640, -609, 0, 3),
    (-608, -577, 1, 2),
    (-576, -545, 2, 2),
    (-544, -513, 3, 3),
    (-512, -481, 0, 2),
    (-480, -449, 1, 1),
    (-448, -417, 2, 2),
    (-416, -385, 3, 3),
    (-384, -353, 0, 1),
    (-352, -321, 1, 2),
    (-320, -289, 2, 3),
    (-288, -257, 3, 3),
    (-256, -225, 0, 2),
    (-224, -193, 1, 2),
    (-192, -161, 2, 2),
    (-160, -129, 3, 1),
    (-128, -97, 0, 1),
    (-96, -65, 1, 3),
    (-64, -33, 2, 1),
    (-32, -1, 3, 2),
    (1, 32, 0, 3),
    (33, 64, 3, 1),
    (65, 96, 2, 1),
    (97, 128, 1, 1),
    (129, 160, 0, 2),
    (161, 192, 3, 2),
    (193, 224, 2, 1),
    (225, 256, 1, 0),
    (257, 288, 0, 2),
    (289, 320, 3, 2),
    (321, 352, 2, 3),
    (353, 384, 1, 3),
    (385, 416, 0, 0),
    (417, 448, 3, 2),
    (449, 480, 2, 1),
    (481, 512, 1, 3),
    (513, 544, 0, 3),
    (545, 576, 3, 3),
    (577, 608, 2, 3),
    (609, 640, 1, 0),
    (641, 672, 0, 3),
    (673, 704, 3, 0),
    (705, 736, 2, 1),
    (737, 768, 1, 1),
]


def _get_phi(k: int) -> float:
    """Compute phase Phi_k for carrier k (ETSI EN 300 401)."""
    for kmin, kmax, i, n in _MODE_I_TABLE:
        if kmin <= k <= kmax:
            return float(np.pi / 2 * (_H_TABLES[i][k - kmin] + n))
    raise ValueError(f"Carrier {k} not in Mode I table")


def _compute_prs_ref_table() -> np.ndarray:
    """Precompute PRS reference in FFT-bin order (2048 complex64).

    Positive carrier k -> bin k, negative carrier -k -> bin T_U - k.
    DC bin 0 and unused bins stay zero.
    """
    ref = np.zeros(T_U, dtype=np.complex64)
    for k in range(1, N_CARRIERS // 2 + 1):
        phi = _get_phi(k)
        ref[k] = np.exp(1j * phi)
        phi_neg = _get_phi(-k)
        ref[T_U - k] = np.exp(1j * phi_neg)
    return ref


PRS_REF_TABLE = _compute_prs_ref_table()
