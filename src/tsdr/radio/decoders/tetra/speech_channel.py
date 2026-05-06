"""TETRA speech channel decoding: type-5 -> two ACELP codec frames.

Speech uses a different channel coding from signaling:
- Rate 1/3 mother code (3 generators, K=5) instead of rate 1/4
- Matrix interleaving (24×18) instead of permutation interleaving
- Three sensitivity classes with separate puncturing and CRC-based BFI
- Two 30ms ACELP frames per full slot (432 bits)

Reference: ETSI EN 300 395-2 V1.3.1.
"""

import numpy as np

from tsdr.radio.decoders.tetra.scramble import descramble_soft, scramble_hard
from tsdr.radio.dsp.viterbi import ViterbiDecoder

# Constants from ETSI EN 300 395-2

SPEECH_K = 5
# Rate 1/3 generators (different from signaling rate 1/4)
SPEECH_G1 = 0x1F  # 1 + D + D2 + D3 + D4
SPEECH_G2 = 0x1B  # 1 + D + D3 + D4
SPEECH_G3 = 0x15  # 1 + D2 + D4
SPEECH_GENERATORS = [SPEECH_G1, SPEECH_G2, SPEECH_G3]

# Class sizes for two-frame (60ms) slot
N0_2 = 102  # Class 0: unprotected (2 × 51)
N1_2 = 112  # Class 1: rate 8/12 protected (2 × 56)
N2_2 = 60  # Class 2: rate 8/18 protected (2 × 30)
N1_2_CODED = 168  # Class 1 after coding
N2_2_CODED = 162  # Class 2 after coding (includes 8-bit CRC + tail)
# N0_2 + N1_2_CODED + N2_2_CODED = 432

# Single-frame class sizes
N0 = 51
N1 = 56
N2 = 30

# Matrix interleave dimensions
LINES = 24
COLUMNS = 18
# LINES * COLUMNS = 432

PUNCT_PERIOD = 8

# fmt: off
# Puncturing matrix for Class 1 (rate 8/12): 3 generators × period 8
# Row 0 = G1, Row 1 = G2, Row 2 = G3
# 1 = transmitted, 0 = punctured
A1 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

# Puncturing matrix for Class 2 (rate 8/18): 3 generators × period 8
A2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0],
], dtype=np.uint8)

# Bit reordering tables: position (1-based) in 137-bit ACELP frame
# From ETSI EN 300 395-2 Table 4
TAB0 = np.array([
    35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 56, 61, 62, 63, 64, 65, 66,
    67, 68, 69, 70, 74, 75, 83, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 101,
    102, 110, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 128, 129, 137,
], dtype=np.int32)

TAB1 = np.array([
    58, 85, 112, 54, 81, 108, 135, 50, 77, 104, 131, 45, 72, 99, 126, 55, 82,
    109, 136, 5, 13, 34, 8, 16, 17, 22, 23, 24, 25, 26, 6, 14, 7, 15, 60, 87,
    114, 46, 73, 100, 127, 44, 71, 98, 125, 33, 49, 76, 103, 130, 59, 86, 113,
    57, 84, 111,
], dtype=np.int32)

TAB2 = np.array([
    18, 19, 20, 21, 31, 32, 53, 80, 107, 134, 1, 2, 3, 4, 9, 10, 11, 12, 27,
    28, 29, 30, 52, 79, 106, 133, 51, 78, 105, 132,
], dtype=np.int32)

# CRC tables for BFI: each row lists the 1-based bit positions in class 2
# that contribute to that CRC bit. 8 CRC bits total.
_CRC_TABS = [
    [1, 5, 8, 9, 13, 15, 16, 17, 19, 21, 22, 24, 25, 31, 32, 35, 36, 38, 40,
     43, 44, 45, 48, 49, 50, 51, 53, 54, 56],
    [2, 6, 9, 10, 14, 16, 17, 18, 20, 22, 23, 25, 26, 32, 33, 36, 37, 39, 41,
     44, 45, 46, 49, 50, 51, 52, 54, 55, 57],
    [3, 7, 10, 11, 15, 17, 18, 19, 21, 23, 24, 26, 27, 33, 34, 37, 38, 40, 42,
     45, 46, 47, 50, 51, 52, 53, 55, 56, 58],
    [1, 4, 5, 9, 11, 12, 13, 15, 17, 18, 20, 21, 27, 28, 31, 32, 34, 36, 39,
     40, 41, 44, 45, 46, 47, 49, 50, 52, 57, 59],
    [2, 5, 6, 10, 12, 13, 14, 16, 18, 19, 21, 22, 28, 29, 32, 33, 35, 37, 40,
     41, 42, 45, 46, 47, 48, 50, 51, 53, 58, 60],
    [3, 6, 7, 11, 13, 14, 15, 17, 19, 20, 22, 23, 29, 30, 33, 34, 36, 38, 41,
     42, 43, 46, 47, 48, 49, 51, 52, 54, 59],
    [4, 7, 8, 12, 14, 15, 16, 18, 20, 21, 23, 24, 30, 31, 34, 35, 37, 39, 42,
     43, 44, 47, 48, 49, 50, 52, 53, 55, 60],
    [1, 2, 3, 4, 8, 13, 14, 16, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 32,
     33, 34, 36, 37, 40, 41, 42, 44, 48, 50, 53, 56, 57, 58, 59, 60],
]
# fmt: on

_speech_viterbi: ViterbiDecoder | None = None


def _get_speech_viterbi() -> ViterbiDecoder:
    global _speech_viterbi
    if _speech_viterbi is None:
        _speech_viterbi = ViterbiDecoder(SPEECH_K, SPEECH_GENERATORS)
    return _speech_viterbi


# Matrix interleaving (24×18)


def deinterleave_speech(soft_bits: np.ndarray) -> np.ndarray:
    """Deinterleave 432 speech soft bits using 24×18 matrix transpose."""
    # output[line * COLUMNS + col] = input[col * LINES + line]
    out = np.empty(432, dtype=soft_bits.dtype)
    for col in range(COLUMNS):
        for line in range(LINES):
            out[line * COLUMNS + col] = soft_bits[col * LINES + line]
    return out


def interleave_speech(soft_bits: np.ndarray) -> np.ndarray:
    """Interleave 432 speech bits using 24×18 matrix transpose (inverse)."""
    out = np.empty(432, dtype=soft_bits.dtype)
    for col in range(COLUMNS):
        for line in range(LINES):
            out[col * LINES + line] = soft_bits[line * COLUMNS + col]
    return out


# Puncturing


def _count_transmitted(punct_matrix: np.ndarray) -> int:
    """Count transmitted bits per period for a puncturing matrix."""
    return int(punct_matrix.sum())


def depuncture(coded: np.ndarray, punct_matrix: np.ndarray) -> np.ndarray:
    """Depuncture coded soft bits, inserting 0.0 erasures at punctured positions.

    punct_matrix is (n_gen, period); 1 = transmitted, 0 = punctured.
    """
    n_gen = punct_matrix.shape[0]
    period = punct_matrix.shape[1]
    transmitted_per_period = _count_transmitted(punct_matrix)
    n_symbols = len(coded) * period // transmitted_per_period

    mother = np.zeros(n_symbols * n_gen, dtype=np.float32)
    coded_idx = 0
    for sym in range(n_symbols):
        p = sym % period
        for g in range(n_gen):
            if punct_matrix[g, p]:
                mother[sym * n_gen + g] = coded[coded_idx]
                coded_idx += 1
    return mother


def puncture(mother: np.ndarray, punct_matrix: np.ndarray) -> np.ndarray:
    """Puncture mother code using puncturing matrix. Inverse of depuncture."""
    n_gen = punct_matrix.shape[0]
    period = punct_matrix.shape[1]
    transmitted_per_period = _count_transmitted(punct_matrix)
    n_symbols = len(mother) // n_gen
    coded_len = n_symbols * transmitted_per_period // period

    coded = np.empty(coded_len, dtype=mother.dtype)
    coded_idx = 0
    for sym in range(n_symbols):
        p = sym % period
        for g in range(n_gen):
            if punct_matrix[g, p]:
                coded[coded_idx] = mother[sym * n_gen + g]
                coded_idx += 1
    return coded


# Convolutional encoding (rate 1/3, for testing)


def _conv_encode_speech(bits: np.ndarray) -> np.ndarray:
    """Convolutional encode with speech rate-1/3 (K=5, G1/G2/G3)."""
    state = 0
    out = []
    for bit in bits:
        reg = (int(bit) << (SPEECH_K - 1)) | state
        state = reg >> 1
        for gen in SPEECH_GENERATORS:
            out.append(bin(reg & gen).count("1") % 2)
    return np.array(out, dtype=np.uint8)


# CRC (BFI)


def check_speech_crc(class2_decoded: np.ndarray) -> bool:
    """Check 8-bit CRC on decoded class 2 bits.

    CRC bits sit at positions [N2_2 .. N2_2+7]; each is the XOR of the
    data-bit positions listed in `_CRC_TABS`.
    """
    for crc_idx, tab in enumerate(_CRC_TABS):
        parity = 0
        for pos in tab:
            parity ^= int(class2_decoded[pos - 1])  # 1-based positions
        crc_bit = int(class2_decoded[N2_2 + crc_idx])
        if parity != crc_bit:
            return False
    return True


# Bit reordering


def reorder_to_codec(
    class0: np.ndarray,
    class1: np.ndarray,
    class2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder decoded class bits into two 137-bit ACELP codec frames.

    Each class interleaves the two frames as (frame0_bit, frame1_bit) pairs;
    TAB0/TAB1/TAB2 give the 1-based position in the 137-bit codec frame.
    """
    frame1 = np.zeros(137, dtype=np.uint8)
    frame2 = np.zeros(137, dtype=np.uint8)

    for i in range(N0):
        frame1[TAB0[i] - 1] = class0[2 * i]
        frame2[TAB0[i] - 1] = class0[2 * i + 1]

    for i in range(N1):
        frame1[TAB1[i] - 1] = class1[2 * i]
        frame2[TAB1[i] - 1] = class1[2 * i + 1]

    for i in range(N2):
        frame1[TAB2[i] - 1] = class2[2 * i]
        frame2[TAB2[i] - 1] = class2[2 * i + 1]

    return frame1, frame2


def codec_to_classes(
    frame1: np.ndarray,
    frame2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of reorder_to_codec: two 137-bit frames -> class0/1/2 bits."""
    class0 = np.empty(N0_2, dtype=np.uint8)
    class1 = np.empty(N1_2, dtype=np.uint8)
    class2 = np.empty(N2_2, dtype=np.uint8)

    for i in range(N0):
        class0[2 * i] = frame1[TAB0[i] - 1]
        class0[2 * i + 1] = frame2[TAB0[i] - 1]

    for i in range(N1):
        class1[2 * i] = frame1[TAB1[i] - 1]
        class1[2 * i + 1] = frame2[TAB1[i] - 1]

    for i in range(N2):
        class2[2 * i] = frame1[TAB2[i] - 1]
        class2[2 * i + 1] = frame2[TAB2[i] - 1]

    return class0, class1, class2


# Bit packing


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack bit array into bytes (MSB first), padding last byte with zeros."""
    padded = np.zeros(((len(bits) + 7) // 8) * 8, dtype=np.uint8)
    padded[: len(bits)] = bits
    return np.packbits(padded).tobytes()


# CRC generation (for encoding/testing)


def _generate_speech_crc(class2_data: np.ndarray) -> np.ndarray:
    """Generate 8 CRC bits for class 2 data (60 bits -> 8 CRC bits)."""
    crc_bits = np.zeros(8, dtype=np.uint8)
    for crc_idx, tab in enumerate(_CRC_TABS):
        parity = 0
        for pos in tab:
            parity ^= int(class2_data[pos - 1])
        crc_bits[crc_idx] = parity
    return crc_bits


# Forward chain (encoder, for testing)


def encode_speech(
    frame1: np.ndarray,
    frame2: np.ndarray,
    scramble_init: int,
) -> np.ndarray:
    """Encode two 137-bit ACELP frames into 432 type-5 soft bits."""
    class0, class1, class2 = codec_to_classes(frame1, frame2)

    # Class 1 has no tail bits; the Viterbi decoder uses block termination.
    coded1 = puncture(_conv_encode_speech(class1), A1)

    class2_full = np.concatenate(
        [class2, _generate_speech_crc(class2), np.zeros(SPEECH_K - 1, dtype=np.uint8)]
    )
    coded2 = puncture(_conv_encode_speech(class2_full), A2)

    combined = np.concatenate([class0, coded1, coded2])
    return scramble_hard(interleave_speech(combined), scramble_init)


# Decode chain


def decode_speech(
    type5_soft: np.ndarray,
    scramble_init: int,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Decode 432 type-5 soft bits into two 137-bit ACELP frames.

    Returns `(frame1, frame2, bfi1, bfi2)`. `bfi` (Bad Frame Indicator) is True
    when the class-2 CRC fails; both frames share that CRC so `bfi1 == bfi2`.
    """
    type4 = descramble_soft(type5_soft, scramble_init)
    type3 = deinterleave_speech(type4)

    class0_soft = type3[:N0_2]
    class1_coded = type3[N0_2 : N0_2 + N1_2_CODED]
    class2_coded = type3[N0_2 + N1_2_CODED :]

    # Class 0 is unprotected, hard-decision direct from soft bits.
    class0 = (class0_soft > 0).astype(np.uint8)

    dec = _get_speech_viterbi()
    class1 = dec.decode(depuncture(class1_coded, A1))[:N1_2]

    # class2_decoded layout: N2_2 data + 8 CRC + (K-1) tail.
    class2_decoded = dec.decode(depuncture(class2_coded, A2))
    bfi = not check_speech_crc(class2_decoded)
    class2 = class2_decoded[:N2_2]

    frame1, frame2 = reorder_to_codec(class0, class1, class2)
    return frame1, frame2, bfi, bfi
