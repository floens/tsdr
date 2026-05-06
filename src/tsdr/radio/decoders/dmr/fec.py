"""DMR forward error correction: Hamming(7,4), Golay(20,8).

Reference: ETSI TS 102 361-1
"""

from dataclasses import dataclass

import numpy as np

from tsdr.radio.decoders.dmr.constants import (
    AMBE_RW,
    AMBE_RX,
    AMBE_RY,
    AMBE_RZ,
    CACH_INTERLEAVE,
    VOICE_FRAME_A_END,
    VOICE_FRAME_A_START,
    VOICE_FRAME_B1_END,
    VOICE_FRAME_B1_START,
    VOICE_FRAME_B2_END,
    VOICE_FRAME_B2_START,
    VOICE_FRAME_C_END,
    VOICE_FRAME_C_START,
)


@dataclass(frozen=True)
class CACHInfo:
    """Decoded Common Announcement Channel."""

    at_flag: int
    timeslot: int
    lcss: int


@dataclass(frozen=True)
class SlotTypeInfo:
    """Decoded Slot Type PDU."""

    color_code: int
    data_type: int


def dibits_to_bits(dibits: np.ndarray) -> np.ndarray:
    """Expand dibits to bits (MSB first per ETSI TS 102 361-1 §B.1.1)."""
    bits = np.zeros(len(dibits) * 2, dtype=np.uint8)
    for i in range(len(dibits)):
        bits[2 * i] = (dibits[i] >> 1) & 1
        bits[2 * i + 1] = dibits[i] & 1
    return bits


# Hamming(7,4)
# Parity check matrix H (3×7): syndrome = H × r mod 2
_HAMMING_7_4_H = np.array(
    [
        [1, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)

# Syndrome -> bit position to flip (0xFF = uncorrectable)
_HAMMING_7_4_CORR = [0xFF] * 8
_HAMMING_7_4_CORR[0b101] = 0
_HAMMING_7_4_CORR[0b111] = 1
_HAMMING_7_4_CORR[0b110] = 2
_HAMMING_7_4_CORR[0b011] = 3
_HAMMING_7_4_CORR[0b100] = 4
_HAMMING_7_4_CORR[0b010] = 5
_HAMMING_7_4_CORR[0b001] = 6


def hamming_7_4_decode(bits: np.ndarray) -> bool:
    """Decode Hamming(7,4) codeword in-place. Returns True if valid/corrected."""
    syndrome = 0
    for row in range(3):
        s = 0
        for col in range(7):
            s += int(bits[col]) * int(_HAMMING_7_4_H[row, col])
        syndrome |= (s % 2) << (2 - row)

    if syndrome == 0:
        return True

    corr_pos = _HAMMING_7_4_CORR[syndrome]
    if corr_pos == 0xFF:
        return False

    bits[corr_pos] ^= 1
    return True


# Golay(20,8)
# Parity check matrix H (12×20)
_GOLAY_20_8_H = np.array(
    [
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

# Pre-compute correction table: syndrome -> up to 3 bit positions to flip
# 0xFF = no correction at that slot
_GOLAY_20_8_CORR: list[list[int]] = [[0xFF, 0xFF, 0xFF] for _ in range(4096)]


def _init_golay_20_8() -> None:
    """Build the Golay(20,8) correction lookup table.

    Most errors first, fewest last.
    Later (fewer-error) entries overwrite earlier ones, ensuring
    the minimum-error correction is always selected.
    """
    h = _GOLAY_20_8_H

    def _syndrome_of(*bits: int) -> int:
        s = 0
        for r in range(12):
            val = 0
            for b in bits:
                val += int(h[r, b])
            s |= (val % 2) << (11 - r)
        return s

    # 3 message bit errors (no parity corrections)
    for i1 in range(8):
        for i2 in range(i1 + 1, 8):
            for i3 in range(i2 + 1, 8):
                syn = _syndrome_of(i1, i2, i3)
                _GOLAY_20_8_CORR[syn] = [i1, i2, i3]

            # 2 message bit errors + 0-1 parity bit errors
            syn = _syndrome_of(i1, i2)
            _GOLAY_20_8_CORR[syn] = [i1, i2, 0xFF]
            for ip in range(12):
                syn_p = syn ^ (1 << (11 - ip))
                _GOLAY_20_8_CORR[syn_p] = [i1, i2, 8 + ip]

        # 1 message bit error + 0-2 parity bit errors
        syn = _syndrome_of(i1)
        _GOLAY_20_8_CORR[syn] = [i1, 0xFF, 0xFF]
        for ip1 in range(12):
            syn_p1 = syn ^ (1 << (11 - ip1))
            _GOLAY_20_8_CORR[syn_p1] = [i1, 8 + ip1, 0xFF]
            for ip2 in range(ip1 + 1, 12):
                syn_p2 = syn_p1 ^ (1 << (11 - ip2))
                _GOLAY_20_8_CORR[syn_p2] = [i1, 8 + ip1, 8 + ip2]

    # 0 message bit errors, 1-3 parity bit errors
    for ip1 in range(12):
        syn = 1 << (11 - ip1)
        _GOLAY_20_8_CORR[syn] = [8 + ip1, 0xFF, 0xFF]
        for ip2 in range(ip1 + 1, 12):
            syn2 = syn ^ (1 << (11 - ip2))
            _GOLAY_20_8_CORR[syn2] = [8 + ip1, 8 + ip2, 0xFF]
            for ip3 in range(ip2 + 1, 12):
                syn3 = syn2 ^ (1 << (11 - ip3))
                _GOLAY_20_8_CORR[syn3] = [8 + ip1, 8 + ip2, 8 + ip3]


_init_golay_20_8()


def golay_20_8_decode(bits: np.ndarray) -> bool:
    """Decode Golay(20,8) codeword in-place. Returns True if valid/corrected."""
    syndrome = 0
    for row in range(12):
        s = 0
        for col in range(20):
            s += int(bits[col]) * int(_GOLAY_20_8_H[row, col])
        syndrome |= (s % 2) << (11 - row)

    if syndrome == 0:
        return True

    corr = _GOLAY_20_8_CORR[syndrome]
    if corr[0] == 0xFF:
        return False

    for pos in corr:
        if pos == 0xFF:
            break
        bits[pos] ^= 1

    return True


# CACH decoding


def decode_cach(dibits: np.ndarray) -> CACHInfo | None:
    """Decode 12-dibit CACH field. Returns None on uncorrectable error."""
    bits = dibits_to_bits(dibits)

    # De-interleave
    deinterleaved = np.zeros(24, dtype=np.uint8)
    for i in range(24):
        deinterleaved[CACH_INTERLEAVE[i]] = bits[i]

    # Hamming(7,4) decode three codewords: bits 0-6, 7-13, 14-20
    # (bits 21-23 are the TACT parity - not Hamming-protected)
    for block in range(3):
        if not hamming_7_4_decode(deinterleaved[block * 7 : block * 7 + 7]):
            return None

    at_flag = int(deinterleaved[0])
    timeslot = int(deinterleaved[1])
    lcss = (int(deinterleaved[2]) << 1) | int(deinterleaved[3])
    return CACHInfo(at_flag=at_flag, timeslot=timeslot, lcss=lcss)


# Slot Type PDU decoding


def decode_slot_type(st1_dibits: np.ndarray, st2_dibits: np.ndarray) -> SlotTypeInfo | None:
    """Decode Slot Type from two 5-dibit halves. Returns None on uncorrectable error."""
    bits = np.concatenate([dibits_to_bits(st1_dibits), dibits_to_bits(st2_dibits)])

    if not golay_20_8_decode(bits):
        return None

    color_code = (bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3]
    data_type = (bits[4] << 3) | (bits[5] << 2) | (bits[6] << 1) | bits[7]
    return SlotTypeInfo(color_code=int(color_code), data_type=int(data_type))


# Voice frame extraction


def _deinterleave_ambe_frame(dibits: np.ndarray) -> bytes:
    """Deinterleave 36 dibits into a 9-byte AMBE+2 frame.

    Uses the rW/rX/rY/rZ tables to scatter dibits into the
    ambe_fr[4][24] codeword matrix, then packs MSB-first into 9 bytes
    matching AmbePlus2Decoder.decode_frame() input format:
    C0(24 bits) + C1(23 bits) + C2(11 bits) + C3(14 bits) = 72 bits.
    """
    ambe_fr = np.zeros((4, 24), dtype=np.uint8)
    for i in range(36):
        ambe_fr[AMBE_RW[i]][AMBE_RX[i]] = (dibits[i] >> 1) & 1
        ambe_fr[AMBE_RY[i]][AMBE_RZ[i]] = dibits[i] & 1

    # Pack codeword matrix to 72 bits MSB-first (matching decode_frame input).
    # Row 0 = C0 (24 bits), row 1 = C1 (23 bits), row 2 = C2 (11 bits),
    # row 3 = C3 (14 bits). Within each row, bit 23 is MSB (transmitted first).
    bits = np.zeros(72, dtype=np.uint8)
    bits[:24] = ambe_fr[0, :24][::-1]
    bits[24:47] = ambe_fr[1, :23][::-1]
    bits[47:58] = ambe_fr[2, :11][::-1]
    bits[58:72] = ambe_fr[3, :14][::-1]
    return bytes(np.packbits(bits))


def extract_voice_frames(burst: np.ndarray) -> list[bytes]:
    """Extract 3 AMBE+2 frames from a 144-dibit voice burst.

    Returns list of 3 nine-byte frames ready for AmbePlus2Decoder.decode_frame().

    Frame layout:
      Frame A: dibits 12-47  (contiguous, 36 dibits)
      Frame B: dibits 48-65 + 90-107  (split around sync/EMB, 18+18 dibits)
      Frame C: dibits 108-143  (contiguous, 36 dibits)
    """
    frame_a_dibits = burst[VOICE_FRAME_A_START:VOICE_FRAME_A_END]
    frame_b_dibits = np.concatenate(
        [
            burst[VOICE_FRAME_B1_START:VOICE_FRAME_B1_END],
            burst[VOICE_FRAME_B2_START:VOICE_FRAME_B2_END],
        ]
    )
    frame_c_dibits = burst[VOICE_FRAME_C_START:VOICE_FRAME_C_END]

    return [
        _deinterleave_ambe_frame(frame_a_dibits),
        _deinterleave_ambe_frame(frame_b_dibits),
        _deinterleave_ambe_frame(frame_c_dibits),
    ]
