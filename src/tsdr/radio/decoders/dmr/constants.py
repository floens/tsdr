"""DMR protocol constants.

ETSI TS 102 361-1: physical layer, burst structure, sync patterns.
"""

from enum import IntEnum, StrEnum

# Physical layer

SYMBOL_RATE = 4800  # symbols/second
CHANNEL_BANDWIDTH = 12_500.0  # Hz

# 4FSK deviation levels (Hz) -> dibit mapping
# +1944 -> dibit 01  (+3)
#  +648 -> dibit 00  (+1)
#  -648 -> dibit 10  (-1)
# -1944 -> dibit 11  (-3)
DEVIATION = 1944.0  # max deviation in Hz

# Burst structure (in dibits)
# A DMR burst is 144 dibits = 288 bits in a 30ms timeslot

BURST_DIBITS = 144
CACH_DIBITS = 12  # Common Announcement Channel
SYNC_DIBITS = 24  # Mid-burst sync pattern
SLOT_TYPE_DIBITS = 10  # 5 dibits each side of sync
EMB_DIBITS = 8  # Embedded signalling (4 each side of sync, voice bursts)
DATA_PART_DIBITS = 49  # Data payload each side
VOICE_PART_DIBITS = 54  # Voice payload each side (3 × 36-dibit AMBE frames)
VOCODER_FRAME_DIBITS = 36  # Single AMBE voice frame (72 bits)

# Sync patterns (24 dibits)
# Dibit encoding: 1 = +648 Hz, 3 = -648 Hz
# Hex representations from ETSI TS 102 361-1

SYNC_BS_DATA = bytes([3, 1, 3, 3, 3, 3, 1, 1, 1, 3, 3, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 1])
SYNC_BS_VOICE = bytes([1, 3, 1, 1, 1, 1, 3, 3, 3, 1, 1, 3, 3, 1, 3, 3, 1, 3, 1, 1, 3, 3, 1, 3])
SYNC_MS_DATA = bytes([3, 1, 1, 1, 3, 1, 1, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 3])
SYNC_MS_VOICE = bytes([1, 3, 3, 3, 1, 3, 3, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 3, 3, 1, 3, 3, 3, 1])


class SyncType(StrEnum):
    """DMR sync pattern types."""

    BS_DATA = "BS_DATA"
    BS_VOICE = "BS_VOICE"
    MS_DATA = "MS_DATA"
    MS_VOICE = "MS_VOICE"


SYNC_PATTERNS: dict[SyncType, bytes] = {
    SyncType.BS_DATA: SYNC_BS_DATA,
    SyncType.BS_VOICE: SYNC_BS_VOICE,
    SyncType.MS_DATA: SYNC_MS_DATA,
    SyncType.MS_VOICE: SYNC_MS_VOICE,
}

# Max dibit errors allowed when matching sync patterns
SYNC_MAX_ERRORS = 2


class DecoderState(StrEnum):
    """Burst collection state machine states."""

    SEARCHING = "searching"
    COLLECTING = "collecting"
    LOCKED = "locked"


# Burst field offsets (in dibits)

CACH_START = 0
DATA1_START = 12
ST1_START = 61  # Slot Type first half (5 dibits)
SYNC_START = 66
ST2_START = 90  # Slot Type second half (5 dibits)
DATA2_START = 95

# Number of dibits before sync (first half of burst)
FIRST_HALF_DIBITS = SYNC_START  # 66
# Number of dibits after sync (second half of burst)
SECOND_HALF_DIBITS = BURST_DIBITS - SYNC_START - SYNC_DIBITS  # 54

# CACH de-interleave table (ETSI TS 102 361-1 §B.2.1)
# Maps transmitted bit position -> de-interleaved bit position
# fmt: off
CACH_INTERLEAVE = [
    0, 7, 8, 9, 1, 10, 11, 12, 2, 13, 14, 15,
    3, 16, 4, 17, 18, 19, 5, 20, 21, 22, 6, 23,
]
# fmt: on


# Voice burst layout (ETSI TS 102 361-1)
# A voice burst carries 3 AMBE+2 frames. Frame B is split around the
# sync/EMB region; frames A and C are contiguous.
#
# Burst A (1st in superframe): dibits 66-89 contain VOICE sync pattern.
# Bursts B-E (continuation):  dibits 66-89 contain EMB + Embedded Signaling
#                              -- no sync pattern, decoder must trust timing.

# fmt: off
VOICE_FRAME_A_START  = 12   # Full AMBE frame (36 dibits, 72 bits)
VOICE_FRAME_A_END    = 48
VOICE_FRAME_B1_START = 48   # Split frame first half (18 dibits)
VOICE_FRAME_B1_END   = 66
VOICE_FRAME_B2_START = 90   # Split frame second half (18 dibits)
VOICE_FRAME_B2_END   = 108
VOICE_FRAME_C_START  = 108  # Full AMBE frame (36 dibits, 72 bits)
VOICE_FRAME_C_END    = 144

EMB1_START           = 66   # EMB part 1 (4 dibits, bursts B-E)
EMB1_END             = 70
ES_START             = 70   # Embedded Signaling (16 dibits)
ES_END               = 86
EMB2_START           = 86   # EMB part 2 (4 dibits, bursts B-E)
EMB2_END             = 90

SUPERFRAME_LEN       = 6    # Bursts per voice superframe
# fmt: on

VOICE_SYNC_TYPES = frozenset({SyncType.BS_VOICE, SyncType.MS_VOICE})

# AMBE+2 deinterleave tables.
# Each of the 36 dibits in a frame maps to two bits in the ambe_fr[4][24]
# codeword matrix: bit 1 -> ambe_fr[rW[i]][rX[i]], bit 0 -> ambe_fr[rY[i]][rZ[i]].
# fmt: off
AMBE_RW = (
    0, 1, 0, 1, 0, 1,  0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1,  0, 1, 0, 1, 0, 2,
    0, 2, 0, 2, 0, 2,  0, 2, 0, 2, 0, 2,
)
AMBE_RX = (
    23, 10, 22, 9, 21, 8,  20, 7, 19, 6, 18, 5,
    17,  4, 16, 3, 15, 2,  14, 1, 13, 0, 12, 10,
    11,  9, 10, 8,  9, 7,   8, 6,  7, 5,  6,  4,
)
AMBE_RY = (
    0, 2, 0, 2, 0, 2,  0, 2, 0, 3, 0, 3,
    1, 3, 1, 3, 1, 3,  1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3,  1, 3, 1, 3, 1, 3,
)
AMBE_RZ = (
     5, 3,  4, 2,  3, 1,   2, 0,  1, 13, 0, 12,
    22, 11, 21, 10, 20, 9, 19, 8, 18,  7, 17, 6,
    16,  5, 15,  4, 14, 3, 13, 2, 12,  1, 11, 0,
)
# fmt: on


class DataType(IntEnum):
    """DMR data types from Slot Type PDU (ETSI TS 102 361-1 §7.1.2)."""

    PI_HEADER = 0
    VOICE_LC_HEADER = 1
    TERMINATOR_WITH_LC = 2
    CSBK = 3
    MBC_HEADER = 4
    MBC_CONTINUATION = 5
    DATA_HEADER = 6
    RATE_12_DATA = 7
    RATE_34_DATA = 8
    IDLE = 9
    RATE_1_DATA = 10
    UNIFIED_SINGLE_BLOCK = 11
