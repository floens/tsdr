"""TETRA burst field extraction.

Extracts data fields from 510-bit (255-symbol) bursts based on burst type.
Offsets follow ETSI EN 300 392-2.
"""

from dataclasses import dataclass

import numpy as np

# Sync burst (510 bits total)
# Tail q11-q22: 0-11, Phase HC: 12-13, Freq F: 14-93, SB1: 94-213,
# Sync train Y: 214-251, BBK: 252-281, SB2: 282-497, Phase HD: 498-499, Tail q1-q10: 500-509
SB_FREQ_OFFSET = 14
SB_FREQ_BITS = 80
SB_BLK1_OFFSET = 94
SB_BLK1_BITS = 120
SB_TRAIN_OFFSET = 214
SB_TRAIN_BITS = 38
SB_BBK_OFFSET = 252
SB_BBK_BITS = 30
SB_BLK2_OFFSET = 282
SB_BLK2_BITS = 216

# Normal downlink burst (510 bits total)
# Tail q11-q22: 0-11, Phase HA: 12-13, BKN1: 14-229, BBK1: 230-243,
# Training N/P: 244-265, BBK2: 266-281, BKN2: 282-497, Phase HB: 498-499, Tail q1-q10: 500-509
NDB_BLK1_OFFSET = 14
NDB_BLK1_BITS = 216
NDB_BBK1_OFFSET = 230
NDB_BBK1_BITS = 14
NDB_TRAIN_OFFSET = 244
NDB_TRAIN_BITS = 22
NDB_BBK2_OFFSET = 266
NDB_BBK2_BITS = 16
NDB_BLK2_OFFSET = 282
NDB_BLK2_BITS = 216


@dataclass(frozen=True)
class SyncBurst:
    """Fields from a synchronization burst."""

    freq_correction: np.ndarray  # 80 soft bits
    sb1: np.ndarray  # 120 soft bits
    bbk: np.ndarray  # 30 soft bits
    sb2: np.ndarray  # 216 soft bits


@dataclass(frozen=True)
class NormalBurst:
    """Fields from a normal downlink burst."""

    bkn1: np.ndarray  # 216 soft bits
    bbk: np.ndarray  # 30 soft bits (14 + 16 combined)
    bkn2: np.ndarray  # 216 soft bits


def extract_sync_burst(soft_bits: np.ndarray) -> SyncBurst:
    """Extract fields from a sync burst's 510 soft bits."""
    return SyncBurst(
        freq_correction=soft_bits[SB_FREQ_OFFSET : SB_FREQ_OFFSET + SB_FREQ_BITS].copy(),
        sb1=soft_bits[SB_BLK1_OFFSET : SB_BLK1_OFFSET + SB_BLK1_BITS].copy(),
        bbk=soft_bits[SB_BBK_OFFSET : SB_BBK_OFFSET + SB_BBK_BITS].copy(),
        sb2=soft_bits[SB_BLK2_OFFSET : SB_BLK2_OFFSET + SB_BLK2_BITS].copy(),
    )


def extract_normal_burst(soft_bits: np.ndarray) -> NormalBurst:
    """Extract fields from a normal downlink burst's 510 soft bits."""
    # Combine the two BBK parts
    bbk = np.concatenate(
        [
            soft_bits[NDB_BBK1_OFFSET : NDB_BBK1_OFFSET + NDB_BBK1_BITS],
            soft_bits[NDB_BBK2_OFFSET : NDB_BBK2_OFFSET + NDB_BBK2_BITS],
        ]
    )
    return NormalBurst(
        bkn1=soft_bits[NDB_BLK1_OFFSET : NDB_BLK1_OFFSET + NDB_BLK1_BITS].copy(),
        bbk=bbk,
        bkn2=soft_bits[NDB_BLK2_OFFSET : NDB_BLK2_OFFSET + NDB_BLK2_BITS].copy(),
    )


def extract_schf(burst: NormalBurst) -> np.ndarray:
    """Extract combined SCH/F (432 soft bits) from a normal-1 burst."""
    return np.concatenate([burst.bkn1, burst.bkn2])
