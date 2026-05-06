"""TETRA burst synchronization via training sequence correlation.

Finds TETRA training sequences in the demodulated symbol stream, locks to
slot boundaries, and identifies burst types (sync vs normal).
"""

from enum import Enum

import numpy as np

from tsdr.radio.decoders.tetra.demod import extract_soft_bits

SLOT_SYMBOLS = 255

# Training sequence offsets in bits (from burst start)
SYNC_TRAIN_BIT_OFFSET = 214  # y_bits
NORM_TRAIN_BIT_OFFSET = 244  # n_bits or p_bits

# Convert to symbol offsets (2 bits per symbol)
SYNC_TRAIN_SYM_OFFSET = SYNC_TRAIN_BIT_OFFSET // 2  # 107
NORM_TRAIN_SYM_OFFSET = NORM_TRAIN_BIT_OFFSET // 2  # 122

# Training sequences (raw bit arrays per ETSI EN 300 392-2)
# fmt: off
Y_BITS = np.array([
    1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
], dtype=np.uint8)
N_BITS = np.array(
    [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    dtype=np.uint8,
)
P_BITS = np.array(
    [0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
    dtype=np.uint8,
)
# fmt: on

# Phase change per dibit (in units of π/4)
_BITS2PHASE = np.array([1, -1, 3, -3], dtype=np.float64)


def bits_to_diff_symbols(bits: np.ndarray) -> np.ndarray:
    """Convert training sequence bits to expected differential complex symbols.

    Dibit packing: sym_in = bit[2k] | (bit[2k+1] << 1).
    """
    sym_in = bits[0::2] | (bits[1::2] << 1)
    phases = _BITS2PHASE[sym_in] * (np.pi / 4)
    result: np.ndarray = np.exp(1j * phases)
    return result


# Precomputed reference differential symbols for each training sequence
Y_REF = bits_to_diff_symbols(Y_BITS)  # 19 symbols
N_REF = bits_to_diff_symbols(N_BITS)  # 11 symbols
P_REF = bits_to_diff_symbols(P_BITS)  # 11 symbols


class SyncState(Enum):
    UNLOCKED = 0
    LOCKED = 1


class BurstResult:
    """Result of processing a single slot."""

    __slots__ = ("burst_type", "soft_bits", "diff_symbols", "correlation")

    def __init__(
        self, burst_type: str, soft_bits: np.ndarray, diff_symbols: np.ndarray, correlation: float
    ):
        self.burst_type = burst_type  # "sync", "normal_1", "normal_2", "unknown"
        self.soft_bits = soft_bits  # 510 soft bits
        self.diff_symbols = diff_symbols  # 255 raw complex differential symbols
        self.correlation = correlation


class SyncDetector:
    """Detects TETRA training sequences and tracks slot boundaries."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.state = SyncState.UNLOCKED
        self._sym_buf = np.array([], dtype=np.complex64)
        self._slot_start = 0
        self._miss_count = 0
        self._max_misses = 10

    def process(self, symbols: np.ndarray) -> list[BurstResult]:
        """Process new symbols, return detected bursts."""
        self._sym_buf = np.concatenate([self._sym_buf, symbols])
        results = []

        while True:
            if self.state == SyncState.UNLOCKED:
                if not self._try_acquire():
                    break
            else:
                burst = self._try_extract_slot()
                if burst is None:
                    break
                results.append(burst)

        # Trim consumed symbols
        if self._slot_start > 0:
            self._sym_buf = self._sym_buf[self._slot_start :]
            self._slot_start = 0

        return results

    def _try_acquire(self) -> bool:
        """Sliding correlation to find sync training sequence."""
        min_len = SLOT_SYMBOLS + 1
        if len(self._sym_buf) < min_len:
            return False

        diff = self._sym_buf[1:] * np.conj(self._sym_buf[:-1])
        # Normalize to unit magnitude for robust correlation
        diff_norm = diff / (np.abs(diff) + 1e-10)

        # Sliding correlation with sync training (y_bits)
        corr = np.correlate(diff_norm, Y_REF, mode="valid")
        peak_idx = int(np.argmax(np.abs(corr)))
        peak_val = np.abs(corr[peak_idx]) / len(Y_REF)

        if peak_val > self.threshold:
            # Sync training found at diff index peak_idx
            # Training starts at symbol offset SYNC_TRAIN_SYM_OFFSET in the burst
            burst_start = peak_idx - SYNC_TRAIN_SYM_OFFSET
            self._slot_start = max(0, burst_start)
            self.state = SyncState.LOCKED
            self._miss_count = 0
            return True

        # Discard symbols we've scanned (keep overlap for next search)
        discard = max(0, len(self._sym_buf) - SLOT_SYMBOLS)
        self._sym_buf = self._sym_buf[discard:]
        return False

    def _try_extract_slot(self) -> BurstResult | None:
        """Extract next slot-aligned burst."""
        # Need SLOT_SYMBOLS + 1 for diff demod (extra symbol as reference)
        end = self._slot_start + SLOT_SYMBOLS + 1
        if end > len(self._sym_buf):
            return None

        slot_syms = self._sym_buf[self._slot_start : end]
        diff = slot_syms[1:] * np.conj(slot_syms[:-1])
        diff_norm = diff / (np.abs(diff) + 1e-10)

        # Identify burst type by correlating at expected training offsets
        sync_c = np.abs(
            np.sum(
                diff_norm[SYNC_TRAIN_SYM_OFFSET : SYNC_TRAIN_SYM_OFFSET + len(Y_REF)]
                * np.conj(Y_REF)
            )
        )
        sync_norm = sync_c / len(Y_REF)

        n_c = np.abs(
            np.sum(
                diff_norm[NORM_TRAIN_SYM_OFFSET : NORM_TRAIN_SYM_OFFSET + len(N_REF)]
                * np.conj(N_REF)
            )
        )
        n_norm = n_c / len(N_REF)

        p_c = np.abs(
            np.sum(
                diff_norm[NORM_TRAIN_SYM_OFFSET : NORM_TRAIN_SYM_OFFSET + len(P_REF)]
                * np.conj(P_REF)
            )
        )
        p_norm = p_c / len(P_REF)

        best_corr = max(sync_norm, n_norm, p_norm)
        if sync_norm >= n_norm and sync_norm >= p_norm:
            burst_type = "sync"
        elif n_norm >= p_norm:
            burst_type = "normal_1"
        else:
            burst_type = "normal_2"

        if best_corr < self.threshold:
            burst_type = "unknown"
            self._miss_count += 1
            if self._miss_count >= self._max_misses:
                self.state = SyncState.UNLOCKED
                # Drop stale buffer so acquisition starts fresh
                self._sym_buf = np.array([], dtype=np.complex64)
                self._slot_start = 0
        else:
            self._miss_count = 0

        soft_bits = extract_soft_bits(diff)
        self._slot_start += SLOT_SYMBOLS
        return BurstResult(
            burst_type=burst_type,
            soft_bits=soft_bits,
            diff_symbols=diff.copy(),
            correlation=best_corr,
        )

    def reset(self) -> None:
        self.state = SyncState.UNLOCKED
        self._sym_buf = np.array([], dtype=np.complex64)
        self._slot_start = 0
        self._miss_count = 0
