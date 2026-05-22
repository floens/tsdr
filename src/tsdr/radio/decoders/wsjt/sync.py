"""FT8 / FT4 sync detection and soft-symbol extraction.

The waterfall is stored as a float32 array of dB magnitudes (per-frame
rfft on a Hann-windowed, half-symbol-overlapping STFT). Float arithmetic
preserves the same candidate ranking as the canonical uint8 form — the
score's overall scale factor cancels in pairwise comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numba as nb
import numpy as np

from .tables import (
    FT4_COSTAS_PATTERNS,
    FT4_GRAY_MAP,
    FT4_LENGTH_SYNC,
    FT4_ND,
    FT4_NUM_SYNC,
    FT4_NUM_TONES,
    FT4_SYMBOL_PERIOD,
    FT4_SYNC_OFFSET,
    FT8_COSTAS_PATTERN,
    FT8_GRAY_MAP,
    FT8_LENGTH_SYNC,
    FT8_ND,
    FT8_NUM_SYNC,
    FT8_NUM_TONES,
    FT8_SYMBOL_PERIOD,
    FT8_SYNC_OFFSET,
)

# Bins below this PSD floor (linear power) are clipped during the log10
# conversion so the waterfall stays well-defined when an FFT bin is exactly
# zero (synthetic test signals, zeroed prebuffer regions).
_PSD_FLOOR = 1e-12


@dataclass
class WaterfallParams:
    """Shape parameters for a slot waterfall."""

    sample_rate: int
    symbol_period: float
    time_osr: int
    freq_osr: int
    f_min: float
    f_max: float

    block_size: int = field(init=False)
    subblock_size: int = field(init=False)
    nfft: int = field(init=False)
    min_bin: int = field(init=False)
    max_bin: int = field(init=False)
    num_bins: int = field(init=False)

    def __post_init__(self) -> None:
        self.block_size = int(self.sample_rate * self.symbol_period)
        self.subblock_size = self.block_size // self.time_osr
        self.nfft = self.block_size * self.freq_osr
        self.min_bin = int(self.f_min * self.symbol_period)
        self.max_bin = int(self.f_max * self.symbol_period) + 1
        self.num_bins = self.max_bin - self.min_bin


@dataclass(frozen=True)
class Waterfall:
    """Magnitude-only waterfall.

    Shape: (num_blocks, time_osr, freq_osr, num_bins), dtype=float32, values
    in dB.
    """

    mag: np.ndarray
    params: WaterfallParams


@dataclass(frozen=True)
class Candidate:
    time_offset: int
    time_sub: int
    freq_sub: int
    freq_offset: int
    score: float


@lru_cache(maxsize=8)
def hann_window(nfft: int) -> np.ndarray:
    # Periodic Hann (zero only at index 0) is the WSJT-X STFT convention.
    # np.hanning is the symmetric form (zero at both ends) and would not match.
    return (2.0 / nfft * np.sin(np.pi * np.arange(nfft) / nfft) ** 2).astype(np.float32)


def compute_waterfall(
    real_audio: np.ndarray,
    params: WaterfallParams,
    *,
    window: np.ndarray | None = None,
) -> Waterfall:
    """Run a Hann-windowed STFT on a real-audio buffer and return a magnitude waterfall.

    ``real_audio`` is float32 / float64 mono at ``params.sample_rate`` Hz; we keep
    only bins in ``[f_min, f_max)``. Pass a pre-built ``window`` of length
    ``params.nfft`` to skip the per-call Hann construction when streaming.
    """
    audio = np.asarray(real_audio, dtype=np.float32).ravel()
    block_size = params.block_size
    subblock = params.subblock_size
    nfft = params.nfft
    num_bins = params.num_bins
    min_bin = params.min_bin
    time_osr = params.time_osr
    freq_osr = params.freq_osr

    if window is None:
        window = hann_window(nfft)

    max_blocks = len(audio) // block_size
    if max_blocks < 1:
        empty = np.zeros((0, time_osr, freq_osr, num_bins), dtype=np.float32)
        return Waterfall(mag=empty, params=params)

    mag = np.empty((max_blocks, time_osr, freq_osr, num_bins), dtype=np.float32)
    last_frame = np.zeros(nfft, dtype=np.float32)

    kept_start = min_bin * freq_osr
    kept_end = kept_start + num_bins * freq_osr

    frame_pos = 0
    for block in range(max_blocks):
        for ts in range(time_osr):
            last_frame[: nfft - subblock] = last_frame[subblock:nfft]
            last_frame[nfft - subblock : nfft] = audio[frame_pos : frame_pos + subblock]
            frame_pos += subblock

            spec = np.fft.rfft(last_frame * window, n=nfft)[kept_start:kept_end]
            mag2 = spec.real * spec.real + spec.imag * spec.imag
            # kept bin layout is (num_bins, freq_osr); waterfall stores (freq_osr, num_bins).
            db_kept = (10.0 * np.log10(mag2 + _PSD_FLOOR)).reshape(num_bins, freq_osr)
            mag[block, ts] = db_kept.T
    return Waterfall(mag=mag, params=params)


# ---------------------------------------------------------------------------
# Sync score (numba kernels)
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _ft8_sync_score(
    mag: np.ndarray,
    time_offset: int,
    time_sub: int,
    freq_sub: int,
    freq_offset: int,
    costas: np.ndarray,
) -> float:
    num_blocks = mag.shape[0]
    score = 0.0
    num_avg = 0
    for m in range(FT8_NUM_SYNC):
        for k in range(FT8_LENGTH_SYNC):
            block = FT8_SYNC_OFFSET * m + k
            block_abs = time_offset + block
            if block_abs < 0:
                continue
            if block_abs >= num_blocks:
                break
            sm = costas[k] + freq_offset
            cur = mag[block_abs, time_sub, freq_sub, sm]
            if sm > freq_offset:
                score += cur - mag[block_abs, time_sub, freq_sub, sm - 1]
                num_avg += 1
            if sm < freq_offset + FT8_NUM_TONES - 1:
                score += cur - mag[block_abs, time_sub, freq_sub, sm + 1]
                num_avg += 1
            if k > 0 and block_abs > 0:
                score += cur - mag[block_abs - 1, time_sub, freq_sub, sm]
                num_avg += 1
            if (k + 1) < FT8_LENGTH_SYNC and (block_abs + 1) < num_blocks:
                score += cur - mag[block_abs + 1, time_sub, freq_sub, sm]
                num_avg += 1
    if num_avg > 0:
        score /= num_avg
    return score


@nb.njit(cache=True)
def _ft4_sync_score(
    mag: np.ndarray,
    time_offset: int,
    time_sub: int,
    freq_sub: int,
    freq_offset: int,
    patterns: np.ndarray,
) -> float:
    num_blocks = mag.shape[0]
    score = 0.0
    num_avg = 0
    for m in range(FT4_NUM_SYNC):
        for k in range(FT4_LENGTH_SYNC):
            block = 1 + FT4_SYNC_OFFSET * m + k
            block_abs = time_offset + block
            if block_abs < 0:
                continue
            if block_abs >= num_blocks:
                break
            sm = patterns[m, k] + freq_offset
            cur = mag[block_abs, time_sub, freq_sub, sm]
            if sm > freq_offset:
                score += cur - mag[block_abs, time_sub, freq_sub, sm - 1]
                num_avg += 1
            if sm < freq_offset + FT4_NUM_TONES - 1:
                score += cur - mag[block_abs, time_sub, freq_sub, sm + 1]
                num_avg += 1
            if k > 0 and block_abs > 0:
                score += cur - mag[block_abs - 1, time_sub, freq_sub, sm]
                num_avg += 1
            if (k + 1) < FT4_LENGTH_SYNC and (block_abs + 1) < num_blocks:
                score += cur - mag[block_abs + 1, time_sub, freq_sub, sm]
                num_avg += 1
    if num_avg > 0:
        score /= num_avg
    return score


@nb.njit(cache=True)
def _scan_candidates_ft8(
    mag: np.ndarray,
    costas: np.ndarray,
    min_score: float,
    time_low: int,
    time_high: int,
) -> np.ndarray:
    """Return a (n, 5) array of (time_offset, time_sub, freq_sub, freq_offset, score)."""
    _, time_osr, freq_osr, num_bins = mag.shape
    num_tones = FT8_NUM_TONES
    max_cands = max(1, (time_high - time_low) * time_osr * freq_osr * (num_bins - num_tones + 1))
    out = np.empty((max_cands, 5), dtype=np.float32)
    cnt = 0
    for ts in range(time_osr):
        for fs in range(freq_osr):
            for to in range(time_low, time_high):
                for fo in range(num_bins - num_tones + 1):
                    s = _ft8_sync_score(mag, to, ts, fs, fo, costas)
                    if s >= min_score:
                        out[cnt, 0] = to
                        out[cnt, 1] = ts
                        out[cnt, 2] = fs
                        out[cnt, 3] = fo
                        out[cnt, 4] = s
                        cnt += 1
    return out[:cnt]


@nb.njit(cache=True)
def _scan_candidates_ft4(
    mag: np.ndarray,
    patterns: np.ndarray,
    min_score: float,
    time_low: int,
    time_high: int,
) -> np.ndarray:
    _, time_osr, freq_osr, num_bins = mag.shape
    num_tones = FT4_NUM_TONES
    max_cands = max(1, (time_high - time_low) * time_osr * freq_osr * (num_bins - num_tones + 1))
    out = np.empty((max_cands, 5), dtype=np.float32)
    cnt = 0
    for ts in range(time_osr):
        for fs in range(freq_osr):
            for to in range(time_low, time_high):
                for fo in range(num_bins - num_tones + 1):
                    s = _ft4_sync_score(mag, to, ts, fs, fo, patterns)
                    if s >= min_score:
                        out[cnt, 0] = to
                        out[cnt, 1] = ts
                        out[cnt, 2] = fs
                        out[cnt, 3] = fo
                        out[cnt, 4] = s
                        cnt += 1
    return out[:cnt]


def find_candidates(
    wf: Waterfall,
    *,
    is_ft4: bool,
    num_candidates: int = 120,
    min_score: float = 10.0,
    time_low: int = -10,
    time_high: int | None = None,
) -> list[Candidate]:
    """Return up to ``num_candidates`` highest-scoring sync hits.

    ``time_high`` is the largest start-time offset (in symbol blocks) the scan
    tries. Defaults: 50 for FT4 (~2.5 s past slot start), 20 for FT8 (~3.2 s).
    Both exceed the geometric head slack (slot − frame: ~2.5 s FT4, ~2.4 s FT8)
    to keep late transmissions in the search; the sync-score kernels short-circuit
    once a block exits the slot, so extra slack is cheap.
    """
    if time_high is None:
        time_high = 50 if is_ft4 else 20
    if is_ft4:
        raw = _scan_candidates_ft4(
            wf.mag, FT4_COSTAS_PATTERNS, np.float32(min_score), time_low, time_high
        )
    else:
        raw = _scan_candidates_ft8(
            wf.mag, FT8_COSTAS_PATTERN, np.float32(min_score), time_low, time_high
        )
    n = raw.shape[0]
    if n == 0:
        return []
    # Partial sort: pull the top-k by score in O(n), then order just those k.
    k = min(num_candidates, n)
    scores_neg = -raw[:, 4]
    if k < n:
        topk = np.argpartition(scores_neg, k - 1)[:k]
    else:
        topk = np.arange(n)
    order = topk[np.argsort(scores_neg[topk])]
    return [
        Candidate(
            time_offset=int(raw[i, 0]),
            time_sub=int(raw[i, 1]),
            freq_sub=int(raw[i, 2]),
            freq_offset=int(raw[i, 3]),
            score=float(raw[i, 4]),
        )
        for i in order
    ]


# ---------------------------------------------------------------------------
# Symbol -> LLR extraction
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _ft8_extract_symbol(bins8: np.ndarray, gray_map: np.ndarray, out: np.ndarray, off: int) -> None:
    # 8 scalar locals avoid a per-call np.empty allocation inside the kernel
    # — this runs 58×num_candidates times per slot.
    s0 = bins8[gray_map[0]]
    s1 = bins8[gray_map[1]]
    s2 = bins8[gray_map[2]]
    s3 = bins8[gray_map[3]]
    s4 = bins8[gray_map[4]]
    s5 = bins8[gray_map[5]]
    s6 = bins8[gray_map[6]]
    s7 = bins8[gray_map[7]]
    out[off + 0] = max(s4, s5, s6, s7) - max(s0, s1, s2, s3)
    out[off + 1] = max(s2, s3, s6, s7) - max(s0, s1, s4, s5)
    out[off + 2] = max(s1, s3, s5, s7) - max(s0, s2, s4, s6)


@nb.njit(cache=True)
def _ft4_extract_symbol(bins4: np.ndarray, gray_map: np.ndarray, out: np.ndarray, off: int) -> None:
    s0 = bins4[gray_map[0]]
    s1 = bins4[gray_map[1]]
    s2 = bins4[gray_map[2]]
    s3 = bins4[gray_map[3]]
    out[off + 0] = max(s2, s3) - max(s0, s1)
    out[off + 1] = max(s1, s3) - max(s0, s2)


@nb.njit(cache=True)
def _ft8_extract_llrs(
    mag: np.ndarray,
    cand_to: int,
    cand_ts: int,
    cand_fs: int,
    cand_fo: int,
    gray_map: np.ndarray,
    log174: np.ndarray,
) -> None:
    num_blocks = mag.shape[0]
    for k in range(FT8_ND):
        # 7 sync tones precede block 1's data; +7 more between data blocks 1 and 2.
        sym_idx = k + (FT8_LENGTH_SYNC if k < 29 else 2 * FT8_LENGTH_SYNC)
        block_abs = cand_to + sym_idx
        if 0 <= block_abs < num_blocks:
            bins8 = mag[block_abs, cand_ts, cand_fs, cand_fo : cand_fo + FT8_NUM_TONES]
            _ft8_extract_symbol(bins8, gray_map, log174, 3 * k)


@nb.njit(cache=True)
def _ft4_extract_llrs(
    mag: np.ndarray,
    cand_to: int,
    cand_ts: int,
    cand_fs: int,
    cand_fo: int,
    gray_map: np.ndarray,
    log174: np.ndarray,
) -> None:
    num_blocks = mag.shape[0]
    for k in range(FT4_ND):
        # FT4 frame: R S4 D29 S4 D29 S4 D29 S4 R. Each Costas block is 4 tones; the
        # leading ramp + 1 Costas = 5 pre-tone slots before the first data block,
        # +4 more between each pair of data blocks (9, then 13 cumulative).
        if k < 29:
            sym_idx = k + 1 + FT4_LENGTH_SYNC
        elif k < 58:
            sym_idx = k + 1 + 2 * FT4_LENGTH_SYNC
        else:
            sym_idx = k + 1 + 3 * FT4_LENGTH_SYNC
        block_abs = cand_to + sym_idx
        if 0 <= block_abs < num_blocks:
            bins4 = mag[block_abs, cand_ts, cand_fs, cand_fo : cand_fo + FT4_NUM_TONES]
            _ft4_extract_symbol(bins4, gray_map, log174, 2 * k)


def extract_llrs(wf: Waterfall, cand: Candidate, *, is_ft4: bool) -> np.ndarray:
    log174 = np.zeros(174, dtype=np.float32)
    if is_ft4:
        _ft4_extract_llrs(
            wf.mag,
            cand.time_offset,
            cand.time_sub,
            cand.freq_sub,
            cand.freq_offset,
            FT4_GRAY_MAP,
            log174,
        )
    else:
        _ft8_extract_llrs(
            wf.mag,
            cand.time_offset,
            cand.time_sub,
            cand.freq_sub,
            cand.freq_offset,
            FT8_GRAY_MAP,
            log174,
        )
    _normalize_logl(log174)
    return log174


@nb.njit(cache=True)
def _normalize_logl(log174: np.ndarray) -> None:
    s = 0.0
    s2 = 0.0
    n = log174.shape[0]
    for i in range(n):
        s += log174[i]
        s2 += log174[i] * log174[i]
    inv_n = 1.0 / n
    variance = (s2 - s * s * inv_n) * inv_n
    if variance <= 0.0:
        return
    norm = (24.0 / variance) ** 0.5
    for i in range(n):
        log174[i] *= norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ft8_params(
    sample_rate: int = 12000, f_min: float = 200.0, f_max: float = 3000.0
) -> WaterfallParams:
    return WaterfallParams(
        sample_rate=sample_rate,
        symbol_period=FT8_SYMBOL_PERIOD,
        time_osr=2,
        freq_osr=2,
        f_min=f_min,
        f_max=f_max,
    )


def ft4_params(
    sample_rate: int = 12000, f_min: float = 200.0, f_max: float = 3000.0
) -> WaterfallParams:
    return WaterfallParams(
        sample_rate=sample_rate,
        symbol_period=FT4_SYMBOL_PERIOD,
        time_osr=2,
        freq_osr=2,
        f_min=f_min,
        f_max=f_max,
    )


__all__ = [
    "Candidate",
    "Waterfall",
    "WaterfallParams",
    "compute_waterfall",
    "extract_llrs",
    "find_candidates",
    "ft4_params",
    "ft8_params",
]
