"""Incremental tests for the DMR decoder.

Each test validates a stage of the DSP pipeline using a real DMR sample
from PI1UTR (438.350 MHz, BrandMeister 204300).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import find_peaks

from tsdr.core.sdr.io import load_iq as _load_iq
from tsdr.radio.decoders.dmr.constants import (
    DEVIATION,
    SYMBOL_RATE,
    SYNC_DIBITS,
    SYNC_MAX_ERRORS,
    SYNC_PATTERNS,
    DataType,
    SyncType,
)
from tsdr.radio.decoders.dmr.decoder import DMRDecoder, _match_sync
from tsdr.radio.decoders.dmr.fec import (
    golay_20_8_decode,
    hamming_7_4_decode,
)
from tsdr.radio.dsp import firwin, lfilter

SAMPLE_FILE = (
    Path(__file__).resolve().parents[2]
    / "samples"
    / "freq=438.35M_sr=250k_dur=9s_gain=24_20260412T1050.cu8.zst"
)
SAMPLE_RATE = 250_000


@pytest.fixture(scope="module")
def dmr_iq():
    """Module-scoped IQ data."""
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample file not found: {SAMPLE_FILE}")
    return _load_iq(SAMPLE_FILE)


@pytest.fixture(scope="module")
def dmr_decoder(dmr_iq):
    """Module-scoped decoder that has processed the full IQ file."""
    decoder = DMRDecoder(sample_rate=SAMPLE_RATE)
    for start in range(0, len(dmr_iq), 32768):
        decoder.demodulate(dmr_iq[start : start + 32768], 0.0)
    return decoder


def decimate_and_demod(iq, sample_rate=SAMPLE_RATE):
    """Run stages 1-2: decimation + FM discriminator."""
    target_rate = 48000.0
    decimation = max(1, round(sample_rate / target_rate))
    decimated_rate = sample_rate / decimation

    cutoff = min(decimated_rate * 0.45, DEVIATION * 3)
    taps = firwin(101, cutoff, fs=sample_rate)
    filtered = lfilter(taps, [1.0], iq)
    decimated = filtered[::decimation]

    product = decimated[1:] * np.conj(decimated[:-1])
    phase_diff = np.angle(product)
    scale = decimated_rate / (2 * np.pi * DEVIATION)
    fm = phase_diff * scale

    return fm, decimated_rate


class TestStage1FMDemod:
    """Stage 1: Verify FM demodulation produces 4FSK levels."""

    def test_four_level_distribution(self, dmr_iq):
        fm, rate = decimate_and_demod(dmr_iq)

        hist, edges = np.histogram(fm, bins=200, range=(-2, 2))
        centers = (edges[:-1] + edges[1:]) / 2

        peaks, props = find_peaks(hist, height=np.max(hist) * 0.05, distance=10)
        peak_positions = sorted(centers[peaks])

        assert len(peak_positions) >= 4, (
            f"Expected at least 4 peaks in 4FSK histogram, got {len(peak_positions)}: {peak_positions}"
        )

    def test_signal_present(self, dmr_iq):
        fm, rate = decimate_and_demod(dmr_iq)

        assert np.std(fm) > 0.2, f"FM signal too weak, std={np.std(fm):.3f}"
        assert np.max(np.abs(fm)) > 0.5, "No significant FM deviation"


class TestStage2SymbolRecovery:
    """Stage 2: Verify M&M recovers symbols at 4800 sym/s."""

    def _run_pipeline(self, dmr_iq):
        decoder = DMRDecoder(sample_rate=SAMPLE_RATE)
        filtered = decoder._antialias.process(dmr_iq)
        decimated = filtered[decoder._decim_phase :: decoder._decimation]
        fm = decoder._fm.process(decimated)
        return decoder._mm.process(fm)

    def test_symbol_rate(self, dmr_iq):
        symbols = self._run_pipeline(dmr_iq)
        duration = len(dmr_iq) / SAMPLE_RATE
        actual_rate = len(symbols) / duration
        assert abs(actual_rate - SYMBOL_RATE) / SYMBOL_RATE < 0.05, (
            f"Symbol rate {actual_rate:.0f} not within 5% of {SYMBOL_RATE}"
        )

    def test_four_level_eye(self, dmr_iq):
        """Recovered symbols should cluster at 4 levels."""
        symbols = self._run_pipeline(dmr_iq)
        hist, edges = np.histogram(symbols, bins=100)
        centers = (edges[:-1] + edges[1:]) / 2

        peaks, _ = find_peaks(hist, height=np.max(hist) * 0.05, distance=5)
        peak_pos = centers[peaks]
        assert np.any(peak_pos > 0), "No positive symbol clusters"
        assert np.any(peak_pos < 0), "No negative symbol clusters"


class TestStage3SyncDetection:
    """Stage 3: Verify DMR sync word detection."""

    def test_sync_found(self, dmr_decoder):
        assert dmr_decoder._syncs_found >= 100, (
            f"Expected at least 100 sync detections, got {dmr_decoder._syncs_found}"
        )

    def test_sync_pattern_matching(self):
        for name, pattern in SYNC_PATTERNS.items():
            arr = np.array(list(pattern), dtype=np.uint8)
            result = _match_sync(arr)
            assert result is not None, f"Failed to match exact {name} pattern"
            assert result[0] == name
            assert result[1] == 0

    def test_sync_with_errors(self):
        pattern = np.array(list(SYNC_PATTERNS[SyncType.BS_DATA]), dtype=np.uint8)

        for n_errors in range(1, SYNC_MAX_ERRORS + 1):
            corrupted = pattern.copy()
            for i in range(n_errors):
                corrupted[i] = 1 if corrupted[i] == 3 else 3
            result = _match_sync(corrupted)
            assert result is not None, f"Failed to match with {n_errors} errors"
            assert result[1] == n_errors

    def test_sync_rejects_noise(self):
        rng = np.random.default_rng(42)
        false_positives = 0
        for _ in range(10000):
            noise = rng.choice([1, 3], size=SYNC_DIBITS).astype(np.uint8)
            if _match_sync(noise) is not None:
                false_positives += 1
        assert false_positives < 10, f"Too many false positives: {false_positives}/10000"


class TestFEC:
    """FEC unit tests: Hamming(7,4) and Golay(20,8)."""

    def test_hamming_7_4_no_error(self):
        # All-zero codeword is valid
        bits = np.zeros(7, dtype=np.uint8)
        assert hamming_7_4_decode(bits) is True
        assert np.all(bits == 0)

    def test_hamming_7_4_single_error(self):
        for pos in range(7):
            bits = np.zeros(7, dtype=np.uint8)
            bits[pos] = 1
            assert hamming_7_4_decode(bits) is True
            assert np.all(bits == 0), f"Failed to correct error at position {pos}"

    def test_golay_20_8_no_error(self):
        bits = np.zeros(20, dtype=np.uint8)
        assert golay_20_8_decode(bits) is True

    def test_golay_20_8_single_error(self):
        for pos in range(20):
            bits = np.zeros(20, dtype=np.uint8)
            bits[pos] = 1
            assert golay_20_8_decode(bits) is True
            # Only check message bits (0-7) - parity bits may be modified
            assert np.all(bits[:8] == 0), (
                f"Message bits wrong after correcting error at position {pos}"
            )

    def test_golay_20_8_double_error(self):
        for p1, p2 in [(0, 5), (3, 10), (7, 19), (1, 15)]:
            bits = np.zeros(20, dtype=np.uint8)
            bits[p1] = 1
            bits[p2] = 1
            assert golay_20_8_decode(bits) is True
            assert np.all(bits[:8] == 0), f"Message bits wrong after correcting errors at {p1},{p2}"

    def test_golay_20_8_triple_error(self):
        for p1, p2, p3 in [(0, 5, 10), (1, 8, 15)]:
            bits = np.zeros(20, dtype=np.uint8)
            bits[p1] = 1
            bits[p2] = 1
            bits[p3] = 1
            assert golay_20_8_decode(bits) is True
            assert np.all(bits[:8] == 0), (
                f"Message bits wrong after correcting errors at {p1},{p2},{p3}"
            )


class TestBurstParsing:
    """Verify burst extraction and CACH/Slot Type decoding on real data."""

    def test_messages_decoded(self, dmr_decoder):
        assert dmr_decoder._bursts_decoded > 100, (
            f"Expected at least 100 decoded bursts, got {dmr_decoder._bursts_decoded}"
        )

    def test_color_code_consistency(self, dmr_decoder):
        """PI1UTR uses color code 1 - most valid bursts should have CC=1."""
        cc_counts = dmr_decoder._color_code_counts
        assert len(cc_counts) > 0, "No slot types successfully decoded"
        most_common_cc = max(cc_counts, key=cc_counts.get)
        assert most_common_cc == 1, f"Expected CC=1 (PI1UTR), got CC={most_common_cc}"
        total = sum(cc_counts.values())
        assert cc_counts[1] / total > 0.5, f"Only {cc_counts[1]}/{total} bursts have CC=1"

    def test_data_types_valid(self, dmr_decoder):
        """Decoded data types should be valid; idle repeater shows mostly IDLE/CSBK."""
        dt_counts = dmr_decoder._data_type_counts
        assert len(dt_counts) > 0, "No data types decoded"
        for dt_val in dt_counts:
            assert dt_val in [e.value for e in DataType], f"Invalid data type: {dt_val}"
        assert DataType.IDLE in dt_counts or DataType.CSBK in dt_counts, (
            f"Expected IDLE or CSBK in idle repeater data, got: {dt_counts}"
        )

    def test_cach_decode_rate(self, dmr_decoder):
        """CACH should decode successfully for most bursts."""
        total = dmr_decoder._cach_ok + dmr_decoder._cach_fail
        assert total > 0, "No bursts decoded"
        rate = dmr_decoder._cach_ok / total
        assert rate > 0.5, f"CACH decode rate too low: {rate:.1%} ({dmr_decoder._cach_ok}/{total})"

    def test_slot_type_decode_rate(self, dmr_decoder):
        """Slot Type should decode successfully for most bursts."""
        total = dmr_decoder._slot_type_ok + dmr_decoder._slot_type_fail
        assert total > 0, "No bursts decoded"
        rate = dmr_decoder._slot_type_ok / total
        assert rate > 0.5, (
            f"Slot Type decode rate too low: {rate:.1%} ({dmr_decoder._slot_type_ok}/{total})"
        )

    def test_streaming_consistency(self, dmr_iq):
        """Chunk size should not affect decode results."""

        # Small chunks (512 samples ≈ 2ms) vs large chunks (65536 ≈ 262ms)
        results = {}
        for chunk_size in [512, 65536]:
            decoder = DMRDecoder(sample_rate=SAMPLE_RATE)
            for start in range(0, len(dmr_iq), chunk_size):
                decoder.demodulate(dmr_iq[start : start + chunk_size], 0.0)
            results[chunk_size] = (decoder._syncs_found, decoder._slot_type_ok)

        assert results[512] == results[65536], (
            f"Results differ: 512-sample chunks={results[512]}, "
            f"65536-sample chunks={results[65536]}"
        )


class TestFrameLocking:
    """Verify frame-locked tracking improves decode reliability."""

    def test_lock_rate(self, dmr_decoder):
        """Most bursts should be found via locked tracking, not free search."""
        total_lock_attempts = dmr_decoder._lock_hits + dmr_decoder._lock_misses
        assert total_lock_attempts > 0, "No locked tracking attempts"
        lock_rate = dmr_decoder._lock_hits / total_lock_attempts
        assert lock_rate > 0.9, (
            f"Lock rate too low: {lock_rate:.1%} ({dmr_decoder._lock_hits}/{total_lock_attempts})"
        )

    def test_lock_recovery(self, dmr_decoder):
        """After losing lock, decoder should recover via free search."""
        assert dmr_decoder._bursts_decoded > 100, (
            f"Too few bursts decoded: {dmr_decoder._bursts_decoded}"
        )
        if dmr_decoder._lock_misses > 0:
            miss_rate = dmr_decoder._lock_misses / dmr_decoder._bursts_decoded
            assert miss_rate < 0.1, (
                f"Too many lock misses: {dmr_decoder._lock_misses}/{dmr_decoder._bursts_decoded}"
            )
