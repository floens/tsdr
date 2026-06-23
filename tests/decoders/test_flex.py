"""Incremental tests for the FLEX decoder.

Each test validates a stage of the DSP pipeline using the real FLEX sample.
"""

from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq as _load_iq
from tsdr.radio.decoders.flex import (
    FLEX_BAUD,
    FLEX_DEVIATION,
    FLEXDecoder,
    bch_correct,
    bch_syndrome,
)
from tsdr.radio.dsp import firwin, lfilter

SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "freq=169.65M_sr=240k_dur=0s_gain=5_20260419T1353.cu8.zst"
)
SAMPLE_RATE = 240_000

# Expected decode health for SAMPLE_FILE.
EXPECTED_MESSAGE = (
    "2029568",
    "P 2 BZB-02 Dier in problemen Rat Verleghstraat Breda 203132",
)
MIN_MESSAGES = 5
MIN_FRAMES = 2
MAX_CODEWORD_FAIL_RATE = 0.01


@pytest.fixture(scope="module")
def flex_iq():
    """Module-scoped IQ data."""
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample file not found: {SAMPLE_FILE}")
    return _load_iq(SAMPLE_FILE)


def decimate_and_demod(iq, sample_rate=SAMPLE_RATE):
    """Run stages 1-2: decimation + FM discriminator."""
    target_rate = 16000.0
    decimation = max(1, round(sample_rate / target_rate))
    decimated_rate = sample_rate / decimation

    # Anti-alias filter
    cutoff = min(decimated_rate * 0.45, FLEX_DEVIATION * 2)
    taps = firwin(101, cutoff, fs=sample_rate)
    filtered = lfilter(taps, [1.0], iq)
    decimated = filtered[::decimation]

    # FM discriminator
    product = decimated[1:] * np.conj(decimated[:-1])
    phase_diff = np.angle(product)
    scale = decimated_rate / (2 * np.pi * FLEX_DEVIATION)
    fm = phase_diff * scale

    return fm, decimated_rate


class TestStage1FMDemod:
    """Stage 1: Verify FM demodulation produces clean 2-FSK levels."""

    def test_bimodal_distribution(self, flex_iq):
        fm, rate = decimate_and_demod(flex_iq)

        # FM output should be bimodal around ±1. Pick the densest bin in each
        # half-plane separately - taking the top-2 globally can return two
        # adjacent bins of the same peak.
        hist, edges = np.histogram(fm, bins=100)
        centers = (edges[:-1] + edges[1:]) / 2

        neg_mask = centers < 0
        pos_mask = centers > 0
        neg_peak = centers[neg_mask][np.argmax(hist[neg_mask])]
        pos_peak = centers[pos_mask][np.argmax(hist[pos_mask])]

        assert abs(neg_peak) > 0.3, f"Negative peak too close to 0: {neg_peak}"
        assert abs(pos_peak) > 0.3, f"Positive peak too close to 0: {pos_peak}"


class TestStage2SymbolRecovery:
    """Stage 2: Verify Mueller-Muller recovers ~1600 symbols/second."""

    @staticmethod
    def _run_frontend(flex_iq):
        decoder = FLEXDecoder(sample_rate=SAMPLE_RATE)
        fm = decoder._channelizer.process(flex_iq)
        symbols = decoder._mm.process(fm)
        return symbols

    def test_symbol_rate(self, flex_iq):
        symbols = self._run_frontend(flex_iq)

        duration = len(flex_iq) / SAMPLE_RATE
        actual_rate = len(symbols) / duration
        assert abs(actual_rate - FLEX_BAUD) / FLEX_BAUD < 0.01, (
            f"Symbol rate {actual_rate:.0f} not within 1% of {FLEX_BAUD}"
        )

    def test_symbols_cluster(self, flex_iq):
        symbols = self._run_frontend(flex_iq)

        positive = symbols[symbols > 0]
        negative = symbols[symbols < 0]

        assert len(positive) > 100, "Too few positive symbols"
        assert len(negative) > 100, "Too few negative symbols"
        assert np.mean(np.abs(symbols)) > 0.3, "Symbols too close to zero"


class TestStage3SyncDetection:
    """Stage 3: Verify sync word detection in the bit stream."""

    def test_sync_found(self, flex_iq):
        decoder = FLEXDecoder(sample_rate=SAMPLE_RATE)

        chunk_size = 32768
        for start in range(0, len(flex_iq), chunk_size):
            decoder.demodulate(flex_iq[start : start + chunk_size], 0.0)

        # FLEX broadcasts a sync word every ~1.875 s, so a healthy ~10 s sample
        # contains >= 5 syncs. The old `>= 1` bound passed even when sync recovery
        # was almost completely broken.
        duration_s = len(flex_iq) / SAMPLE_RATE
        expected_min = max(3, int(duration_s / 2.0))
        assert decoder._syncs_found >= expected_min, (
            f"Expected at least {expected_min} sync detections in {duration_s:.0f}s of data, "
            f"got {decoder._syncs_found}"
        )


class TestStage5BCH:
    """Stage 5: BCH(31,21) error correction unit tests."""

    def test_no_error(self):
        """Valid codeword should pass."""
        # Construct a valid BCH(31,21) codeword
        # Info bits: 0x1FFFFF (all ones = idle), encode with BCH
        # Known idle codeword in FLEX: info=0x1FFFFF
        # We'll test with a manually constructed codeword
        # For now, test that syndrome of 0 gives no correction needed
        s1, s3 = bch_syndrome(0)
        assert s1 == 0
        assert s3 == 0

    def test_single_bit_correction(self):
        """Single bit error should be corrected."""
        # Start with all-zero codeword (valid BCH)
        original = 0x00000000
        for bit in range(1, 32):  # bits 1..31 are BCH-protected
            corrupted = original ^ (1 << bit)
            corrected, ok = bch_correct(corrupted)
            assert ok, f"Failed to correct single bit error at position {bit}"
            # After correction, BCH part should be zero again
            # (parity bit 0 may differ)
            assert (corrected >> 1) == 0, f"Correction wrong at bit {bit}"

    def test_double_bit_correction(self):
        """Double bit error should be corrected."""
        original = 0x00000000
        # Test a few double-bit error combinations
        test_pairs = [(1, 5), (3, 10), (7, 20), (15, 25)]
        for b1, b2 in test_pairs:
            corrupted = original ^ (1 << b1) ^ (1 << b2)
            corrected, ok = bch_correct(corrupted)
            assert ok, f"Failed to correct double bit error at positions {b1},{b2}"
            assert (corrected >> 1) == 0, f"Correction wrong at bits {b1},{b2}"


class TestStage6FullDecode:
    """Stage 6: Full decode - run entire pipeline on sample."""

    def test_full_pipeline(self, flex_iq):
        decoder = FLEXDecoder(sample_rate=SAMPLE_RATE)

        chunk_size = 32768
        for start in range(0, len(flex_iq), chunk_size):
            decoder.demodulate(flex_iq[start : start + chunk_size], 0.0)

        duration_s = len(flex_iq) / SAMPLE_RATE
        expected_min = max(3, int(duration_s / 2.0))
        assert decoder._syncs_found >= expected_min, (
            f"Expected at least {expected_min} sync words, got {decoder._syncs_found}"
        )

        assert decoder._frames_decoded >= MIN_FRAMES, (
            f"Expected at least {MIN_FRAMES} frames decoded, got {decoder._frames_decoded}"
        )

        total_cw = decoder._codewords_ok + decoder._codewords_fail
        assert total_cw > 0, "No codewords processed"
        fail_rate = decoder._codewords_fail / total_cw
        assert fail_rate <= MAX_CODEWORD_FAIL_RATE, (
            f"Codeword fail rate {fail_rate:.3f} exceeds {MAX_CODEWORD_FAIL_RATE} "
            f"({decoder._codewords_fail}/{total_cw})"
        )

        messages = decoder.get_messages()
        assert len(messages) >= MIN_MESSAGES, (
            f"Expected at least {MIN_MESSAGES} decoded messages, got {len(messages)}"
        )

        capcode, text = EXPECTED_MESSAGE
        needle = f"[{capcode}] {text}"
        assert any(needle in m.text for m in messages), (
            f"Pinned message {needle!r} not found in decoded output. "
            f"Got: {[m.text for m in messages]}"
        )

    def test_streaming_consistency(self, flex_iq):
        """Chunked streaming must produce the same sync count regardless of chunk size."""
        d1 = FLEXDecoder(sample_rate=SAMPLE_RATE)
        for start in range(0, len(flex_iq), 8192):
            d1.demodulate(flex_iq[start : start + 8192], 0.0)

        d2 = FLEXDecoder(sample_rate=SAMPLE_RATE)
        for start in range(0, len(flex_iq), 131072):
            d2.demodulate(flex_iq[start : start + 131072], 0.0)

        assert d1._syncs_found == d2._syncs_found, (
            f"Chunk size affects sync count: 8K→{d1._syncs_found}, 128K→{d2._syncs_found}"
        )
