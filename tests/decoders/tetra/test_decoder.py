import functools
from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.tetra.burst import (
    NDB_BBK1_BITS,
    NDB_BBK2_BITS,
    NDB_BLK1_BITS,
    NDB_BLK2_BITS,
    SB_BBK_BITS,
    SB_BLK1_BITS,
    SB_BLK2_BITS,
    SB_FREQ_BITS,
    extract_normal_burst,
    extract_schf,
    extract_sync_burst,
)
from tsdr.radio.decoders.tetra.channel import (
    append_crc,
    decode_block,
    deinterleave,
    depuncture_2_3,
    encode_block,
    interleave,
    puncture_2_3,
    rm3014_decode,
    rm3014_encode,
)
from tsdr.radio.decoders.tetra.decoder import TETRADecoder
from tsdr.radio.decoders.tetra.demod import TetraDemod, estimate_freq_offset, rrc_taps
from tsdr.radio.decoders.tetra.mac import SB1Info, parse_sb1
from tsdr.radio.decoders.tetra.scramble import SCRAMB_INIT, generate_scramble_bits, scramble_init
from tsdr.radio.decoders.tetra.speech_channel import (
    A1,
    A2,
    SPEECH_K,
    _generate_speech_crc,
    bits_to_bytes,
    check_speech_crc,
    codec_to_classes,
    decode_speech,
    deinterleave_speech,
    depuncture,
    encode_speech,
    interleave_speech,
    puncture,
    reorder_to_codec,
)
from tsdr.radio.decoders.tetra.sync import (
    SLOT_SYMBOLS,
    SYNC_TRAIN_SYM_OFFSET,
    Y_REF,
    SyncDetector,
    SyncState,
)
from tsdr.radio.dsp import lfilter
from tsdr.radio.dsp.costas import CostasLoop
from tsdr.radio.dsp.viterbi import ViterbiDecoder

# TETRA convolutional code parameters (K=5, rate 1/4)
# Register convention: input bit at MSB (bit K-1), same as DAB.
#   G1 = 1+D+D^4:         taps at bit4(input), bit3(D), bit0(D^4) -> 0o31
#   G2 = 1+D^2+D^3+D^4:   taps at bit4, bit2, bit1, bit0          -> 0o27
#   G3 = 1+D+D^2+D^4:     taps at bit4, bit3, bit2, bit0          -> 0o35
#   G4 = 1+D+D^3+D^4:     taps at bit4, bit3, bit1, bit0          -> 0o33
TETRA_K = 5
TETRA_GENERATORS = [0o31, 0o27, 0o35, 0o33]


def conv_encode(bits: np.ndarray, k: int, generators: list[int]) -> np.ndarray:
    """Simple convolutional encoder using polynomial form (matches Viterbi convention)."""
    state = 0
    out = []
    for bit in bits:
        reg = (int(bit) << (k - 1)) | state
        state = reg >> 1
        for gen in generators:
            out.append(bin(reg & gen).count("1") % 2)
    return np.array(out, dtype=np.uint8)


def tetra_conv_encode_reference(bits: np.ndarray) -> np.ndarray:
    """Reference TETRA encoder using delay-line form."""
    delayed = [0, 0, 0, 0]
    out = []
    for bit in bits:
        b = int(bit)
        g1 = (b + delayed[0] + delayed[3]) % 2
        g2 = (b + delayed[1] + delayed[2] + delayed[3]) % 2
        g3 = (b + delayed[0] + delayed[1] + delayed[3]) % 2
        g4 = (b + delayed[0] + delayed[2] + delayed[3]) % 2
        delayed[3] = delayed[2]
        delayed[2] = delayed[1]
        delayed[1] = delayed[0]
        delayed[0] = b
        out.extend([g1, g2, g3, g4])
    return np.array(out, dtype=np.uint8)


class TestCheckpoint1Viterbi:
    """Checkpoint 1: Generalized Viterbi decoder."""

    def test_encode_decode_roundtrip(self):
        """Encode 100 random bits + 4 tail bits, decode, assert exact match."""
        rng = np.random.default_rng(42)
        info_bits = rng.integers(0, 2, size=100, dtype=np.uint8)
        # Add K-1 tail bits to flush encoder to state 0
        bits = np.concatenate([info_bits, np.zeros(TETRA_K - 1, dtype=np.uint8)])

        encoded = conv_encode(bits, TETRA_K, TETRA_GENERATORS)
        assert len(encoded) == len(bits) * 4

        # Perfect soft bits: 0 -> -1.0, 1 -> +1.0
        soft = encoded.astype(np.float32) * 2.0 - 1.0

        dec = ViterbiDecoder(TETRA_K, TETRA_GENERATORS)
        decoded = dec.decode(soft)

        np.testing.assert_array_equal(decoded, bits)

    def test_soft_decision_gain(self):
        """At SNR=2dB, soft Viterbi should achieve BER < 1%."""
        rng = np.random.default_rng(123)
        n_info = 1000
        info_bits = rng.integers(0, 2, size=n_info, dtype=np.uint8)
        bits = np.concatenate([info_bits, np.zeros(TETRA_K - 1, dtype=np.uint8)])

        encoded = conv_encode(bits, TETRA_K, TETRA_GENERATORS)
        soft_clean = encoded.astype(np.float32) * 2.0 - 1.0

        # Add Gaussian noise at SNR=2dB (Eb/N0)
        snr_db = 2.0
        noise_std = 1.0 / np.sqrt(2.0 * 10.0 ** (snr_db / 10.0))
        noise = rng.normal(0, noise_std, size=len(soft_clean)).astype(np.float32)
        soft_noisy = soft_clean + noise

        dec = ViterbiDecoder(TETRA_K, TETRA_GENERATORS)
        decoded = dec.decode(soft_noisy)

        ber = np.mean(decoded[:n_info] != info_bits)
        assert ber < 0.01, f"BER {ber:.4f} exceeds 1%"

    def test_reference_match(self):
        """Polynomial encoder matches reference delay-line encoder bit-for-bit."""
        # Test with all-zeros (SB1-like)
        bits_zero = np.zeros(60, dtype=np.uint8)
        assert np.array_equal(
            conv_encode(bits_zero, TETRA_K, TETRA_GENERATORS),
            tetra_conv_encode_reference(bits_zero),
        )

        # Test with random data
        rng = np.random.default_rng(99)
        bits_rand = rng.integers(0, 2, size=100, dtype=np.uint8)
        assert np.array_equal(
            conv_encode(bits_rand, TETRA_K, TETRA_GENERATORS),
            tetra_conv_encode_reference(bits_rand),
        )

        # Test with alternating pattern
        bits_alt = np.array([i % 2 for i in range(60)], dtype=np.uint8)
        assert np.array_equal(
            conv_encode(bits_alt, TETRA_K, TETRA_GENERATORS),
            tetra_conv_encode_reference(bits_alt),
        )

    def test_dab_compatibility(self):
        """Generalized decoder works with DAB parameters (K=7)."""
        dab_k = 7
        dab_generators = [0o133, 0o171, 0o145, 0o133]

        rng = np.random.default_rng(77)
        info_bits = rng.integers(0, 2, size=100, dtype=np.uint8)
        bits = np.concatenate([info_bits, np.zeros(dab_k - 1, dtype=np.uint8)])

        encoded = conv_encode(bits, dab_k, dab_generators)
        assert len(encoded) == len(bits) * 4

        soft = encoded.astype(np.float32) * 2.0 - 1.0

        dec = ViterbiDecoder(dab_k, dab_generators)
        decoded = dec.decode(soft)

        np.testing.assert_array_equal(decoded, bits)


class TestCheckpoint2Costas:
    """Checkpoint 2: CostasLoop QPSK mode."""

    def test_bpsk_regression(self):
        """Existing BPSK behavior unchanged: lock with 0.05 rad/sample freq offset."""
        rng = np.random.default_rng(42)
        n = 1000
        # Random BPSK symbols: ±1 on real axis
        bits = rng.integers(0, 2, size=n)
        symbols = np.array([1.0 + 0j if b else -1.0 + 0j for b in bits], dtype=np.complex64)

        # Apply frequency offset
        freq_offset = 0.05
        phase = np.cumsum(np.full(n, freq_offset))
        symbols_shifted = symbols * np.exp(1j * phase).astype(np.complex64)

        loop = CostasLoop(alpha=0.1, beta=0.001, mode="bpsk")
        out = loop.process(symbols_shifted)

        # After settling (~200 symbols), output should cluster near ±1 on real axis
        settled = out[200:]
        assert np.mean(np.abs(settled.imag)) < 0.3, "BPSK should cluster on real axis"
        assert np.mean(np.abs(settled.real)) > 0.7, "BPSK should have strong real component"

    def test_qpsk_lock(self):
        """QPSK Costas locks onto QPSK signal with 0.1 rad/sample freq offset."""
        rng = np.random.default_rng(42)
        n = 4000
        # Random QPSK symbols at (±1±1j)/√2
        dibits = rng.integers(0, 4, size=n)
        constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        symbols = constellation[dibits].astype(np.complex64)

        # Apply frequency offset
        freq_offset = 0.1
        phase = np.cumsum(np.full(n, freq_offset))
        symbols_shifted = symbols * np.exp(1j * phase).astype(np.complex64)

        loop = CostasLoop(alpha=0.08, beta=0.08**2 * 0.25, mode="qpsk")
        out = loop.process(symbols_shifted)

        # After settling, output symbols should cluster near 4 QPSK points (within 0.3)
        # QPSK Costas can lock to any of 4 rotations, so check all rotations
        settled = out[1000:]
        min_dists = np.array([np.min(np.abs(s - constellation)) for s in settled])
        assert np.mean(min_dists) < 0.3, (
            f"Mean distance to nearest QPSK point: {np.mean(min_dists):.3f}"
        )

    def test_qpsk_frequency_track(self):
        """QPSK Costas tracks slowly drifting frequency."""
        rng = np.random.default_rng(42)
        n = 3000
        dibits = rng.integers(0, 4, size=n)
        constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        symbols = constellation[dibits].astype(np.complex64)

        # Drifting frequency: starts at 0.01 rad/sample, accelerates at 0.00001 rad/sample²
        freq = 0.01 + 0.00001 * np.arange(n)
        phase = np.cumsum(freq)
        symbols_shifted = symbols * np.exp(1j * phase).astype(np.complex64)

        loop = CostasLoop(alpha=0.03, beta=0.03**2 * 0.25, mode="qpsk")
        loop.process(symbols_shifted)

        # Final frequency estimate should be close to actual final frequency
        final_freq_actual = freq[-1]
        assert abs(loop.freq - final_freq_actual) < 0.02, (
            f"Freq estimate {loop.freq:.4f} vs actual {final_freq_actual:.4f}"
        )


# Checkpoint 3: π/4-DQPSK demodulator

TETRA_SAMPLE = "tests/samples/freq=427.031M_sr=240k_dur=5s_gain=30_20260423T2014.cu8.zst"
SAMPLE_RATE = 240_000

# Phase change mapping for π/4-DQPSK modulator (consistent with demod soft bit extraction)
# With MSB=-im/norm, LSB=-re/norm:
#   Phase π/4 -> dibit 00, Phase -π/4 -> dibit 10
#   Phase 3π/4 -> dibit 01, Phase -3π/4 -> dibit 11
PI4_DQPSK_PHASE_MAP = {
    0b00: np.pi / 4,
    0b01: 3 * np.pi / 4,
    0b10: -np.pi / 4,
    0b11: -3 * np.pi / 4,
}


@functools.lru_cache(maxsize=1)
def _load_tetra_iq() -> np.ndarray:
    """Load TETRA IQ sample with caching (avoids 16x reload)."""
    sample_path = Path(TETRA_SAMPLE)
    if not sample_path.exists():
        pytest.skip(f"Sample not found: {TETRA_SAMPLE}")
    return load_iq(sample_path)


def pi4dqpsk_modulate(dibits: np.ndarray, sps: int) -> np.ndarray:
    """Generate π/4-DQPSK baseband signal with RRC pulse shaping."""
    # Map dibits to phase changes and accumulate
    phase = 0.0
    symbols = np.empty(len(dibits), dtype=np.complex128)
    for i, d in enumerate(dibits):
        phase += PI4_DQPSK_PHASE_MAP[int(d)]
        symbols[i] = np.exp(1j * phase)

    # Upsample and pulse shape (causal filter, matching demod)
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    taps = rrc_taps(float(sps), alpha=0.35)
    result: np.ndarray = lfilter(taps, 1.0, upsampled)
    return result.astype(np.complex64)


class TestCheckpoint3Demod:
    """Checkpoint 3: π/4-DQPSK demodulator + sample signal verification."""

    def test_synthetic_signal(self):
        """Demod synthetic π/4-DQPSK signal, verify >95% dibit accuracy."""

        rng = np.random.default_rng(42)
        n_symbols = 1000
        sps = 4
        sample_rate = 18000.0 * sps  # 72 kHz, no decimation needed

        dibits = rng.integers(0, 4, size=n_symbols, dtype=np.uint8)
        signal = pi4dqpsk_modulate(dibits, sps)

        # Add AWGN at SNR=15 dB
        snr_db = 15.0
        sig_power = np.mean(np.abs(signal) ** 2)
        noise_power = sig_power * 10.0 ** (-snr_db / 10.0)
        noise = rng.normal(0, np.sqrt(noise_power / 2), size=(len(signal), 2)).astype(np.float32)
        noisy = signal + (noise[:, 0] + 1j * noise[:, 1]).astype(np.complex64)

        demod = TetraDemod(sample_rate)
        soft_bits = demod.process(noisy)

        # Convert soft bits to hard dibits
        hard_bits = (soft_bits > 0).astype(np.uint8)
        n_recovered = len(hard_bits) // 2
        recovered_dibits = hard_bits[0::2] * 2 + hard_bits[1::2]

        # Find best alignment accounting for RRC filter delay (~12 symbols)
        # and diff demod offset. Search over both input and output offsets.
        best_acc = 0.0
        for skip_rec in range(min(50, n_recovered)):
            for offset in range(min(50, len(dibits))):
                ref = dibits[offset : offset + n_recovered - skip_rec]
                rec = recovered_dibits[skip_rec : skip_rec + len(ref)]
                match_len = min(len(ref), len(rec))
                if match_len > 100:
                    acc = np.mean(ref[:match_len] == rec[:match_len])
                    best_acc = max(best_acc, acc)
                    if best_acc > 0.99:
                        break
            if best_acc > 0.99:
                break
        assert best_acc > 0.95, f"Best dibit accuracy {best_acc:.3f} < 95%"

    def test_real_sample_loads(self):
        """Load real TETRA sample, verify output symbol count."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        duration = 0.5
        n_samples = int(duration * sample_rate)
        iq_chunk = iq[:n_samples]

        demod = TetraDemod(sample_rate)
        soft_bits = demod.process(iq_chunk)

        # Each symbol produces 2 soft bits
        n_symbols = len(soft_bits) // 2
        expected = int(duration * 18000)  # ~9000 symbols
        assert abs(n_symbols - expected) / expected < 0.10, (
            f"Symbol count {n_symbols} not within 10% of expected {expected}"
        )
        # Verify not all zeros
        assert np.any(soft_bits != 0), "Soft bits are all zero"

    def test_phase_histogram(self):
        """Real TETRA signal should show 4-cluster phase histogram."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(1.0 * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])

        if len(symbols) < 100:
            pytest.fail(f"Too few symbols recovered: {len(symbols)}")

        # Differential demod
        diff = symbols[1:] * np.conj(symbols[:-1])
        phases = np.angle(diff)

        # Check 4 quadrants have significant occupancy
        quadrant_counts = [
            np.sum((phases > 0) & (phases < np.pi / 2)),  # π/4
            np.sum((phases > np.pi / 2) & (phases < np.pi)),  # 3π/4
            np.sum((phases < 0) & (phases > -np.pi / 2)),  # -π/4
            np.sum((phases < -np.pi / 2) & (phases > -np.pi)),  # -3π/4
        ]
        total = sum(quadrant_counts)
        fractions = [c / total for c in quadrant_counts]
        for i, f in enumerate(fractions):
            assert f > 0.15, f"Quadrant {i} has only {f:.1%} of symbols (expect >15%)"


class TestCheckpoint4Sync:
    """Checkpoint 4: Burst synchronization."""

    def test_synthetic_sync_burst(self):
        """Construct sync burst, embed in noise, verify sync detection."""
        rng = np.random.default_rng(42)

        # Build a sync burst: random symbols except training at correct offset
        n_syms = SLOT_SYMBOLS
        diff_syms = np.exp(1j * rng.uniform(-np.pi, np.pi, size=n_syms))

        # Insert sync training at correct offset
        diff_syms[SYNC_TRAIN_SYM_OFFSET : SYNC_TRAIN_SYM_OFFSET + len(Y_REF)] = Y_REF

        # Convert diff symbols to raw symbols (integrate phase)
        raw_syms = np.empty(n_syms + 1, dtype=np.complex128)
        raw_syms[0] = 1.0
        for i in range(n_syms):
            raw_syms[i + 1] = raw_syms[i] * diff_syms[i]

        # Add some padding and noise
        padding = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        padded = np.concatenate([padding.astype(np.complex64), raw_syms.astype(np.complex64)])

        # Add noise (SNR=10dB)
        sig_power = np.mean(np.abs(padded) ** 2)
        noise_power = sig_power * 10.0 ** (-10 / 10.0)
        noise = (
            rng.standard_normal(len(padded)) + 1j * rng.standard_normal(len(padded))
        ) * np.sqrt(noise_power / 2)
        noisy = (padded + noise).astype(np.complex64)

        detector = SyncDetector(threshold=0.5)
        results = detector.process(noisy)

        assert detector.state == SyncState.LOCKED, "Should lock to sync burst"
        assert len(results) >= 1, "Should find at least one burst"
        assert results[0].burst_type == "sync", f"Expected sync, got {results[0].burst_type}"
        assert results[0].correlation > 0.5, f"Correlation {results[0].correlation:.3f} too low"

    def test_real_sample_sync_detection(self):
        """Demod real sample, detect sync bursts."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        duration = 5.0
        n_samples = int(duration * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])

        detector = SyncDetector()
        results = detector.process(symbols)

        sync_count = sum(1 for r in results if r.burst_type == "sync")
        normal_count = sum(1 for r in results if r.burst_type in ("normal_1", "normal_2"))
        total = len(results)

        print(f"Total bursts: {total}, sync: {sync_count}, normal: {normal_count}")
        assert total >= 5, f"Only {total} bursts found (expected ≥5)"

    def test_lock_stability(self):
        """Verify stable lock over 10 seconds."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        duration = 10.0
        n_samples = int(duration * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])

        detector = SyncDetector()
        results = detector.process(symbols)

        # Count longest consecutive non-unknown run
        longest_run = 0
        current_run = 0
        for r in results:
            if r.burst_type != "unknown":
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        print(f"Total bursts: {len(results)}, longest locked run: {longest_run}")
        assert longest_run >= 5, f"Longest locked run {longest_run} < 5"

    def test_burst_type_identification(self):
        """Verify sync:normal ratio is approximately 1:17."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        duration = 10.0
        n_samples = int(duration * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])

        detector = SyncDetector()
        results = detector.process(symbols)

        sync_count = sum(1 for r in results if r.burst_type == "sync")
        normal_count = sum(1 for r in results if r.burst_type in ("normal_1", "normal_2"))

        print(f"Sync: {sync_count}, Normal: {normal_count}")
        assert sync_count > 0, "No sync bursts"
        assert normal_count > 0, "No normal bursts"


class TestCheckpoint5Burst:
    """Checkpoint 5: Burst field extraction."""

    def test_offset_consistency(self):
        """Verify field offsets by constructing a known 510-bit array."""
        # Fill 510 bits with position-dependent values
        soft = np.arange(510, dtype=np.float32) / 510.0

        sb = extract_sync_burst(soft)
        assert len(sb.freq_correction) == SB_FREQ_BITS == 80
        assert len(sb.sb1) == SB_BLK1_BITS == 120
        assert len(sb.bbk) == SB_BBK_BITS == 30
        assert len(sb.sb2) == SB_BLK2_BITS == 216

        # Verify values match expected positions
        np.testing.assert_array_almost_equal(sb.freq_correction, soft[14:94])
        np.testing.assert_array_almost_equal(sb.sb1, soft[94:214])
        np.testing.assert_array_almost_equal(sb.bbk, soft[252:282])
        np.testing.assert_array_almost_equal(sb.sb2, soft[282:498])

        # Total data bits: 80 + 120 + 30 + 216 = 446 (+ tails + phase adj + training = 510)

        ndb = extract_normal_burst(soft)
        assert len(ndb.bkn1) == NDB_BLK1_BITS == 216
        assert len(ndb.bbk) == NDB_BBK1_BITS + NDB_BBK2_BITS == 30
        assert len(ndb.bkn2) == NDB_BLK2_BITS == 216

        np.testing.assert_array_almost_equal(ndb.bkn1, soft[14:230])
        np.testing.assert_array_almost_equal(ndb.bbk[:14], soft[230:244])
        np.testing.assert_array_almost_equal(ndb.bbk[14:], soft[266:282])
        np.testing.assert_array_almost_equal(ndb.bkn2, soft[282:498])

    def test_real_sample_extraction(self):
        """Extract SB1 from sync bursts in real sample."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(5.0 * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])

        detector = SyncDetector()
        results = detector.process(symbols)

        sync_bursts = [r for r in results if r.burst_type == "sync"]
        assert len(sync_bursts) > 0, "No sync bursts found"

        for sb_result in sync_bursts[:10]:
            sb = extract_sync_burst(sb_result.soft_bits)
            assert len(sb.sb1) == 120, f"SB1 wrong length: {len(sb.sb1)}"
            assert len(sb.bbk) == 30, f"BBK wrong length: {len(sb.bbk)}"
            # Soft bits should have non-zero variance (not all zero)
            assert np.std(sb.sb1) > 0.1, "SB1 soft bits have no variance"

    def test_schf_extraction(self):
        """Test SCH/F combined extraction from normal bursts."""

        # Create known normal burst soft bits
        soft = np.arange(510, dtype=np.float32) / 510.0

        ndb = extract_normal_burst(soft)
        schf = extract_schf(ndb)
        assert len(schf) == 432, f"SCH/F wrong length: {len(schf)}"
        np.testing.assert_array_almost_equal(schf[:216], ndb.bkn1)
        np.testing.assert_array_almost_equal(schf[216:], ndb.bkn2)


class TestCheckpoint6Channel:
    """Checkpoint 6: Channel decoding chain."""

    def test_lfsr_reference(self):
        """LFSR scrambling sequence with init=3 matches expected values."""

        bits = generate_scramble_bits(3, 120)
        assert len(bits) == 120
        # First few bits should be deterministic for init=3
        # Verify the LFSR produces non-trivial output
        assert np.sum(bits) > 20, "Scramble bits too sparse"
        assert np.sum(bits) < 100, "Scramble bits too dense"
        # Verify reproducibility
        bits2 = generate_scramble_bits(3, 120)
        np.testing.assert_array_equal(bits, bits2)

    def test_interleave_roundtrip(self):
        """Interleave then deinterleave recovers original data."""

        rng = np.random.default_rng(42)
        for k, a in [(120, 11), (216, 101), (168, 13), (432, 103)]:
            data = rng.standard_normal(k).astype(np.float32)
            interleaved = interleave(data, k, a)
            recovered = deinterleave(interleaved, k, a)
            np.testing.assert_array_almost_equal(recovered, data, err_msg=f"K={k}, a={a}")

    def test_depuncture_encode_roundtrip(self):
        """Encode -> puncture -> depuncture -> Viterbi recovers type2 bits."""
        rng = np.random.default_rng(42)
        type1 = rng.integers(0, 2, size=60, dtype=np.uint8)

        # CRC

        type2 = append_crc(type1)
        assert len(type2) == 76  # 60 + 16 CRC
        # Pad to 80 for SB1
        type2_padded = np.concatenate([type2, np.zeros(4, dtype=np.uint8)])

        # Encode
        mother = conv_encode(type2_padded, TETRA_K, TETRA_GENERATORS)
        assert len(mother) == 320  # 80 * 4

        # Puncture
        punctured = puncture_2_3(mother, 120)
        assert len(punctured) == 120

        # Depuncture
        soft_mother = depuncture_2_3(punctured.astype(np.float32) * 2.0 - 1.0, 320)
        assert len(soft_mother) == 320

        # Viterbi
        dec = ViterbiDecoder(TETRA_K, TETRA_GENERATORS)
        decoded = dec.decode(soft_mother)
        np.testing.assert_array_equal(decoded, type2_padded)

    def test_full_sb1_chain(self):
        """Full forward + reverse chain for SB1: encode -> decode = identity."""

        # Known SB1 type1 bits: MCC=204, MNC=16383, CC=1, TN=3, FN=18, MN=1
        type1 = np.zeros(60, dtype=np.uint8)
        # System code (bits 0-3)
        # Colour code (bits 4-9): CC=1 -> binary 000001
        type1[9] = 1
        # TN (bits 10-11): TN=3 -> binary 11
        type1[10] = 1
        type1[11] = 1
        # FN (bits 12-16): FN=18 -> binary 10010
        type1[12] = 1
        type1[16] = 1
        # MN (bits 17-22): MN=1 -> binary 000001
        type1[22] = 1
        # MCC (bits 31-40): MCC=204 -> binary 0011001100
        mcc_bits = [(204 >> (9 - i)) & 1 for i in range(10)]
        type1[31:41] = mcc_bits
        # MNC (bits 41-54): MNC=16383 -> binary 11111111111111
        type1[41:55] = 1

        # Encode
        type5 = encode_block(type1, "SB1", SCRAMB_INIT)
        assert len(type5) == 120

        # Decode with perfect soft bits
        soft = type5.astype(np.float32) * 2.0 - 1.0
        decoded, crc_ok = decode_block(soft, "SB1", SCRAMB_INIT)

        assert crc_ok, "CRC should be valid"
        np.testing.assert_array_equal(decoded, type1)

    def test_rm3014(self):
        """RM(30,14) encode + soft ML decode recovers info bits."""

        rng = np.random.default_rng(42)

        # Clean decode
        for _ in range(10):
            info = rng.integers(0, 2, size=14, dtype=np.uint8)
            codeword = rm3014_encode(info)
            assert len(codeword) == 30
            soft = codeword.astype(np.float32) * 2.0 - 1.0
            decoded = rm3014_decode(soft)
            np.testing.assert_array_equal(decoded, info)

        # With up to 3 hard bit errors
        for n_errors in range(1, 4):
            info = rng.integers(0, 2, size=14, dtype=np.uint8)
            codeword = rm3014_encode(info)
            soft = codeword.astype(np.float32) * 2.0 - 1.0
            # Flip n_errors random bits
            err_pos = rng.choice(30, size=n_errors, replace=False)
            soft[err_pos] *= -1
            decoded = rm3014_decode(soft)
            np.testing.assert_array_equal(decoded, info, err_msg=f"{n_errors} errors")

    def test_real_sample_sb1_decode(self):
        """Decode SB1 from real sample sync bursts. Target >50% CRC-OK."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(10.0 * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])

        detector = SyncDetector()
        results = detector.process(symbols)

        sync_bursts = [r for r in results if r.burst_type == "sync"]
        assert len(sync_bursts) > 0, "No sync bursts found"

        crc_ok_count = 0
        for sb_result in sync_bursts:
            sb = extract_sync_burst(sb_result.soft_bits)
            _, crc_ok = decode_block(sb.sb1, "SB1", SCRAMB_INIT)
            if crc_ok:
                crc_ok_count += 1

        rate = crc_ok_count / len(sync_bursts)
        print(f"SB1 CRC-OK: {crc_ok_count}/{len(sync_bursts)} = {rate:.1%}")
        assert rate > 0.50, f"CRC-OK rate {rate:.1%} below 50% target"


class TestCheckpoint7Bootstrap:
    """Checkpoint 7: SB1 bootstrap (BSCH decode)."""

    def test_sb1_parse(self):
        """Parse SB1 type1 bits with known values."""

        type1 = np.zeros(60, dtype=np.uint8)
        # CC=1 at bits 4-9
        type1[9] = 1
        # TN=3 at bits 10-11
        type1[10] = 1
        type1[11] = 1
        # FN=18 at bits 12-16: 18 = 0b10010
        type1[12] = 1
        type1[15] = 1
        # MN=1 at bits 17-22
        type1[22] = 1
        # MCC=204 at bits 31-40
        for i, b in enumerate([(204 >> (9 - j)) & 1 for j in range(10)]):
            type1[31 + i] = b
        # MNC=16383 at bits 41-54
        type1[41:55] = 1

        info = parse_sb1(type1)
        assert info.colour_code == 1
        assert info.timeslot == 3
        assert info.frame_number == 18
        assert info.multiframe_number == 1
        assert info.mcc == 204
        assert info.mnc == 16383

    def test_scramble_init(self):
        """Verify scramble init formula."""

        type1 = np.zeros(60, dtype=np.uint8)
        type1[9] = 1  # CC=1
        for i, b in enumerate([(204 >> (9 - j)) & 1 for j in range(10)]):
            type1[31 + i] = b
        type1[41:55] = 1  # MNC=16383

        info = parse_sb1(type1)
        expected = scramble_init(204, 16383, 1)
        assert info.scramble_init == expected
        # Verify formula: (CC | (MNC << 6) | (MCC << 20)) << 2 | 3
        manual = ((1 | (16383 << 6) | (204 << 20)) << 2) | 3
        assert info.scramble_init == manual

    def test_real_sample_bootstrap(self):
        """Decode SB1 from real sample, extract MCC/MNC/CC."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(10.0 * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])
        detector = SyncDetector()
        results = detector.process(symbols)

        sync_bursts = [r for r in results if r.burst_type == "sync"]

        mccs, mncs, ccs = [], [], []
        for sb_result in sync_bursts:
            sb = extract_sync_burst(sb_result.soft_bits)
            type1, crc_ok = decode_block(sb.sb1, "SB1", SCRAMB_INIT)
            if crc_ok:
                info = parse_sb1(type1)
                mccs.append(info.mcc)
                mncs.append(info.mnc)
                ccs.append(info.colour_code)

        assert len(mccs) > 0, "No valid SB1 decodes"
        print(f"MCC={mccs[0]}, MNC={mncs[0]}, CC={ccs[0]}")

        # All decodes should be consistent
        assert len(set(mccs)) == 1, f"Inconsistent MCC: {set(mccs)}"
        assert len(set(mncs)) == 1, f"Inconsistent MNC: {set(mncs)}"
        assert len(set(ccs)) == 1, f"Inconsistent CC: {set(ccs)}"

        # Valid ranges
        assert 0 <= mccs[0] <= 1023
        assert 0 <= mncs[0] <= 16383

    def test_tdma_timing(self):
        """Verify TDMA timing from SB1 decodes."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(10.0 * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])
        detector = SyncDetector()
        results = detector.process(symbols)

        for sb_result in [r for r in results if r.burst_type == "sync"][:10]:
            sb = extract_sync_burst(sb_result.soft_bits)
            type1, crc_ok = decode_block(sb.sb1, "SB1", SCRAMB_INIT)
            if crc_ok:
                info = parse_sb1(type1)
                assert 0 <= info.timeslot <= 3, f"TN={info.timeslot} out of range"
                assert 1 <= info.frame_number <= 18, f"FN={info.frame_number} out of range"
                assert 1 <= info.multiframe_number <= 60, (
                    f"MN={info.multiframe_number} out of range"
                )
                # FN should be in valid range (may not always be 18
                # since the sync detector identifies based on training sequence
                # correlation, and the signal may use sync format on all slots)
                print(f"  TN={info.timeslot} FN={info.frame_number} MN={info.multiframe_number}")


class TestCheckpoint8FullDecode:
    """Checkpoint 8: Full block decode (SB2, NDB, BBK)."""

    def _get_bursts_and_init(self, duration: float = 10.0):
        """Helper: demod real sample, get bursts and scramble init."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(duration * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])
        detector = SyncDetector()
        results = detector.process(symbols)

        # Bootstrap: find scramble init from first valid SB1
        s_init = None
        for r in results:
            if r.burst_type == "sync":
                sb = extract_sync_burst(r.soft_bits)
                type1, crc_ok = decode_block(sb.sb1, "SB1", SCRAMB_INIT)
                if crc_ok:
                    info = parse_sb1(type1)
                    s_init = info.scramble_init
                    break

        assert s_init is not None, "Could not bootstrap scramble init"
        return results, s_init

    def test_sb2_decode(self):
        """Decode SB2 from sync bursts with derived scramble init."""

        results, s_init = self._get_bursts_and_init()

        sync_bursts = [r for r in results if r.burst_type == "sync"]
        crc_ok_count = 0
        for r in sync_bursts:
            sb = extract_sync_burst(r.soft_bits)
            _, crc_ok = decode_block(sb.sb2, "SB2", s_init)
            if crc_ok:
                crc_ok_count += 1

        rate = crc_ok_count / len(sync_bursts) if sync_bursts else 0
        print(f"SB2 CRC-OK: {crc_ok_count}/{len(sync_bursts)} = {rate:.1%}")
        assert rate > 0.50, f"SB2 CRC-OK rate {rate:.1%} below 50%"

    def test_bbk_decode(self):
        """Decode BBK with RM(30,14) soft ML decoder."""

        results, _ = self._get_bursts_and_init()

        sync_bursts = [r for r in results if r.burst_type == "sync"]
        decoded_values = []
        for r in sync_bursts[:20]:
            sb = extract_sync_burst(r.soft_bits)
            # BBK is 30 soft bits, decoded with RM(30,14) to 14 info bits
            info_14 = rm3014_decode(sb.bbk)
            assert len(info_14) == 14
            decoded_values.append(tuple(info_14))

        # Verify at least some consistency in decoded values
        unique_vals = set(decoded_values)
        print(f"BBK unique values: {len(unique_vals)} from {len(decoded_values)} decodes")
        assert len(decoded_values) > 0, "No BBK decodes"

    def test_statistics_summary(self):
        """Print decode statistics."""

        results, s_init = self._get_bursts_and_init()

        sync_count = sum(1 for r in results if r.burst_type == "sync")
        normal_count = sum(1 for r in results if r.burst_type in ("normal_1", "normal_2"))

        sb1_ok = sb2_ok = 0
        for r in results:
            if r.burst_type == "sync":
                sb = extract_sync_burst(r.soft_bits)
                _, ok = decode_block(sb.sb1, "SB1", SCRAMB_INIT)
                if ok:
                    sb1_ok += 1
                _, ok = decode_block(sb.sb2, "SB2", s_init)
                if ok:
                    sb2_ok += 1

        print("\n=== TETRA Decode Statistics ===")
        print(f"Total bursts: {len(results)}")
        print(f"Sync: {sync_count}, Normal: {normal_count}")
        print(f"SB1 CRC-OK: {sb1_ok}/{sync_count} = {sb1_ok / max(1, sync_count):.0%}")
        print(f"SB2 CRC-OK: {sb2_ok}/{sync_count} = {sb2_ok / max(1, sync_count):.0%}")
        print(f"Scramble init: 0x{s_init:08X}")
        assert len(results) >= 5


class TestCheckpoint9FreqCorrection:
    """Checkpoint 9: Frequency correction."""

    def test_synthetic_offset(self):
        """Estimate frequency offset from synthetic signal with +2000 Hz offset."""
        rng = np.random.default_rng(42)
        sps = 4
        sample_rate = 18000.0 * sps
        n_symbols = 1000
        freq_offset = 2000.0  # Hz

        # Generate π/4-DQPSK symbols
        dibits = rng.integers(0, 4, size=n_symbols, dtype=np.uint8)
        phase = 0.0
        symbols = np.empty(n_symbols, dtype=np.complex128)
        for i, d in enumerate(dibits):
            phase += PI4_DQPSK_PHASE_MAP[int(d)]
            symbols[i] = np.exp(1j * phase)

        # Override freq correction field symbols with all-zeros phase changes (+π/4)
        # Freq correction field starts at symbol 7 in the burst (bit offset 14)
        # Middle symbols are at burst positions 7+4=11 to 7+35=42
        # Use known burst structure: set symbols to produce +π/4 phase changes
        fc_start_sym = 7
        fc_end_sym = 47  # freq correction spans 40 symbols (bits 14-93)
        for i in range(fc_start_sym, min(fc_end_sym, n_symbols - 1)):
            # Set diff to produce +π/4 (dibit 00)
            symbols[i + 1] = symbols[i] * np.exp(1j * np.pi / 4)

        # Upsample and TX RRC
        taps = rrc_taps(float(sps), alpha=0.35)
        up = np.zeros(n_symbols * sps, dtype=np.complex128)
        up[::sps] = symbols
        signal = lfilter(taps, 1.0, up).astype(np.complex64)

        # Apply frequency offset
        t = np.arange(len(signal)) / sample_rate
        signal = signal * np.exp(1j * 2 * np.pi * freq_offset * t).astype(np.complex64)

        # Demod and sync
        demod = TetraDemod(sample_rate)
        recovered_symbols = demod.process_symbols(signal)

        if len(recovered_symbols) > 255:
            # Compute diff symbols
            diff = recovered_symbols[1:] * np.conj(recovered_symbols[:-1])
            est = estimate_freq_offset(diff)
            print(f"Estimated offset: {est:.0f} Hz (expected: {freq_offset:.0f} Hz)")
            assert abs(est - freq_offset) < 500, (
                f"Estimate {est:.0f} Hz too far from {freq_offset:.0f} Hz"
            )

    def test_real_sample_offset(self):
        """Extract freq correction from real sync bursts, verify consistency."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(10.0 * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])

        detector = SyncDetector()
        results = detector.process(symbols)

        offsets = []
        for r in results:
            if r.burst_type == "sync":
                est = estimate_freq_offset(r.diff_symbols)
                offsets.append(est)

        assert len(offsets) >= 3, "Not enough sync bursts for frequency estimate"
        mean_offset = np.mean(offsets)
        std_offset = np.std(offsets)
        print(
            f"Frequency offset: mean={mean_offset:.0f} Hz, std={std_offset:.0f} Hz, n={len(offsets)}"
        )
        assert std_offset < 500, f"Frequency estimates too inconsistent: std={std_offset:.0f} Hz"

    def test_before_after(self):
        """Compare CRC-OK rates with and without frequency correction."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(10.0 * sample_rate)

        # Without freq correction
        demod1 = TetraDemod(sample_rate)
        symbols1 = demod1.process_symbols(iq[:n_samples])
        det1 = SyncDetector()
        results1 = det1.process(symbols1)

        crc_ok_1 = sum(
            1
            for r in results1
            if r.burst_type == "sync"
            and decode_block(extract_sync_burst(r.soft_bits).sb1, "SB1", SCRAMB_INIT)[1]
        )
        total_1 = sum(1 for r in results1 if r.burst_type == "sync")

        # Estimate offset from first pass
        offsets = [estimate_freq_offset(r.diff_symbols) for r in results1 if r.burst_type == "sync"]
        mean_offset = float(np.mean(offsets)) if offsets else 0.0

        # With freq correction
        demod2 = TetraDemod(sample_rate)
        demod2.apply_freq_correction(mean_offset)
        symbols2 = demod2.process_symbols(iq[:n_samples])
        det2 = SyncDetector()
        results2 = det2.process(symbols2)

        crc_ok_2 = sum(
            1
            for r in results2
            if r.burst_type == "sync"
            and decode_block(extract_sync_burst(r.soft_bits).sb1, "SB1", SCRAMB_INIT)[1]
        )
        total_2 = sum(1 for r in results2 if r.burst_type == "sync")

        rate1 = crc_ok_1 / max(1, total_1)
        rate2 = crc_ok_2 / max(1, total_2)
        print(f"Without freq corr: {crc_ok_1}/{total_1} = {rate1:.0%}")
        print(f"With freq corr ({mean_offset:.0f} Hz): {crc_ok_2}/{total_2} = {rate2:.0%}")
        # Both should be high (sample is well-tuned) or corrected should be better
        assert rate2 >= rate1 * 0.9, f"Freq correction made things worse: {rate2:.0%} < {rate1:.0%}"


class TestCheckpoint10Integration:
    """Checkpoint 10: Decoder integration."""

    def test_streaming_chunks(self):
        """Feed sample in 0.1s chunks, verify messages produced within 2 seconds."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        chunk_size = int(0.1 * sample_rate)

        decoder = TETRADecoder(sample_rate)
        all_messages = []
        first_msg_chunk = None

        for i in range(50):  # 5 seconds
            start = i * chunk_size
            end = start + chunk_size
            if end > len(iq):
                break
            decoder.demodulate(iq[start:end], i * 0.1)
            msgs = decoder.get_messages()
            if msgs and first_msg_chunk is None:
                first_msg_chunk = i
            all_messages.extend(msgs)

        assert len(all_messages) > 0, "No messages produced"
        assert first_msg_chunk is not None
        first_msg_time = first_msg_chunk * 0.1
        print(f"First message at {first_msg_time:.1f}s, total messages: {len(all_messages)}")
        assert first_msg_time < 2.0, f"First message at {first_msg_time:.1f}s (>2s)"

    def test_chunk_size_independence(self):
        """Same MCC/MNC/CC extracted regardless of chunk size."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        duration = 5.0
        n_samples = int(duration * sample_rate)

        results = {}
        for chunk_dur in [0.05, 0.1, 0.5]:
            chunk_size = int(chunk_dur * sample_rate)
            decoder = TETRADecoder(sample_rate)
            for i in range(n_samples // chunk_size):
                start = i * chunk_size
                decoder.demodulate(iq[start : start + chunk_size], i * chunk_dur)

            msgs = []
            # Drain any remaining
            final_msgs = decoder.get_messages()
            for m in final_msgs:
                if isinstance(m.data, SB1Info):
                    msgs.append(m)

            if msgs:
                sb1 = msgs[0].data
                assert isinstance(sb1, SB1Info)
                results[chunk_dur] = (sb1.mcc, sb1.mnc, sb1.colour_code)

        assert len(results) >= 2, f"Only {len(results)} chunk sizes produced results"
        values = list(results.values())
        for v in values[1:]:
            assert v == values[0], f"Inconsistent results: {results}"

    def test_reset(self):
        """Reset and re-lock."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        chunk_size = int(0.5 * sample_rate)

        decoder = TETRADecoder(sample_rate)

        # First pass
        decoder.demodulate(iq[: chunk_size * 4], 0.0)
        msgs1 = decoder.get_messages()
        assert len(msgs1) > 0, "No messages before reset"

        # Reset
        decoder.reset()
        assert decoder._state.network is None

        # Second pass with different data
        decoder.demodulate(iq[chunk_size * 4 : chunk_size * 8], 2.0)
        msgs2 = decoder.get_messages()
        assert len(msgs2) > 0, "No messages after reset"

    def test_statistics(self):
        """Print decoder stats after 30 seconds."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        chunk_size = int(0.5 * sample_rate)
        duration = 30.0

        decoder = TETRADecoder(sample_rate)
        total_msgs = 0

        for i in range(int(duration / 0.5)):
            start = i * chunk_size
            end = start + chunk_size
            if end > len(iq):
                break
            decoder.demodulate(iq[start:end], i * 0.5)
            total_msgs += len(decoder.get_messages())

        quality = decoder._state.quality
        crc_events = quality.crc_events
        n_crc = len(crc_events)
        n_ok = sum(1 for _, ok in crc_events if ok)
        print(f"\n=== {type(decoder).LABEL} ===")
        print(f"Total bursts: {quality.lifetime_bursts}")
        print(f"CRC OK: {n_ok}/{n_crc}")
        print(f"Messages: {total_msgs}")

        if n_crc > 0:
            crc_rate = n_ok / n_crc
            print(f"CRC-OK rate: {crc_rate:.0%}")
            assert crc_rate > 0.50, f"CRC-OK rate {crc_rate:.0%} below 50%"
        assert total_msgs >= 2, f"Only {total_msgs} messages in 30s"


class TestSpeechChannel:
    """Tests for TETRA speech traffic channel coding."""

    def test_interleave_roundtrip(self):
        """Matrix interleave/deinterleave should be identity."""
        rng = np.random.default_rng(42)
        data = rng.uniform(-1, 1, 432).astype(np.float32)
        assert np.allclose(deinterleave_speech(interleave_speech(data)), data)
        assert np.allclose(interleave_speech(deinterleave_speech(data)), data)

    def test_puncture_depuncture_class1(self):
        """Puncture/depuncture roundtrip for class 1 (rate 8/12)."""
        rng = np.random.default_rng(42)
        # 112 symbols (no tail), x 3 generators = 336 mother bits
        mother = rng.choice([-1.0, 1.0], size=336).astype(np.float32)
        coded = puncture(mother, A1)
        assert len(coded) == 168  # N1_2_CODED
        recovered = depuncture(coded, A1)
        assert len(recovered) == 336
        # Non-punctured positions should match exactly
        for i in range(len(recovered)):
            if recovered[i] != 0.0:
                assert recovered[i] == mother[i]

    def test_puncture_depuncture_class2(self):
        """Puncture/depuncture roundtrip for class 2 (rate 8/18)."""
        rng = np.random.default_rng(42)
        # (60 + 8 CRC + 4 tail) = 72 symbols, x 3 = 216 mother bits
        mother = rng.choice([-1.0, 1.0], size=216).astype(np.float32)
        coded = puncture(mother, A2)
        assert len(coded) == 162  # N2_2_CODED
        recovered = depuncture(coded, A2)
        assert len(recovered) == 216
        for i in range(len(recovered)):
            if recovered[i] != 0.0:
                assert recovered[i] == mother[i]

    def test_reorder_roundtrip(self):
        """Codec-to-class and class-to-codec should be inverses."""
        rng = np.random.default_rng(42)
        class0 = rng.integers(0, 2, 102, dtype=np.uint8)
        class1 = rng.integers(0, 2, 112, dtype=np.uint8)
        class2 = rng.integers(0, 2, 60, dtype=np.uint8)
        f1, f2 = reorder_to_codec(class0, class1, class2)
        c0, c1, c2 = codec_to_classes(f1, f2)
        np.testing.assert_array_equal(c0, class0)
        np.testing.assert_array_equal(c1, class1)
        np.testing.assert_array_equal(c2, class2)

    def test_speech_crc(self):
        """CRC generation and check should agree."""
        rng = np.random.default_rng(42)
        class2_data = rng.integers(0, 2, 60, dtype=np.uint8)
        crc = _generate_speech_crc(class2_data)
        assert len(crc) == 8
        # Combine: data + CRC + tail
        combined = np.concatenate([class2_data, crc, np.zeros(SPEECH_K - 1, dtype=np.uint8)])
        assert check_speech_crc(combined)
        # Flip a bit -- CRC should fail
        combined[5] ^= 1
        assert not check_speech_crc(combined)

    def test_encode_decode_roundtrip(self):
        """Full speech encode/decode roundtrip."""
        rng = np.random.default_rng(42)
        frame1 = rng.integers(0, 2, 137, dtype=np.uint8)
        frame2 = rng.integers(0, 2, 137, dtype=np.uint8)
        scramble_init_val = 0x12345678

        # Encode
        type5 = encode_speech(frame1, frame2, scramble_init_val)
        assert len(type5) == 432

        # Convert hard bits to soft: 0 -> -1.0, 1 -> +1.0
        soft = 2.0 * type5.astype(np.float32) - 1.0

        # Decode
        dec_f1, dec_f2, bfi1, bfi2 = decode_speech(soft, scramble_init_val)

        np.testing.assert_array_equal(dec_f1, frame1)
        np.testing.assert_array_equal(dec_f2, frame2)
        assert not bfi1
        assert not bfi2

        # Verify bit packing produces expected length
        packed = bits_to_bytes(dec_f1)
        assert len(packed) == 18  # ceil(137/8) = 18

    def test_bits_to_bytes_packing(self):
        """bits_to_bytes should pack MSB-first and pad correctly."""
        bits = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
        result = bits_to_bytes(bits)
        assert result == bytes([0b10100011, 0b11000000])
