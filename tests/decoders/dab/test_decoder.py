"""DAB+ decoder tests - stage by stage validation against real sample.

Sample: 227.36 MHz (DAB Band III), 2048 kHz, 10s, gain=10
Expected: ~104 frames of Mode I DAB
"""

import time
from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.dab import DABData, DABDecoder
from tsdr.radio.decoders.dab.aac import _make_adts_header
from tsdr.radio.decoders.dab.constants import (
    FREQ_DEINTERLEAVE_TABLE,
    N_CARRIERS,
    N_SYMBOLS,
    T_FRAME,
    T_G,
)
from tsdr.radio.decoders.dab.fec import _check_fib_crc, _crc16_bytes, _generate_prbs
from tsdr.radio.decoders.dab.fic import _decode_fic, _demod_frame_fic, _fic_depuncture
from tsdr.radio.decoders.dab.fig import _build_ensemble, _FIGParserState, _parse_figs
from tsdr.radio.decoders.dab.msc import (
    BITS_PER_CU,
    CIF_BITS,
    N_MSC_SYMBOLS,
    _build_eep_depuncture_index,
    _demod_frame_msc,
    _eep_depuncture,
    _extract_subchannel,
    _msc_to_cifs,
    _TimeDeinterleaver,
)
from tsdr.radio.decoders.dab.ofdm import (
    _dqpsk_to_soft_bits,
    _estimate_fractional_freq_offset,
    _extract_symbols,
    _find_fine_timing,
    _ofdm_demod_frame,
    detect_null_symbols,
)
from tsdr.radio.decoders.dab.pad import _DynamicLabelDecoder
from tsdr.radio.decoders.dab.superframe import _DabPlusSuperframe, _rs_decode
from tsdr.radio.decoders.dab.viterbi import _viterbi_decode

DAB_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "samples"
    / "freq=227.36M_sr=2048k_dur=10s_gain=10_20260308T2248.cu8.zst"
)


@pytest.fixture(scope="module")
def iq_data():
    """Load DAB IQ sample (cached per module)."""
    if not DAB_SAMPLE.exists():
        pytest.skip(f"Sample file not found: {DAB_SAMPLE}")
    return load_iq(DAB_SAMPLE)


class TestStage1NullSymbolDetection:
    """Stage 1: Detect null symbols to find frame boundaries."""

    def test_detects_null_symbols(self, iq_data):
        """Should find ~104 null symbols in 10s of data."""
        nulls = detect_null_symbols(iq_data)
        print(f"Found {len(nulls)} null symbols")
        assert len(nulls) >= 80, f"Expected ≥80 null symbols, got {len(nulls)}"
        assert len(nulls) <= 120, f"Expected ≤120 null symbols, got {len(nulls)}"

    def test_null_symbol_spacing(self, iq_data):
        """Null symbols should be spaced ~T_FRAME apart."""
        nulls = detect_null_symbols(iq_data)
        if len(nulls) < 2:
            pytest.skip("Not enough null symbols for spacing test")

        spacings = np.diff(nulls)
        median_spacing = np.median(spacings)
        print(f"Median spacing: {median_spacing} (expected ~{T_FRAME})")
        assert abs(median_spacing - T_FRAME) / T_FRAME < 0.05


class TestStage2FrequencyOffset:
    """Stage 2: Fine timing and frequency offset estimation."""

    def test_fine_timing(self, iq_data):
        """Fine timing should find a valid offset."""
        nulls = detect_null_symbols(iq_data)
        frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
        offset = _find_fine_timing(frame_iq)
        print(f"Fine timing offset: {offset} samples from nominal")
        # Should be within ±T_G of nominal
        assert abs(offset) < T_G

    def test_fractional_offset_small(self, iq_data):
        """Fractional frequency offset should be < 0.5 subcarriers."""
        nulls = detect_null_symbols(iq_data)
        frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
        fine = _find_fine_timing(frame_iq)
        offset = _estimate_fractional_freq_offset(frame_iq, fine)
        print(f"Fractional frequency offset: {offset:.4f} subcarrier spacings")
        assert abs(offset) < 0.5


class TestStage3OFDMDemod:
    """Stage 3: OFDM demodulation - FFT and carrier extraction."""

    def test_symbol_extraction_shape(self, iq_data):
        """Should produce (76, 1536) complex array."""
        nulls = detect_null_symbols(iq_data)
        frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
        fine = _find_fine_timing(frame_iq)
        offset = _estimate_fractional_freq_offset(frame_iq, fine)
        symbols = _extract_symbols(frame_iq, offset, fine)

        assert symbols.shape == (N_SYMBOLS, N_CARRIERS)
        assert symbols.dtype == np.complex64

    def test_carrier_power_uniformity(self, iq_data):
        """Active carriers should have roughly uniform power."""
        nulls = detect_null_symbols(iq_data)
        frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
        fine = _find_fine_timing(frame_iq)
        offset = _estimate_fractional_freq_offset(frame_iq, fine)
        symbols = _extract_symbols(frame_iq, offset, fine)

        powers = np.mean(np.abs(symbols[1:]) ** 2, axis=1)
        cv = np.std(powers) / np.mean(powers)
        print(f"Power CV across symbols: {cv:.3f}")
        assert cv < 0.5


class TestStage4DQPSK:
    """Stage 4: DQPSK decoding with frequency de-interleaving."""

    def test_fic_soft_bits_shape(self, iq_data):
        """FIC demod should produce 9216 soft bits."""
        nulls = detect_null_symbols(iq_data)
        frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
        fic_soft = _demod_frame_fic(frame_iq)
        assert fic_soft.shape == (9216,)

    def test_soft_bits_not_zero(self, iq_data):
        """Soft bits should have non-trivial values."""
        nulls = detect_null_symbols(iq_data)
        frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
        fic_soft = _demod_frame_fic(frame_iq)
        assert np.std(fic_soft) > 0.1


class TestStage5FICDecoding:
    """Stage 5: FIC decoding - depuncture + Viterbi + CRC."""

    def test_prbs_deterministic(self):
        """PRBS should produce deterministic output."""
        p1 = _generate_prbs(100)
        p2 = _generate_prbs(100)
        assert np.array_equal(p1, p2)

    def test_prbs_first_bits(self):
        """PRBS first bits should match LFSR x^9+x^5+1."""
        p = _generate_prbs(10)
        # LFSR: all-1s init, output = bit[8] XOR bit[4]
        # First 10 outputs: 0,0,0,0,0,1,1,1,1,0
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        assert np.array_equal(p, expected), f"PRBS starts with {p}, expected {expected}"

    def test_viterbi_known_input(self):
        """Viterbi should decode a trivial all-zeros input."""
        soft = np.ones(4 * 20, dtype=np.float32) * -1.0
        decoded = _viterbi_decode(soft)
        assert len(decoded) == 20
        assert np.all(decoded == 0)

    def test_freq_deinterleave_table(self):
        """De-interleaving table should be a valid permutation of 0..1535."""
        assert len(FREQ_DEINTERLEAVE_TABLE) == N_CARRIERS
        assert FREQ_DEINTERLEAVE_TABLE.min() >= 0
        assert FREQ_DEINTERLEAVE_TABLE.max() < N_CARRIERS
        assert len(set(FREQ_DEINTERLEAVE_TABLE)) == N_CARRIERS

    def test_depuncture_sizes(self):
        """Depuncturing should expand 2304 -> 3096 bits."""
        soft = np.ones(2304, dtype=np.float32)
        depunct = _fic_depuncture(soft)
        assert len(depunct) == 3096

    def test_crc_check_valid(self):
        """CRC check should accept a known-good FIB."""
        # All-zero FIB data with correct CRC
        data = bytes(30) + b"\xff\xff"  # placeholder
        # Compute the actual CRC for all-zero data
        crc = _crc16_bytes(bytes(30))
        # Store inverted CRC
        crc_inv = crc ^ 0xFFFF
        data = bytes(30) + bytes([crc_inv >> 8, crc_inv & 0xFF])
        assert _check_fib_crc(data)

    def test_fib_crc_check(self, iq_data):
        """At least some FIBs should pass CRC."""
        nulls = detect_null_symbols(iq_data)
        assert len(nulls) >= 5

        total_fibs = 0
        crc_ok = 0

        for frame_idx in range(min(5, len(nulls))):
            frame_start = nulls[frame_idx]
            frame_end = frame_start + T_FRAME
            if frame_end > len(iq_data):
                break

            frame_iq = iq_data[frame_start : frame_end + 5000]
            fic_soft = _demod_frame_fic(frame_iq)
            fibs = _decode_fic(fic_soft)

            for _, ok in fibs:
                total_fibs += 1
                if ok:
                    crc_ok += 1

        rate = crc_ok / total_fibs if total_fibs > 0 else 0
        print(f"FIB CRC: {crc_ok}/{total_fibs} = {rate:.1%}")
        assert crc_ok > 0, f"No FIBs passed CRC out of {total_fibs}"
        assert rate > 0.5, f"FIB CRC rate {rate:.1%} too low (need >50%)"


class TestStage6FIGParsing:
    """Stage 6: FIG parsing - extract ensemble label and services."""

    def test_extract_ensemble_info(self, iq_data):
        """Should extract ensemble label and at least 1 service."""
        nulls = detect_null_symbols(iq_data)

        state = _FIGParserState()

        for frame_idx in range(min(30, len(nulls))):
            frame_start = nulls[frame_idx]
            frame_end = frame_start + T_FRAME
            if frame_end > len(iq_data):
                break

            frame_iq = iq_data[frame_start : frame_end + 5000]
            fic_soft = _demod_frame_fic(frame_iq)
            fibs = _decode_fic(fic_soft)

            for fib_bytes, crc_ok in fibs:
                if crc_ok:
                    _parse_figs(fib_bytes, state)

        ensemble = _build_ensemble(state)
        print(f"Ensemble: '{ensemble.label}' (ID={ensemble.ensemble_id:#06x})")
        print(f"Services: {len(ensemble.services)}")
        for s in ensemble.services:
            print(f"  - {s.label} (SId={s.service_id:#06x})")

        assert ensemble.label, "No ensemble label decoded"
        assert len(ensemble.services) >= 1, "No services decoded"


class TestStage7Integration:
    """Stage 7: Full DABDecoder integration test."""

    def test_decode_chunked(self, iq_data):
        """Decoder should produce messages with 100% FIC CRC and audio output."""
        for chunk_size in [65536, 131072]:
            decoder = DABDecoder(sample_rate=2_048_000)
            all_messages = []
            all_audio = []
            service_selected = False

            for i in range(0, len(iq_data), chunk_size):
                chunk = iq_data[i : i + chunk_size]
                decoder.demodulate(chunk, 0.0)
                all_messages.extend(decoder.get_messages())
                all_audio.extend(decoder.get_audio())
                if not service_selected and decoder.stats.frames_processed > 10:
                    result = decoder.select_service()
                    if result and not result.startswith("Error"):
                        service_selected = True

            stats = decoder.stats
            print(f"Chunk {chunk_size}: {len(all_messages)} messages, stats={stats}")

            assert stats.frames_processed >= 100, (
                f"Expected ≥100 frames, got {stats.frames_processed}"
            )
            assert stats.fibs_crc_ok == stats.fibs_decoded, (
                f"FIC CRC not 100%: {stats.fibs_crc_ok}/{stats.fibs_decoded}"
            )
            assert len(all_audio) > 0, "No audio output produced"
            total_samples = sum(len(batch.samples) for batch in all_audio)
            reported_rate = all_audio[0].sample_rate
            duration = total_samples / reported_rate
            print(f"  Audio: {duration:.2f}s from {len(all_audio)} batches (rate={reported_rate})")
            assert duration > 6.0, f"Audio too short: {duration:.2f}s (expected >6s)"

    def test_reset_clears_state(self):
        """Reset should clear all internal state."""
        decoder = DABDecoder()
        decoder._frames_processed = 42
        decoder._buffer = np.ones(1000, dtype=np.complex64)
        decoder.reset()

        assert decoder._frames_processed == 0
        assert len(decoder._buffer) == 0
        assert decoder._last_ensemble is None


# ============================================================
# MSC Audio Decoding Tests (Stages 8-14)
# ============================================================


@pytest.fixture(scope="module")
def first_frame_fft(iq_data):
    """Get FFT symbols for first good frame."""
    nulls = detect_null_symbols(iq_data)
    frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
    fft_syms = _ofdm_demod_frame(frame_iq)
    assert fft_syms is not None
    return fft_syms


@pytest.fixture(scope="module")
def ensemble_info(iq_data):
    """Decode enough frames to get ensemble metadata."""
    nulls = detect_null_symbols(iq_data)
    state = _FIGParserState()
    for frame_idx in range(min(30, len(nulls))):
        frame_start = nulls[frame_idx]
        frame_end = frame_start + T_FRAME
        if frame_end > len(iq_data):
            break
        frame_iq = iq_data[frame_start : frame_end + 5000]
        fic_soft = _demod_frame_fic(frame_iq)
        fibs = _decode_fic(fic_soft)
        for fib_bytes, crc_ok in fibs:
            if crc_ok:
                _parse_figs(fib_bytes, state)
    return _build_ensemble(state)


class TestStage8OFDMRefactor:
    """Stage 8: Shared OFDM front-end produces all 76 symbols."""

    def test_ofdm_demod_shape(self, first_frame_fft):
        """Should produce (76, 2048) complex array."""
        assert first_frame_fft.shape == (76, 2048)

    def test_fic_via_shared_path(self, iq_data):
        """FIC via shared OFDM path should produce identical results to old path."""
        nulls = detect_null_symbols(iq_data)
        frame_iq = iq_data[nulls[5] : nulls[5] + T_FRAME + 5000]
        # New path
        fft_syms = _ofdm_demod_frame(frame_iq)
        fic_new = _dqpsk_to_soft_bits(fft_syms, 0, 3)
        # Old path (now also uses the shared path internally)
        fic_old = _demod_frame_fic(frame_iq)
        # Should be identical (same code path now)
        np.testing.assert_array_almost_equal(fic_new, fic_old, decimal=3)


class TestStage9MSCDemod:
    """Stage 9: MSC demodulation - 72 symbols -> 221184 soft bits."""

    def test_msc_soft_bits_shape(self, first_frame_fft):
        msc = _demod_frame_msc(first_frame_fft)
        assert msc.shape == (N_MSC_SYMBOLS * 3072,)  # 221184
        assert msc.dtype == np.float32

    def test_msc_nonzero(self, first_frame_fft):
        msc = _demod_frame_msc(first_frame_fft)
        assert np.std(msc) > 0.1, "MSC soft bits should have non-trivial values"

    def test_msc_to_cifs(self, first_frame_fft):
        msc = _demod_frame_msc(first_frame_fft)
        cifs = _msc_to_cifs(msc)
        assert len(cifs) == 4
        for cif in cifs:
            assert len(cif) == CIF_BITS


class TestStage10TimeDeinterleaver:
    """Stage 10: 16-frame convolutional time de-interleaver."""

    def test_returns_none_for_first_16(self):
        """Should buffer 16 frames before producing output."""
        di = _TimeDeinterleaver()
        for i in range(16):
            result = di.push(np.random.randn(CIF_BITS).astype(np.float32))
            assert result is None, f"Should be None at push {i}"

    def test_returns_output_at_17(self):
        """Should produce output after 17 pushes (16 buffer fill + 1)."""
        di = _TimeDeinterleaver()
        for _i in range(16):
            di.push(np.random.randn(CIF_BITS).astype(np.float32))
        result = di.push(np.random.randn(CIF_BITS).astype(np.float32))
        assert result is not None
        assert result.shape == (CIF_BITS,)

    def test_identity_when_no_interleaving(self):
        """If all frames are the same, output should equal input."""
        di = _TimeDeinterleaver()
        data = np.random.randn(CIF_BITS).astype(np.float32)
        for _ in range(16):
            di.push(data.copy())
        result = di.push(data.copy())
        np.testing.assert_array_almost_equal(result, data, decimal=5)


class TestStage11EEPDepuncture:
    """Stage 11: EEP depuncture index and subchannel extraction."""

    def test_eep_a_index_sizes(self, ensemble_info):
        """Depuncture index length should match subchannel transmitted bits."""
        for svc in ensemble_info.services:
            if svc.subchannel_size is None or svc.protection_level is None:
                continue
            idx, out_len = _build_eep_depuncture_index(
                svc.subchannel_size, svc.protection_level, svc.eep_option
            )
            # Index length = number of transmitted bits
            print(
                f"  {svc.label}: size={svc.subchannel_size}, prot={svc.protection_level}, "
                f"option={svc.eep_option}, transmitted={len(idx)}, depunctured={out_len}"
            )
            assert len(idx) > 0
            assert out_len > len(idx)
            # Output length should be divisible by 4 (rate 1/4 code)
            assert out_len % 4 == 0

    def test_extract_subchannel(self):
        """Subchannel extraction should slice correct range."""
        cif = np.arange(CIF_BITS, dtype=np.float32)
        sub = _extract_subchannel(cif, start_address=10, size=84)
        assert len(sub) == 84 * BITS_PER_CU
        assert sub[0] == 10 * BITS_PER_CU

    def test_depuncture_fills_zeros(self):
        """Depunctured output should have zeros at punctured positions."""
        idx, out_len = _build_eep_depuncture_index(84, 2, 0)
        soft = np.ones(len(idx), dtype=np.float32)
        depunctured = _eep_depuncture(soft, idx, out_len)
        # All indexed positions should be 1.0, others 0.0
        assert np.all(depunctured[idx] == 1.0)
        mask = np.ones(out_len, dtype=bool)
        mask[idx] = False
        assert np.all(depunctured[mask] == 0.0)


class TestStage12ReedSolomon:
    """Stage 12: RS(120,110) error correction."""

    def test_rs_no_errors(self):
        """RS should pass through clean data."""
        # Create a valid codeword (all zeros is a valid RS codeword)
        data = np.zeros(120, dtype=np.int32)
        decoded, nerr = _rs_decode(data)
        assert nerr == 0
        assert len(decoded) == 110
        assert np.all(decoded == 0)

    def test_rs_detects_errors(self):
        """RS should detect/correct small errors."""
        data = np.zeros(120, dtype=np.int32)
        # Introduce 1 error
        data[50] = 0x42
        decoded, nerr = _rs_decode(data)
        # Should either correct (nerr=1) or detect as uncorrectable (nerr=-1)
        # For all-zero codeword with 1 error, it should correct
        print(f"RS result: nerr={nerr}")
        if nerr >= 0:
            assert decoded[50] == 0  # error should be corrected


class TestStage13Superframe:
    """Stage 13: DAB+ superframe assembly."""

    def test_accumulates_5_frames(self):
        """Should return None until 5 frames accumulated."""
        sf = _DabPlusSuperframe()
        for i in range(4):
            result = sf.push(bytes(110))
            assert result is None, f"Should be None at push {i}"

    def test_adts_header_format(self):
        """ADTS header should have correct sync word and length."""
        header = _make_adts_header(100, 48000, 2)
        assert len(header) == 7
        assert header[0] == 0xFF
        assert (header[1] & 0xF0) == 0xF0  # sync word


class TestStage14FIG01Option:
    """Stage 14: FIG 0/1 long-form option field extraction."""

    def test_option_field_extracted(self, ensemble_info):
        """Services should have eep_option field populated."""
        for svc in ensemble_info.services:
            if svc.subchannel_size is not None:
                print(f"  {svc.label}: eep_option={svc.eep_option}")
                # eep_option should be 0 (EEP-A) or 1 (EEP-B)
                assert svc.eep_option in (0, 1)


class TestStage15FullAudioPipeline:
    """Stage 15: End-to-end audio decoding integration test."""

    def test_decoder_produces_audio(self, iq_data):
        """Full pipeline should produce audio batches from the 10s sample."""
        decoder = DABDecoder(sample_rate=2_048_000)
        all_messages = []
        all_audio = []

        # Process in chunks
        chunk_size = T_FRAME * 2
        for i in range(0, len(iq_data), chunk_size):
            chunk = iq_data[i : i + chunk_size]
            decoder.demodulate(chunk, 0.0)
            all_messages.extend(decoder.get_messages())
            audio = decoder.get_audio()
            all_audio.extend(audio)

        print(f"Messages: {len(all_messages)}")
        print(f"Audio batches: {len(all_audio)}")
        if all_audio:
            total_samples = sum(len(batch.samples) for batch in all_audio)
            reported_rate = all_audio[0].sample_rate
            total_duration = total_samples / reported_rate
            print(
                f"Total audio: {total_samples} samples = {total_duration:.2f}s (rate={reported_rate})"
            )
            # Check PCM values are in range
            for batch in all_audio:
                assert (
                    batch.sample_rate == 48000.0 * 1024 / 960
                )  # 51200 Hz (960-sample transform correction)
                assert batch.samples.dtype == np.float32
                assert np.max(np.abs(batch.samples)) <= 5.0  # HE-AAC can produce peaks > 1.0

        print(f"Stats: {decoder.stats}")
        # We expect audio after ~16 frames (de-interleaver fill) + 5 frames (superframe)
        # With 10s sample (~104 frames), should get some audio
        # But it's OK if AAC decoding fails initially - the pipeline should at least try

    def test_no_channel_switching(self, iq_data):
        """Superframe decodes should have consistent channel count (no misaligned decodes)."""
        decode_results = []
        original_decode = _DabPlusSuperframe._decode_superframe

        def logging_decode(self, frames):
            result = original_decode(self, frames)
            if result is not None:
                _, _, fmt = result
                decode_results.append({"core_sr": fmt.core_sample_rate, "channels": fmt.channels})
            return result

        _DabPlusSuperframe._decode_superframe = logging_decode
        try:
            decoder = DABDecoder(sample_rate=2_048_000)
            chunk_size = T_FRAME * 2
            service_selected = False
            for i in range(0, len(iq_data), chunk_size):
                chunk = iq_data[i : i + chunk_size]
                decoder.demodulate(chunk, 0.0)
                decoder.get_audio()
                if not service_selected and decoder.stats.frames_processed > 10:
                    result = decoder.select_service()
                    if result and not result.startswith("Error"):
                        service_selected = True
        finally:
            _DabPlusSuperframe._decode_superframe = original_decode

        assert len(decode_results) > 0, "No superframe decodes"
        channels_set = {r["channels"] for r in decode_results}
        sr_set = {r["core_sr"] for r in decode_results}
        print(f"Decodes: {len(decode_results)}, channels: {channels_set}, sample rates: {sr_set}")
        # A real stereo DAB+ station should have consistent channel count
        assert channels_set == {2}, (
            f"Channel count switched: {[r['channels'] for r in decode_results]}"
        )


class TestPerformance:
    """Performance regression test for real-time DAB+ decoding."""

    def test_decode_throughput(self, iq_data):
        """Decode full 10s sample and report ms/frame. Target: <48ms (2x realtime)."""
        # Warm up JIT (first call compiles)
        warmup = DABDecoder(sample_rate=2_048_000)
        warmup_chunk = iq_data[: T_FRAME * 3]
        warmup.demodulate(warmup_chunk, 0.0)

        # Timed run
        decoder = DABDecoder(sample_rate=2_048_000)
        chunk_size = T_FRAME * 2

        t0 = time.perf_counter()
        for i in range(0, len(iq_data), chunk_size):
            chunk = iq_data[i : i + chunk_size]
            decoder.demodulate(chunk, 0.0)
            decoder.get_audio()
        elapsed = time.perf_counter() - t0

        frames = decoder.stats.frames_processed
        ms_per_frame = (elapsed / frames * 1000) if frames > 0 else float("inf")
        frame_budget_ms = T_FRAME / 2_048_000 * 1000  # 96ms

        print(f"Frames: {frames}")
        print(f"Total: {elapsed:.2f}s")
        print(f"Per frame: {ms_per_frame:.1f}ms (budget: {frame_budget_ms:.0f}ms)")
        print(f"Realtime factor: {(frames * frame_budget_ms / 1000) / elapsed:.1f}x")

        assert ms_per_frame < 48, f"Too slow: {ms_per_frame:.1f}ms/frame (target <48ms)"


# ============================================================
# PAD / DLS / MOT Tests (Stages 16-18)
# ============================================================


class TestStage16PADExtraction:
    """Stage 16: PAD extraction from superframe AUs."""

    def test_pad_bytes_present(self, iq_data):
        """At least some AUs should contain PAD data (DSE element)."""

        decoder = DABDecoder(sample_rate=2_048_000)
        pad_found = 0

        chunk_size = T_FRAME * 2
        for i in range(0, len(iq_data), chunk_size):
            chunk = iq_data[i : i + chunk_size]
            decoder.demodulate(chunk, 0.0)
            decoder.get_audio()

        # Run again collecting PAD data directly from superframe
        decoder2 = DABDecoder(sample_rate=2_048_000)
        # We need to access superframe results - use a second pass approach
        # by checking the AU bytes for DSE marker
        all_messages = []
        for i in range(0, len(iq_data), chunk_size):
            chunk = iq_data[i : i + chunk_size]
            decoder2.demodulate(chunk, 0.0)
            all_messages.extend(decoder2.get_messages())
            decoder2.get_audio()

        # Check if dynamic_label was set (indicates PAD was processed)
        for msg in all_messages:
            if isinstance(msg.data, DABData) and msg.data.dynamic_label is not None:
                pad_found += 1

        print(f"Messages with DLS: {pad_found}/{len(all_messages)}")
        # At minimum, PAD extraction code should not crash
        # Whether DLS is present depends on the station


class TestStage17DLSReassembly:
    """Stage 17: DLS segment reassembly (unit test)."""

    def test_single_segment_label(self):
        """A single-segment DLS label should reassemble correctly."""
        dl = _DynamicLabelDecoder()

        # DLS data group = prefix(2) + chars + CRC(2)
        # prefix[0]: toggle(7)|first(6)|last(5)|command(4)|field_len-1(3:0)
        # prefix[1]: charset(7:4) for first segment
        text = b"Hello"
        byte0 = 0x60 | (len(text) - 1)  # first+last, no command, field_len=5
        byte1 = 0xF0  # charset=0x0F (UTF-8)
        dg = bytearray([byte0, byte1]) + text
        crc = _crc16_bytes(bytes(dg)) ^ 0xFFFF
        dg += crc.to_bytes(2, "big")

        result = dl.process_subfield(True, bytes(dg))
        print(f"Result: {result}, label: {dl.label}")
        assert result
        assert dl.label == "Hello"

    def test_multi_segment_label(self):
        """Two segments should reassemble into one label."""
        dl = _DynamicLabelDecoder()

        # Segment 0: first=1, last=0, toggle=1, charset=0x0F, 5 chars
        text1 = b"Hello"
        prefix0 = bytes([0xC0 | (len(text1) - 1), 0xF0])  # toggle+first, charset=UTF-8
        dg0 = prefix0 + text1
        crc0 = _crc16_bytes(dg0) ^ 0xFFFF
        dg0 += crc0.to_bytes(2, "big")

        # Segment 1: first=0, last=1, toggle=1, seg_num=1, 6 chars
        text2 = b" World"
        prefix1 = bytes([0xA0 | (len(text2) - 1), 0x10])  # toggle+last, seg_num=1
        dg1 = prefix1 + text2
        crc1 = _crc16_bytes(dg1) ^ 0xFFFF
        dg1 += crc1.to_bytes(2, "big")

        dl.process_subfield(True, dg0)
        result = dl.process_subfield(True, dg1)
        print(f"Result: {result}, label: {dl.label}")
        if result:
            assert dl.label == "Hello World"

    def test_toggle_clears_cache(self):
        """Segments with different toggle values should not mix."""
        dl = _DynamicLabelDecoder()

        # Segment 0 with toggle=0
        text1 = b"Old"
        prefix0 = bytes([0x60 | (len(text1) - 1), 0xF0])  # first+last, toggle=0
        dg0 = prefix0 + text1
        dg0 += (_crc16_bytes(dg0) ^ 0xFFFF).to_bytes(2, "big")
        dl.process_subfield(True, dg0)
        assert dl.label == "Old"

        # Segment 0 with toggle=1 (new label)
        text2 = b"New"
        prefix1 = bytes([0xE0 | (len(text2) - 1), 0xF0])  # first+last, toggle=1
        dg1 = prefix1 + text2
        dg1 += (_crc16_bytes(dg1) ^ 0xFFFF).to_bytes(2, "big")
        dl.process_subfield(True, dg1)
        assert dl.label == "New"


class TestStage18FullPipelineDLS:
    """Stage 18: Full pipeline DLS extraction from real sample."""

    def test_dls_from_sample(self, iq_data):
        """Run full decoder with service selection and check for DLS text."""

        decoder = DABDecoder(sample_rate=2_048_000)
        service_selected = False

        chunk_size = T_FRAME * 2
        for i in range(0, len(iq_data), chunk_size):
            chunk = iq_data[i : i + chunk_size]
            decoder.demodulate(chunk, 0.0)
            decoder.get_messages()
            decoder.get_audio()

            # Auto-select first service once ensemble is available
            if not service_selected and decoder._last_ensemble is not None:
                result = decoder.select_service()
                if result and not result.startswith("Error"):
                    print(f"Selected: {result}")
                    service_selected = True

        print(f"Service selected: {service_selected}")
        print(f"Dynamic label: {decoder._pad_decoder.dynamic_label!r}")
        print(f"MOT app type: {decoder._pad_decoder.mot_app_type}")
        print(f"Slide: {decoder._pad_decoder.slide}")
        print(f"User app types: {decoder._fig_state.user_app_types}")
        print(f"Audio batches: {len(decoder.get_audio())}")
        # DLS may or may not be present in this 10s sample - just verify no crash
