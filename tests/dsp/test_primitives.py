import numpy as np

from tsdr.radio.dsp import CostasLoop, FMDiscriminator, MuellerMuller

# Signal generators


def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at specified SNR."""
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(42)
    if np.iscomplexobj(signal):
        noise = rng.normal(0, np.sqrt(noise_power / 2), len(signal)) + 1j * rng.normal(
            0, np.sqrt(noise_power / 2), len(signal)
        )
    else:
        noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def make_fsk_signal(bits: np.ndarray, sps: float, snr_db: float = 20.0) -> np.ndarray:
    """2-FSK: ±1 repeated sps times, with AWGN."""
    symbols = 2.0 * bits.astype(np.float64) - 1.0
    n_samples = int(len(symbols) * sps) + int(sps)
    signal = np.zeros(n_samples, dtype=np.float64)
    for i, sym in enumerate(symbols):
        start = int(i * sps)
        end = int((i + 1) * sps)
        end = min(end, n_samples)
        signal[start:end] = sym
    return awgn(signal, snr_db)


def make_bpsk_signal(
    bits: np.ndarray,
    sps: float,
    snr_db: float = 20.0,
    phase_offset: float = 0.0,
    freq_offset: float = 0.0,
) -> np.ndarray:
    """Complex BPSK baseband with optional impairments and AWGN."""
    symbols = 2.0 * bits.astype(np.float64) - 1.0
    n_samples = int(len(symbols) * sps) + int(sps)
    signal = np.zeros(n_samples, dtype=np.complex128)
    for i, sym in enumerate(symbols):
        start = int(i * sps)
        end = int((i + 1) * sps)
        end = min(end, n_samples)
        signal[start:end] = sym

    # Apply phase and frequency offset
    t = np.arange(n_samples)
    signal = signal * np.exp(1j * (phase_offset + freq_offset * t))
    return awgn(signal, snr_db)


def make_tone_iq(
    freq: float, sample_rate: float, n_samples: int, snr_db: float = 30.0
) -> np.ndarray:
    """Complex sinusoid with AWGN for FM discriminator tests."""
    t = np.arange(n_samples) / sample_rate
    signal = np.exp(2j * np.pi * freq * t).astype(np.complex64)
    return awgn(signal, snr_db)


def _ber(recovered_bits: np.ndarray, original_bits: np.ndarray) -> float:
    """Compute bit error rate, aligning by cross-correlation."""
    if len(recovered_bits) < 10:
        return 1.0
    # Try several offsets to find best alignment
    best_ber = 1.0
    min_len = min(len(recovered_bits), len(original_bits))
    for offset in range(-5, 6):
        if offset >= 0:
            r = recovered_bits[offset:min_len]
            o = original_bits[: min_len - offset]
        else:
            r = recovered_bits[: min_len + offset]
            o = original_bits[-offset:min_len]
        n = min(len(r), len(o))
        if n < 10:
            continue
        errors = np.sum(r[:n] != o[:n])
        ber = errors / n
        best_ber = min(best_ber, ber)
    return best_ber


# MuellerMuller tests


class TestMuellerMuller:
    def test_real_recovery_clean(self):
        """2-FSK at sps=19.53, no noise, verify BER=0%."""
        rng = np.random.default_rng(123)
        bits = rng.integers(0, 2, size=200)
        sps = 19.53
        signal = make_fsk_signal(bits, sps, snr_db=40.0)
        mm = MuellerMuller(sps)
        symbols = mm.process(signal)
        recovered = (symbols > 0).astype(np.uint8)
        # Skip convergence period
        assert _ber(recovered[20:], bits[20:]) == 0.0

    def test_real_recovery_noisy(self):
        """2-FSK with snr_db=10, verify BER < 5%."""
        rng = np.random.default_rng(456)
        bits = rng.integers(0, 2, size=500)
        sps = 19.53
        signal = make_fsk_signal(bits, sps, snr_db=10.0)
        mm = MuellerMuller(sps)
        symbols = mm.process(signal)
        recovered = (symbols > 0).astype(np.uint8)
        assert _ber(recovered[30:], bits[30:]) < 0.05

    def test_complex_recovery_clean(self):
        """BPSK at sps=8.4, verify symbol signs match."""
        rng = np.random.default_rng(789)
        bits = rng.integers(0, 2, size=200)
        sps = 8.4
        signal = make_bpsk_signal(bits, sps, snr_db=40.0)
        mm = MuellerMuller(sps, gain=0.008)
        symbols = mm.process(signal)
        recovered = (np.real(symbols) > 0).astype(np.uint8)
        assert _ber(recovered[20:], bits[20:]) == 0.0

    def test_complex_recovery_noisy(self):
        """BPSK at snr_db=10, verify BER < 5%."""
        rng = np.random.default_rng(101)
        bits = rng.integers(0, 2, size=500)
        sps = 8.4
        signal = make_bpsk_signal(bits, sps, snr_db=10.0)
        mm = MuellerMuller(sps, gain=0.008)
        symbols = mm.process(signal)
        recovered = (np.real(symbols) > 0).astype(np.uint8)
        assert _ber(recovered[30:], bits[30:]) < 0.05

    def test_chunk_boundary_continuity(self):
        """Process same signal whole vs chunked, assert outputs match."""
        rng = np.random.default_rng(202)
        bits = rng.integers(0, 2, size=300)
        sps = 10.0
        signal = make_fsk_signal(bits, sps, snr_db=40.0)

        # Whole
        mm_whole = MuellerMuller(sps)
        whole_out = mm_whole.process(signal)

        # Chunked with several chunk sizes
        for chunk_size in [256, 512, 1024]:
            mm_chunked = MuellerMuller(sps)
            parts = []
            for i in range(0, len(signal), chunk_size):
                chunk = signal[i : i + chunk_size]
                out = mm_chunked.process(chunk)
                if len(out) > 0:
                    parts.append(out)
            chunked_out = np.concatenate(parts) if parts else np.array([])

            n = min(len(whole_out), len(chunked_out))
            assert n > 0, f"No output for chunk_size={chunk_size}"
            np.testing.assert_allclose(
                whole_out[:n],
                chunked_out[:n],
                atol=1e-6,
                err_msg=f"Mismatch for chunk_size={chunk_size}",
            )

    def test_chunk_boundary_with_noise(self):
        """Chunk boundaries don't amplify errors under noise."""
        rng = np.random.default_rng(303)
        bits = rng.integers(0, 2, size=300)
        sps = 10.0
        signal = make_fsk_signal(bits, sps, snr_db=15.0)

        mm_whole = MuellerMuller(sps)
        whole_out = mm_whole.process(signal)

        for chunk_size in [256, 512]:
            mm_chunked = MuellerMuller(sps)
            parts = []
            for i in range(0, len(signal), chunk_size):
                out = mm_chunked.process(signal[i : i + chunk_size])
                if len(out) > 0:
                    parts.append(out)
            chunked_out = np.concatenate(parts) if parts else np.array([])

            n = min(len(whole_out), len(chunked_out))
            assert n > 0
            np.testing.assert_allclose(
                whole_out[:n],
                chunked_out[:n],
                atol=1e-6,
                err_msg=f"Noise chunk mismatch at chunk_size={chunk_size}",
            )

    def test_fractional_sps(self):
        """Non-integer sps (10.3), verify correct symbol count and no drift."""
        rng = np.random.default_rng(404)
        bits = rng.integers(0, 2, size=200)
        sps = 10.3
        signal = make_fsk_signal(bits, sps, snr_db=40.0)
        mm = MuellerMuller(sps)
        symbols = mm.process(signal)
        # Should produce approximately len(bits) symbols
        expected = len(bits)
        assert abs(len(symbols) - expected) < expected * 0.1

    def test_nudge(self):
        """Verify nudge(0.25) shifts mu by sps/4."""
        sps = 10.0
        mm = MuellerMuller(sps)
        mu_before = mm._mu
        mm.nudge(0.25)
        expected = (mu_before + sps * 0.25) % sps
        assert abs(mm._mu - expected) < 1e-10

    def test_short_input(self):
        """< 32 samples -> empty array."""
        mm = MuellerMuller(10.0)
        result = mm.process(np.zeros(20, dtype=np.float64))
        assert len(result) == 0

    def test_auto_detect_dtype(self):
        """Same class handles both real and complex without constructor flag."""
        rng = np.random.default_rng(505)

        # Real signal
        bits_r = rng.integers(0, 2, size=100)
        signal_r = make_fsk_signal(bits_r, 10.0, snr_db=30.0)
        mm = MuellerMuller(10.0)
        out_r = mm.process(signal_r)
        assert not np.iscomplexobj(out_r)

        # Complex signal (fresh instance)
        bits_c = rng.integers(0, 2, size=100)
        signal_c = make_bpsk_signal(bits_c, 10.0, snr_db=30.0)
        mm2 = MuellerMuller(10.0)
        out_c = mm2.process(signal_c)
        assert np.iscomplexobj(out_c)


# CostasLoop tests


class TestCostasLoop:
    def test_phase_correction_clean(self):
        """BPSK with 45 deg offset -> imag part -> 0 after convergence."""
        rng = np.random.default_rng(606)
        bits = rng.integers(0, 2, size=500)
        symbols = (2.0 * bits - 1.0).astype(np.complex128)
        rotated = symbols * np.exp(1j * np.pi / 4)

        costas = CostasLoop()
        out = costas.process(rotated)

        # After convergence, imaginary part should be near zero
        tail = out[100:]
        assert np.mean(np.abs(np.imag(tail))) < 0.2

    def test_phase_correction_noisy(self):
        """Phase correction at snr_db=15 still converges."""
        rng = np.random.default_rng(707)
        bits = rng.integers(0, 2, size=500)
        symbols = (2.0 * bits - 1.0).astype(np.complex128)
        rotated = symbols * np.exp(1j * np.pi / 4)
        noisy = awgn(rotated, 15.0)

        costas = CostasLoop()
        out = costas.process(noisy)

        tail = out[100:]
        assert np.mean(np.abs(np.imag(tail))) < 0.4

    def test_frequency_offset(self):
        """Small freq offset -> verify tracking."""
        rng = np.random.default_rng(808)
        bits = rng.integers(0, 2, size=1000)
        symbols = (2.0 * bits - 1.0).astype(np.complex128)
        # Small frequency offset: 0.01 rad/symbol
        t = np.arange(len(symbols))
        rotated = symbols * np.exp(1j * 0.01 * t)

        costas = CostasLoop()
        out = costas.process(rotated)

        # After convergence, real part should dominate
        tail = out[200:]
        real_power = np.mean(np.real(tail) ** 2)
        imag_power = np.mean(np.imag(tail) ** 2)
        assert real_power > 5 * imag_power

    def test_chunk_continuity(self):
        """Whole vs chunked -> identical output."""
        rng = np.random.default_rng(909)
        bits = rng.integers(0, 2, size=500)
        symbols = (2.0 * bits - 1.0).astype(np.complex128) * np.exp(1j * 0.3)

        costas_whole = CostasLoop()
        whole_out = costas_whole.process(symbols)

        costas_chunked = CostasLoop()
        parts = []
        chunk_size = 100
        for i in range(0, len(symbols), chunk_size):
            out = costas_chunked.process(symbols[i : i + chunk_size])
            parts.append(out)
        chunked_out = np.concatenate(parts)

        np.testing.assert_allclose(whole_out, chunked_out, atol=1e-10)

    def test_reset(self):
        """Verify reset zeros state."""
        costas = CostasLoop()
        rng = np.random.default_rng(111)
        symbols = rng.normal(size=100) + 1j * rng.normal(size=100)
        costas.process(symbols)
        assert costas.phase != 0.0 or costas.freq != 0.0

        costas.reset()
        assert costas.phase == 0.0
        assert costas.freq == 0.0


# FMDiscriminator tests


class TestFMDiscriminator:
    def test_known_tone_clean(self):
        """Single-freq sinusoid -> constant output = freq/deviation."""
        sample_rate = 32000.0
        deviation = 4800.0
        freq = 2400.0  # Half deviation -> output should be ~0.5
        iq = make_tone_iq(freq, sample_rate, 1000, snr_db=60.0)

        fm = FMDiscriminator(sample_rate, deviation)
        out = fm.process(iq)

        # Skip first sample (transition from zero prev_sample)
        expected = freq / deviation
        assert abs(np.mean(out[10:]) - expected) < 0.05

    def test_known_tone_noisy(self):
        """Same at snr_db=20 -> output mean matches, bounded variance."""
        sample_rate = 32000.0
        deviation = 4800.0
        freq = 2400.0
        iq = make_tone_iq(freq, sample_rate, 2000, snr_db=20.0)

        fm = FMDiscriminator(sample_rate, deviation)
        out = fm.process(iq)

        expected = freq / deviation
        assert abs(np.mean(out[10:]) - expected) < 0.15
        assert np.std(out[10:]) < 0.5

    def test_chunk_continuity(self):
        """Whole vs chunked -> identical output."""
        sample_rate = 32000.0
        deviation = 4800.0
        freq = 1000.0
        iq = make_tone_iq(freq, sample_rate, 2000, snr_db=40.0)

        fm_whole = FMDiscriminator(sample_rate, deviation)
        whole_out = fm_whole.process(iq)

        fm_chunked = FMDiscriminator(sample_rate, deviation)
        parts = []
        chunk_size = 300
        for i in range(0, len(iq), chunk_size):
            out = fm_chunked.process(iq[i : i + chunk_size])
            parts.append(out)
        chunked_out = np.concatenate(parts)

        np.testing.assert_allclose(whole_out, chunked_out, atol=1e-10)

    def test_fsk_roundtrip(self):
        """Generate FSK IQ -> discriminate -> verify ±1 output levels."""
        sample_rate = 32000.0
        deviation = 4800.0
        sps = 20

        rng = np.random.default_rng(222)
        bits = rng.integers(0, 2, size=100)
        freqs = np.where(bits, deviation, -deviation)

        # Generate FM IQ
        phase = np.zeros(len(bits) * sps)
        for i, f in enumerate(freqs):
            phase[i * sps : (i + 1) * sps] = f
        phase_accum = np.cumsum(phase) * 2 * np.pi / sample_rate
        iq = np.exp(1j * phase_accum).astype(np.complex64)

        fm = FMDiscriminator(sample_rate, deviation)
        out = fm.process(iq)

        # Check mid-symbol values are close to ±1
        for i in range(5, len(bits) - 1):
            mid = i * sps + sps // 2
            expected = 1.0 if bits[i] == 1 else -1.0
            assert abs(out[mid] - expected) < 0.3, f"bit {i}: expected {expected}, got {out[mid]}"
