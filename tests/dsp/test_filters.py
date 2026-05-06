from math import gcd

import numpy as np
import scipy.signal

from tsdr.radio.dsp import FMDiscriminator, butter, firwin, lfilter, lfilter_zi, resample_poly
from tsdr.radio.dsp._kernels import (
    StreamingDecimFilter,
    StreamingFilter,
    StreamingPolyphaseResampler,
    _fm_discriminator_f32,
    _freq_shift_f32_to_c64,
    _iq_metrics_c64,
    _lfilter_fir_f32,
    _lfilter_iir,
    _lfilter_iir_f32,
    _sint8_iq_to_complex64,
    _uint8_iq_to_complex64,
    fir_decim_f32_into,
)


class TestFirwin:
    def test_hamming_lowpass(self):
        for n, fc in [(51, 0.3), (128, 0.5), (21, 0.1)]:
            ours = firwin(n, fc)
            ref = scipy.signal.firwin(n, fc)
            np.testing.assert_allclose(ours, ref, atol=1e-12, err_msg=f"n={n} fc={fc}")

    def test_hamming_lowpass_fs(self):
        ours = firwin(64, 5000, fs=48000)
        ref = scipy.signal.firwin(64, 5000, fs=48000)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_nuttall_window(self):
        ours = firwin(71, 2400, fs=19000, window="nuttall")
        ref = scipy.signal.firwin(71, 2400, fs=19000, window="nuttall")
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_kaiser_window(self):
        n_taps = 41
        ours = firwin(n_taps, 1.0 / 5, window=("kaiser", 5.0))
        ref = scipy.signal.firwin(n_taps, 1.0 / 5, window=("kaiser", 5.0))
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_bandpass(self):
        ours = firwin(256, [0.38, 0.42], pass_zero=False)
        ref = scipy.signal.firwin(256, [0.38, 0.42], pass_zero=False)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_bandpass_fs(self):
        ours = firwin(256, [18500, 19500], pass_zero=False, fs=250000)
        ref = scipy.signal.firwin(256, [18500, 19500], pass_zero=False, fs=250000)
        np.testing.assert_allclose(ours, ref, atol=1e-12)


class TestLfilterFIR:
    def test_real_no_state(self):
        taps = firwin(51, 0.3)
        x = np.random.default_rng(42).standard_normal(200).astype(np.float32)
        ours = lfilter(taps, [1.0], x)
        ref = scipy.signal.lfilter(taps, [1.0], x)
        np.testing.assert_allclose(ours, ref, atol=1e-5)

    def test_real_with_state(self):
        taps = firwin(64, 5000, fs=48000).astype(np.float32)
        zi = lfilter_zi(taps, [1.0]).astype(np.float32) * 0.0
        zi_ref = scipy.signal.lfilter_zi(taps, [1.0]).astype(np.float32) * 0.0
        x = np.random.default_rng(7).standard_normal(300).astype(np.float32)

        y_ours, zf_ours = lfilter(taps, [1.0], x, zi=zi)
        y_ref, zf_ref = scipy.signal.lfilter(taps, [1.0], x, zi=zi_ref)
        np.testing.assert_allclose(y_ours, y_ref, atol=1e-5)
        np.testing.assert_allclose(zf_ours, zf_ref, atol=1e-5)

    def test_complex_with_state(self):
        taps = firwin(64, 0.2).astype(np.float32)
        zi = lfilter_zi(taps, [1.0]).astype(np.complex64) * 0.0
        zi_ref = scipy.signal.lfilter_zi(taps, [1.0]).astype(np.complex64) * 0.0
        rng = np.random.default_rng(99)
        x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)

        y_ours, zf_ours = lfilter(taps, [1.0], x, zi=zi)
        y_ref, zf_ref = scipy.signal.lfilter(taps, [1.0], x, zi=zi_ref)
        np.testing.assert_allclose(y_ours, y_ref, atol=1e-4)
        np.testing.assert_allclose(zf_ours, zf_ref, atol=1e-4)

    def test_streaming_continuity(self):
        """Chunked processing must equal single-shot."""
        taps = firwin(33, 0.25)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500).astype(np.float32)

        # Single-shot
        zi0 = np.zeros(len(taps) - 1, dtype=np.float32)
        y_single, _ = lfilter(taps, [1.0], x, zi=zi0.copy())

        # Chunked
        zi = zi0.copy()
        chunks = [x[:100], x[100:350], x[350:]]
        parts = []
        for chunk in chunks:
            y_part, zi = lfilter(taps, [1.0], chunk, zi=zi)
            parts.append(y_part)
        y_chunked = np.concatenate(parts)

        np.testing.assert_allclose(y_chunked, y_single, atol=1e-6)


class TestLfilterIIR:
    def test_deemphasis(self):
        """First-order IIR de-emphasis filter (common in FM demod)."""
        tau = 75e-6
        fs = 48000
        d = 1.0 - np.exp(-1.0 / (fs * tau))
        b_iir = np.array([d])
        a_iir = np.array([1.0, -(1.0 - d)])

        rng = np.random.default_rng(3)
        x = rng.standard_normal(400).astype(np.float32)

        zi = lfilter_zi(b_iir, a_iir)
        zi_ref = scipy.signal.lfilter_zi(b_iir, a_iir)

        y_ours, zf_ours = lfilter(b_iir, a_iir, x, zi=zi)
        y_ref, zf_ref = scipy.signal.lfilter(b_iir, a_iir, x, zi=zi_ref)
        np.testing.assert_allclose(y_ours, y_ref, atol=1e-6)
        np.testing.assert_allclose(zf_ours, zf_ref, atol=1e-6)

    def test_iir_streaming(self):
        """IIR streaming state must be continuous across chunks."""
        tau = 75e-6
        fs = 48000
        d = 1.0 - np.exp(-1.0 / (fs * tau))
        b_iir = np.array([d])
        a_iir = np.array([1.0, -(1.0 - d)])

        rng = np.random.default_rng(5)
        x = rng.standard_normal(600).astype(np.float32)

        zi0 = lfilter_zi(b_iir, a_iir)
        y_single, _ = lfilter(b_iir, a_iir, x, zi=zi0.copy())

        zi = zi0.copy()
        parts = []
        for chunk in [x[:200], x[200:400], x[400:]]:
            y_part, zi = lfilter(b_iir, a_iir, chunk, zi=zi)
            parts.append(y_part)
        y_chunked = np.concatenate(parts)

        np.testing.assert_allclose(y_chunked, y_single, atol=1e-6)


class TestLfilterZi:
    def test_fir(self):
        taps = firwin(51, 0.3)
        ours = lfilter_zi(taps, [1.0])
        ref = scipy.signal.lfilter_zi(taps, [1.0])
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_iir(self):
        tau = 75e-6
        fs = 48000
        d = 1.0 - np.exp(-1.0 / (fs * tau))
        b_iir = np.array([d])
        a_iir = np.array([1.0, -(1.0 - d)])
        ours = lfilter_zi(b_iir, a_iir)
        ref = scipy.signal.lfilter_zi(b_iir, a_iir)
        np.testing.assert_allclose(ours, ref, atol=1e-12)

    def test_butter_coeffs(self):
        b, a = butter(4, [0.1, 0.4], btype="band")
        ours = lfilter_zi(b, a)
        ref = scipy.signal.lfilter_zi(b, a)
        np.testing.assert_allclose(ours, ref, atol=1e-8)


class TestButter:
    def test_bandpass_4th_order(self):
        """Match the exact SSB call site."""
        ours_b, ours_a = butter(4, [0.05, 0.3], btype="band")
        ref_b, ref_a = scipy.signal.butter(4, [0.05, 0.3], btype="band")
        np.testing.assert_allclose(ours_b, ref_b, atol=1e-10)
        np.testing.assert_allclose(ours_a, ref_a, atol=1e-10)

    def test_lowpass(self):
        ours_b, ours_a = butter(4, 0.3)
        ref_b, ref_a = scipy.signal.butter(4, 0.3)
        np.testing.assert_allclose(ours_b, ref_b, atol=1e-10)
        np.testing.assert_allclose(ours_a, ref_a, atol=1e-10)


class TestResamplePoly:
    def test_decimate(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(1000).astype(np.float32)
        ours = resample_poly(x, 1, 5)
        ref = scipy.signal.resample_poly(x, 1, 5)
        # Output lengths should match
        min_len = min(len(ours), len(ref))
        # Allow some edge differences, check bulk
        np.testing.assert_allclose(ours[2 : min_len - 2], ref[2 : min_len - 2], atol=0.1)

    def test_decimate_complex(self):
        rng = np.random.default_rng(2)
        x = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64)
        ours = resample_poly(x, 1, 10)
        ref = scipy.signal.resample_poly(x, 1, 10)
        min_len = min(len(ours), len(ref))
        np.testing.assert_allclose(ours[2 : min_len - 2], ref[2 : min_len - 2], atol=0.1)

    def test_rational_resample(self):
        """Non-trivial up/down (exercised in test_dmr_decoder)."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(500).astype(np.float32)
        ours = resample_poly(x, 24, 125)
        ref = scipy.signal.resample_poly(x, 24, 125)
        min_len = min(len(ours), len(ref))
        np.testing.assert_allclose(ours[5 : min_len - 5], ref[5 : min_len - 5], atol=0.2)

    def test_identity(self):
        x = np.arange(100, dtype=np.float32)
        result = resample_poly(x, 1, 1)
        np.testing.assert_array_equal(result, x)


class TestFMDiscriminatorKernel:
    def test_matches_numpy_reference(self):
        """Kernel output must match the conjugate-product numpy reference."""
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
        scale = 250000.0 / (2.0 * np.pi * 75000.0)

        # Numpy reference: conj-product then atan2
        padded = np.concatenate([[np.complex64(0)], iq])
        product = padded[1:] * np.conj(padded[:-1])
        ref = (np.angle(product) * scale).astype(np.float32)

        out, _, _ = _fm_discriminator_f32(iq, 0.0, 0.0, scale)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_streaming_continuity(self):
        """Chunked processing must equal single-shot."""
        rng = np.random.default_rng(7)
        iq = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64)
        scale = 48000.0 / (2.0 * np.pi * 5000.0)

        out_single, _, _ = _fm_discriminator_f32(iq, 0.0, 0.0, scale)

        # Chunked
        parts = []
        prev_re, prev_im = 0.0, 0.0
        for chunk in [iq[:100], iq[100:350], iq[350:]]:
            out, prev_re, prev_im = _fm_discriminator_f32(chunk, prev_re, prev_im, scale)
            parts.append(out)
        out_chunked = np.concatenate(parts)

        np.testing.assert_allclose(out_chunked, out_single, atol=1e-6)

    def test_fm_discriminator_class(self):
        """FMDiscriminator class must produce same output as kernel."""
        rng = np.random.default_rng(11)
        iq = (rng.standard_normal(300) + 1j * rng.standard_normal(300)).astype(np.complex64)

        disc = FMDiscriminator(sample_rate=250000, deviation=75000)
        result = disc.process(iq)

        scale = 250000.0 / (2.0 * np.pi * 75000.0)
        kernel_out, _, _ = _fm_discriminator_f32(iq, 0.0, 0.0, scale)
        np.testing.assert_allclose(result, kernel_out, atol=1e-5)


# IQ conversion kernels


class TestIQConvert:
    def test_uint8_matches_numpy(self):
        rng = np.random.default_rng(1)
        raw = rng.integers(0, 256, size=2000, dtype=np.uint8)

        # Numpy reference
        pairs = raw.reshape(-1, 2)
        i_ref = (pairs[:, 0].astype(np.float32) - 127.5) / 127.5
        q_ref = (pairs[:, 1].astype(np.float32) - 127.5) / 127.5
        ref = i_ref + 1j * q_ref

        result = _uint8_iq_to_complex64(raw)
        np.testing.assert_allclose(result, ref, atol=1e-6)

    def test_sint8_matches_numpy(self):
        rng = np.random.default_rng(2)
        raw = rng.integers(-128, 128, size=2000, dtype=np.int8)

        # Numpy reference
        pairs = raw.reshape(-1, 2)
        i_ref = pairs[:, 0].astype(np.float32) / 127.0
        q_ref = pairs[:, 1].astype(np.float32) / 127.0
        ref = i_ref + 1j * q_ref

        result = _sint8_iq_to_complex64(raw)
        np.testing.assert_allclose(result, ref, atol=1e-6)

    def test_uint8_edge_values(self):
        """Test boundary values 0 and 255."""
        raw = np.array([0, 0, 255, 255, 128, 128], dtype=np.uint8)
        result = _uint8_iq_to_complex64(raw)
        assert result[0].real < -0.99  # 0 -> ~-1.0
        assert result[1].real > 0.99  # 255 -> ~+1.0
        np.testing.assert_allclose(result[2].real, (128 - 127.5) / 127.5, atol=1e-6)


class TestPolyphaseResampler:
    def test_sine_wave_frequency_preserved(self):
        """A sine wave resampled from 50kHz to 48kHz must preserve frequency."""
        source_rate = 50000
        target_rate = 48000
        freq = 1000.0  # 1 kHz tone
        duration = 0.1  # 100ms

        t = np.arange(int(source_rate * duration)) / source_rate
        mono = np.sin(2 * np.pi * freq * t).astype(np.float32)
        stereo = np.column_stack([mono, mono])

        g = gcd(target_rate, source_rate)
        up, down = target_rate // g, source_rate // g
        n_taps = 2 * 10 * max(up, down) + 1
        resampler = StreamingPolyphaseResampler(up, down, n_taps)
        result = resampler.process(stereo)

        # Check output length is approximately correct
        expected_len = int(len(mono) * target_rate / source_rate)
        assert abs(result.shape[0] - expected_len) <= 2

        # Check the resampled signal still has energy at 1kHz using FFT
        spectrum = np.abs(np.fft.rfft(result[50:-50, 0]))  # skip edges
        freqs = np.fft.rfftfreq(len(result[50:-50, 0]), 1.0 / target_rate)
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - freq) < 50  # within 50 Hz

    def test_streaming_continuity(self):
        """Chunked processing must produce same length and similar output as single-shot."""
        source_rate = 50000
        target_rate = 48000
        g = gcd(target_rate, source_rate)
        up, down = target_rate // g, source_rate // g
        n_taps = 2 * 10 * max(up, down) + 1

        rng = np.random.default_rng(42)
        mono = rng.standard_normal(2000).astype(np.float32)
        stereo = np.column_stack([mono, mono])

        # Single-shot
        r1 = StreamingPolyphaseResampler(up, down, n_taps)
        out_single = r1.process(stereo)

        # Chunked (3 chunks covering all 2000 samples)
        r2 = StreamingPolyphaseResampler(up, down, n_taps)
        parts = []
        for start, end in [(0, 500), (500, 1200), (1200, 2000)]:
            parts.append(r2.process(stereo[start:end]))
        out_chunked = np.concatenate(parts)

        # Total output length must match
        assert out_chunked.shape[0] == out_single.shape[0]
        # Values must match
        np.testing.assert_allclose(out_chunked, out_single, atol=1e-5)


class TestLfilterIIRF32:
    def test_matches_f64_version(self):
        """Float32 IIR kernel matches float64 within tolerance."""
        b = np.array([0.05, 0.0], dtype=np.float64)
        a = np.array([1.0, -0.95], dtype=np.float64)
        x = np.random.randn(2000).astype(np.float32)

        zi_f64 = np.zeros(1, dtype=np.float64)
        y_f64 = _lfilter_iir(b, a, x.astype(np.float64), zi_f64)

        zi_f32 = np.zeros(1, dtype=np.float32)
        y_f32 = _lfilter_iir_f32(b, a, x, zi_f32)

        np.testing.assert_allclose(y_f32, y_f64, atol=1e-4, rtol=1e-3)

    def test_streaming_filter_f32_iir(self):
        """StreamingFilter with dtype=float32 uses float32 IIR path."""
        b = np.array([0.1])
        a = np.array([1.0, -0.9])
        sf = StreamingFilter(b, a, dtype=np.float32)
        x = np.random.randn(500).astype(np.float32)
        y = sf.process(x)
        assert y.dtype == np.float32


# fir_decim_f32_into


class TestFirDecimF32:
    def test_matches_filter_then_slice(self):
        """Decimating FIR matches lfilter + slice."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(12500).astype(np.float32)
        taps = firwin(128, 15000, fs=250000).astype(np.float64)
        m = 5

        # Reference: full filter then slice
        zi = np.zeros(len(taps) - 1, dtype=np.float32)
        y_full = _lfilter_fir_f32(taps, x, zi)
        y_ref = y_full[::m]

        # Decimating kernel
        flipped = np.ascontiguousarray(taps[::-1], dtype=np.float32)
        history = np.zeros(len(taps) - 1, dtype=np.float32)
        padded = np.empty(len(x) + len(taps) - 1, dtype=np.float32)
        y_out = np.empty(len(x) // m + 2, dtype=np.float32)
        n_out, _ = fir_decim_f32_into(x, flipped, m, history, 0, padded, y_out)

        np.testing.assert_allclose(y_out[:n_out], y_ref, atol=1e-5)

    def test_streaming_continuity(self):
        """Chunked processing equals single-shot."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal(10000).astype(np.float32)
        taps = firwin(64, 0.3).astype(np.float64)
        m = 3

        flipped = np.ascontiguousarray(taps[::-1], dtype=np.float32)

        # Single shot
        h1 = np.zeros(len(taps) - 1, dtype=np.float32)
        p1 = np.empty(len(x) + len(taps) - 1, dtype=np.float32)
        y1 = np.empty(len(x) // m + 2, dtype=np.float32)
        n1, _ = fir_decim_f32_into(x, flipped, m, h1, 0, p1, y1)

        # Three chunks
        h2 = np.zeros(len(taps) - 1, dtype=np.float32)
        chunks = [x[:3000], x[3000:7000], x[7000:]]
        parts = []
        phase = 0
        for chunk in chunks:
            p2 = np.empty(len(chunk) + len(taps) - 1, dtype=np.float32)
            y2 = np.empty(len(chunk) // m + 2, dtype=np.float32)
            n, phase = fir_decim_f32_into(chunk, flipped, m, h2, phase, p2, y2)
            parts.append(y2[:n].copy())

        y_chunked = np.concatenate(parts)
        np.testing.assert_allclose(y_chunked, y1[:n1], atol=1e-5)

    def test_m1_equals_nondecimating(self):
        """With m=1, decimating FIR equals standard FIR."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(5000).astype(np.float32)
        taps = firwin(32, 0.4).astype(np.float64)

        zi = np.zeros(len(taps) - 1, dtype=np.float32)
        y_ref = _lfilter_fir_f32(taps, x, zi)

        flipped = np.ascontiguousarray(taps[::-1], dtype=np.float32)
        history = np.zeros(len(taps) - 1, dtype=np.float32)
        padded = np.empty(len(x) + len(taps) - 1, dtype=np.float32)
        y_out = np.empty(len(x) + 2, dtype=np.float32)
        n_out, _ = fir_decim_f32_into(x, flipped, 1, history, 0, padded, y_out)

        np.testing.assert_allclose(y_out[:n_out], y_ref, atol=1e-5)


# StreamingDecimFilter wrapper


class TestStreamingDecimFilter:
    def test_streaming_continuity(self):
        """StreamingDecimFilter gives same result across chunked calls."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000).astype(np.float32)
        taps = firwin(64, 0.3).astype(np.float64)

        f1 = StreamingDecimFilter(taps, decimation=4)
        y_single = f1.process(x).copy()

        f2 = StreamingDecimFilter(taps, decimation=4)
        parts = [
            f2.process(x[:3000]).copy(),
            f2.process(x[3000:7000]).copy(),
            f2.process(x[7000:]).copy(),
        ]
        y_chunked = np.concatenate(parts)

        np.testing.assert_allclose(y_chunked, y_single, atol=1e-5)


# _freq_shift_f32_to_c64


class TestFreqShift:
    def test_matches_numpy(self):
        """Numba freq shift kernel matches numpy reference."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(12000).astype(np.float32)
        carrier_freq = 2 * np.pi * 57000 / 250000
        phase_in = 0.3

        # Numpy reference
        t = np.arange(len(audio))
        carrier_phase = phase_in + carrier_freq * t
        ref = audio * np.exp(-1j * carrier_phase)

        # Kernel
        out = np.empty(len(audio), dtype=np.complex64)
        _freq_shift_f32_to_c64(audio, carrier_freq, phase_in, out)

        np.testing.assert_allclose(out.real, ref.real.astype(np.float32), atol=1e-4)
        np.testing.assert_allclose(out.imag, ref.imag.astype(np.float32), atol=1e-4)


# _iq_metrics_c64


class TestIQMetrics:
    def test_matches_numpy(self):
        """Single-pass IQ metrics match numpy reference."""
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(10000) + 1j * rng.standard_normal(10000)).astype(np.complex64)
        iq *= 0.5  # scale to reasonable range

        rms, peak, clip_pct = _iq_metrics_c64(iq)

        mag = np.abs(iq)
        ref_rms = float(np.sqrt(np.mean(mag**2)))
        ref_peak = float(np.max(mag))
        clipped = (np.abs(iq.real) >= 0.99) | (np.abs(iq.imag) >= 0.99)
        ref_clip = 100.0 * float(np.mean(clipped))

        np.testing.assert_allclose(rms, ref_rms, rtol=1e-5)
        np.testing.assert_allclose(peak, ref_peak, rtol=1e-5)
        np.testing.assert_allclose(clip_pct, ref_clip, atol=1e-6)
