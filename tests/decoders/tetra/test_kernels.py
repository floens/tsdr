"""Bit-parity tests for numba kernels in src/tsdr/radio/decoders/tetra/_kernels.py.

Every kernel swap must produce output that matches the reference implementation
bit-for-bit (integer) or within a tiny tolerance (float). These tests are the
safety net for streaming-state bugs: "one long call" must equal "many small
calls".
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.decoders.tetra._kernels import (
    bits_to_uint as kernel_bits_to_uint,
)
from tsdr.radio.decoders.tetra._kernels import (
    crc16_ccitt as kernel_crc16_ccitt,
)
from tsdr.radio.decoders.tetra._kernels import (
    deinterleave as kernel_deinterleave,
)
from tsdr.radio.decoders.tetra._kernels import (
    depuncture_2_3 as kernel_depuncture_2_3,
)
from tsdr.radio.decoders.tetra._kernels import (
    descramble_soft as kernel_descramble_soft,
)
from tsdr.radio.decoders.tetra._kernels import (
    fir_decim_c64,
    fir_filter_c64,
)
from tsdr.radio.decoders.tetra._kernels import (
    generate_scramble_bits as kernel_generate_scramble_bits,
)
from tsdr.radio.decoders.tetra.channel import (
    crc16_ccitt as ref_crc16_ccitt,
)
from tsdr.radio.decoders.tetra.channel import (
    deinterleave as ref_deinterleave,
)
from tsdr.radio.decoders.tetra.channel import (
    depuncture_2_3 as ref_depuncture_2_3,
)
from tsdr.radio.dsp import firwin, lfilter
from tsdr.radio.dsp._kernels import apply_freq_shift_c64
from tsdr.radio.dsp.mm import _mm_process_complex


def ref_bits_to_uint(bits: np.ndarray, start: int, length: int) -> int:
    """Pure-Python reference for the numba bits_to_uint kernel."""
    val = 0
    for i in range(length):
        val = (val << 1) | int(bits[start + i] & 1)
    return val


def _random_complex64(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(n).astype(np.float32)
    im = rng.standard_normal(n).astype(np.float32)
    return (re + 1j * im).astype(np.complex64)


def test_fir_filter_matches_scipy_lfilter_one_shot():
    """Single call from zero state: output must match scipy.lfilter within 1e-5.

    Note: the kernel uses direct form I (raw sample history as state) while
    scipy uses direct form II transposed (zi). They produce identical outputs
    but the state representation differs, so only the output is compared here.
    The streaming test below verifies state correctness across chunk boundaries.
    """
    n_taps = 41
    taps = firwin(n_taps, 0.1).astype(np.float32)
    x = _random_complex64(5000, seed=1)
    history = np.zeros(n_taps - 1, dtype=np.complex64)

    y_ref, _ = lfilter(taps, 1.0, x, zi=np.zeros(n_taps - 1))
    y_kernel, new_history = fir_filter_c64(x, taps, history)

    assert y_kernel.dtype == np.complex64
    assert new_history.dtype == np.complex64
    assert new_history.shape == (n_taps - 1,)
    assert y_kernel.shape == y_ref.shape
    np.testing.assert_allclose(
        y_kernel.astype(np.complex128), y_ref.astype(np.complex128), atol=1e-5, rtol=1e-5
    )
    # New history should be the last K-1 samples of x (direct form I state).
    np.testing.assert_array_equal(new_history, x[-(n_taps - 1) :])


def test_fir_filter_streaming_equals_single_shot():
    """Chunked calls must produce the same total output as one big call."""
    n_taps = 41
    taps = firwin(n_taps, 0.1).astype(np.float32)
    x = _random_complex64(10000, seed=2)
    history0 = np.zeros(n_taps - 1, dtype=np.complex64)

    # Single shot reference
    y_whole, _ = fir_filter_c64(x, taps, history0)

    # Chunked: feed in irregular chunk sizes, including some smaller than K-1
    chunks = [0, 513, 1024, 1, 2000, 777, 5685]  # sum = 10000
    assert sum(chunks) == 10000
    history = history0.copy()
    y_pieces: list[np.ndarray] = []
    start = 0
    for size in chunks:
        piece, history = fir_filter_c64(x[start : start + size], taps, history)
        y_pieces.append(piece)
        start += size
    y_streamed = np.concatenate(y_pieces)

    np.testing.assert_array_equal(y_whole, y_streamed)


def test_fir_filter_streaming_with_tiny_chunks():
    """Chunks smaller than K-1 must still maintain correct history across calls."""
    n_taps = 41
    taps = firwin(n_taps, 0.1).astype(np.float32)
    x = _random_complex64(500, seed=5)
    history0 = np.zeros(n_taps - 1, dtype=np.complex64)

    y_whole, _ = fir_filter_c64(x, taps, history0)

    # Feed one sample at a time to hit the "chunk smaller than history" branch.
    history = history0.copy()
    y_pieces: list[np.ndarray] = []
    for i in range(len(x)):
        piece, history = fir_filter_c64(x[i : i + 1], taps, history)
        y_pieces.append(piece)
    y_streamed = np.concatenate(y_pieces)

    np.testing.assert_array_equal(y_whole, y_streamed)


def test_fir_filter_larger_tap_count():
    """Exercise a larger FIR to cover the RRC filter's tap count (~50-80)."""
    n_taps = 77
    taps = firwin(n_taps, 0.25).astype(np.float32)
    x = _random_complex64(3000, seed=3)
    history = np.zeros(n_taps - 1, dtype=np.complex64)

    y_ref, _ = lfilter(taps, 1.0, x, zi=np.zeros(n_taps - 1))
    y_kernel, _ = fir_filter_c64(x, taps, history)

    np.testing.assert_allclose(
        y_kernel.astype(np.complex128), y_ref.astype(np.complex128), atol=1e-5, rtol=1e-5
    )


def test_fir_filter_empty_input():
    """Empty input must return empty output and unchanged state."""
    n_taps = 41
    taps = firwin(n_taps, 0.1).astype(np.float32)
    x = np.empty(0, dtype=np.complex64)
    history = _random_complex64(n_taps - 1, seed=4)

    y, new_history = fir_filter_c64(x, taps, history)
    assert y.shape == (0,)
    np.testing.assert_array_equal(new_history, history)


# Decimating FIR kernel


def test_fir_decim_matches_fir_then_slice():
    """Single-shot: fir_decim_c64 must equal fir_filter_c64[::m] within float32 precision.

    The two kernels compile to different vectorized loops under `fastmath=True`,
    so LLVM may reorder the FMAs differently between them. That produces ~1e-7
    bit-level drift. Streaming equivalence within a single kernel is tested
    bit-exactly in `test_fir_decim_streaming_matches_single_shot`.
    """
    for n_taps, m in [(41, 4), (281, 28), (55, 7)]:
        taps = firwin(n_taps, 0.8 / m).astype(np.float32)
        x = _random_complex64(10000, seed=50 + n_taps)
        history = np.zeros(n_taps - 1, dtype=np.complex64)

        # Reference: full FIR, then slice.
        y_full, _ = fir_filter_c64(x, taps, history)
        y_ref = y_full[::m]

        # Kernel: decimating FIR directly.
        y_kernel, _, _ = fir_decim_c64(x, taps, m, history, 0)

        assert y_kernel.shape == y_ref.shape
        np.testing.assert_allclose(
            y_kernel.astype(np.complex128),
            y_ref.astype(np.complex128),
            atol=1e-5,
            rtol=1e-5,
        )


def test_fir_decim_streaming_matches_single_shot():
    """Chunked decimating FIR must equal a single big call, bit-exactly."""
    n_taps, m = 281, 28
    taps = firwin(n_taps, 0.8 / m).astype(np.float32)
    x = _random_complex64(200000, seed=51)
    history0 = np.zeros(n_taps - 1, dtype=np.complex64)

    # Single shot
    y_whole, _, _ = fir_decim_c64(x, taps, m, history0, 0)

    # Irregular chunks (mix of large, tiny, and sub-m chunks to exercise
    # phase rollover and history carry-over in every branch of the kernel).
    chunks = [5000, 13, 27, 50000, 100, 3, 144857]
    assert sum(chunks) == 200000
    history = history0.copy()
    phase = 0
    pieces: list[np.ndarray] = []
    start = 0
    for size in chunks:
        piece, history, phase = fir_decim_c64(x[start : start + size], taps, m, history, phase)
        pieces.append(piece)
        start += size
    y_stream = np.concatenate(pieces)

    np.testing.assert_array_equal(y_whole, y_stream)


def test_fir_decim_chunk_smaller_than_m():
    """Sub-m chunks must still advance phase/history so a stream of
    one-sample chunks reproduces a single big call exactly."""
    n_taps, m = 281, 28
    taps = firwin(n_taps, 0.8 / m).astype(np.float32)
    x = _random_complex64(2000, seed=52)

    # Single shot reference
    y_whole, _, _ = fir_decim_c64(x, taps, m, np.zeros(n_taps - 1, dtype=np.complex64), 0)

    # Feed one sample at a time.
    history = np.zeros(n_taps - 1, dtype=np.complex64)
    phase = 0
    pieces: list[np.ndarray] = []
    for i in range(len(x)):
        piece, history, phase = fir_decim_c64(x[i : i + 1], taps, m, history, phase)
        pieces.append(piece)
    y_stream = np.concatenate(pieces)

    np.testing.assert_array_equal(y_whole, y_stream)


def test_fir_decim_empty_input():
    """Empty input: no outputs, history/phase unchanged."""
    n_taps, m = 41, 4
    taps = firwin(n_taps, 0.1).astype(np.float32)
    x = np.empty(0, dtype=np.complex64)
    history = _random_complex64(n_taps - 1, seed=53)

    y, new_history, phase_out = fir_decim_c64(x, taps, m, history, 2)
    assert y.shape == (0,)
    np.testing.assert_array_equal(new_history, history)
    assert phase_out == 2


# MuellerMuller kernel


def _mm_reference_complex(
    samples: np.ndarray,
    sps: float,
    gain: float,
    mu_in: float,
    i_in_start: int,
    prev_out0: complex,
    prev_out1: complex,
    prev_rail0: complex,
    prev_rail1: complex,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Pure-Python reference with identical semantics to the old mm.py code."""
    n_samples = len(samples)
    sps_int = int(sps)
    max_symbols = n_samples // sps_int + 10
    dt = samples.dtype

    out = np.zeros(max_symbols, dtype=dt)
    out_rail = np.zeros(max_symbols, dtype=dt)

    out[0] = prev_out0
    out[1] = prev_out1
    out_rail[0] = prev_rail0
    out_rail[1] = prev_rail1

    i_in = i_in_start
    i_out = 2
    mu = mu_in

    while i_out < max_symbols and i_in < n_samples - 1:
        frac = mu
        sample = samples[i_in] * (1 - frac) + samples[i_in + 1] * frac
        out[i_out] = sample
        out_rail[i_out] = complex(
            1.0 if sample.real > 0 else -1.0,
            1.0 if sample.imag > 0 else -1.0,
        )

        x = (out_rail[i_out] - out_rail[i_out - 2]) * out[i_out - 1].conjugate()
        y = (out[i_out] - out[i_out - 2]) * out_rail[i_out - 1].conjugate()
        mm_val = (y - x).real

        mu += sps + gain * mm_val
        step = int(mu)
        i_in += step
        mu -= step
        i_out += 1

    return out, out_rail, i_out, i_in, mu


def test_mm_process_complex_matches_reference():
    """Numba MM kernel must match the pure-Python reference within float32 precision.

    Exact bit-for-bit equality isn't achievable: the Python reference upcasts
    scalars to complex128 in arithmetic expressions, while numba keeps
    everything in complex64 throughout. Errors accumulate from float32
    precision over the MM loop's ~N_symbols iterations.
    """
    sps = 4.0
    gain = 0.01
    mu_in = 0.01
    i_in_start = 0
    samples = _random_complex64(4000, seed=10)

    ref = _mm_reference_complex(
        samples,
        sps,
        gain,
        mu_in,
        i_in_start,
        complex(0),
        complex(0),
        complex(0),
        complex(0),
    )
    kernel = _mm_process_complex(
        samples,
        sps,
        gain,
        mu_in,
        i_in_start,
        complex(0),
        complex(0),
        complex(0),
        complex(0),
    )

    ref_out, ref_rail, ref_nout, ref_iin, ref_mu = ref
    k_out, k_rail, k_nout, k_iin, k_mu = kernel

    assert k_nout == ref_nout
    assert k_iin == ref_iin
    assert abs(k_mu - ref_mu) < 1e-3  # float32 drift over ~1000 iterations
    # Compare only the valid portion of the output
    np.testing.assert_allclose(
        k_out[:k_nout].astype(np.complex128),
        ref_out[:ref_nout].astype(np.complex128),
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_array_equal(k_rail[:k_nout], ref_rail[:ref_nout])


# Frequency shift kernel


def _freq_shift_reference(
    x: np.ndarray, offset_hz: float, sample_rate: float, phase_in: float
) -> tuple[np.ndarray, float]:
    """Pure-numpy reference matching the original _apply_freq_shift semantics."""
    n = len(x)
    phase_inc = -2 * np.pi * offset_hz / sample_rate
    phases = phase_in + phase_inc * np.arange(n)
    new_phase = phases[-1] + phase_inc
    new_phase = new_phase % (2 * np.pi)
    result = x * np.exp(1j * phases).astype(np.complex64)
    return result, float(new_phase)


def test_apply_freq_shift_matches_reference_zero_phase():
    x = _random_complex64(4096, seed=20)
    y_ref, p_ref = _freq_shift_reference(x, 123.4, 2_048_000.0, 0.0)
    y_kernel, p_kernel = apply_freq_shift_c64(x, 123.4, 2_048_000.0, 0.0)

    assert y_kernel.dtype == np.complex64
    np.testing.assert_allclose(
        y_kernel.astype(np.complex128),
        y_ref.astype(np.complex128),
        atol=1e-4,
        rtol=1e-4,
    )
    assert abs(p_kernel - p_ref) < 1e-4


def test_apply_freq_shift_streaming_continues_phase():
    x = _random_complex64(10000, seed=21)
    sr = 2_048_000.0
    offset = -55.3

    # Single shot
    y_whole, p_whole = apply_freq_shift_c64(x, offset, sr, 0.0)

    # Chunked
    chunks = [1000, 500, 3000, 1, 5499]
    assert sum(chunks) == 10000
    y_pieces: list[np.ndarray] = []
    p = 0.0
    start = 0
    for size in chunks:
        piece, p = apply_freq_shift_c64(x[start : start + size], offset, sr, p)
        y_pieces.append(piece)
        start += size
    y_stream = np.concatenate(y_pieces)

    np.testing.assert_allclose(
        y_whole.astype(np.complex128),
        y_stream.astype(np.complex128),
        atol=1e-4,
        rtol=1e-4,
    )


# Scramble LFSR kernel


def _scramble_reference(init: int, length: int) -> np.ndarray:
    """Pure-Python reference matching scramble.py:21 generate_scramble_bits."""
    lfsr = init & 0xFFFFFFFF
    out = np.empty(length, dtype=np.uint8)
    taps = [32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 4, 2, 1]
    for i in range(length):
        bit = 0
        for tap in taps:
            bit ^= (lfsr >> (32 - tap)) & 1
        lfsr = (lfsr >> 1) | (bit << 31)
        out[i] = bit
    return out


def test_generate_scramble_bits_matches_reference():
    for init in (0x12345678, 0xDEADBEEF, 0x00000003):
        for length in (100, 432, 1024):
            ref = _scramble_reference(init, length)
            kernel = kernel_generate_scramble_bits(init, length)
            np.testing.assert_array_equal(kernel, ref)


def test_descramble_soft_matches_reference():
    rng = np.random.default_rng(seed=30)
    soft = rng.standard_normal(432).astype(np.float32)
    init = 0xCAFEBABE

    scramble = _scramble_reference(init, len(soft))
    signs = np.ones(len(soft), dtype=np.float32)
    signs[scramble == 1] = -1.0
    ref = soft * signs

    kernel = kernel_descramble_soft(soft, init)
    np.testing.assert_array_equal(kernel, ref)


# Channel-coding kernels


def test_deinterleave_matches_reference():
    rng = np.random.default_rng(seed=40)
    # Sizes match BLOCK_PARAMS (type345, a).
    for k, a in [(120, 11), (216, 101), (168, 13), (432, 103)]:
        soft = rng.standard_normal(k).astype(np.float32)
        np.testing.assert_array_equal(kernel_deinterleave(soft, k, a), ref_deinterleave(soft, k, a))


def test_depuncture_2_3_matches_reference():
    rng = np.random.default_rng(seed=41)
    # Sizes match the real BLOCK_PARAMS: (type345_len, type2*4).
    for n_punctured, mother_len in [
        (120, 320),  # SB1: type345=120, mother=80*4
        (216, 576),  # SB2/NDB: type345=216, mother=144*4
        (168, 448),  # SCH_HU
        (432, 1152),  # SCH_F
    ]:
        type345 = rng.standard_normal(n_punctured).astype(np.float32)
        np.testing.assert_array_equal(
            kernel_depuncture_2_3(type345, mother_len),
            ref_depuncture_2_3(type345, mother_len),
        )


def test_crc16_ccitt_matches_reference():
    rng = np.random.default_rng(seed=42)
    for length in (16, 76, 124, 432):
        bits = rng.integers(0, 2, size=length, dtype=np.uint8)
        assert kernel_crc16_ccitt(bits) == ref_crc16_ccitt(bits)


def test_bits_to_uint_matches_reference():
    rng = np.random.default_rng(seed=43)
    bits = rng.integers(0, 2, size=200, dtype=np.uint8)
    for start, length in [(0, 8), (5, 16), (13, 24), (100, 32), (150, 7)]:
        assert kernel_bits_to_uint(bits, start, length) == ref_bits_to_uint(bits, start, length)


def test_mm_process_complex_with_nonzero_prev_state():
    """Kernel must honor prev_out/prev_rail state carry-over."""
    sps = 5.12
    gain = 0.02
    mu_in = 0.37
    i_in_start = 3
    samples = _random_complex64(2000, seed=11)
    prev_out0 = complex(0.3, -0.4)
    prev_out1 = complex(-0.2, 0.6)
    prev_rail0 = complex(1.0, -1.0)
    prev_rail1 = complex(-1.0, 1.0)

    ref = _mm_reference_complex(
        samples,
        sps,
        gain,
        mu_in,
        i_in_start,
        prev_out0,
        prev_out1,
        prev_rail0,
        prev_rail1,
    )
    kernel = _mm_process_complex(
        samples,
        sps,
        gain,
        mu_in,
        i_in_start,
        prev_out0,
        prev_out1,
        prev_rail0,
        prev_rail1,
    )

    ref_out, ref_rail, ref_nout, ref_iin, ref_mu = ref
    k_out, k_rail, k_nout, k_iin, k_mu = kernel

    assert k_nout == ref_nout
    assert k_iin == ref_iin
    assert abs(k_mu - ref_mu) < 1e-3  # float32 drift over ~1000 iterations
    np.testing.assert_allclose(
        k_out[:k_nout].astype(np.complex128),
        ref_out[:ref_nout].astype(np.complex128),
        atol=1e-3,
        rtol=1e-3,
    )
