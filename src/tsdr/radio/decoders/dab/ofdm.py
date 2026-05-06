import dataclasses

import numba as nb
import numpy as np

from .constants import (
    _FFT_BIN_TABLE,
    FREQ_DEINTERLEAVE_TABLE,
    N_CARRIERS,
    N_SYMBOLS,
    PRS_REF_TABLE,
    T_FRAME,
    T_G,
    T_NULL,
    T_S,
    T_U,
)


@dataclasses.dataclass
class OFDMState:
    """Inter-frame OFDM state for frequency tracking."""

    accumulated_hz: float = 0.0  # Running PLL frequency correction (Hz)


def detect_null_symbols(iq: np.ndarray, threshold: float = 0.15) -> list[int]:
    """Detect null symbols (frame boundaries) in IQ data.

    Returns list of sample indices at the start of each null symbol.
    Positions are approximate (quantized to block boundaries); the OFDM
    demodulator uses PRS correlation for sample-accurate timing.
    """
    block = T_NULL // 4
    n_blocks = len(iq) // block
    if n_blocks < 4:
        return []

    trimmed = iq[: n_blocks * block].reshape(n_blocks, block)
    block_power = np.mean(np.abs(trimmed) ** 2, axis=1)

    ref = np.median(block_power)
    if ref < 1e-12:
        return []

    null_mask = block_power < (threshold * ref)

    null_starts = []
    in_null = False
    run_start = 0
    for i in range(n_blocks):
        if null_mask[i] and not in_null:
            run_start = i
            in_null = True
        elif not null_mask[i] and in_null:
            if i - run_start >= 2:
                null_starts.append(run_start * block)
            in_null = False
    if in_null and n_blocks - run_start >= 2:
        null_starts.append(run_start * block)

    if not null_starts:
        return []
    merged = [null_starts[0]]
    for s in null_starts[1:]:
        if s - merged[-1] > T_FRAME // 2:
            merged.append(s)

    return merged


@nb.njit(cache=True)
def _fine_timing_coarse_jit(
    frame_iq: np.ndarray,
    positions: np.ndarray,
    offsets: np.ndarray,
    t_u: int,
    t_g: int,
) -> tuple[float, int]:
    """JIT-compiled coarse timing search via guard interval correlation."""
    best_corr = 0.0
    best_pos = 0
    for idx in range(len(positions)):
        pos = positions[idx]
        acc_re = np.float64(0.0)
        acc_im = np.float64(0.0)
        for j in range(t_g):
            a = frame_iq[pos + t_u + j]
            b = frame_iq[pos + j]
            acc_re += a.real * b.real + a.imag * b.imag
            acc_im += a.real * b.imag - a.imag * b.real
        corr = np.sqrt(acc_re * acc_re + acc_im * acc_im)
        if corr > best_corr:
            best_corr = corr
            best_pos = offsets[idx]
    return best_corr, best_pos


def _find_prs_timing(prs_candidate: np.ndarray) -> int | None:
    """Find PRS start via cross-correlation with known reference.

    Args:
        prs_candidate: T_U samples from the estimated PRS region.

    Returns:
        Peak index (offset within the T_U window where useful part starts),
        or None if no clear peak found.
    """
    fft_prs = np.fft.fft(prs_candidate)
    corr = np.fft.ifft(fft_prs * np.conj(PRS_REF_TABLE))
    impulse = np.abs(corr)

    avg = float(np.mean(impulse))
    if avg < 1e-12:
        return None

    # Search the full T_U range - the null detector can be off by hundreds
    # of samples, so a narrow search window around T_G misses the true peak.
    peak_idx = int(np.argmax(impulse))
    peak_val = impulse[peak_idx]

    if peak_val < 3 * avg:
        return None

    return peak_idx


def _find_fine_timing(frame_iq: np.ndarray) -> int:
    """Find precise PRS start using cyclic prefix correlation.

    Returns offset from T_NULL (can be negative).
    """
    search_range = 2000
    step = 4
    n_iq = len(frame_iq)

    offsets = np.arange(-search_range, search_range + 1, step)
    positions = T_NULL + offsets
    valid = (positions >= 0) & (positions + T_S <= n_iq)
    offsets = offsets[valid]
    positions = positions[valid]

    best_corr_jit, best_pos_jit = _fine_timing_coarse_jit(frame_iq, positions, offsets, T_U, T_G)
    best_corr = float(best_corr_jit)
    best_pos: int = int(best_pos_jit)

    # Fine search (step 1) around best coarse position
    for fine_off in range(best_pos - step, best_pos + step + 1):
        fine_pos = T_NULL + fine_off
        if fine_pos < 0 or fine_pos + T_S > n_iq:
            continue
        corr = abs(
            np.vdot(
                frame_iq[fine_pos + T_U : fine_pos + T_U + T_G], frame_iq[fine_pos : fine_pos + T_G]
            )
        )
        if corr > best_corr:
            best_corr = corr
            best_pos = fine_off

    return best_pos


@nb.njit(cache=True)
def _freq_offset_correlations_jit(
    frame_iq: np.ndarray, fine_offset: int, n_syms: int, t_null: int, t_s: int, t_u: int, t_g: int
) -> np.ndarray:
    """JIT-compiled CP correlation for frequency offset estimation."""
    n_iq = len(frame_iq)
    corrs = np.zeros(n_syms, dtype=np.complex128)
    count = 0
    for k in range(n_syms):
        sym_start = t_null + fine_offset + k * t_s
        if sym_start < 0 or sym_start + t_s > n_iq:
            continue
        acc = np.complex128(0.0)
        for j in range(t_g):
            a = frame_iq[sym_start + t_u + j]
            b = frame_iq[sym_start + j]
            acc += (a.real - 1j * a.imag) * b
        corrs[count] = acc
        count += 1
    return corrs[:count]


@nb.njit(cache=True)
def _extract_and_correct_symbols_jit(
    frame_iq: np.ndarray,
    fine_offset: int,
    freq_offset: float,
    n_total: int,
    t_null: int,
    t_s: int,
    t_u: int,
    t_g: int,
) -> np.ndarray:
    """JIT-compiled symbol extraction with inline frequency correction."""
    stacked = np.empty((n_total, t_u), dtype=np.complex64)
    two_pi_f_over_tu = 2.0 * np.pi * freq_offset / t_u
    do_correct = abs(freq_offset) > 1e-6
    for k in range(n_total):
        sym_start = t_null + fine_offset + k * t_s + t_g
        for j in range(t_u):
            s = frame_iq[sym_start + j]
            if do_correct:
                t = sym_start + j
                phase = -two_pi_f_over_tu * t
                c = np.cos(phase)
                si = np.sin(phase)
                sr = s.real * c - s.imag * si
                simag = s.real * si + s.imag * c
                stacked[k, j] = sr + 1j * simag
            else:
                stacked[k, j] = s
    return stacked


def _ofdm_demod_frame(frame_iq: np.ndarray, state: OFDMState | None = None) -> np.ndarray | None:
    """Demodulate all OFDM symbols in a frame via batched FFT.

    Uses a cumulative PLL: the accumulated frequency correction from previous
    frames is applied first, then the residual offset is measured on the
    corrected data and integrated into the accumulator (IIR gain 0.1).

    Returns:
        Complex array of shape (76, 2048) - raw FFT output per symbol
        (PRS + 75 data), or None if the frame is too short.
    """
    # When PLL is locked (accumulated_hz is non-trivial), apply it directly.
    # On bootstrap (accumulated_hz ≈ 0), estimate from raw data and seed the accumulator.
    if state is not None and abs(state.accumulated_hz) < 0.5:
        # Bootstrap: estimate full offset from raw data's CP correlation
        boot_offset = _find_fine_timing(frame_iq)
        boot_corrs = _freq_offset_correlations_jit(
            frame_iq, boot_offset, N_SYMBOLS, T_NULL, T_S, T_U, T_G
        )
        if len(boot_corrs) > 0:
            state.accumulated_hz = float(-np.angle(np.mean(boot_corrs))) / np.pi * 500.0

    if state is not None and abs(state.accumulated_hz) > 0.5:
        t = np.arange(len(frame_iq))
        frame_iq = frame_iq * np.exp(
            -1j * 2.0 * np.pi * state.accumulated_hz / 2_048_000.0 * t
        ).astype(np.complex64)

    fine_offset = _find_fine_timing(frame_iq)
    if len(frame_iq) >= T_NULL + T_U:
        prs_idx = _find_prs_timing(frame_iq[T_NULL : T_NULL + T_U])
        if prs_idx is not None:
            fine_offset = prs_idx - T_G

    n_total = N_SYMBOLS  # 76: PRS + 75 data symbols
    first_sym_start = T_NULL + fine_offset + T_G
    last_sym_end = T_NULL + fine_offset + (n_total - 1) * T_S + T_G + T_U
    if first_sym_start < 0 or last_sym_end > len(frame_iq):
        return None  # accumulated_hz preserved for next frame

    stacked = _extract_and_correct_symbols_jit(
        frame_iq, fine_offset, 0.0, n_total, T_NULL, T_S, T_U, T_G
    )

    # residual offset from corrected data's CP
    corrs = _freq_offset_correlations_jit(frame_iq, fine_offset, N_SYMBOLS, T_NULL, T_S, T_U, T_G)
    residual_hz = float(-np.angle(np.mean(corrs))) / np.pi * 500.0 if len(corrs) > 0 else 0.0

    if state is not None:
        state.accumulated_hz += 0.1 * residual_hz
        # Coarse snap: keep fine correction within ±500 Hz (half carrier spacing)
        if abs(state.accumulated_hz) > 500:
            carriers = round(state.accumulated_hz / 1000)
            state.accumulated_hz -= carriers * 1000

    return np.fft.fft(stacked, axis=1)


def _dqpsk_to_soft_bits(fft_syms: np.ndarray, start_sym: int, n_syms: int) -> np.ndarray:
    """DQPSK demod + frequency de-interleave for a range of symbols (batched).

    Args:
        fft_syms: Shape (76, 2048) from _ofdm_demod_frame.
        start_sym: First symbol index (0-based in fft_syms, so PRS=0).
        n_syms: Number of differential symbols to produce.

    Returns:
        Soft bits array of shape (n_syms * 3072,).
    """
    # Batch extract carriers for all symbols at once: (n_syms+1, 1536)
    all_carriers = fft_syms[start_sym : start_sym + n_syms + 1][:, _FFT_BIN_TABLE]

    # Differential products across all symbols: (n_syms, 1536)
    r1 = all_carriers[1:] * np.conj(all_carriers[:-1])

    # Frequency de-interleave: (n_syms, 1536)
    r1_deint = r1[:, FREQ_DEINTERLEAVE_TABLE]

    # Constellation rotation by π/2: DQPSK differential products land on the
    # diagonals (±45°, ±135°). Our soft-bit mapping (-imag, +real) needs the
    # constellation axis-aligned to produce discriminating soft values for Viterbi.
    r1_deint = r1_deint * np.complex64(1j)

    # Normalize
    norm = np.abs(r1_deint.real) + np.abs(r1_deint.imag)
    norm = np.maximum(norm, 1e-10)

    # Assemble soft bits: (n_syms, 3072) -> flat
    soft = np.empty((n_syms, 3072), dtype=np.float32)
    soft[:, :N_CARRIERS] = -r1_deint.imag / norm
    soft[:, N_CARRIERS : 2 * N_CARRIERS] = r1_deint.real / norm

    return soft.ravel()


def _dqpsk_constellation(
    fft_syms: np.ndarray, start_sym: int = 0, n_syms: int | None = None
) -> np.ndarray:
    """Extract DQPSK constellation points from FFT symbols.

    Returns complex array of differential products (one per carrier per symbol),
    unnormalized so amplitude variation is visible.
    """
    if n_syms is None:
        n_syms = fft_syms.shape[0] - 1 - start_sym
    points = np.zeros(n_syms * N_CARRIERS, dtype=np.complex64)
    for i in range(n_syms):
        prev = fft_syms[start_sym + i]
        curr = fft_syms[start_sym + i + 1]
        r1 = curr[_FFT_BIN_TABLE] * np.conj(prev[_FFT_BIN_TABLE])
        points[i * N_CARRIERS : (i + 1) * N_CARRIERS] = r1
    return points


def _extract_symbols(
    frame_iq: np.ndarray, freq_offset: float = 0.0, fine_offset: int = 0
) -> np.ndarray:
    """Extract frequency-domain OFDM symbols from a frame (diagnostics).

    Returns complex array of shape (76, 1536) - carrier-domain after FFT.
    """
    n_samples = len(frame_iq)
    symbols = np.zeros((N_SYMBOLS, N_CARRIERS), dtype=np.complex64)

    if abs(freq_offset) > 1e-6:
        t = np.arange(n_samples)
        frame_iq = frame_iq * np.exp(-2j * np.pi * freq_offset * t / T_U).astype(np.complex64)

    for k in range(N_SYMBOLS):
        sym_start = T_NULL + fine_offset + k * T_S + T_G
        sym_end = sym_start + T_U
        if sym_end > n_samples or sym_start < 0:
            break
        fft_out = np.fft.fft(frame_iq[sym_start:sym_end])
        # Carrier order: -768..-1 (bins 1280..2047), +1..+768 (bins 1..768)
        symbols[k] = np.concatenate([fft_out[1280:2048], fft_out[1:769]])

    return symbols


def _estimate_fractional_freq_offset(frame_iq: np.ndarray, fine_offset: int = 0) -> float:
    """Estimate fractional subcarrier frequency offset from cyclic prefix correlation."""
    correlations = []
    for k in range(min(10, N_SYMBOLS)):
        sym_start = T_NULL + fine_offset + k * T_S
        if sym_start < 0 or sym_start + T_S > len(frame_iq):
            continue
        guard = frame_iq[sym_start : sym_start + T_G]
        tail = frame_iq[sym_start + T_U : sym_start + T_U + T_G]
        if len(guard) == T_G and len(tail) == T_G:
            correlations.append(np.vdot(tail, guard))

    if not correlations:
        return 0.0
    return float(np.angle(np.mean(correlations)) / (2 * np.pi))
