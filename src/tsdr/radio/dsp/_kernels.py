import logging
import math
from fractions import Fraction

import numba as nb
import numpy as np

from tsdr.radio.dsp.filters import firwin as _firwin

logger = logging.getLogger(__name__)

# A rational resampler's prototype filter is `2 * taps_per_phase * max(up, down)`.
# Legitimate audio ratios top out near 12.8k taps (11025->48k); anything past
# this bound means a caller derived up/down from an unbounded runtime rate.
_RESAMPLER_TAPS_WARN = 50_000


@nb.njit(cache=True, fastmath=True)
def _lfilter_fir_f32(
    b: np.ndarray,
    x: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """FIR filter on real float32 input. *zi* is mutated in place."""
    n_x = x.shape[0]
    n_z = zi.shape[0]
    y = np.empty(n_x, dtype=np.float32)

    for n in range(n_x):
        y[n] = np.float32(b[0]) * x[n] + zi[0]
        for k in range(n_z - 1):
            zi[k] = np.float32(b[k + 1]) * x[n] + zi[k + 1]
        if n_z > 0:
            zi[n_z - 1] = np.float32(b[k + 2 if n_z > 1 else 1]) * x[n]

    return y


@nb.njit(cache=True, fastmath=True)
def _lfilter_fir_c64(
    b: np.ndarray,
    x: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """FIR filter on complex64 input with real taps. *zi* is mutated in place."""
    n_x = x.shape[0]
    n_z = zi.shape[0]
    y = np.empty(n_x, dtype=np.complex64)

    for n in range(n_x):
        xr = x[n].real
        xi = x[n].imag
        b0 = np.float32(b[0])
        yr = b0 * xr + zi[0].real
        yi = b0 * xi + zi[0].imag
        for k in range(n_z - 1):
            bk = np.float32(b[k + 1])
            zi[k] = np.complex64(complex(bk * xr + zi[k + 1].real, bk * xi + zi[k + 1].imag))
        if n_z > 0:
            blast = np.float32(b[n_z])
            zi[n_z - 1] = np.complex64(complex(blast * xr, blast * xi))
        y[n] = np.complex64(complex(yr, yi))

    return y


@nb.njit(cache=True)
def _lfilter_iir(
    b: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """General IIR filter (direct-form II transposed). *zi* mutated in place."""
    n_x = x.shape[0]
    n_z = zi.shape[0]
    y = np.empty(n_x, dtype=zi.dtype)

    for n in range(n_x):
        xn = np.float64(x[n])
        yn = b[0] * xn + zi[0]
        y[n] = yn
        for k in range(n_z - 1):
            zi[k] = b[k + 1] * xn - a[k + 1] * yn + zi[k + 1]
        if n_z > 0:
            zi[n_z - 1] = b[n_z] * xn - a[n_z] * yn

    return y


@nb.njit(cache=True, fastmath=True)
def _lfilter_iir_f32(
    b: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """IIR filter in float32 (direct-form II transposed). *zi* mutated in place.

    Same algorithm as ``_lfilter_iir`` but avoids float64 promotion.
    Sufficient precision for low-order filters (de-emphasis, AGC smoothing).
    """
    n_x = x.shape[0]
    n_z = zi.shape[0]
    y = np.empty(n_x, dtype=np.float32)

    for n in range(n_x):
        xn = np.float32(x[n])
        yn = np.float32(b[0]) * xn + zi[0]
        y[n] = yn
        for k in range(n_z - 1):
            zi[k] = np.float32(b[k + 1]) * xn - np.float32(a[k + 1]) * yn + zi[k + 1]
        if n_z > 0:
            zi[n_z - 1] = np.float32(b[n_z]) * xn - np.float32(a[n_z]) * yn

    return y


@nb.njit(cache=True, fastmath=True)
def _freq_shift_f32_to_c64(
    audio: np.ndarray,
    carrier_freq: float,
    phase_in: float,
    out: np.ndarray,
) -> float:
    """Shift real float32 signal to baseband via complex carrier mixing.

    Computes ``out[i] = audio[i] * exp(-j * phase)`` sample-by-sample,
    incrementing phase by *carrier_freq* each step.  Writes to caller-
    allocated *out* (complex64).  Returns the final carrier phase.
    """
    n = audio.shape[0]
    phase = phase_in
    two_pi = 2.0 * math.pi
    for i in range(n):
        c = math.cos(phase)
        s = math.sin(phase)
        v = audio[i]
        out[i] = np.complex64(complex(v * c, -v * s))
        phase += carrier_freq
        if phase > two_pi:
            phase -= two_pi
        elif phase < -two_pi:
            phase += two_pi
    return phase


@nb.njit(cache=True, fastmath=True)
def apply_freq_shift_c64(
    x: np.ndarray,
    offset_hz: float,
    sample_rate: float,
    phase_in: float,
) -> tuple[np.ndarray, float]:
    """Fused frequency-shift kernel: multiply x[n] by exp(j * (phase_in + phase_inc*n))."""
    n = x.shape[0]
    y = np.empty(n, dtype=np.complex64)
    two_pi = 2.0 * math.pi
    phase_inc = -two_pi * offset_hz / sample_rate
    phase = phase_in
    for i in range(n):
        c = math.cos(phase)
        s = math.sin(phase)
        xi = x[i]
        xr = xi.real
        xj = xi.imag
        # (xr + j*xj) * (c + j*s) = (xr*c - xj*s) + j*(xr*s + xj*c)
        y[i] = np.complex64(complex(xr * c - xj * s, xr * s + xj * c))
        phase += phase_inc
    phase = phase % two_pi
    return y, phase


@nb.njit(cache=True, fastmath=True)
def _fm_discriminator_f32(
    iq: np.ndarray,
    prev_re: float,
    prev_im: float,
    scale: float,
) -> tuple[np.ndarray, float, float]:
    """FM discriminator via conjugate-product: atan2(imag, real) of iq[i]*conj(prev).

    Output is scaled by *scale* (typically sample_rate / (2*pi*deviation)) so
    that +-deviation Hz maps to ±1.0.  Returns (audio, new_prev_re, new_prev_im).
    """
    n = iq.shape[0]
    out = np.empty(n, dtype=np.float32)
    pr = np.float32(prev_re)
    pi_ = np.float32(prev_im)
    for i in range(n):
        xr = iq[i].real
        xi = iq[i].imag
        # conjugate product: iq[i] * conj(prev)
        prod_re = xr * pr + xi * pi_
        prod_im = xi * pr - xr * pi_
        out[i] = np.float32(math.atan2(prod_im, prod_re) * scale)
        pr = xr
        pi_ = xi
    return out, float(pr), float(pi_)


@nb.njit(cache=True, fastmath=True)
def _dpll_bitsync(
    x: np.ndarray,
    inc: float,
    k: float,
    phase: float,
    prev: float,
) -> tuple[np.ndarray, float, float]:
    """DPLL bit-clock recovery on an NRZ soft-decision signal.

    Emits one hard bit per symbol period, sampled at the eye centre (phase
    wrap). On each sign transition of *x* the phase is nudged so transitions
    land mid-period (phase 0.5), locking the bit clock. Returns
    (bits, new_phase, new_prev).
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.uint8)
    count = 0
    ph = phase
    pv = prev
    for i in range(n):
        v = x[i]
        if (v >= 0.0) != (pv >= 0.0):
            ph += k * (0.5 - ph)
        pv = v
        ph += inc
        if ph >= 1.0:
            ph -= 1.0
            out[count] = 1 if v >= 0.0 else 0
            count += 1
    return out[:count], ph, pv


@nb.njit(cache=True, fastmath=True)
def _decayavg(avg: float, value: float, weight: float) -> float:
    """One-pole running average, ``avg + (value - avg) / weight``."""
    if weight <= 1.0:
        return value
    return avg + (value - avg) / weight


@nb.njit(cache=True, fastmath=True)
def _fsk_gated_bitsync(
    mark_abs: np.ndarray,
    space_abs: np.ndarray,
    bit_sample_count: float,
    env_state: np.ndarray,
    acc_state: np.ndarray,
    avg_state: np.ndarray,
    evt_state: np.ndarray,
    out_bits: np.ndarray,
) -> int:
    """Non-coherent 2-FSK bit-sync: mark/space envelopes -> one soft bit per baud.

    An adaptive-threshold discriminator (per-tone envelope + noise-floor trackers)
    feeds three early/prompt/late gated integrators that recover the bit clock and
    emit one soft bit (sign = mark(+)/space(-)) per symbol. All loop-carried state
    lives in the four ``*_state`` arrays (mutated in place), so processing resumes
    exactly across streaming chunks. Returns the number of soft bits written.

    The soft value keeps float ``log1p`` precision; do not truncate it to int,
    which discards the low-confidence bits the FEC layer relies on.
    """
    mark_env = env_state[0]
    space_env = env_state[1]
    mark_noise = env_state[2]
    space_noise = env_state[3]
    early_acc = acc_state[0]
    prompt_acc = acc_state[1]
    late_acc = acc_state[2]
    avg_early = avg_state[0]
    avg_prompt = avg_state[1]
    avg_late = avg_state[2]
    next_early = evt_state[0]
    next_prompt = evt_state[1]
    next_late = evt_state[2]
    sc = evt_state[3]

    bsc = bit_sample_count
    w_up = bsc / 4.0
    w_env_dn = bsc * 16.0
    w_noise_dn = bsc / 4.0
    w_noise_up = bsc * 48.0
    period = int(bsc * 8.0)

    n = mark_abs.shape[0]
    out_count = 0
    for i in range(n):
        ma = mark_abs[i]
        sa = space_abs[i]

        # Re-centre the sampling instants every 8 bits toward maximum eye opening.
        if period > 0 and sc > 0.5 and (int(sc) % period) == 0:
            slope = avg_late - avg_early
            if avg_prompt * 1.05 < avg_early and avg_prompt * 1.05 < avg_late:
                if avg_early > avg_late:
                    slope = np.fmod((next_early - next_prompt) - bsc, bsc)
                    avg_late = avg_prompt
                    avg_prompt = avg_early
                else:
                    slope = np.fmod((next_late - next_prompt) + bsc, bsc)
                    avg_early = avg_prompt
                    avg_prompt = avg_late
            else:
                slope = slope / 1024.0
            if slope != 0.0:
                next_early += slope
                next_prompt += slope
                next_late += slope

        # Envelope tracks the peak (fast up, slow down); noise tracks the floor.
        mark_env = _decayavg(mark_env, ma, w_up if ma > mark_env else w_env_dn)
        mark_noise = _decayavg(mark_noise, ma, w_noise_dn if ma < mark_noise else w_noise_up)
        space_env = _decayavg(space_env, sa, w_up if sa > space_env else w_env_dn)
        space_noise = _decayavg(space_noise, sa, w_noise_dn if sa < space_noise else w_noise_up)
        nf = (space_noise + mark_noise) / 2.0

        mc = ma
        if mc > mark_env:
            mc = mark_env
        if mc < nf:
            mc = nf
        scl = sa
        if scl > space_env:
            scl = space_env
        if scl < nf:
            scl = nf

        # Mark-space discriminator with automatic threshold correction.
        logic = (
            (mc - nf) * (mark_env - nf)
            - (scl - nf) * (space_env - nf)
            - 0.5 * ((mark_env - nf) * (mark_env - nf) - (space_env - nf) * (space_env - nf))
        )
        ms = math.log1p(abs(logic))
        if logic < 0.0:
            ms = -ms

        early_acc += ms
        prompt_acc += ms
        late_acc += ms

        if sc >= next_early:
            avg_early = _decayavg(avg_early, abs(early_acc), 64.0)
            next_early += bsc
            early_acc = 0.0
        if sc >= next_late:
            avg_late = _decayavg(avg_late, abs(late_acc), 64.0)
            next_late += bsc
            late_acc = 0.0
        if sc >= next_prompt:
            avg_prompt = _decayavg(avg_prompt, abs(prompt_acc), 64.0)
            next_prompt += bsc
            out_bits[out_count] = np.float32(prompt_acc)
            out_count += 1
            prompt_acc = 0.0

        sc += 1.0

    env_state[0] = mark_env
    env_state[1] = space_env
    env_state[2] = mark_noise
    env_state[3] = space_noise
    acc_state[0] = early_acc
    acc_state[1] = prompt_acc
    acc_state[2] = late_acc
    avg_state[0] = avg_early
    avg_state[1] = avg_prompt
    avg_state[2] = avg_late
    evt_state[0] = next_early
    evt_state[1] = next_prompt
    evt_state[2] = next_late
    evt_state[3] = sc
    return out_count


@nb.njit(cache=True, fastmath=True)
def _iq_metrics_c64(iq: np.ndarray) -> tuple[float, float, float]:
    """Compute IQ signal metrics in a single pass (no intermediate arrays).

    Returns (rms, peak, clip_pct) where:
    - rms = sqrt(mean(|iq|^2))
    - peak = max(|iq|)
    - clip_pct = percentage of samples where |re| >= 0.99 or |im| >= 0.99
    """
    n = iq.shape[0]
    sum_sq = 0.0
    max_sq = 0.0
    clip_count = 0
    for i in range(n):
        re = iq[i].real
        im = iq[i].imag
        sq = re * re + im * im
        sum_sq += sq
        if sq > max_sq:
            max_sq = sq
        if re >= 0.99 or re <= -0.99 or im >= 0.99 or im <= -0.99:
            clip_count += 1
    rms = math.sqrt(sum_sq / n) if n > 0 else 0.0
    peak = math.sqrt(max_sq)
    clip_pct = 100.0 * clip_count / n if n > 0 else 0.0
    return rms, peak, clip_pct


@nb.njit(cache=True, fastmath=True)
def _uint8_iq_to_complex64(raw: np.ndarray) -> np.ndarray:
    """Convert interleaved uint8 I/Q pairs to complex64 in one pass.

    Maps [0..255] to [-1.0..+1.0] via (x - 127.5) / 127.5.
    """
    n = raw.shape[0] // 2
    out = np.empty(n, dtype=np.complex64)
    inv = np.float32(1.0 / 127.5)
    bias = np.float32(127.5)
    for i in range(n):
        out[i] = np.complex64(
            complex(
                (np.float32(raw[2 * i]) - bias) * inv,
                (np.float32(raw[2 * i + 1]) - bias) * inv,
            )
        )
    return out


@nb.njit(cache=True, fastmath=True)
def _sint8_iq_to_complex64(raw: np.ndarray) -> np.ndarray:
    """Convert interleaved sint8 I/Q pairs to complex64 in one pass.

    Maps [-128..127] to [-1.0..+1.0] via x / 127.0.
    """
    n = raw.shape[0] // 2
    out = np.empty(n, dtype=np.complex64)
    inv = np.float32(1.0 / 127.0)
    for i in range(n):
        out[i] = np.complex64(
            complex(
                np.float32(raw[2 * i]) * inv,
                np.float32(raw[2 * i + 1]) * inv,
            )
        )
    return out


@nb.njit(cache=True, fastmath=True)
def fir_decim_c64_into(
    x: np.ndarray,
    flipped: np.ndarray,
    m: int,
    history: np.ndarray,
    phase_in: int,
    padded_scratch: np.ndarray,
    y_out: np.ndarray,
) -> tuple[int, int]:
    """Streaming decimating FIR on complex64 samples (direct form I).

    Computes only the decimated outputs (skips m-1 out of every m), so a
    decimation factor of 10 does 10x less work than filter-then-slice.
    ``m=1`` handles the non-decimating case.

    Writes into caller-allocated ``padded_scratch`` and ``y_out`` and updates
    ``history`` in place -- no per-call numpy allocations.

    Parameters
    ----------
    x : complex64[N]
        Input samples.
    flipped : float32[K]
        FIR coefficients pre-flipped: ``flipped[i] = taps[K-1-i]``.
    m : int
        Decimation factor.
    history : complex64[K-1]
        Streaming state from the previous call. Updated in place.
    phase_in : int
        Input-index offset for the first output. Pass back ``phase_out``
        from the previous call.
    padded_scratch : complex64[>= N + K - 1]
        Caller-allocated scratch buffer.
    y_out : complex64[>= ceil((N + m - 1) / m)]
        Caller-allocated output buffer.

    Returns
    -------
    n_out : int
        Number of valid output samples in ``y_out[:n_out]``.
    phase_out : int
        Phase offset for the next call.
    """
    n = x.shape[0]
    k = flipped.shape[0]
    h_len = k - 1

    if phase_in >= n:
        n_out = 0
    else:
        n_out = (n - phase_in + m - 1) // m

    if n_out > 0:
        for i in range(h_len):
            padded_scratch[i] = history[i]
        for i in range(n):
            padded_scratch[h_len + i] = x[i]

        for i in range(n_out):
            base = phase_in + i * m
            acc_re = np.float32(0.0)
            acc_im = np.float32(0.0)
            for j in range(k):
                xp = padded_scratch[base + j]
                t = flipped[j]
                acc_re += t * xp.real
                acc_im += t * xp.imag
            y_out[i] = np.complex64(complex(acc_re, acc_im))

    # Update history: last K-1 samples of the incoming stream
    if n >= h_len:
        for i in range(h_len):
            history[i] = x[n - h_len + i]
    elif n > 0:
        for i in range(h_len - n):
            history[i] = history[n + i]
        for i in range(n):
            history[h_len - n + i] = x[i]

    if phase_in >= n:
        phase_out = phase_in - n
    else:
        phase_out = phase_in + n_out * m - n

    return n_out, phase_out


@nb.njit(cache=True, fastmath=True)
def fir_decim_f32_into(
    x: np.ndarray,
    flipped: np.ndarray,
    m: int,
    history: np.ndarray,
    phase_in: int,
    padded_scratch: np.ndarray,
    y_out: np.ndarray,
) -> tuple[int, int]:
    """Streaming decimating FIR on float32 samples (direct form I).

    Float32 counterpart of ``fir_decim_c64_into``.  See that function for
    full parameter documentation.  The inner dot-product loop has no
    loop-carried dependencies, enabling SIMD auto-vectorization.
    """
    n = x.shape[0]
    k = flipped.shape[0]
    h_len = k - 1

    if phase_in >= n:
        n_out = 0
    else:
        n_out = (n - phase_in + m - 1) // m

    if n_out > 0:
        for i in range(h_len):
            padded_scratch[i] = history[i]
        for i in range(n):
            padded_scratch[h_len + i] = x[i]

        for i in range(n_out):
            base = phase_in + i * m
            acc = np.float32(0.0)
            for j in range(k):
                acc += flipped[j] * padded_scratch[base + j]
            y_out[i] = acc

    # Update history: last K-1 samples of the incoming stream
    if n >= h_len:
        for i in range(h_len):
            history[i] = x[n - h_len + i]
    elif n > 0:
        for i in range(h_len - n):
            history[i] = history[n + i]
        for i in range(n):
            history[h_len - n + i] = x[i]

    if phase_in >= n:
        phase_out = phase_in - n
    else:
        phase_out = phase_in + n_out * m - n

    return n_out, phase_out


@nb.njit(cache=True, fastmath=True)
def _polyphase_resample_f32(
    x: np.ndarray,
    poly_bank: np.ndarray,
    up: int,
    down: int,
    history: np.ndarray,
    time_register: int,
) -> tuple[np.ndarray, int]:
    """Polyphase rational resampler for float32.

    Instead of zero-inserting by ``up`` then filtering the bloated stream,
    this directly computes each output sample from the correct polyphase
    branch, reducing work by a factor of ``up``.

    Parameters
    ----------
    x : float32[N]
        Input samples for one channel.
    poly_bank : float32[up, taps_per_phase]
        Polyphase filter bank.  ``poly_bank[p]`` is the sub-filter for
        phase ``p``, stored with taps in convolution order.
    up, down : int
        Rational resampling ratio ``up / down``.
    history : float32[taps_per_phase - 1]
        Last ``taps_per_phase - 1`` input samples from previous call.
        Mutated in place.
    time_register : int
        Fractional time accumulator (units of output phases, in ``[0, up)``).
        Determines the polyphase branch and input offset for the first output.

    Returns
    -------
    out : float32[n_out]
        Resampled output.
    time_register_out : int
        Updated time register for the next call.
    """
    n_in = x.shape[0]
    taps_per_phase = poly_bank.shape[1]
    h_len = taps_per_phase - 1

    # Count output samples
    n_out = 0
    t = time_register
    for _ in range(n_in * up + 1):
        inp_idx = t // up
        if inp_idx >= n_in:
            break
        n_out += 1
        t += down

    out = np.empty(n_out, dtype=np.float32)

    # Build padded = [history | x]
    padded_len = h_len + n_in
    padded = np.empty(padded_len, dtype=np.float32)
    for i in range(h_len):
        padded[i] = history[i]
    for i in range(n_in):
        padded[h_len + i] = x[i]

    t = time_register
    for i in range(n_out):
        inp_idx = t // up
        phase = t % up
        # Convolve poly_bank[phase] with padded[inp_idx .. inp_idx + taps_per_phase]
        base = inp_idx  # index into padded (already offset by h_len in padded layout)
        acc = np.float32(0.0)
        for j in range(taps_per_phase):
            acc += poly_bank[phase, j] * padded[base + j]
        out[i] = acc
        t += down

    # Update history: last h_len input samples
    if n_in >= h_len:
        for i in range(h_len):
            history[i] = x[n_in - h_len + i]
    elif n_in > 0:
        for i in range(h_len - n_in):
            history[i] = history[n_in + i]
        for i in range(n_in):
            history[h_len - n_in + i] = x[i]

    time_register_out = t - n_in * up
    return out, time_register_out


def lfilter(
    b: np.ndarray | list[float],
    a: np.ndarray | float | list[float],
    x: np.ndarray,
    *,
    zi: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Filter data with an IIR or FIR filter (direct-form II transposed)."""
    b_ = np.atleast_1d(np.asarray(b, dtype=np.float64))
    a_ = np.atleast_1d(np.asarray(a, dtype=np.float64))

    is_fir = len(a_) == 1 and a_[0] == 1.0
    is_complex = np.iscomplexobj(x)

    # State vector length
    n_z = max(len(b_), len(a_)) - 1

    if zi is not None:
        z = zi.copy()
    else:
        if is_complex:
            z = np.zeros(n_z, dtype=np.complex64)
        elif is_fir:
            z = np.zeros(n_z, dtype=np.float32)
        else:
            z = np.zeros(n_z, dtype=np.float64)

    z_out: np.ndarray
    if is_fir:
        if is_complex:
            x_c64 = np.ascontiguousarray(x, dtype=np.complex64)
            z_c64 = np.ascontiguousarray(z, dtype=np.complex64)
            y = _lfilter_fir_c64(b_, x_c64, z_c64)
            z_out = z_c64
        else:
            x_f32 = np.ascontiguousarray(x, dtype=np.float32)
            z_f32 = np.ascontiguousarray(z, dtype=np.float32)
            y = _lfilter_fir_f32(b_, x_f32, z_f32)
            z_out = z_f32
    else:
        # General IIR path
        if a_[0] != 1.0:
            b_ = b_ / a_[0]
            a_ = a_ / a_[0]
        # Pad b and a to equal length
        n = max(len(b_), len(a_))
        if len(b_) < n:
            b_ = np.r_[b_, np.zeros(n - len(b_))]
        if len(a_) < n:
            a_ = np.r_[a_, np.zeros(n - len(a_))]
        z_f64 = np.ascontiguousarray(z, dtype=np.float64)
        x_f64 = np.ascontiguousarray(x, dtype=np.float64)
        y = _lfilter_iir(b_, a_, x_f64, z_f64)
        z_out = z_f64

    if zi is not None:
        return y, z_out
    result: np.ndarray = y
    return result


class StreamingFilter:
    """Pre-configured streaming filter that calls numba kernels directly.

    For hot paths where taps and coefficients never change. Skips all
    per-call type checking, coefficient conversion, and state copying
    that ``lfilter()`` does.
    """

    def __init__(
        self,
        b: np.ndarray | list[float],
        a: np.ndarray | float | list[float],
        *,
        dtype: np.dtype | type = np.float32,
    ):
        b_ = np.atleast_1d(np.asarray(b, dtype=np.float64))
        a_ = np.atleast_1d(np.asarray(a, dtype=np.float64))
        self._is_fir = len(a_) == 1 and a_[0] == 1.0
        self._is_complex = np.issubdtype(dtype, np.complexfloating)

        if not self._is_fir:
            if a_[0] != 1.0:
                b_ = b_ / a_[0]
                a_ = a_ / a_[0]
            n = max(len(b_), len(a_))
            if len(b_) < n:
                b_ = np.r_[b_, np.zeros(n - len(b_))]
            if len(a_) < n:
                a_ = np.r_[a_, np.zeros(n - len(a_))]

        self._b = np.ascontiguousarray(b_)
        self._a = np.ascontiguousarray(a_)
        n_z = max(len(self._b), len(self._a)) - 1

        # Use float32 IIR when caller requested float32 (avoids f64 promotion)
        self._iir_f32 = not self._is_fir and not self._is_complex and dtype == np.float32

        if self._is_complex:
            self._zi = np.zeros(n_z, dtype=np.complex64)
        elif self._is_fir or self._iir_f32:
            self._zi = np.zeros(n_z, dtype=np.float32)
        else:
            self._zi = np.zeros(n_z, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter *x*, updating internal state in place."""
        result: np.ndarray
        if self._is_fir:
            if self._is_complex:
                result = _lfilter_fir_c64(
                    self._b,
                    np.ascontiguousarray(x, dtype=np.complex64),
                    self._zi,
                )
            else:
                result = _lfilter_fir_f32(
                    self._b,
                    np.ascontiguousarray(x, dtype=np.float32),
                    self._zi,
                )
        elif self._iir_f32:
            result = _lfilter_iir_f32(
                self._b,
                self._a,
                np.ascontiguousarray(x, dtype=np.float32),
                self._zi,
            )
        else:
            result = _lfilter_iir(
                self._b,
                self._a,
                np.ascontiguousarray(x, dtype=np.float64),
                self._zi,
            )
        return result

    def reset(self) -> None:
        """Zero the filter state."""
        self._zi[:] = 0


class StreamingDecimFilter:
    """Decimating FIR filter with pre-allocated scratch buffers.

    Wraps ``fir_decim_f32_into`` for float32 or ``fir_decim_c64_into`` for
    complex64. Computes only the decimated outputs (m=1 for non-decimating).
    The Direct Form I inner loop is a pure dot product, enabling SIMD.

    The returned array is owned by this filter -- do not hold references
    across ``process()`` calls.
    """

    def __init__(
        self,
        taps: np.ndarray,
        decimation: int = 1,
        *,
        dtype: type = np.float32,
        expected_input_size: int = 20_000,
    ):
        self._decimation = decimation
        self._is_complex = np.issubdtype(dtype, np.complexfloating)
        n_taps = len(taps)
        self._n_taps = n_taps
        self._flipped = np.ascontiguousarray(taps[::-1], dtype=np.float32)
        h_dtype = np.complex64 if self._is_complex else np.float32
        self._history = np.zeros(n_taps - 1, dtype=h_dtype)
        self._phase = 0

        # Pre-allocate scratch buffers
        self._scratch_size = expected_input_size
        self._padded = np.empty(self._scratch_size + n_taps - 1, dtype=h_dtype)
        self._y_out = np.empty(self._scratch_size // max(decimation, 1) + 2, dtype=h_dtype)

    def _grow(self, n: int) -> None:
        self._scratch_size = n
        h_dtype = np.complex64 if self._is_complex else np.float32
        self._padded = np.empty(n + self._n_taps - 1, dtype=h_dtype)
        self._y_out = np.empty(n // max(self._decimation, 1) + 2, dtype=h_dtype)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter and decimate *x*. Returned array is owned by this filter."""
        n = len(x)
        if n > self._scratch_size:
            self._grow(n)

        if self._is_complex:
            x_c = np.ascontiguousarray(x, dtype=np.complex64)
            n_out, self._phase = fir_decim_c64_into(
                x_c,
                self._flipped,
                self._decimation,
                self._history,
                self._phase,
                self._padded,
                self._y_out,
            )
        else:
            x_f = np.ascontiguousarray(x, dtype=np.float32)
            n_out, self._phase = fir_decim_f32_into(
                x_f,
                self._flipped,
                self._decimation,
                self._history,
                self._phase,
                self._padded,
                self._y_out,
            )
        return self._y_out[:n_out]

    def reset(self) -> None:
        """Zero filter state."""
        self._history[:] = 0
        self._phase = 0


class StreamingPolyphaseResampler:
    """Polyphase rational resampler with persistent state across calls.

    Decomposes a prototype lowpass FIR into ``up`` polyphase branches and
    computes only the needed output samples. For up=24, down=25 this does
    ~600x less work than zero-insert + full-rate FIR + decimate.
    """

    def __init__(
        self, up: int, down: int, n_taps: int, window: tuple[str, float] = ("kaiser", 5.0)
    ):
        self.up = up
        self.down = down

        if n_taps > _RESAMPLER_TAPS_WARN:
            logger.warning("resampler_taps_large up=%d down=%d n_taps=%d", up, down, n_taps)

        # Clamp the cutoff below Nyquist: a unity ratio (up == down == 1, from a
        # near-equal target/source) gives max_rate 1, and firwin rejects the
        # resulting cutoff of exactly 1.0.
        max_rate = max(up, down)
        h = _firwin(n_taps, min(1.0 / max_rate, 0.999), window=window)
        h_scaled = (h * up).astype(np.float32)

        # Pad to multiple of up
        n_padded = ((len(h_scaled) + up - 1) // up) * up
        h_padded = np.zeros(n_padded, dtype=np.float32)
        h_padded[: len(h_scaled)] = h_scaled

        # Build polyphase bank: poly_bank[phase, tap]
        taps_per_phase = n_padded // up
        self._poly_bank = np.ascontiguousarray(h_padded.reshape(taps_per_phase, up).T)
        self._taps_per_phase = taps_per_phase

        # Per-channel state
        self._histories: list[np.ndarray] = []
        self._time_register = 0

    def _ensure_channels(self, n_ch: int) -> None:
        """Lazily create per-channel history buffers."""
        while len(self._histories) < n_ch:
            self._histories.append(np.zeros(self._taps_per_phase - 1, dtype=np.float32))

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Resample stereo/mono audio, maintaining state across calls.

        Args:
            audio: float32 array of shape (n_samples, n_channels)

        Returns:
            Resampled float32 array of shape (n_out, n_channels)
        """
        n_ch = audio.shape[1]
        self._ensure_channels(n_ch)

        channels = []
        # All channels share the same time_register since they have the same
        # input/output timing relationship
        for ch in range(n_ch):
            x = np.ascontiguousarray(audio[:, ch], dtype=np.float32)
            out, new_time = _polyphase_resample_f32(
                x,
                self._poly_bank,
                self.up,
                self.down,
                self._histories[ch],
                self._time_register,
            )
            channels.append(out)

        # Update shared time register from last channel
        self._time_register = new_time
        return np.column_stack(channels)

    def reset(self) -> None:
        """Zero all state."""
        for h in self._histories:
            h[:] = 0
        self._time_register = 0


def make_rational_resampler(
    target_rate: float,
    source_rate: float,
    *,
    taps_per_phase: int = 10,
    max_denominator: int = 1000,
) -> StreamingPolyphaseResampler:
    """Build a resampler for ``target_rate / source_rate``, bounding the rational
    approximation so a fractional source rate (e.g. a KiwiSDR's GPS-corrected
    12001.116 Hz) can't produce a coprime ratio like 48000/12001 and a
    million-tap prototype filter. The sub-0.01% rate error the bound introduces
    is inaudible and absorbed by downstream buffering.
    """
    ratio = Fraction(target_rate / source_rate).limit_denominator(max_denominator)
    up = ratio.numerator
    down = ratio.denominator
    n_taps = 2 * taps_per_phase * max(up, down) + 1
    return StreamingPolyphaseResampler(up, down, n_taps)


@nb.njit(cache=True, fastmath=True)
def _dc_blocker_f32(
    x: np.ndarray,
    state: np.ndarray,
    rate: float,
) -> np.ndarray:
    """Single-pole IIR DC blocker.

    y[n] = x[n] - offset
    offset += y[n] * rate
    state[0] holds offset across calls; mutated in place.
    """
    n = x.shape[0]
    y = np.empty(n, dtype=np.float32)
    offset = state[0]
    r = np.float32(rate)
    for i in range(n):
        v = x[i] - offset
        y[i] = v
        offset += v * r
    state[0] = offset
    return y


@nb.njit(cache=True, fastmath=True)
def _agc_f32(
    x: np.ndarray,
    state: np.ndarray,
    attack: float,
    decay: float,
    setpoint: float,
    max_gain: float,
) -> np.ndarray:
    """Sample-by-sample AGC.

    Tracks |x| via asymmetric one-pole; output gain = min(setpoint/amp, max_gain).
    state[0] holds amp across calls; mutated in place.
    """
    n = x.shape[0]
    y = np.empty(n, dtype=np.float32)
    amp = state[0]
    a = np.float32(attack)
    d = np.float32(decay)
    inv_a = np.float32(1.0) - a
    inv_d = np.float32(1.0) - d
    sp = np.float32(setpoint)
    mg = np.float32(max_gain)
    for i in range(n):
        v = x[i]
        in_amp = abs(v)
        if in_amp > amp:
            amp = amp * inv_a + in_amp * a
        else:
            amp = amp * inv_d + in_amp * d
        if amp > np.float32(1e-12):
            gain = sp / amp
            if gain > mg:
                gain = mg
        else:
            gain = mg
        y[i] = v * gain
    state[0] = amp
    return y


@nb.njit(cache=True, fastmath=True)
def _morse_envelope_kernel(
    env: np.ndarray,
    smooth_state: np.ndarray,
    noise_floor: np.ndarray,
    sig_avg: np.ndarray,
    is_on: np.ndarray,
    smooth_alpha: float,
    noise_alpha: float,
    sig_alpha: float,
    hyst_high: float,
    hyst_low: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a CW keying envelope, run adaptive level trackers, apply Schmitt
    hysteresis, and emit on/off transition events.

    State arrays (1-element each) are mutated in place so the kernel can be
    called repeatedly across chunks without losing continuity.

    Returns ``(offsets, signs)``:
        offsets: int32 sample indices into ``env`` where the binary state changed
        signs:   int8, +1 for off->on, -1 for on->off
    """
    n = env.shape[0]
    offs = np.empty(n, dtype=np.int32)
    sgns = np.empty(n, dtype=np.int8)
    nt = 0

    smoothed = np.float32(smooth_state[0])
    nf = np.float32(noise_floor[0])
    sa = np.float32(sig_avg[0])
    state = is_on[0]

    sm_a = np.float32(smooth_alpha)
    nf_a = np.float32(noise_alpha)
    sa_a = np.float32(sig_alpha)
    high = np.float32(hyst_high)
    low = np.float32(hyst_low)

    for i in range(n):
        v = env[i]
        smoothed = smoothed + sm_a * (v - smoothed)
        # Slow noise-floor tracker only when likely silent (smoothed below sig_avg).
        if smoothed < sa:
            nf = nf + nf_a * (smoothed - nf)
        # Fast signal-level tracker on every sample.
        sa = sa + sa_a * (smoothed - sa)
        upper = high * sa
        lower = low * sa
        if state == 0 and smoothed > upper:
            offs[nt] = i
            sgns[nt] = 1
            nt += 1
            state = 1
        elif state == 1 and smoothed < lower:
            offs[nt] = i
            sgns[nt] = -1
            nt += 1
            state = 0

    smooth_state[0] = smoothed
    noise_floor[0] = nf
    sig_avg[0] = sa
    is_on[0] = state
    return offs[:nt].copy(), sgns[:nt].copy()
