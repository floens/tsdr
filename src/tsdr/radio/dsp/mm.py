"""Mueller-Muller symbol timing recovery with linear interpolation.

Auto-detects real vs complex input per process() call. The algorithm is
structurally identical - only the rail decision differs (scalar sign vs
component-wise sign).

The complex branch is numba-jit'd (kernel + wrapper in this file, matching
the `viterbi.py` pattern). MuellerMuller is a shared DSP primitive so the
kernel stays in `dsp/` with zero coupling to any specific decoder.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _mm_process_complex(
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
    """Mueller-Muller inner loop for complex samples (numba-jit'd).

    Returns (out, out_rail, n_out, i_in_final, mu_final). `out` and
    `out_rail` are allocated here; caller slices `out[2:n_out]` as the
    symbols output. Indices 0 and 1 hold the previous-call tail state.

    The math mirrors the reference Python implementation but expands
    `.conjugate()` and the complex `*` into explicit real/imag scalar ops:
    numba doesn't autovectorize `.conjugate()` on indexed complex elements.
    """
    n_samples = samples.shape[0]
    sps_int = int(sps)
    max_symbols = n_samples // sps_int + 10

    out = np.empty(max_symbols, dtype=samples.dtype)
    out_rail = np.empty(max_symbols, dtype=samples.dtype)

    out[0] = prev_out0
    out[1] = prev_out1
    out_rail[0] = prev_rail0
    out_rail[1] = prev_rail1

    i_in = i_in_start
    i_out = 2
    mu = mu_in

    while i_out < max_symbols and i_in < n_samples - 1:
        frac = mu
        s0 = samples[i_in]
        s1 = samples[i_in + 1]
        sample_re = s0.real * (1.0 - frac) + s1.real * frac
        sample_im = s0.imag * (1.0 - frac) + s1.imag * frac
        out[i_out] = complex(sample_re, sample_im)

        rail_re = 1.0 if sample_re > 0.0 else -1.0
        rail_im = 1.0 if sample_im > 0.0 else -1.0
        out_rail[i_out] = complex(rail_re, rail_im)

        # x = (out_rail[i_out] - out_rail[i_out-2]) * conj(out[i_out-1])
        dr_re = rail_re - out_rail[i_out - 2].real
        dr_im = rail_im - out_rail[i_out - 2].imag
        po_re = out[i_out - 1].real
        po_im = -out[i_out - 1].imag  # conjugate
        x_re = dr_re * po_re - dr_im * po_im
        # x_im unused: we only need (y - x).real below

        # y = (out[i_out] - out[i_out-2]) * conj(out_rail[i_out-1])
        dy_re = sample_re - out[i_out - 2].real
        dy_im = sample_im - out[i_out - 2].imag
        pr_re = out_rail[i_out - 1].real
        pr_im = -out_rail[i_out - 1].imag  # conjugate
        y_re = dy_re * pr_re - dy_im * pr_im

        mm_val = y_re - x_re  # == (y - x).real

        mu += sps + gain * mm_val
        step = int(mu)
        i_in += step
        mu -= step
        i_out += 1

    return out, out_rail, i_out, i_in, mu


class MuellerMuller:
    """Mueller-Muller symbol timing recovery with linear interpolation.

    Auto-detects real vs complex input per process() call.
    """

    def __init__(self, sps: float, gain: float = 0.01):
        self.sps = sps
        self.gain = gain
        self._mu = 0.01
        self._prev_out = np.zeros(2, dtype=np.float64)
        self._prev_rail = np.zeros(2, dtype=np.float64)
        self._tail: np.ndarray = np.array([], dtype=np.float64)
        self._tail_context = 0
        self._is_complex: bool | None = None
        # Pre-allocated work buffer (avoids per-call np.concatenate)
        self._work_buf: np.ndarray | None = None
        self._work_size = 0

    def _init_dtype(self, is_complex: bool) -> None:
        """Initialize/reinitialize arrays for the given dtype."""
        self._is_complex = is_complex
        dt = np.complex128 if is_complex else np.float64
        self._prev_out = self._prev_out.astype(dt)
        self._prev_rail = self._prev_rail.astype(dt)
        if len(self._tail) > 0:
            self._tail = self._tail.astype(dt)

    def process(self, samples: np.ndarray) -> np.ndarray:
        is_complex = np.iscomplexobj(samples)

        # Match internal dtype to input
        if self._is_complex is None or self._is_complex != is_complex:
            self._init_dtype(is_complex)

        dt = samples.dtype
        # Ensure consistent complex dtype (fixes complex128 tail + complex64 input)
        if is_complex and self._tail.dtype != dt and len(self._tail) > 0:
            self._tail = self._tail.astype(dt)

        if len(self._tail) > 0:
            total = len(self._tail) + len(samples)
            if total > self._work_size or self._work_buf is None or self._work_buf.dtype != dt:
                self._work_size = max(total, 1024)
                self._work_buf = np.empty(self._work_size, dtype=dt)
            tail_len = len(self._tail)
            self._work_buf[:tail_len] = self._tail
            self._work_buf[tail_len:total] = samples
            samples = self._work_buf[:total]
            i_in_start = self._tail_context
        else:
            i_in_start = 0

        if len(samples) < 32:
            return np.array([], dtype=dt)

        sps = self.sps
        mu = self._mu
        gain = self.gain
        n_samples = len(samples)

        if is_complex:
            out, out_rail, i_out, i_in, mu = _mm_process_complex(
                samples,
                sps,
                gain,
                mu,
                i_in_start,
                complex(self._prev_out[0]),
                complex(self._prev_out[1]),
                complex(self._prev_rail[0]),
                complex(self._prev_rail[1]),
            )
        else:
            max_symbols = n_samples // int(sps) + 10
            out = np.zeros(max_symbols, dtype=dt)
            out_rail = np.zeros(max_symbols, dtype=dt)

            out[0] = self._prev_out[0]
            out[1] = self._prev_out[1]
            out_rail[0] = self._prev_rail[0]
            out_rail[1] = self._prev_rail[1]

            i_in = i_in_start
            i_out = 2
            while i_out < max_symbols and i_in < n_samples - 1:
                frac = mu
                sample = samples[i_in] * (1 - frac) + samples[i_in + 1] * frac
                out[i_out] = sample
                out_rail[i_out] = 1.0 if sample > 0 else -1.0

                x = (out_rail[i_out] - out_rail[i_out - 2]) * out[i_out - 1]
                y = (out[i_out] - out[i_out - 2]) * out_rail[i_out - 1]
                mm_val = y - x

                mu += sps + gain * mm_val
                i_in += int(mu)
                mu -= int(mu)
                i_out += 1

        context_len = min(int(sps) + 2, i_in)
        self._tail = samples[i_in - context_len :].copy()
        self._tail_context = context_len

        self._mu = mu
        if i_out >= 4:
            self._prev_out[0] = out[i_out - 2]
            self._prev_out[1] = out[i_out - 1]
            self._prev_rail[0] = out_rail[i_out - 2]
            self._prev_rail[1] = out_rail[i_out - 1]

        result: np.ndarray = out[2:i_out]
        return result

    def nudge(self, fraction: float = 0.25) -> None:
        """Shift mu by fraction * sps."""
        self._mu = (self._mu + self.sps * fraction) % self.sps

    def reset(self) -> None:
        dt = np.complex128 if self._is_complex else np.float64
        self._mu = 0.01
        self._prev_out = np.zeros(2, dtype=dt)
        self._prev_rail = np.zeros(2, dtype=dt)
        self._tail = np.array([], dtype=dt)
        self._tail_context = 0
