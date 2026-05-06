"""π/4-DQPSK demodulator for TETRA signals.

Pipeline: LPF+decimate -> RRC filter -> π/8 rotation -> MuellerMuller -> differential demod -> soft bits.

The π/8 rotation moves all 8 constellation points off the I/Q axes, fixing the
Mueller-Muller component-wise sign decision. Differential demod cancels the rotation.
"""

import numpy as np

from tsdr.radio.decoders.tetra._kernels import fir_decim_c64_into
from tsdr.radio.dsp import firwin
from tsdr.radio.dsp._kernels import apply_freq_shift_c64
from tsdr.radio.dsp.mm import MuellerMuller

# Default scratch size in seconds of IQ. Anything larger grows lazily in
# `_grow_scratch_for`.
_DEFAULT_SCRATCH_SECONDS = 1.0

SYMBOL_RATE = 18000.0
TARGET_RATE = 72000.0


def _flip_taps(taps: np.ndarray) -> np.ndarray:
    """Reverse a FIR tap array into the positive-stride layout `fir_decim_c64_into` expects."""
    return np.ascontiguousarray(taps[::-1]).astype(np.float32)


def rrc_taps(sps: float, alpha: float, ntaps_per_sym: int = 11) -> np.ndarray:
    """Design root raised cosine filter taps."""
    n_taps = int(ntaps_per_sym * sps) | 1  # ensure odd
    t = (np.arange(n_taps) - (n_taps - 1) / 2.0) / sps  # normalized to symbol periods

    h = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - alpha + 4.0 * alpha / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-10:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(
                np.pi * ti * (1 + alpha)
            )
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h**2))
    return h.astype(np.float32)


def extract_soft_bits(diff: np.ndarray) -> np.ndarray:
    """Extract soft dibits from differential symbols.

    After differential demod, ideal points are at phases {π/4, 3π/4, -3π/4, -π/4}
    = {(1+j)/√2, (-1+j)/√2, (-1-j)/√2, (1-j)/√2}. Standard QPSK demapping:
    - Soft bit 0 (MSB): -imag / norm  (im < 0 -> bit 1)
    - Soft bit 1 (LSB): -real / norm  (re < 0 -> bit 1)

    Sign convention: positive -> bit 1, negative -> bit 0 (matches Viterbi decoder).
    Verified empirically against real TETRA signal with CRC check.
    """
    re = diff.real.astype(np.float32)
    im = diff.imag.astype(np.float32)
    norm = np.abs(re) + np.abs(im)
    norm = np.maximum(norm, 1e-10)

    msb = -im / norm
    lsb = -re / norm

    # Interleave: [msb0, lsb0, msb1, lsb1, ...]
    soft = np.empty(2 * len(diff), dtype=np.float32)
    soft[0::2] = msb
    soft[1::2] = lsb
    return soft


# Frequency correction field: 80 bits = 40 diff symbols
# Bits 0-7 (syms 0-3): f1..f8 = all 1s
# Bits 8-71 (syms 4-35): all 0s -> constant phase change per symbol
# Bits 72-79 (syms 36-39): f73..f80 = all 1s
# All-zeros dibits produce phase +π/4 per symbol
_FC_EXPECTED_PHASE = np.pi / 4  # expected phase change per symbol for dibit 00
_FC_SYM_START = 4  # first middle symbol (skip preamble)
_FC_SYM_END = 36  # last middle symbol + 1
_FC_FIELD_SYM_OFFSET = 7  # symbol offset of freq correction field in burst (bit 14 / 2)


def estimate_freq_offset(diff_symbols: np.ndarray, symbol_rate: float = SYMBOL_RATE) -> float:
    """Estimate frequency offset from a sync burst's differential symbols.

    Uses the frequency correction field (burst symbols 7-46): the middle 32 symbols
    (indices 4-35 within the field) are all-zeros dibits producing +π/4 phase change.
    Any deviation from +π/4 indicates frequency offset.
    """
    # Extract freq correction field diff symbols from the full burst
    fc_start = _FC_FIELD_SYM_OFFSET + _FC_SYM_START
    fc_end = _FC_FIELD_SYM_OFFSET + _FC_SYM_END
    if fc_end > len(diff_symbols):
        return 0.0

    fc_syms = diff_symbols[fc_start:fc_end]

    # Average phase change: use the mean of the complex values then take the angle
    avg_phase = np.angle(np.mean(fc_syms / np.abs(fc_syms)))

    # Frequency offset from expected
    phase_error = avg_phase - _FC_EXPECTED_PHASE
    # Wrap to [-π, π]
    phase_error = (phase_error + np.pi) % (2 * np.pi) - np.pi

    return float(phase_error * symbol_rate / (2 * np.pi))


class TetraDemod:
    """π/4-DQPSK demodulator for TETRA."""

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self._decim = max(1, round(sample_rate / TARGET_RATE))
        self._dec_rate = sample_rate / self._decim
        self._sps = self._dec_rate / SYMBOL_RATE

        # Anti-alias LPF + decim (fused via fir_decim_c64_into) and RRC
        # matched filter (M=1 through the same kernel). Each filter owns a
        # tap-flipped view, streaming history, and preallocated scratch.
        max_chunk = int(sample_rate * _DEFAULT_SCRATCH_SECONDS)
        max_dec_chunk = max_chunk // max(1, self._decim) + 2

        self._lpf_taps: np.ndarray | None
        if self._decim > 1:
            cutoff = 0.8 / self._decim  # normalized to Nyquist
            n_taps = self._decim * 10 + 1
            self._lpf_taps = firwin(n_taps, cutoff).astype(np.float32)
            self._lpf_flipped = _flip_taps(self._lpf_taps)
            self._lpf_zi = np.zeros(n_taps - 1, dtype=np.complex64)
            self._lpf_padded_scratch = np.empty(max_chunk + n_taps - 1, dtype=np.complex64)
            self._lpf_y_scratch = np.empty(max_dec_chunk, dtype=np.complex64)
        else:
            # decim=1: no LPF pass needed. Empty arrays keep the attributes
            # single-typed for mypy; they're never read when _lpf_taps is None.
            self._lpf_taps = None
            self._lpf_flipped = np.empty(0, dtype=np.float32)
            self._lpf_zi = np.empty(0, dtype=np.complex64)
            self._lpf_padded_scratch = np.empty(0, dtype=np.complex64)
            self._lpf_y_scratch = np.empty(0, dtype=np.complex64)
        self._lpf_phase = 0

        self._rrc = rrc_taps(self._sps, alpha=0.35)
        self._rrc_flipped = _flip_taps(self._rrc)
        self._rrc_zi = np.zeros(len(self._rrc) - 1, dtype=np.complex64)
        self._rrc_padded_scratch = np.empty(max_dec_chunk + len(self._rrc) - 1, dtype=np.complex64)
        self._rrc_y_scratch = np.empty(max_dec_chunk, dtype=np.complex64)

        # Constant complex-64 phase rotation applied after the matched filter
        # (moves DQPSK constellation off the I/Q axes for MM). Hoisted out of
        # the per-call hot path.
        self._pi_8_rot = np.complex64(np.exp(1j * np.pi / 8))

        # No Costas loop: π/4-DQPSK has 8-point constellation which produces zero
        # QPSK error for all symbols. Differential demod inherently cancels carrier phase.
        # Frequency correction is handled via sync burst (Checkpoint 9).

        # Timing recovery
        self._mm = MuellerMuller(sps=self._sps, gain=0.01)

        # Differential demod state
        self._prev_sym = np.complex64(0)

        # Constellation buffer (~0.1s of differential symbols)
        # Differential products cancel carrier phase, showing the DQPSK constellation.
        self._constellation_size = int(SYMBOL_RATE // 9)
        self._constellation_buf = np.zeros(self._constellation_size, dtype=np.complex64)
        self._constellation_pos = 0
        self._constellation_prev = np.complex64(0)
        self._constellation_points: np.ndarray | None = None

        # Frequency correction state
        self._freq_offset_hz = 0.0  # applied bulk frequency shift
        self._freq_phase = 0.0  # running phase accumulator for freq shift

    def apply_freq_correction(self, offset_hz: float) -> None:
        """Set bulk frequency correction to apply to incoming IQ samples."""
        self._freq_offset_hz = offset_hz

    def _apply_freq_shift(self, x: np.ndarray) -> np.ndarray:
        """Apply bulk frequency correction at the decimated rate.

        Runs in `_front_end` AFTER LPF+decim, so we mix only the ~N/M samples
        that survive decimation instead of all N input samples. For TETRA's
        sub-kHz offsets (well inside the ~29 kHz LPF passband) this is
        mathematically equivalent to mixing at the input rate up to a
        constant phase offset that the differential demod cancels.
        """
        if self._freq_offset_hz == 0.0:
            return x
        y, self._freq_phase = apply_freq_shift_c64(
            x, self._freq_offset_hz, self._dec_rate, self._freq_phase
        )
        result: np.ndarray = y
        return result

    def _grow_scratch_for(self, n_in: int) -> None:
        """Grow preallocated FIR scratch buffers if `n_in` exceeds their capacity.

        Amortized cost: each caller that feeds a larger chunk than seen before
        pays a one-time allocation; steady-state streaming hits the no-op path.
        """
        if self._lpf_taps is not None:
            need_lpf_padded = n_in + len(self._lpf_taps) - 1
            if self._lpf_padded_scratch.shape[0] < need_lpf_padded:
                self._lpf_padded_scratch = np.empty(need_lpf_padded, dtype=np.complex64)
            need_lpf_y = n_in // self._decim + 2
            if self._lpf_y_scratch.shape[0] < need_lpf_y:
                self._lpf_y_scratch = np.empty(need_lpf_y, dtype=np.complex64)
            n_after_decim = need_lpf_y
        else:
            n_after_decim = n_in
        need_rrc_padded = n_after_decim + len(self._rrc) - 1
        if self._rrc_padded_scratch.shape[0] < need_rrc_padded:
            self._rrc_padded_scratch = np.empty(need_rrc_padded, dtype=np.complex64)
        if self._rrc_y_scratch.shape[0] < n_after_decim:
            self._rrc_y_scratch = np.empty(n_after_decim, dtype=np.complex64)

    def _front_end(self, iq: np.ndarray) -> np.ndarray:
        """LPF+decim, freq shift, RRC, π/8 rotation, timing recovery.

        The shared head of `process` and `process_symbols`. Returns recovered
        complex symbols; callers branch on whether to run differential demod +
        soft-bit extraction or to update the constellation buffer.
        """
        self._grow_scratch_for(len(iq))
        x = iq.astype(np.complex64, copy=False)

        if self._lpf_taps is not None:
            n_out, self._lpf_phase = fir_decim_c64_into(
                x,
                self._lpf_flipped,
                self._decim,
                self._lpf_zi,
                self._lpf_phase,
                self._lpf_padded_scratch,
                self._lpf_y_scratch,
            )
            x = self._lpf_y_scratch[:n_out]

        # Freq shift at the decimated rate: M-times fewer samples to mix.
        x = self._apply_freq_shift(x)

        # M=1 through the same decimating kernel: no separate non-decimating
        # variant, one FIR implementation to maintain.
        n_in = len(x)
        fir_decim_c64_into(
            x,
            self._rrc_flipped,
            1,
            self._rrc_zi,
            0,
            self._rrc_padded_scratch,
            self._rrc_y_scratch,
        )
        # In-place multiply is safe: the slice is a view into _rrc_y_scratch,
        # which is fully overwritten at the start of every call.
        x = self._rrc_y_scratch[:n_in]
        x *= self._pi_8_rot

        return self._mm.process(x)

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Process IQ samples, return soft bits (flat float32 array)."""
        symbols = self._front_end(iq)
        if len(symbols) == 0:
            return np.array([], dtype=np.float32)

        diff = np.empty(len(symbols), dtype=np.complex64)
        diff[0] = symbols[0] * np.conj(self._prev_sym)
        diff[1:] = symbols[1:] * np.conj(symbols[:-1])
        self._prev_sym = symbols[-1]

        return extract_soft_bits(diff)

    def process_symbols(self, iq: np.ndarray) -> np.ndarray:
        """Process IQ samples, return recovered complex symbols (before diff demod)."""
        symbols = self._front_end(iq)

        # Collect differential products for constellation display.
        # Raw symbols have arbitrary carrier phase (no Costas loop);
        # differential demod cancels it, revealing the DQPSK points.
        if len(symbols) > 0:
            diff = np.empty(len(symbols), dtype=np.complex64)
            diff[0] = symbols[0] * np.conj(self._constellation_prev)
            diff[1:] = symbols[1:] * np.conj(symbols[:-1])
            self._constellation_prev = symbols[-1]

            n = len(diff)
            buf = self._constellation_buf
            pos = self._constellation_pos
            if n >= len(buf):
                buf[:] = diff[-len(buf) :]
                self._constellation_pos = 0
            else:
                space = len(buf) - pos
                if n <= space:
                    buf[pos : pos + n] = diff
                    self._constellation_pos = pos + n
                else:
                    buf[pos:] = diff[:space]
                    buf[: n - space] = diff[space:]
                    self._constellation_pos = n - space
            self._constellation_points = buf.copy()

        return symbols

    def get_constellation(self) -> np.ndarray | None:
        points = self._constellation_points
        self._constellation_points = None
        return points

    def reset(self) -> None:
        if self._lpf_taps is not None:
            self._lpf_zi[:] = 0
        self._lpf_phase = 0
        self._rrc_zi[:] = 0
        self._mm.reset()
        self._prev_sym = np.complex64(0)
        self._freq_offset_hz = 0.0
        self._freq_phase = 0.0
