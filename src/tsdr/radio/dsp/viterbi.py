"""Generalized soft-decision Viterbi decoder for convolutional codes.

Parameterized by constraint length K and generator polynomials.
Supports any rate 1/n code. Uses Numba JIT for the forward pass and traceback.

Register convention: input bit at MSB position (bit K-1), same as the DAB decoder.
Soft bit convention: positive -> bit 1, negative -> bit 0.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _viterbi_forward_jit(
    syms: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    bm_a: np.ndarray,
    bm_b: np.ndarray,
    hist_a: np.ndarray,
    hist_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """JIT-compiled Viterbi forward pass with scalar loops."""
    n_symbols = syms.shape[0]
    n_states = pred_a.shape[0]
    n_gen = syms.shape[1]
    inf = np.float32(1e9)

    pm = np.full(n_states, inf, dtype=np.float32)
    pm[0] = np.float32(0.0)
    history = np.empty((n_symbols, n_states), dtype=np.int32)

    for t in range(n_symbols):
        new_pm = np.full(n_states, inf, dtype=np.float32)
        for st in range(n_states):
            bm_val_a = np.float32(0.0)
            bm_val_b = np.float32(0.0)
            for g in range(n_gen):
                bm_val_a += bm_a[st, g] * syms[t, g]
                bm_val_b += bm_b[st, g] * syms[t, g]
            ma = pm[pred_a[st]] + bm_val_a
            mb = pm[pred_b[st]] + bm_val_b
            if ma <= mb:
                new_pm[st] = ma
                history[t, st] = hist_a[st]
            else:
                new_pm[st] = mb
                history[t, st] = hist_b[st]
        pm = new_pm

    return pm, history


@nb.njit(cache=True)
def _viterbi_traceback_jit(history: np.ndarray, final_state: int) -> np.ndarray:
    """JIT-compiled Viterbi traceback."""
    n_symbols = history.shape[0]
    decoded = np.zeros(n_symbols, dtype=np.uint8)
    state = final_state
    for t in range(n_symbols - 1, -1, -1):
        h = history[t, state]
        decoded[t] = h & 1
        state = h >> 1
    return decoded


class ViterbiDecoder:
    """Soft-decision Viterbi decoder parameterized by constraint length K and generator polynomials."""

    def __init__(self, k: int, generators: list[int]):
        self.k = k
        self.generators = generators
        self.n_states = 1 << (k - 1)
        self.rate_inv = len(generators)
        self._build_tables()

    def _build_tables(self):
        k = self.k
        n_states = self.n_states
        n_gen = self.rate_inv

        next_state = np.zeros((n_states, 2), dtype=np.int32)
        output = np.zeros((n_states, 2), dtype=np.int32)

        for state in range(n_states):
            for inp in range(2):
                reg = (inp << (k - 1)) | state
                next_state[state][inp] = reg >> 1

                out = 0
                for g_idx, gen in enumerate(self.generators):
                    bit = bin(reg & gen).count("1") % 2
                    out |= bit << (n_gen - 1 - g_idx)
                output[state][inp] = out

        # Branch metric signs: +1 for expected 0, -1 for expected 1
        bm_signs = np.zeros((n_states, 2, n_gen), dtype=np.float32)
        for s in range(n_states):
            for inp in range(2):
                o = output[s][inp]
                for b in range(n_gen):
                    bm_signs[s, inp, b] = -1.0 if ((o >> (n_gen - 1 - b)) & 1) else 1.0

        # Predecessor arrays: each state has exactly 2 incoming transitions
        pred_a = np.zeros(n_states, dtype=np.int32)
        pred_b = np.zeros(n_states, dtype=np.int32)
        pred_a_inp = np.zeros(n_states, dtype=np.int32)
        pred_b_inp = np.zeros(n_states, dtype=np.int32)
        pred_a_bm_idx = np.zeros(n_states, dtype=np.int32)
        pred_b_bm_idx = np.zeros(n_states, dtype=np.int32)

        pred_count = np.zeros(n_states, dtype=np.int32)
        for s in range(n_states):
            for inp in range(2):
                ns = next_state[s][inp]
                bm_idx = s * 2 + inp
                if pred_count[ns] == 0:
                    pred_a[ns] = s
                    pred_a_inp[ns] = inp
                    pred_a_bm_idx[ns] = bm_idx
                else:
                    pred_b[ns] = s
                    pred_b_inp[ns] = inp
                    pred_b_bm_idx[ns] = bm_idx
                pred_count[ns] += 1

        bm_flat = bm_signs.reshape(n_states * 2, n_gen)
        self._pred_a = pred_a
        self._pred_b = pred_b
        self._bm_a = bm_flat[pred_a_bm_idx].copy()
        self._bm_b = bm_flat[pred_b_bm_idx].copy()
        self._hist_a = (pred_a << 1) | pred_a_inp
        self._hist_b = (pred_b << 1) | pred_b_inp

    def decode(self, soft_bits: np.ndarray) -> np.ndarray:
        """Decode soft bits. Length must be multiple of rate_inv.

        Soft bit convention: positive -> bit 1, negative -> bit 0.
        Returns hard bits (uint8 array of 0/1).
        """
        syms = soft_bits.astype(np.float32).reshape(-1, self.rate_inv)
        pm, history = _viterbi_forward_jit(
            syms,
            self._pred_a,
            self._pred_b,
            self._bm_a,
            self._bm_b,
            self._hist_a,
            self._hist_b,
        )
        final_state = int(np.argmin(pm))
        result: np.ndarray = _viterbi_traceback_jit(history, final_state)
        return result
