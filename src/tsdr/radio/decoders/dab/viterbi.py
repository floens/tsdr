import numba as nb
import numpy as np

from .constants import VITERBI_GENERATORS, VITERBI_K, VITERBI_STATES


def _build_viterbi_tables():
    """Precompute state transition and output tables for the convolutional code."""
    n_states = VITERBI_STATES
    next_state = np.zeros((n_states, 2), dtype=np.int32)
    output = np.zeros((n_states, 2), dtype=np.int32)

    for state in range(n_states):
        for inp in range(2):
            reg = (inp << (VITERBI_K - 1)) | state
            next_state[state][inp] = reg >> 1

            out = 0
            for g_idx, gen in enumerate(VITERBI_GENERATORS):
                bit = bin(reg & gen).count("1") % 2
                out |= bit << (3 - g_idx)
            output[state][inp] = out

    return next_state, output


_VITERBI_NEXT_STATE, _VITERBI_OUTPUT = _build_viterbi_tables()

# Precompute branch metric sign patterns: +1 for expected 0, -1 for expected 1
# Shape: (64, 2, 4) - [state][input][bit_position]
_VITERBI_BM_SIGNS = np.zeros((VITERBI_STATES, 2, 4), dtype=np.float32)
for _s in range(VITERBI_STATES):
    for _inp in range(2):
        _out = _VITERBI_OUTPUT[_s][_inp]
        for _b in range(4):
            _VITERBI_BM_SIGNS[_s, _inp, _b] = -1.0 if ((_out >> (3 - _b)) & 1) else 1.0
del _s, _inp, _out, _b

# Flat arrays for vectorized Viterbi: 128 transitions (64 states × 2 inputs)
_VITERBI_NS_FLAT = _VITERBI_NEXT_STATE.ravel().astype(np.int32)  # (128,)
_VITERBI_SRC = np.repeat(np.arange(VITERBI_STATES, dtype=np.int32), 2)  # (128,)
_VITERBI_INP = np.tile(np.array([0, 1], dtype=np.uint8), VITERBI_STATES)  # (128,)

# For each target state: its two incoming transitions (predecessor state, input bit,
# and index into the flat BM array). Each state has exactly 2 predecessors.
_VITERBI_PRED_A = np.zeros(VITERBI_STATES, dtype=np.int32)
_VITERBI_PRED_B = np.zeros(VITERBI_STATES, dtype=np.int32)
_VITERBI_PRED_A_INP = np.zeros(VITERBI_STATES, dtype=np.uint8)
_VITERBI_PRED_B_INP = np.zeros(VITERBI_STATES, dtype=np.uint8)
_VITERBI_PRED_A_BM_IDX = np.zeros(VITERBI_STATES, dtype=np.int32)
_VITERBI_PRED_B_BM_IDX = np.zeros(VITERBI_STATES, dtype=np.int32)
_pred_count = np.zeros(VITERBI_STATES, dtype=np.int32)
for _s in range(VITERBI_STATES):
    for _inp in range(2):
        _ns = _VITERBI_NEXT_STATE[_s][_inp]
        _bm_idx = _s * 2 + _inp
        if _pred_count[_ns] == 0:
            _VITERBI_PRED_A[_ns] = _s
            _VITERBI_PRED_A_INP[_ns] = _inp
            _VITERBI_PRED_A_BM_IDX[_ns] = _bm_idx
        else:
            _VITERBI_PRED_B[_ns] = _s
            _VITERBI_PRED_B_INP[_ns] = _inp
            _VITERBI_PRED_B_BM_IDX[_ns] = _bm_idx
        _pred_count[_ns] += 1
del _pred_count, _s, _inp, _ns, _bm_idx

# Direct branch metric sign matrices for each predecessor pair: (64, 4)
_bm_flat = _VITERBI_BM_SIGNS.reshape(128, 4)
_VITERBI_BM_A = _bm_flat[_VITERBI_PRED_A_BM_IDX]  # (64, 4)
_VITERBI_BM_B = _bm_flat[_VITERBI_PRED_B_BM_IDX]  # (64, 4)
del _bm_flat

# Packed predecessor + input_bit into int32: (predecessor << 1) | input_bit
_VITERBI_HIST_A = (_VITERBI_PRED_A.astype(np.int32) << 1) | _VITERBI_PRED_A_INP.astype(np.int32)
_VITERBI_HIST_B = (_VITERBI_PRED_B.astype(np.int32) << 1) | _VITERBI_PRED_B_INP.astype(np.int32)


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
    inf = np.float32(1e9)

    pm = np.full(n_states, inf, dtype=np.float32)
    pm[0] = np.float32(0.0)
    history = np.empty((n_symbols, n_states), dtype=np.int32)

    for t in range(n_symbols):
        s0 = syms[t, 0]
        s1 = syms[t, 1]
        s2 = syms[t, 2]
        s3 = syms[t, 3]
        new_pm = np.full(n_states, inf, dtype=np.float32)
        for st in range(n_states):
            # Branch metric for predecessor A
            bm_val_a = bm_a[st, 0] * s0 + bm_a[st, 1] * s1 + bm_a[st, 2] * s2 + bm_a[st, 3] * s3
            ma = pm[pred_a[st]] + bm_val_a
            # Branch metric for predecessor B
            bm_val_b = bm_b[st, 0] * s0 + bm_b[st, 1] * s1 + bm_b[st, 2] * s2 + bm_b[st, 3] * s3
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


def _viterbi_decode(soft_bits: np.ndarray) -> np.ndarray:
    """Viterbi decode rate 1/4 convolutional code (numba JIT-compiled)."""
    syms = soft_bits.reshape(-1, 4)
    pm, history = _viterbi_forward_jit(
        syms,
        _VITERBI_PRED_A,
        _VITERBI_PRED_B,
        _VITERBI_BM_A,
        _VITERBI_BM_B,
        _VITERBI_HIST_A,
        _VITERBI_HIST_B,
    )
    final_state = int(np.argmin(pm))
    result: np.ndarray = _viterbi_traceback_jit(history, final_state)
    return result
