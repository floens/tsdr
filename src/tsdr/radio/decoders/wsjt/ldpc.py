"""LDPC(174, 91) belief-propagation decoder.

The per-bit LLR is ``log P(1)/P(0)`` — **positive LLR votes for bit = 1**,
negative votes for bit = 0. This matches ``_ft8_extract_symbol`` which sets
``logl[k] = max(mags where bit_k=1) - max(mags where bit_k=0)``.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from .tables import LDPC_MN, LDPC_N, LDPC_NM, LDPC_NUM_ROWS


@nb.njit(cache=True, fastmath=True)
def _fast_tanh(x: float) -> float:
    if x < -4.97:
        return -1.0
    if x > 4.97:
        return 1.0
    x2 = x * x
    a = x * (945.0 + x2 * (105.0 + x2))
    b = 945.0 + x2 * (420.0 + x2 * 15.0)
    return a / b


@nb.njit(cache=True, fastmath=True)
def _fast_atanh(x: float) -> float:
    x2 = x * x
    a = x * (945.0 + x2 * (-735.0 + x2 * 64.0))
    b = 945.0 + x2 * (-1050.0 + x2 * 225.0)
    return a / b


@nb.njit(cache=True)
def _ldpc_check(codeword: np.ndarray, nm: np.ndarray, num_rows: np.ndarray) -> int:
    """Return the number of parity-check violations for a hard 174-bit codeword."""
    errors = 0
    for m in range(nm.shape[0]):
        x = nb.uint8(0)
        for i in range(num_rows[m]):
            x ^= codeword[nm[m, i]]
        if x != 0:
            errors += 1
    return errors


@nb.njit(cache=True, fastmath=True)
def _bp_decode_jit(
    codeword: np.ndarray,
    max_iters: int,
    nm: np.ndarray,
    mn: np.ndarray,
    num_rows: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Belief-propagation LDPC decoder.

    Returns (plain, min_errors): the best hard-decision codeword found and
    the corresponding parity-error count. ``min_errors == 0`` means a valid
    codeword.
    """
    n = nm.shape[1]  # max columns per check (=7)
    m_count = nm.shape[0]
    n_count = mn.shape[0]

    tov = np.zeros((n_count, 3), dtype=np.float32)
    toc = np.zeros((m_count, n), dtype=np.float32)
    plain = np.zeros(n_count, dtype=np.uint8)
    best_plain = np.zeros(n_count, dtype=np.uint8)
    min_errors = m_count

    for _iter in range(max_iters):
        plain_sum = 0
        for nn in range(n_count):
            v = codeword[nn] + tov[nn, 0] + tov[nn, 1] + tov[nn, 2]
            plain[nn] = 1 if v > 0.0 else 0
            plain_sum += plain[nn]

        if plain_sum == 0:
            # all-zero is a degenerate codeword: prohibited by spec
            break

        errors = _ldpc_check(plain, nm, num_rows)
        if errors < min_errors:
            min_errors = errors
            for nn in range(n_count):
                best_plain[nn] = plain[nn]
            if errors == 0:
                break

        # bit -> check messages
        for mm in range(m_count):
            for n_idx in range(num_rows[mm]):
                nn = nm[mm, n_idx]
                tnm = codeword[nn]
                for m_idx in range(3):
                    if mn[nn, m_idx] != mm:
                        tnm += tov[nn, m_idx]
                toc[mm, n_idx] = _fast_tanh(-tnm * 0.5)

        # check -> bit messages
        for nn in range(n_count):
            for m_idx in range(3):
                mm = mn[nn, m_idx]
                tmn = 1.0
                for n_idx in range(num_rows[mm]):
                    if nm[mm, n_idx] != nn:
                        tmn *= toc[mm, n_idx]
                tov[nn, m_idx] = -2.0 * _fast_atanh(tmn)

    return best_plain, min_errors


@nb.njit(cache=True, fastmath=True)
def _ldpc_decode_jit(
    codeword: np.ndarray,
    max_iters: int,
    nm: np.ndarray,
    mn: np.ndarray,
    num_rows: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Sum-product LDPC decoder (alternative, slower path).

    Uses M*N storage so it's the heavier of the two. ``bp_decode`` is normally
    preferred; this one exists so cross-checks against a sum-product reference
    can be run.
    """
    m_count = nm.shape[0]
    n_count = mn.shape[0]

    m = np.empty((m_count, n_count), dtype=np.float32)
    e = np.zeros((m_count, n_count), dtype=np.float32)
    plain = np.zeros(n_count, dtype=np.uint8)
    best_plain = np.zeros(n_count, dtype=np.uint8)

    for j in range(m_count):
        for i in range(n_count):
            m[j, i] = codeword[i]

    min_errors = m_count

    for _iter in range(max_iters):
        for j in range(m_count):
            rj = num_rows[j]
            for ii1 in range(rj):
                i1 = nm[j, ii1]
                a = 1.0
                for ii2 in range(rj):
                    i2 = nm[j, ii2]
                    if i2 != i1:
                        a *= _fast_tanh(-m[j, i2] * 0.5)
                e[j, i1] = -2.0 * _fast_atanh(a)

        for i in range(n_count):
            ell = codeword[i]
            for j in range(3):
                ell += e[mn[i, j], i]
            plain[i] = 1 if ell > 0.0 else 0

        errors = _ldpc_check(plain, nm, num_rows)
        if errors < min_errors:
            min_errors = errors
            for nn in range(n_count):
                best_plain[nn] = plain[nn]
            if errors == 0:
                break

        for i in range(n_count):
            for ji1 in range(3):
                j1 = mn[i, ji1]
                ell = codeword[i]
                for ji2 in range(3):
                    if ji1 != ji2:
                        j2 = mn[i, ji2]
                        ell += e[j2, i]
                m[j1, i] = ell

    return best_plain, min_errors


def bp_decode(llr: np.ndarray, max_iters: int = 25) -> tuple[np.ndarray, int]:
    """Decode a 174-LLR codeword with belief propagation.

    Args:
        llr: float32 array of length 174, with ``llr[i] = log P(0)/P(1)``.
        max_iters: maximum BP iterations.

    Returns:
        (plain, errors): plain is a uint8 array of length 174; errors is the
        parity-error count for the best hard decision found (0 means a valid
        codeword). The first 91 bits of ``plain`` are the systematic payload.
    """
    cw = np.ascontiguousarray(llr, dtype=np.float32)
    if cw.size != LDPC_N:
        raise ValueError(f"llr must have {LDPC_N} elements, got {cw.size}")
    plain, errors = _bp_decode_jit(cw, int(max_iters), LDPC_NM, LDPC_MN, LDPC_NUM_ROWS)
    return plain, int(errors)


def ldpc_decode(llr: np.ndarray, max_iters: int = 25) -> tuple[np.ndarray, int]:
    """Decode with the sum-product variant.

    Slower than :func:`bp_decode`; provided for cross-checks.
    """
    cw = np.ascontiguousarray(llr, dtype=np.float32)
    if cw.size != LDPC_N:
        raise ValueError(f"llr must have {LDPC_N} elements, got {cw.size}")
    plain, errors = _ldpc_decode_jit(cw, int(max_iters), LDPC_NM, LDPC_MN, LDPC_NUM_ROWS)
    return plain, int(errors)


def ldpc_check(codeword: np.ndarray) -> int:
    """Count parity-check violations for a 174-bit hard codeword."""
    cw = np.ascontiguousarray(codeword, dtype=np.uint8)
    if cw.size != LDPC_N:
        raise ValueError(f"codeword must have {LDPC_N} elements, got {cw.size}")
    return int(_ldpc_check(cw, LDPC_NM, LDPC_NUM_ROWS))


__all__ = ["bp_decode", "ldpc_check"]
