import numba
import numpy as np


@numba.njit(cache=True)
def syn_filt(a: np.ndarray, x: np.ndarray, mem: np.ndarray, update_mem: bool) -> np.ndarray:
    """IIR synthesis filter 1/A(z).

    a: LPC coefficients (11,), a[0] = 1.0
    x: input signal (L_subfr,)
    mem: filter memory (10,), last 10 output samples
    update_mem: if True, update mem in-place with last 10 outputs

    Returns output signal (L_subfr,).

    Output samples are saturated to int16 range.
    """
    p = len(a) - 1  # LPC order = 10
    lg = len(x)
    y = np.empty(lg + p, dtype=np.float64)

    # Copy memory into temp buffer
    for i in range(p):
        y[i] = mem[i]

    # Filter with int16 saturation on output
    for i in range(lg):
        s = x[i]
        for j in range(1, p + 1):
            s -= a[j] * y[p + i - j]
        # Saturate to int16 range (matches C's extract_h behavior)
        if s > 32767.0:
            s = 32767.0
        elif s < -32768.0:
            s = -32768.0
        y[p + i] = s

    if update_mem:
        for i in range(p):
            mem[i] = y[lg + i]

    return y[p:]


@numba.njit(cache=True)
def pred_lt(
    exc: np.ndarray,
    offset: int,
    t0: int,
    frac: int,
    l_subfr: int,
    coef_1_3: np.ndarray,
    coef_m1_3: np.ndarray,
) -> None:
    """Fractional pitch prediction. Writes into exc[offset:offset+l_subfr].

    exc: full excitation buffer (history + current frame)
    offset: start index for current subframe in exc
    t0: integer pitch lag
    frac: fractional part (-1, 0, or 1)
    l_subfr: subframe length (60)
    coef_1_3: FIR coefficients for frac=+1 (32,)
    coef_m1_3: FIR coefficients for frac=-1 (32,)
    """
    if frac == 0:
        for i in range(l_subfr):
            exc[offset + i] = exc[offset + i - t0]
    elif frac == 1:
        for i in range(l_subfr):
            s = 0.0
            k = offset + i - t0
            for j in range(32):
                s += exc[k + j - 16] * coef_1_3[j]
            exc[offset + i] = 2.0 * s
    elif frac == -1:
        for i in range(l_subfr):
            s = 0.0
            k = offset + i - t0
            for j in range(32):
                s += exc[k + j - 15] * coef_m1_3[j]
            exc[offset + i] = 2.0 * s
