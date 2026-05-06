import numpy as np

from .kernels import syn_filt
from .tables import (
    DICO1_CLSP,
    DICO2_CLSP,
    DICO3_CLSP,
    GAMMA3,
    GAMMA4,
    L_SUBFR,
    T_QUA_ENER,
    P,
)

# Bit allocation for 23 parameters
BITNO = np.array([8, 9, 9, 8, 14, 1, 1, 6, 5, 14, 1, 1, 6, 5, 14, 1, 1, 6, 5, 14, 1, 1, 6])


def unpack_frame(coded: bytes) -> np.ndarray:
    """Unpack 18 bytes (137 bits MSB-first) into 23 integer parameters."""
    bits = np.unpackbits(np.frombuffer(coded, dtype=np.uint8))[:137]
    params = np.empty(23, dtype=np.int32)
    pos = 0
    for i in range(23):
        n = BITNO[i]
        val = 0
        for _b in range(n):
            val = (val << 1) | int(bits[pos])
            pos += 1
        params[i] = val
    return params


def decode_lsp(indices: np.ndarray, lsp_old: np.ndarray) -> np.ndarray:
    """Decode 3 codebook indices into 10 LSP values (cosine domain, float)."""
    lsp = np.empty(10, dtype=np.float64)
    lsp[0:3] = DICO1_CLSP[indices[0]]
    lsp[3:6] = DICO2_CLSP[indices[1]]
    lsp[6:10] = DICO3_CLSP[indices[2]]

    # Minimum distance enforcement between codebook boundaries
    # 917/32768 ≈ 0.028 (50 Hz around 1000 Hz)
    gap = 917.0 / 32768.0 - (lsp[2] - lsp[3])
    if gap > 0:
        lsp[2] += gap / 2.0
        lsp[3] -= gap / 2.0

    # 1245/32768 ≈ 0.038 (50 Hz around 1600 Hz)
    gap = 1245.0 / 32768.0 - (lsp[5] - lsp[6])
    if gap > 0:
        lsp[5] += gap / 2.0
        lsp[6] -= gap / 2.0

    # Verify ordering (LSPs are in descending order in cosine domain)
    ordered = True
    for i in range(9):
        if lsp[i] <= lsp[i + 1]:
            ordered = False
            break

    if not ordered:
        lsp[:] = lsp_old
    return lsp


def get_lsp_pol(lsp5: np.ndarray) -> np.ndarray:
    """Compute polynomial from 5 LSP values.

    Builds the symmetric/antisymmetric polynomial used in LSP-to-LPC conversion.
    """
    f = np.zeros(6, dtype=np.float64)
    f[0] = 1.0
    f[1] = -2.0 * lsp5[0]

    for i in range(2, 6):
        f[i] = f[i - 2]
        for k in range(i, 1, -1):
            f[k] = f[k] + f[k - 2] - 2.0 * lsp5[i - 1] * f[k - 1]
        f[1] -= 2.0 * lsp5[i - 1]  # boundary: f[1] -= 2*lsp * f[0], f[0]=1

    return f


def lsp_to_az(lsp: np.ndarray) -> np.ndarray:
    """Convert 10 LSPs to 11 LPC coefficients. a[0] = 1.0."""
    # Split into even/odd
    f1 = get_lsp_pol(lsp[0::2])  # lsp[0,2,4,6,8]
    f2 = get_lsp_pol(lsp[1::2])  # lsp[1,3,5,7,9]

    # f1[i] += f1[i-1]; f2[i] -= f2[i-1]; for i=5..1
    for i in range(5, 0, -1):
        f1[i] = f1[i] + f1[i - 1]
        f2[i] = f2[i] - f2[i - 1]

    a = np.empty(11, dtype=np.float64)
    a[0] = 1.0
    for i in range(1, 6):
        a[i] = 0.5 * (f1[i] + f2[i])
        a[11 - i] = 0.5 * (f1[i] - f2[i])

    return a


def interpolate_lpc(lsp_old: np.ndarray, lsp_new: np.ndarray) -> np.ndarray:
    """Interpolate LSPs for 4 subframes and convert to LPC.

    Returns (4, 11) array of LPC coefficients.
    """
    a_sf = np.empty((4, 11), dtype=np.float64)
    weights_new = [0.25, 0.50, 0.75]
    for sf in range(3):
        w = weights_new[sf]
        lsp_interp = (1.0 - w) * lsp_old + w * lsp_new
        a_sf[sf] = lsp_to_az(lsp_interp)
    a_sf[3] = lsp_to_az(lsp_new)
    return a_sf


def decode_pitch_sf0(index: int) -> tuple[int, int, int, int]:
    """Decode pitch for first subframe (8-bit index).

    Returns (t0, t0_frac, t0_min, t0_max).
    """
    if index < 197:
        t0 = (index + 2) // 3 + 19
        t0_frac = index - t0 * 3 + 58
    else:
        t0 = index - 112
        t0_frac = 0

    t0_min = t0 - 5
    if t0_min < 20:
        t0_min = 20
    t0_max = t0_min + 9
    if t0_max > 143:
        t0_max = 143
        t0_min = t0_max - 9

    return t0, t0_frac, t0_min, t0_max


def decode_pitch_sfn(index: int, t0_min: int) -> tuple[int, int]:
    """Decode pitch for subframes 1-3 (5-bit index).

    Returns (t0, t0_frac).
    """
    i = (index + 2) // 3 - 1
    t0 = t0_min + i
    t0_frac = index - 2 - i * 3
    return t0, t0_frac


def pond_ai(a: np.ndarray, gamma: float) -> np.ndarray:
    """Spectral expansion: a_exp[i] = a[i] * gamma^i."""
    a_exp = a.copy()
    g = gamma
    for i in range(1, len(a)):
        a_exp[i] = a[i] * g
        g *= gamma
    return a_exp


def build_noise_filter(a: np.ndarray, t0: int) -> np.ndarray:
    """Build noise shaping filter from LPC coefficients."""
    ap3 = pond_ai(a, GAMMA3)
    ap4 = pond_ai(a, GAMMA4)

    f_padded = np.zeros(64 + L_SUBFR, dtype=np.float64)
    f = f_padded[64:]  # f[0..59], with f[-64..-1] = 0

    # f[0..10] = ap3[0..10], f[11..59] = 0
    f[: P + 1] = ap3

    # Filter through 1/ap4(z) with zero memory
    mem = np.zeros(P, dtype=np.float64)
    f[:] = syn_filt(ap4, f, mem, False)

    # Pitch contribution with fixed gain of 0.8 (26216/32768 ≈ 0.8)
    for i in range(t0, L_SUBFR):
        f[i] += 0.8 * f[i - t0]

    return f_padded


def decode_algebraic_code(index: int, sign: int, shift: int, f_padded: np.ndarray) -> np.ndarray:
    """Decode 4-pulse algebraic codebook vector.

    index: 14-bit codebook index
    sign: 0 or 1 (negate if 1)
    shift: 0 or 1 (offset pulse positions)
    f_padded: noise filter with 64-sample zero prefix
    """
    # Pulse positions
    pos0 = (index & 0x1F) * 2  # bits 0-4, even positions 0..58
    pos1 = ((index >> 5) & 0x7) * 8 + 2  # bits 5-7, positions 2,10,...,58
    pos2 = ((index >> 8) & 0x7) * 8 + 4  # bits 8-10, positions 4,12,...,60
    pos3 = ((index >> 11) & 0x7) * 8 + 6  # bits 11-13, positions 6,14,...,62

    base = 64 - shift
    gain_i0 = 2896.0 / 2048.0  # sqrt(2) in Q11 -> real

    cod = np.empty(L_SUBFR, dtype=np.float64)
    for i in range(L_SUBFR):
        # cod[i] = gain_i0*F[p0+i] - F[p1+i] + F[p2+i] - F[p3+i]
        p0_val = f_padded[base - pos0 + i] if 0 <= base - pos0 + i < len(f_padded) else 0.0
        p1_val = f_padded[base - pos1 + i] if 0 <= base - pos1 + i < len(f_padded) else 0.0
        p2_val = f_padded[base - pos2 + i] if 0 <= base - pos2 + i < len(f_padded) else 0.0
        p3_val = f_padded[base - pos3 + i] if 0 <= base - pos3 + i < len(f_padded) else 0.0

        cod[i] = gain_i0 * p0_val - p1_val + p2_val - p3_val

    if sign != 0:
        cod = -cod

    return cod


def lpc_gain(a: np.ndarray) -> float:
    """Energy of impulse response of 1/A(z) for 60 samples."""
    # Impulse response: feed [1, 0, 0, ...] through 1/A(z)
    h = np.zeros(L_SUBFR, dtype=np.float64)
    h[0] = 1.0
    mem = np.zeros(P, dtype=np.float64)
    h = syn_filt(a, h, mem, False)
    return float(np.sum(h * h))


def decode_gains(
    index: int,
    bfi: bool,
    a: np.ndarray,
    adaptive_exc: np.ndarray,
    code: np.ndarray,
    last_ener_pit: float,
    last_ener_cod: float,
) -> tuple[float, float, float, float]:
    """Decode pitch and code gains.

    All energy values (last_ener_pit, last_ener_cod, ener_plt, ener_c) live in
    the same Q8-as-float domain (Q8 integer divided by 256).

    Returns (gain_pit_Q12, gain_cod_Q0, last_ener_pit, last_ener_cod).
    """
    ener_lpc = lpc_gain(a)

    # Both energy formulas reduce to:
    #   energy_q8 = log2(signal_energy_float) + log2(lpc_gain_float) - 5.32
    energy_offset = -5.32

    # Adaptive codebook energy
    ener_plt_lin = float(np.sum(adaptive_exc**2)) + 1.0  # +1 avoids log2(0)
    ener_plt = np.log2(ener_plt_lin) + np.log2(max(ener_lpc, 1e-30)) + energy_offset

    # Innovation codebook energy
    code_energy = float(np.sum(code**2))
    ener_c = np.log2(max(code_energy, 1e-30)) + np.log2(max(ener_lpc, 1e-30)) + energy_offset

    # BFI handling
    if bfi:
        last_ener_pit = max(last_ener_pit - 0.5, 0.0)
        last_ener_cod = max(last_ener_cod - 0.5, 0.0)
    else:
        # Prediction
        pred_pit = max(0.5 * last_ener_pit + 0.25 * last_ener_cod - 3.0, 0.0)
        pred_cod = max(0.5 * last_ener_cod + 0.25 * last_ener_pit - 3.0, 0.0)

        # Decode from table (T_QUA_ENER is Q8 integer / 256 = real)
        last_ener_pit = T_QUA_ENER[index, 0] + pred_pit
        last_ener_cod = T_QUA_ENER[index, 1] + pred_cod

        # Limit
        last_ener_pit = min(last_ener_pit, 27.0)
        last_ener_cod = min(last_ener_cod, 25.0)

    # Pitch gain (Q12): pow(2, (last_ener_pit - ener_plt)/2 + 12)
    # The +12 puts the result in Q12 scale (2^12 = 4096 = 1.0 in Q12)
    gain_pit_q12 = 2.0 ** (0.5 * (last_ener_pit - ener_plt) + 12.0)
    gain_pit_q12 = min(gain_pit_q12, 4915.0)  # 4915 = 1.2 in Q12

    # Code gain (Q0): pow(2, (last_ener_cod - ener_c)/2)
    gain_cod_q0 = 2.0 ** (0.5 * (last_ener_cod - ener_c))

    return gain_pit_q12, gain_cod_q0, last_ener_pit, last_ener_cod
