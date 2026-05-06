import numpy as np

# Window functions


def _window_hamming(m: int) -> np.ndarray:
    """Symmetric Hamming window of length *m*."""
    if m <= 1:
        return np.ones(m)
    n = np.arange(m)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (m - 1))


def _window_nuttall(m: int) -> np.ndarray:
    """Symmetric Nuttall (4-term Blackman-Nuttall) window."""
    if m <= 1:
        return np.ones(m)
    n = np.arange(m)
    a0, a1, a2, a3 = 0.3635819, 0.4891775, 0.1365995, 0.0106411
    x = 2.0 * np.pi * n / (m - 1)
    return a0 - a1 * np.cos(x) + a2 * np.cos(2.0 * x) - a3 * np.cos(3.0 * x)


def _window_kaiser(m: int, beta: float) -> np.ndarray:
    """Symmetric Kaiser window with parameter *beta*."""
    if m <= 1:
        return np.ones(m)
    n = np.arange(m)
    alpha = (m - 1) / 2.0
    return np.i0(beta * np.sqrt(1.0 - ((n - alpha) / alpha) ** 2)) / np.i0(beta)


def _get_window(window: str | tuple[str, float], m: int) -> np.ndarray:
    if isinstance(window, tuple):
        name, param = window[0], window[1]
        if name == "kaiser":
            return _window_kaiser(m, param)
        raise ValueError(f"Unknown parameterized window: {name!r}")
    if window == "hamming":
        return _window_hamming(m)
    if window == "nuttall":
        return _window_nuttall(m)
    raise ValueError(f"Unknown window: {window!r}")


def firwin(
    numtaps: int,
    cutoff: float | list[float] | np.ndarray,
    *,
    fs: float | None = None,
    window: str | tuple[str, float] = "hamming",
    pass_zero: bool = True,
) -> np.ndarray:
    """Design a FIR filter using the window method."""
    cutoff_arr = np.atleast_1d(np.asarray(cutoff, dtype=np.float64))

    # Normalise to [0, 1] relative to Nyquist
    if fs is not None:
        cutoff_arr = cutoff_arr / (fs / 2.0)

    if np.any(cutoff_arr <= 0) or np.any(cutoff_arr >= 1):
        raise ValueError(f"Cutoff frequencies must be in (0, 1) Nyquist, got {cutoff_arr}")

    # Ideal impulse responses (windowed-sinc)
    alpha = (numtaps - 1) / 2.0
    n = np.arange(numtaps)
    m = n - alpha
    # Avoid division by zero at the centre tap
    m_safe = np.where(m == 0, 1.0, m)

    if cutoff_arr.size == 1:
        # Lowpass
        fc = cutoff_arr[0]
        h = np.where(m == 0, fc, np.sin(np.pi * fc * m_safe) / (np.pi * m_safe))
    elif cutoff_arr.size == 2:
        # Bandpass (pass_zero=False) or band-stop
        f1, f2 = cutoff_arr[0], cutoff_arr[1]
        h_low = np.where(m == 0, f1, np.sin(np.pi * f1 * m_safe) / (np.pi * m_safe))
        h_high = np.where(m == 0, f2, np.sin(np.pi * f2 * m_safe) / (np.pi * m_safe))
        if not pass_zero:
            # Bandpass: difference of two lowpass
            h = h_high - h_low
        else:
            # Band-stop: complement of bandpass
            h = np.where(m == 0, 1.0, 0.0) - (h_high - h_low)
    else:
        raise ValueError("cutoff must have 1 or 2 elements")

    # Apply window
    w = _get_window(window, numtaps)
    h *= w

    # Normalise gain
    if pass_zero:
        # Lowpass / band-stop: unity gain at DC
        h /= np.sum(h)
    else:
        # Bandpass: unity gain at centre frequency
        f_centre = (cutoff_arr[0] + cutoff_arr[1]) / 2.0
        gain = np.abs(np.sum(h * np.exp(-1j * np.pi * f_centre * n)))
        if gain > 0:
            h /= gain

    result: np.ndarray = h
    return result


def lfilter_zi(
    b: np.ndarray | list[float],
    a: np.ndarray | list[float] | float,
) -> np.ndarray:
    """Compute steady-state initial conditions for lfilter."""
    b_ = np.atleast_1d(np.asarray(b, dtype=np.float64))
    a_ = np.atleast_1d(np.asarray(a, dtype=np.float64))

    # Normalise so a[0] == 1
    if a_[0] != 1.0:
        b_ = b_ / a_[0]
        a_ = a_ / a_[0]

    n = max(len(a_), len(b_))

    # Pad to equal length
    if len(b_) < n:
        b_ = np.r_[b_, np.zeros(n - len(b_))]
    if len(a_) < n:
        a_ = np.r_[a_, np.zeros(n - len(a_))]

    # Build the companion matrix system (I - A) * zi = B
    # For direct-form II transposed, at steady state with x=1:
    #   z[k] - z[k+1] + a[k+1]*z[0] = b[k+1] - a[k+1]*b[0]
    #   z[n-2]         + a[n-1]*z[0] = b[n-1] - a[n-1]*b[0]
    m = n - 1
    if m == 0:
        return np.array([], dtype=np.float64)

    lhs = np.eye(m)
    for i in range(m - 1):
        lhs[i, i + 1] = -1.0
    lhs[:, 0] += a_[1:n]

    rhs = b_[1:n] - a_[1:n] * b_[0]

    result: np.ndarray = np.linalg.solve(lhs, rhs)
    return result


# butter


def _cplx_poly(roots: np.ndarray) -> np.ndarray:
    """Polynomial coefficients from roots: (z - r0)(z - r1)..."""
    p = np.array([1.0], dtype=np.complex128)
    for r in roots:
        p = np.convolve(p, [1.0, -r])
    return p


def butter(
    order: int, wn: float | list[float] | np.ndarray, *, btype: str = "low"
) -> tuple[np.ndarray, np.ndarray]:
    """Butterworth digital filter design returning (b, a) coefficients.

    *wn* is normalised to [0, 1] where 1 = Nyquist.
    """
    wn_arr = np.atleast_1d(np.asarray(wn, dtype=np.float64))

    # Sampling frequency for normalised digital filters
    fs = 2.0
    fs2 = 2.0 * fs  # = 4.0, used in bilinear transform

    # Pre-warp to analog frequencies
    warped = 2.0 * fs * np.tan(np.pi * wn_arr / fs)

    # Analog Butterworth prototype: poles on the unit circle in the left half plane
    k_range = np.arange(order)
    proto_poles = np.exp(1j * np.pi * (2 * k_range + order + 1) / (2 * order))

    if btype == "low":
        wc = warped[0]
        s_poles = wc * proto_poles
        s_zeros = np.array([], dtype=np.complex128)
        k_analog = wc**order
    elif btype == "band":
        bw = warped[1] - warped[0]
        w0 = np.sqrt(warped[0] * warped[1])
        bp_poles = np.empty(2 * order, dtype=np.complex128)
        for i, p in enumerate(proto_poles):
            sp = p * bw / 2.0
            dp = np.sqrt(sp * sp - w0 * w0 + 0j)
            bp_poles[2 * i] = sp + dp
            bp_poles[2 * i + 1] = sp - dp
        s_poles = bp_poles
        s_zeros = np.zeros(order, dtype=np.complex128)
        k_analog = bw**order
    else:
        raise ValueError(f"Unsupported btype: {btype!r}")

    # Bilinear transform: z = (1 + s/fs2) / (1 - s/fs2)
    z_poles = (1.0 + s_poles / fs2) / (1.0 - s_poles / fs2)

    if len(s_zeros) > 0:
        z_zeros = (1.0 + s_zeros / fs2) / (1.0 - s_zeros / fs2)
    else:
        z_zeros = np.array([], dtype=np.complex128)

    # Zeros at infinity become zeros at z = -1
    n_inf_zeros = len(s_poles) - len(s_zeros)
    z_zeros = np.append(z_zeros, -np.ones(n_inf_zeros))

    # Digital gain via bilinear gain formula
    k_digital = (
        k_analog * np.real(np.prod(fs2 - s_zeros) / np.prod(fs2 - s_poles))
        if len(s_zeros) > 0
        else k_analog * np.real(1.0 / np.prod(fs2 - s_poles))
    )

    # Convert zpk to tf
    num = _cplx_poly(z_zeros)
    den = _cplx_poly(z_poles)

    b = np.real(k_digital * num)
    a = np.real(den)

    # Normalise a[0] to 1
    b = b / a[0]
    a = a / a[0]

    return b, a


def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Polyphase resampling by rational factor *up/down*. Stateless."""
    if up == down:
        return x.copy()

    max_rate = max(up, down)
    half_len = 10 * max_rate
    n_taps = 2 * half_len + 1
    h = firwin(n_taps, 1.0 / max_rate, window=("kaiser", 5.0)) * up

    n_in = len(x)

    # Zero-insert by `up`, convolve centred, then downsample
    if up > 1:
        x_up = np.zeros(n_in * up, dtype=x.dtype)
        x_up[::up] = x
    else:
        x_up = x

    # mode='same' centres the filter so there is no group-delay shift
    y = np.convolve(x_up, h, mode="same")

    return y[::down]
