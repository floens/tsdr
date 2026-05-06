from __future__ import annotations

# TODO: should these be moved?
from typing import TYPE_CHECKING

import numpy as np

from tsdr.core.sdr.datatypes import SignalStatistics

if TYPE_CHECKING:
    from tsdr.radio.dsp.fft import FFTPlan


def compute_fft(
    iq_samples: np.ndarray,
    window: np.ndarray,
    inv_norm_sq: float,
    plan: FFTPlan | None = None,
) -> np.ndarray:
    """Compute FFT power spectrum from IQ samples.

    Args:
        iq_samples: Complex IQ samples (numpy array, dtype=complex64/128)
        window: Pre-computed window function array
        inv_norm_sq: Pre-computed 1/(coherent_gain * N)^2 normalization factor
        plan: Pre-configured FFTPlan for FFTW acceleration.
              Falls back to numpy when None.

    Returns:
        Power spectrum in dB (numpy array, float32)
        - Zero frequency at center (fftshift applied)
        - Values in dBFS scale: 10 * log10(power)
    """
    windowed = iq_samples * window
    if plan is not None:
        fft_result = np.fft.fftshift(plan.execute(windowed))
    else:
        fft_result = np.fft.fftshift(np.fft.fft(windowed))

    # |X|^2 directly avoids the sqrt inside np.abs.
    power = fft_result.real**2 + fft_result.imag**2
    power *= inv_norm_sq

    # Floor before log10 to avoid log(0).
    power_db: np.ndarray = 10.0 * np.log10(power + 1e-10)

    return power_db.astype(np.float32)


def compute_statistics(
    spectrum: np.ndarray,
    center_frequency: float,
    sample_rate: float,
    channel_bandwidth: float | None = None,
) -> SignalStatistics:
    """Compute statistics from power spectrum.

    Args:
        spectrum: Power spectrum in dB (fftshifted, center bin = center freq)
        center_frequency: Center frequency in Hz
        sample_rate: Sample rate in Hz
        channel_bandwidth: Channel bandwidth in Hz for SNR calculation.
    """
    n_bins = len(spectrum)
    center_bin = n_bins // 2

    peak_bin = int(np.argmax(spectrum))
    peak_power = float(spectrum[peak_bin])

    average_power = float(np.mean(spectrum))

    # Estimate noise floor (lower 20th percentile via partial sort)
    k = max(1, n_bins // 5)
    noise_floor = float(np.partition(spectrum, k)[k])
    bin_offset = peak_bin - center_bin

    # Frequency resolution
    freq_resolution = sample_rate / n_bins

    # Peak frequency = center_frequency + (bin_offset * freq_resolution)
    peak_frequency = center_frequency + (bin_offset * freq_resolution)

    # Channel SNR (SDR++ style): peak in-channel vs mean of adjacent sidebands
    # All in dB domain - no linear conversion needed
    channel_snr = None
    if channel_bandwidth is not None and channel_bandwidth > 0:
        half_bw_bins = int(channel_bandwidth / 2 / freq_resolution)
        if 0 < half_bw_bins < n_bins // 2:
            sig_lo = center_bin - half_bw_bins
            sig_hi = center_bin + half_bw_bins
            # Adjacent noise sidebands: 2x channel width on each side
            noise_lo = max(0, sig_lo - half_bw_bins * 2)
            noise_hi = min(n_bins, sig_hi + half_bw_bins * 2)
            signal_peak = float(np.max(spectrum[sig_lo:sig_hi]))
            left_noise = spectrum[noise_lo:sig_lo]
            right_noise = spectrum[sig_hi:noise_hi]
            noise_bins = np.concatenate([left_noise, right_noise])
            if len(noise_bins) > 0:
                channel_snr = signal_peak - float(np.mean(noise_bins))

    return SignalStatistics(
        peak_power=peak_power,
        average_power=average_power,
        peak_frequency=peak_frequency,
        peak_bin=peak_bin,
        noise_floor=noise_floor,
        dynamic_range=peak_power - noise_floor,
        channel_snr=channel_snr,
    )


def apply_dc_offset_correction(iq_samples: np.ndarray) -> np.ndarray:
    """Remove DC offset from IQ samples.

    Many SDR devices have a DC spike at the center frequency due to
    imperfections in the mixer. Subtracting the mean removes it.
    """
    mean: np.ndarray = np.mean(iq_samples)  # type: ignore[assignment]
    result: np.ndarray = iq_samples - mean  # type: ignore[assignment]
    return result


def apply_iq_imbalance_correction(iq_samples: np.ndarray) -> np.ndarray:
    """Correct IQ imbalance.

    IQ imbalance occurs when I and Q channels have different gains or phases.
    This applies a simple correction by normalizing the I and Q channels.

    Args:
        iq_samples: Complex IQ samples

    Returns:
        IQ-corrected samples

    Example:
        >>> iq_corrected = apply_iq_imbalance_correction(iq_samples)
    """
    # Normalize I and Q to have the same variance: scale Q to match I's std.
    i_samples = iq_samples.real
    q_samples = iq_samples.imag

    i_std = np.std(i_samples)
    q_std = np.std(q_samples)

    if i_std > 0 and q_std > 0:
        scale_factor = i_std / q_std
        q_normalized: np.ndarray = q_samples * scale_factor  # type: ignore[assignment]
        result: np.ndarray = i_samples + 1j * q_normalized  # type: ignore[assignment]
        return result
    else:
        return iq_samples
