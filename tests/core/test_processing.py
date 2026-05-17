"""Unit tests for compute_statistics and find_peaks on synthetic spectra."""

from __future__ import annotations

import numpy as np

from tsdr.core.sdr.processing import find_peaks


def _make_spectrum(
    n_bins: int,
    *,
    noise_db: float = -80.0,
    peaks: dict[int, float] | None = None,
) -> np.ndarray:
    """Flat noise floor with optional peaks injected at specific bin indices."""
    spectrum = np.full(n_bins, noise_db, dtype=np.float32)
    if peaks:
        for bin_idx, db in peaks.items():
            spectrum[bin_idx] = db
    return spectrum


def test_find_peaks_empty_below_threshold() -> None:
    spectrum = _make_spectrum(1024)
    peaks = find_peaks(spectrum, 100e6, 2.4e6)
    assert peaks == ()


def test_find_peaks_returns_strong_signal() -> None:
    n = 1024
    spectrum = _make_spectrum(n, peaks={300: -30.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6)
    assert len(peaks) == 1
    freq_res = 2.4e6 / n
    expected_freq = 100e6 + (300 - n // 2) * freq_res
    assert abs(peaks[0][0] - expected_freq) < 1e-6
    assert peaks[0][1] == -30.0


def test_find_peaks_threshold_rejects() -> None:
    n = 1024
    # Median = noise_db = -80. Threshold = -70. -75 < threshold → rejected.
    spectrum = _make_spectrum(n, peaks={300: -75.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6, threshold_db_above_median=10.0)
    assert peaks == ()


def test_find_peaks_multiple_distinct() -> None:
    n = 1024
    spectrum = _make_spectrum(n, peaks={200: -40.0, 600: -35.0, 900: -30.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6)
    assert len(peaks) == 3
    # Ascending by freq.
    freqs = [p[0] for p in peaks]
    assert freqs == sorted(freqs)


def test_find_peaks_cluster_suppression() -> None:
    n = 1024
    # Three closely-spaced peaks; cluster_bins=5 should keep only the strongest.
    spectrum = _make_spectrum(n, peaks={500: -40.0, 502: -35.0, 504: -30.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6, cluster_bins=5)
    assert len(peaks) == 1
    assert peaks[0][1] == -30.0


def test_find_peaks_top_n_cap() -> None:
    n = 4096
    # 50 peaks 50 bins apart, all above threshold.
    injected = {100 + i * 50: -40.0 + i * 0.1 for i in range(50)}
    spectrum = _make_spectrum(n, peaks=injected)
    peaks = find_peaks(spectrum, 100e6, 2.4e6, max_peaks=10)
    assert len(peaks) == 10


def test_find_peaks_tiny_spectrum() -> None:
    # Spectrum too small to have interior bins → return empty.
    assert find_peaks(np.array([1.0, 2.0], dtype=np.float32), 100e6, 2.4e6) == ()
    assert find_peaks(np.array([], dtype=np.float32), 100e6, 2.4e6) == ()
