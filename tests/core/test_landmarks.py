from __future__ import annotations

import numpy as np

from tsdr.core.bandplans import Band, Bandplan
from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.landmarks import find_landmarks, next_target
from tsdr.core.memories import Memory


def _memory(freq: int, name: str = "m") -> Memory:
    return Memory(id=name, frequency=freq, name=name, mode="AM", bandwidth=10_000)


def _bandplan(*bands: tuple[int, int]) -> Bandplan:
    return Bandplan(
        name="test",
        country_name="x",
        country_code="x",
        author_name="x",
        author_url="x",
        filename="test.json",
        bands=tuple(
            Band(name=f"b{i}", type="t", start=lo, end=hi) for i, (lo, hi) in enumerate(bands)
        ),
    )


def _fft_with_peak_at(peak_freq: float) -> FFTUpdateEvent:
    """Synthesize a spectrum with one discoverable peak at (approximately) peak_freq."""
    n_bins = 128
    sample_rate = 2_400_000.0
    center_freq = 14_000_000.0
    freq_resolution = sample_rate / n_bins
    center_bin = n_bins // 2
    spectrum = np.full(n_bins, -80.0, dtype=np.float32)
    peak_bin = int(round((peak_freq - center_freq) / freq_resolution)) + center_bin
    if 0 <= peak_bin < n_bins:
        spectrum[peak_bin] = -30.0  # well above noise floor + threshold
    return FFTUpdateEvent(
        device_id="d",
        spectrum=spectrum,
        frequencies=np.empty(n_bins, dtype=np.float32),
        center_frequency=center_freq,
        sample_rate=sample_rate,
    )


def test_find_landmarks_memories_only() -> None:
    memories = [_memory(14_000_000), _memory(14_200_000)]
    out = find_landmarks(memories, None, None)
    assert out == {14_000_000.0, 14_200_000.0}


def test_find_landmarks_bandplan_edges() -> None:
    bp = _bandplan((14_000_000, 14_350_000), (50_000_000, 54_000_000))
    out = find_landmarks([], bp, None)
    assert out == {14_000_000.0, 14_350_000.0, 50_000_000.0, 54_000_000.0}


def test_find_landmarks_tunable_range_filters() -> None:
    memories = [_memory(10), _memory(1_000_000_000)]
    out = find_landmarks(memories, None, (0.0, 1_000.0))
    assert out == {10.0}


def test_next_target_picks_nearest_in_direction() -> None:
    memories = [_memory(14_100_000), _memory(14_200_000), _memory(13_900_000)]
    assert next_target(1, 14_000_000, None, memories, None, None) == 14_100_000
    assert next_target(-1, 14_000_000, None, memories, None, None) == 13_900_000


def test_next_target_merges_peaks() -> None:
    # Spectrum peak ~+50 kHz; memory at +200 kHz. Peak is closer → wins.
    fft = _fft_with_peak_at(14_050_000)
    memories = [_memory(14_200_000)]
    target = next_target(1, 14_000_000, fft, memories, None, None)
    assert target is not None and 14_000_000 < target < 14_200_000


def test_next_target_empty_returns_none() -> None:
    assert next_target(1, 14_000_000, None, [], None, None) is None


def test_next_target_zero_direction() -> None:
    memories = [_memory(14_100_000)]
    assert next_target(0, 14_000_000, None, memories, None, None) is None


def test_next_target_skips_current_freq() -> None:
    # A landmark exactly at current_freq is *not* in either direction.
    memories = [_memory(14_000_000), _memory(14_100_000)]
    assert next_target(1, 14_000_000, None, memories, None, None) == 14_100_000
