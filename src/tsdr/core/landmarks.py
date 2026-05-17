from __future__ import annotations

from tsdr.core.bandplans import Bandplan
from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.memories import Memory
from tsdr.core.sdr.processing import find_peaks


def find_landmarks(
    memories: list[Memory],
    bandplan: Bandplan | None,
    tunable_range: tuple[float, float] | None,
) -> set[float]:
    """Memory frequencies + bandplan band edges within tunable range."""
    lo, hi = tunable_range if tunable_range is not None else (float("-inf"), float("inf"))
    out: set[float] = set()
    for m in memories:
        f = float(m.frequency)
        if lo <= f <= hi:
            out.add(f)
    if bandplan is not None:
        for band in bandplan.bands:
            for edge in (float(band.start), float(band.end)):
                if lo <= edge <= hi:
                    out.add(edge)
    return out


def next_target(
    direction: int,
    current_freq: float,
    fft: FFTUpdateEvent | None,
    memories: list[Memory],
    bandplan: Bandplan | None,
    tunable_range: tuple[float, float] | None,
) -> float | None:
    """Nearest landmark or FFT peak strictly in `direction`. None if none."""
    if direction == 0:
        return None
    candidates = find_landmarks(memories, bandplan, tunable_range)
    if fft is not None:
        peaks = find_peaks(fft.spectrum, fft.center_frequency, fft.sample_rate)
        candidates.update(freq for freq, _ in peaks)
    in_dir = (c for c in candidates if (c - current_freq) * direction > 0)
    return min(in_dir, key=lambda c: abs(c - current_freq), default=None)
