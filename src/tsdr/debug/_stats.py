import time
from collections import deque


class StatsOverlay:
    """Track update rate and sample counts for visualization windows."""

    def __init__(self, max_history: int = 100):
        self.update_times: deque[float] = deque(maxlen=max_history)
        self.sample_counts: deque[int] = deque(maxlen=max_history)

    def record_update(self, sample_count: int) -> None:
        """Record an update with the given sample count."""
        self.update_times.append(time.perf_counter())
        self.sample_counts.append(sample_count)

    def get_updates_per_second(self) -> float:
        """Calculate updates per second from recent history."""
        if len(self.update_times) < 2:
            return 0.0
        elapsed = self.update_times[-1] - self.update_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.update_times) - 1) / elapsed

    def get_total_samples(self) -> int:
        """Get total samples from recent history."""
        return sum(self.sample_counts)

    def get_stats_text(self) -> str:
        """Get formatted stats string for display."""
        ups = self.get_updates_per_second()
        total = self.get_total_samples()
        return f"{ups:.1f} ups | {total:,} samples"
