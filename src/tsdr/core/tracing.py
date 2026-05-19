import logging
import threading
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import overload

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpanStats:
    """Statistics for a single span type with time-windowed storage."""

    # Store (timestamp, duration_ms) tuples for rolling window
    samples: deque[tuple[float, float]] = field(default_factory=deque)

    def add(self, duration_ms: float) -> None:
        """Add a sample with current timestamp."""
        self.samples.append((perf_counter(), duration_ms))

    def prune(self, window_seconds: float) -> None:
        """Remove samples older than window."""
        cutoff = perf_counter() - window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def get_recent_durations(self, window_seconds: float) -> list[float]:
        """Get durations within the time window."""
        self.prune(window_seconds)
        return [d for _, d in self.samples]

    @property
    def durations(self) -> list[float]:
        return [d for _, d in self.samples]

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def min_ms(self) -> float:
        durations = self.durations
        return min(durations) if durations else 0

    @property
    def max_ms(self) -> float:
        durations = self.durations
        return max(durations) if durations else 0

    @property
    def avg_ms(self) -> float:
        durations = self.durations
        return sum(durations) / len(durations) if durations else 0

    @property
    def p99_ms(self) -> float:
        durations = self.durations
        if not durations:
            return 0
        return float(np.percentile(durations, 99))


# Thread-local storage for parent span tracking
_local = threading.local()

# Global stats with lock protection
_stats_lock = threading.Lock()
_stats: dict[str, SpanStats] = {}


def _get_parent_name() -> str | None:
    return getattr(_local, "parent_name", None)


def _set_parent_name(name: str | None) -> None:
    _local.parent_name = name


def _get_child_time_stack() -> list[float]:
    stack: list[float] | None = getattr(_local, "child_time_stack", None)
    if stack is None:
        stack = []
        _local.child_time_stack = stack
    return stack


def _record_span(name: str, duration_ms: float) -> None:
    with _stats_lock:
        if name not in _stats:
            _stats[name] = SpanStats()
        _stats[name].add(duration_ms)


@contextmanager
def span(name: str):
    """Context manager for tracing a block of code.

    Uses dot notation for nested spans (e.g., process.compute_fft).
    Automatically records a "{name}.other" span for time not covered
    by child spans.
    """
    parent = _get_parent_name()
    full_name = f"{parent}.{name}" if parent else name

    _set_parent_name(full_name)
    stack = _get_child_time_stack()
    stack.append(0.0)

    start = perf_counter()
    try:
        yield
    finally:
        duration_ms = (perf_counter() - start) * 1000
        _set_parent_name(parent)

        child_time_ms = stack.pop()
        _record_span(full_name, duration_ms)

        # Record unaccounted time if this span had children
        if child_time_ms > 0:
            other_ms = duration_ms - child_time_ms
            if other_ms > 0:
                _record_span(f"{full_name}.other", other_ms)

        # Report our duration to parent's accumulator
        if stack:
            stack[-1] += duration_ms


@dataclass(frozen=True, slots=True)
class SpanStatsSnapshot:
    count: int
    min_ms: float
    max_ms: float
    avg_ms: float
    p99_ms: float


def get_stats() -> dict[str, SpanStatsSnapshot]:
    with _stats_lock:
        return {
            name: SpanStatsSnapshot(
                count=s.count,
                min_ms=s.min_ms,
                max_ms=s.max_ms,
                avg_ms=s.avg_ms,
                p99_ms=s.p99_ms,
            )
            for name, s in _stats.items()
        }


def clear_stats() -> None:
    with _stats_lock:
        _stats.clear()


def get_smoothed_stats(window_seconds: float = 5.0) -> dict[str, float]:
    """Get averaged stats over a rolling time window.

    Args:
        window_seconds: Time window for averaging (default 5 seconds)

    Returns:
        Dict mapping span name to average duration in ms
    """
    with _stats_lock:
        result = {}
        for name, stats in _stats.items():
            durations = stats.get_recent_durations(window_seconds)
            if durations:
                result[name] = sum(durations) / len(durations)
        return result


def log_stats(threshold_ms: float = 0, phase: str = "") -> None:
    with _stats_lock:
        if not _stats:
            logger.info("tracing_stats_empty phase=%s", phase)
            return

        logger.info("tracing_stats_begin phase=%s", phase)
        for name in sorted(_stats.keys()):
            stats = _stats[name]
            if stats.avg_ms >= threshold_ms:
                logger.info(
                    "tracing_stats name=%s count=%d min_ms=%.2f max_ms=%.2f avg_ms=%.2f p99_ms=%.2f",
                    name,
                    stats.count,
                    stats.min_ms,
                    stats.max_ms,
                    stats.avg_ms,
                    stats.p99_ms,
                )


@overload
def traced[**P, T](name_or_func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def traced[**P, T](
    name_or_func: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def traced[**P, T](
    name_or_func: str | Callable[P, T] | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for tracing functions. Use as @traced or @traced("custom_name")."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name_or_func if isinstance(name_or_func, str) else func.__name__

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with span(span_name):
                return func(*args, **kwargs)

        return wrapper

    if callable(name_or_func):
        return decorator(name_or_func)
    return decorator
