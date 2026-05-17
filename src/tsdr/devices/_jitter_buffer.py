"""Jitter buffer for network SDR sources.

Absorbs bursty TCP arrivals into a steady stream the I/O worker can read
without exposing network jitter to the DSP pipeline. A background producer
thread fills a fixed-capacity byte ring; `read()` blocks until the
configured pre-fill watermark is reached, then delivers requested bytes.
On underflow (ring < 10% of watermark), the buffer re-arms its fill state
and blocks subsequent reads until the watermark recovers.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Final

from tsdr.core.sdr.exceptions import DeviceError

logger = logging.getLogger(__name__)

# Underflow triggers when the ring drops below this fraction of prime.
# Higher = more frequent but shorter refills (less audio downtime per event).
# Lower = rarer but longer refills.
_UNDERFLOW_FRACTION: Final[float] = 0.5
# Ring is sized at twice prime so a burst arriving while almost full has
# headroom to land before old data is overwritten.
_CAPACITY_MULTIPLE: Final[int] = 2
# Time to wait for the producer thread to exit on stop(). The caller must
# have already unblocked any pending I/O (e.g. socket shutdown); a join
# beyond this means the producer is stuck and we leak the thread.
_STOP_JOIN_TIMEOUT_S: Final[float] = 2.0


class _ByteRing:
    """Fixed-capacity FIFO byte ring with overflow-drops-oldest semantics.

    Not thread-safe; the caller (JitterBuffer) serializes access via its
    condition variable.
    """

    def __init__(self, capacity: int) -> None:
        self._buf = bytearray(capacity)
        self._capacity = capacity
        self._head = 0
        self._tail = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def reset(self) -> None:
        self._head = 0
        self._tail = 0
        self._size = 0

    def write(self, chunk: bytes) -> None:
        """Append chunk; drop oldest bytes on overflow."""
        n = len(chunk)
        if n == 0 or self._capacity == 0:
            return
        if n >= self._capacity:
            # Chunk alone exceeds capacity: keep only the newest tail.
            self._buf[:] = chunk[-self._capacity :]
            self._head = 0
            self._tail = 0
            self._size = self._capacity
            return
        overflow = (self._size + n) - self._capacity
        if overflow > 0:
            self._head = (self._head + overflow) % self._capacity
            self._size -= overflow
        end_space = self._capacity - self._tail
        if n <= end_space:
            self._buf[self._tail : self._tail + n] = chunk
        else:
            self._buf[self._tail :] = chunk[:end_space]
            self._buf[: n - end_space] = chunk[end_space:]
        self._tail = (self._tail + n) % self._capacity
        self._size += n

    def pop(self, n: int) -> bytes:
        """Remove and return n bytes from the head."""
        if n <= 0 or n > self._size:
            return b""
        end_space = self._capacity - self._head
        if n <= end_space:
            data = bytes(self._buf[self._head : self._head + n])
        else:
            data = bytes(self._buf[self._head :]) + bytes(self._buf[: n - end_space])
        self._head = (self._head + n) % self._capacity
        self._size -= n
        return data

    def peek_tail(self, n: int) -> bytes:
        """Return the newest n bytes without removing them."""
        if n <= 0 or n > self._size or self._capacity == 0:
            return b""
        start = (self._tail - n) % self._capacity
        end_space = self._capacity - start
        if n <= end_space:
            return bytes(self._buf[start : start + n])
        return bytes(self._buf[start:]) + bytes(self._buf[: n - end_space])


class JitterBuffer:
    """Absorb network jitter between bursty receive and steady consumption.

    Background thread runs producer(chunk_size) in a loop and fills a
    fixed-capacity byte ring. read(count) blocks until count bytes are
    available; while in fill state (startup or post-underflow) it blocks
    until ring >= prefill_target. A producer exception is captured and
    re-raised on the next read() so the caller never deadlocks.
    """

    def __init__(
        self,
        prefill_seconds: float,
        sample_rate: float,
        bytes_per_sample: int,
        producer_chunk_bytes: int = 65536,
    ) -> None:
        self._prefill_seconds = float(prefill_seconds)
        self._sample_rate = float(sample_rate)
        self._bytes_per_sample = int(bytes_per_sample)
        self._chunk_bytes = int(producer_chunk_bytes)

        self._cond = threading.Condition()
        self._ring = _ByteRing(0)
        self._prime_target = 0
        self._underflow_target = 0

        self._filling = True
        self._rebuffer_count = 0
        self._error: Exception | None = None
        self._stop = False

        # Wall-clock pacing: scheduled time of the next allowed read return.
        # Prevents the consumer from outpacing the producer when the ring
        # holds pre-filled data (which would otherwise drain in one burst
        # and instantly trigger underflow).
        self._next_read_ts: float = 0.0

        self._producer: Callable[[int], bytes] | None = None
        self._thread: threading.Thread | None = None

        # If sample_rate is known at construction, size the ring eagerly so
        # the producer thread can begin draining as soon as start() is
        # called. Otherwise it idles until set_sample_rate establishes a
        # nonzero capacity.
        if self._sample_rate > 0 and self._prefill_seconds > 0:
            with self._cond:
                self._reconfigure_locked(self._prefill_seconds, self._sample_rate)

    def start(self, producer: Callable[[int], bytes]) -> None:
        """Begin the producer thread.

        Producer is called in a loop with `producer_chunk_bytes` until
        `stop()` or until producer raises. The thread idles while the ring
        capacity is zero; subsequent producer calls run with the lock
        released so they can block on I/O without stalling readers.

        Resets per-lifecycle state so the same JitterBuffer instance can
        be reused after a stop (e.g. a device restart): clears any
        stop/error flags, drops stale ring contents, re-arms fill state.
        """
        if self._thread is not None:
            raise RuntimeError("JitterBuffer already started")
        self._producer = producer
        with self._cond:
            self._stop = False
            self._error = None
            self._filling = True
            self._ring.reset()
            self._rebuffer_count = 0
            self._next_read_ts = 0.0
        self._thread = threading.Thread(
            target=self._loop,
            name=f"jitter-{id(self):x}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the producer to exit and join it.

        Caller is responsible for unblocking any pending I/O in the
        producer (e.g. socket.shutdown) before calling stop; otherwise
        the join will time out and leak a daemon thread.

        If the join times out the zombie thread is kept in `_thread` so
        a subsequent `start()` raises RuntimeError instead of silently
        spawning a second producer that would race on the ring.
        """
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "JitterBuffer producer thread did not exit within %.1fs; "
                    "did the caller unblock the underlying I/O?",
                    _STOP_JOIN_TIMEOUT_S,
                )
            else:
                self._thread = None

    def read(self, count: int) -> bytes:
        """Block until `count` bytes are available, then return them.

        Wall-clock paced: returns no faster than the configured sample
        rate × bytes-per-sample. Without this the consumer drains any
        pre-filled ring in one burst and immediately trips underflow,
        producing a sawtooth instead of a steady delivery cadence. Once
        the buffer has primed, this pacing keeps the fill level near
        prime so it can absorb future jitter.

        Raises:
            ValueError: count negative or larger than ring capacity.
            DeviceError: producer raised an exception, or buffer was stopped.
        """
        if count == 0:
            return b""
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")

        # Pacing (outside the lock so the producer can keep filling).
        # Schedule reads as absolute wall-clock slots: each cycle bumps
        # the deadline by `required_dur` regardless of when this cycle
        # actually returned. Using the actual return time (via max(...,
        # now)) would bake OS sleep slop into the next deadline and
        # compound it indefinitely — see CLAUDE.md "Pipeline Pacing".
        rate = self._sample_rate
        bps = self._bytes_per_sample
        if rate > 0 and bps > 0:
            now = time.monotonic()
            required_dur = count / (rate * bps)
            if self._next_read_ts == 0.0 or self._next_read_ts < now - required_dur:
                # First call, or schedule fell more than one cycle behind
                # (initial fill wait, rebuffer wait, pipeline backpressure).
                # Reset to avoid burst-draining the buffer playing catch-up.
                # Cap is one cycle so OS sleep slop (always < required_dur)
                # doesn't trigger reset, preserving drift correction.
                self._next_read_ts = now
            if now < self._next_read_ts:
                time.sleep(self._next_read_ts - now)
            self._next_read_ts += required_dur

        with self._cond:
            if count > self._ring.capacity:
                raise ValueError(
                    f"read({count}) exceeds ring capacity {self._ring.capacity}; "
                    "increase network_buffer_seconds or reduce buffer_samples"
                )
            while True:
                if self._error is not None:
                    raise self._error
                if self._stop:
                    raise DeviceError("JitterBuffer closed")
                if self._filling:
                    if self._ring.size >= self._prime_target:
                        self._filling = False
                    else:
                        self._cond.wait()
                        continue
                if self._ring.size >= count:
                    data = self._ring.pop(count)
                    if self._ring.size < self._underflow_target and not self._filling:
                        self._filling = True
                        self._rebuffer_count += 1
                        logger.debug(
                            "JitterBuffer underflow (size=%d < target=%d), rebuffer #%d",
                            self._ring.size,
                            self._underflow_target,
                            self._rebuffer_count,
                        )
                    return data
                self._cond.wait()

    def set_prefill_seconds(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError(f"prefill_seconds must be positive, got {seconds}")
        with self._cond:
            self._reconfigure_locked(seconds, self._sample_rate)

    def set_sample_rate(self, rate: float) -> None:
        if rate < 0:
            raise ValueError(f"sample_rate must be non-negative, got {rate}")
        with self._cond:
            self._reconfigure_locked(self._prefill_seconds, rate)

    @property
    def fill_bytes(self) -> int:
        return self._ring.size

    @property
    def target_bytes(self) -> int:
        return self._prime_target

    @property
    def fill_seconds(self) -> float:
        bps = self._sample_rate * self._bytes_per_sample
        return self._ring.size / bps if bps > 0 else 0.0

    @property
    def target_seconds(self) -> float:
        return self._prefill_seconds

    @property
    def fill_fraction(self) -> float:
        return self._ring.size / self._prime_target if self._prime_target > 0 else 0.0

    @property
    def rebuffering(self) -> bool:
        return self._filling

    @property
    def rebuffer_count(self) -> int:
        return self._rebuffer_count

    def _reconfigure_locked(self, prefill_seconds: float, sample_rate: float) -> None:
        """Recompute capacity/watermarks; reallocate ring only if capacity changed."""
        self._prefill_seconds = prefill_seconds
        self._sample_rate = sample_rate
        bps = sample_rate * self._bytes_per_sample
        new_capacity = max(int(_CAPACITY_MULTIPLE * prefill_seconds * bps), 0)
        new_prime = max(int(prefill_seconds * bps), 1) if new_capacity > 0 else 0
        new_underflow = max(int(_UNDERFLOW_FRACTION * new_prime), 1) if new_prime > 0 else 0

        self._prime_target = new_prime
        self._underflow_target = new_underflow

        if new_capacity != self._ring.capacity:
            # Capacity change → reallocate ring, preserve newest bytes.
            keep = min(self._ring.size, new_capacity)
            data = self._ring.peek_tail(keep) if keep > 0 else b""
            self._ring = _ByteRing(new_capacity)
            if keep > 0:
                self._ring.write(data)

        if self._ring.size < self._underflow_target:
            self._filling = True

        self._cond.notify_all()

    def _loop(self) -> None:
        """Producer thread main loop."""
        assert self._producer is not None
        while True:
            with self._cond:
                while not self._stop and self._ring.capacity == 0:
                    self._cond.wait()
                if self._stop:
                    return
                chunk_size = self._chunk_bytes
            # Producer call outside the lock — may block on I/O.
            try:
                chunk = self._producer(chunk_size)
            except Exception as e:  # noqa: BLE001 - producer wraps arbitrary I/O
                with self._cond:
                    self._error = e
                    self._cond.notify_all()
                logger.debug("JitterBuffer producer raised: %s", e)
                return
            if not chunk:
                with self._cond:
                    self._error = DeviceError("Producer returned no bytes (EOF)")
                    self._cond.notify_all()
                return
            with self._cond:
                if self._stop:
                    return
                self._ring.write(chunk)
                self._cond.notify_all()
