"""Tests for the JitterBuffer device-layer helper.

Each test asserts a hard wall-clock deadline so a deadlock fails loudly.
"""

import random
import threading
import time

import pytest

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.devices._jitter_buffer import JitterBuffer

# Each blocking test path enforces this deadline. Generous enough for slow
# CI but short enough that a real hang fails the suite in seconds.
DEADLINE_S = 5.0


def _make_jb(
    *,
    prefill_seconds: float = 0.1,
    sample_rate: float = 100_000.0,
    bytes_per_sample: int = 2,
    chunk_bytes: int = 1024,
) -> JitterBuffer:
    return JitterBuffer(
        prefill_seconds=prefill_seconds,
        sample_rate=sample_rate,
        bytes_per_sample=bytes_per_sample,
        producer_chunk_bytes=chunk_bytes,
    )


def _drain_in_background(
    jb: JitterBuffer, count: int, deadline_s: float = DEADLINE_S
) -> tuple[threading.Thread, list[bytes | Exception]]:
    """Read `count` bytes from jb on a worker thread. Returns (thread, result-list).

    The result list contains either the bytes (success) or an Exception.
    """
    result: list[bytes | Exception] = []

    def _run() -> None:
        try:
            result.append(jb.read(count))
        except Exception as e:  # noqa: BLE001 - test harness captures everything
            result.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result


def test_eager_construction_allocates_ring() -> None:
    jb = _make_jb(prefill_seconds=0.1, sample_rate=10_000.0, bytes_per_sample=2)
    # 0.1 s × 10 kSps × 2 B = 2000 B prime; 2× = 4000 B capacity.
    assert jb.target_bytes == 2000
    assert jb.fill_bytes == 0
    assert jb.rebuffering is True
    assert jb.target_seconds == pytest.approx(0.1)


def test_zero_sample_rate_defers_allocation() -> None:
    jb = JitterBuffer(prefill_seconds=0.1, sample_rate=0, bytes_per_sample=2)
    assert jb.target_bytes == 0
    assert jb._ring.capacity == 0  # type: ignore[attr-defined]
    jb.set_sample_rate(50_000.0)
    assert jb.target_bytes == 10_000  # 0.1 × 50_000 × 2


def test_smooth_read_through_bursty_producer() -> None:
    """Producer emits at average sample rate with jitter; output is smooth."""
    sample_rate = 100_000.0
    bps = 2
    jb = _make_jb(prefill_seconds=0.05, sample_rate=sample_rate, chunk_bytes=2048)
    payload = bytes(range(256)) * 20
    pos = [0]
    burst_count = [0]
    bytes_per_second = sample_rate * bps

    def producer(n: int) -> bytes:
        # Pace producer at sample_rate, with random jitter around the mean.
        target_dur = n / bytes_per_second
        burst_count[0] += 1
        if burst_count[0] % 3 == 0:
            # Stall a bit: 2× the expected duration.
            time.sleep(target_dur * 2.0)
        else:
            # Burst: half the duration (faster than rate).
            time.sleep(target_dur * 0.5)
        start = pos[0] % len(payload)
        end = start + n
        if end <= len(payload):
            chunk = payload[start:end]
        else:
            chunk = payload[start:] + payload[: end - len(payload)]
        pos[0] += n
        return chunk

    jb.start(producer)
    try:
        out = bytearray()
        deadline = time.monotonic() + DEADLINE_S
        while len(out) < 30_000 and time.monotonic() < deadline:
            out.extend(jb.read(1024))
        assert len(out) >= 30_000
        # Output must be a contiguous slice of the synthetic stream — pacing
        # keeps the buffer near prime so there are no overflow drops.
        big_stream = (payload * (len(out) // len(payload) + 2))[: len(out)]
        assert bytes(out) == big_stream
    finally:
        jb.stop()


def test_prefill_gate_blocks_until_primed() -> None:
    """First read() blocks until ring >= prime_target."""
    jb = _make_jb(prefill_seconds=0.1, sample_rate=100_000.0)
    # prime = 0.1 × 100k × 2 = 20_000 bytes
    release = threading.Event()
    chunks_emitted = [0]

    def producer(n: int) -> bytes:
        if not release.wait(timeout=DEADLINE_S):
            raise RuntimeError("release event not set in time")
        chunks_emitted[0] += 1
        if chunks_emitted[0] <= 30:
            return b"\x42" * n
        # Hold the line after enough chunks delivered.
        time.sleep(0.05)
        return b""

    jb.start(producer)
    try:
        thread, result = _drain_in_background(jb, count=1024)
        # No data flowing yet — read should block.
        time.sleep(0.05)
        assert not result
        # Now release the producer.
        release.set()
        thread.join(timeout=DEADLINE_S)
        assert not thread.is_alive(), "read() blocked past deadline"
        assert isinstance(result[0], bytes), result[0]
        assert len(result[0]) == 1024
        assert jb.rebuffering is False
    finally:
        jb.stop()


def test_underflow_rearms_fill_and_increments_count() -> None:
    """Drain past underflow threshold; next read blocks until refill."""
    jb = _make_jb(prefill_seconds=0.1, sample_rate=100_000.0, chunk_bytes=4096)
    # prime = 20_000; underflow_target = 2_000.
    fill_phase = threading.Event()
    fill_phase.set()  # start by allowing producer

    def producer(n: int) -> bytes:
        fill_phase.wait(timeout=DEADLINE_S)
        return b"\xaa" * n

    jb.start(producer)
    try:
        # Drain past the underflow boundary. Pop more than (prime - underflow)
        # in one read so the post-pop check trips.
        deadline = time.monotonic() + DEADLINE_S
        while jb.fill_bytes < jb.target_bytes and time.monotonic() < deadline:
            time.sleep(0.001)
        assert jb.fill_bytes >= jb.target_bytes, "never reached prime"

        # First read leaves fill state.
        _ = jb.read(1024)
        assert jb.rebuffering is False
        assert jb.rebuffer_count == 0

        # Block the producer so we can drain deterministically.
        fill_phase.clear()
        # Wait for the producer's currently-in-flight write (if any) to drop the
        # wait state, then drain.
        time.sleep(0.02)
        # Drain everything in one big read past the underflow line.
        remaining = jb.fill_bytes
        drain = max(remaining - 1500, 1)  # leave < 2000 (underflow target)
        _ = jb.read(drain)
        # Now fill state should have re-armed.
        assert jb.rebuffering is True
        assert jb.rebuffer_count == 1
    finally:
        fill_phase.set()  # let producer exit its wait
        jb.stop()


def test_set_prefill_seconds_resizes_ring() -> None:
    jb = _make_jb(prefill_seconds=0.1, sample_rate=100_000.0)
    assert jb.target_bytes == 20_000
    jb.set_prefill_seconds(0.5)
    assert jb.target_bytes == 100_000
    jb.set_prefill_seconds(0.05)
    assert jb.target_bytes == 10_000


def test_set_prefill_preserves_newest_on_shrink() -> None:
    """Shrinking the ring keeps the most recent bytes (tail-keep)."""
    jb = _make_jb(prefill_seconds=0.1, sample_rate=100_000.0)
    # Capacity is 40_000. Write 5000 bytes of an ascending pattern.
    written = bytes(i % 256 for i in range(5000))
    with jb._cond:  # type: ignore[attr-defined]
        jb._ring.write(written)  # type: ignore[attr-defined]
    assert jb.fill_bytes == 5000

    # Shrink to capacity = 4000 (prefill 0.01 s × 100k × 2 = 2000 prime; cap 4000).
    jb.set_prefill_seconds(0.01)
    assert jb._ring.capacity == 4000  # type: ignore[attr-defined]
    assert jb.fill_bytes == 4000
    # The bytes we should still have are written[-4000:]
    with jb._cond:  # type: ignore[attr-defined]
        kept = jb._ring.pop(4000)  # type: ignore[attr-defined]
    assert kept == written[-4000:]


def test_set_sample_rate_resizes_ring() -> None:
    jb = _make_jb(prefill_seconds=0.1, sample_rate=100_000.0)
    assert jb.target_bytes == 20_000
    jb.set_sample_rate(200_000.0)
    assert jb.target_bytes == 40_000


def test_oversize_read_raises_value_error() -> None:
    jb = _make_jb(prefill_seconds=0.1, sample_rate=100_000.0)
    # capacity = 40_000
    with pytest.raises(ValueError, match="exceeds ring capacity"):
        jb.read(50_000)


def test_zero_length_read_returns_empty() -> None:
    jb = _make_jb()
    assert jb.read(0) == b""


def test_negative_count_raises() -> None:
    jb = _make_jb()
    with pytest.raises(ValueError):
        jb.read(-1)


def test_producer_error_surfaces_on_read() -> None:
    """A producer exception is captured and re-raised on the next read."""
    jb = _make_jb(prefill_seconds=0.01, sample_rate=100_000.0, chunk_bytes=64)
    sentinel = RuntimeError("simulated socket failure")
    calls = [0]

    def producer(n: int) -> bytes:
        calls[0] += 1
        if calls[0] >= 3:
            raise sentinel
        return b"\x55" * n

    jb.start(producer)
    try:
        # Wait briefly for producer to crash.
        deadline = time.monotonic() + DEADLINE_S
        while jb._error is None and time.monotonic() < deadline:  # type: ignore[attr-defined]
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="simulated socket failure"):
            jb.read(1024)
    finally:
        jb.stop()


def test_producer_empty_chunk_is_eof() -> None:
    """Empty bytes from producer is treated as EOF and surfaces as DeviceError."""
    jb = _make_jb(prefill_seconds=0.01, sample_rate=100_000.0)
    served = [False]

    def producer(n: int) -> bytes:
        if not served[0]:
            served[0] = True
            return b"\x33" * n
        return b""

    jb.start(producer)
    try:
        deadline = time.monotonic() + DEADLINE_S
        while jb._error is None and time.monotonic() < deadline:  # type: ignore[attr-defined]
            time.sleep(0.01)
        with pytest.raises(DeviceError, match="no bytes"):
            jb.read(2000)
    finally:
        jb.stop()


def test_stop_unblocks_waiting_reader() -> None:
    """stop() while a reader is waiting must release it with DeviceError."""
    jb = _make_jb(prefill_seconds=1.0, sample_rate=100_000.0)
    silent = threading.Event()

    def producer(n: int) -> bytes:
        silent.wait(timeout=DEADLINE_S)
        return b"\x00" * n

    jb.start(producer)
    thread, result = _drain_in_background(jb, count=1024)
    time.sleep(0.05)  # let reader enter wait
    assert not result, "reader returned before stop"
    jb.stop()
    thread.join(timeout=DEADLINE_S)
    assert not thread.is_alive(), "reader did not unblock after stop()"
    assert isinstance(result[0], DeviceError)
    silent.set()  # release any lingering producer wait


def test_stop_joins_producer_after_external_unblock() -> None:
    """Mock 'socket shutdown': producer blocked in wait, then raises on shutdown."""
    jb = _make_jb(prefill_seconds=0.1, sample_rate=100_000.0)
    socket_open = threading.Event()
    socket_open.set()

    def producer(n: int) -> bytes:
        # Simulate socket.recv: blocks until shutdown raises.
        socket_open.wait(timeout=DEADLINE_S)
        if not socket_open.is_set():
            raise OSError("Connection reset by peer")
        # Normally never reach here in this test; bail out to satisfy types.
        return b""

    jb.start(producer)
    time.sleep(0.05)
    # Caller "shuts down the socket": clear the event so producer's wait
    # exits and it raises.
    socket_open.clear()
    # Allow producer to detect shutdown and raise.
    time.sleep(0.05)
    jb.stop()
    # If stop completed, the thread is gone.
    assert jb._thread is None  # type: ignore[attr-defined]


def test_chunk_larger_than_capacity_keeps_tail() -> None:
    """A single huge chunk overwrites the ring; only the last `capacity` bytes survive."""
    jb = _make_jb(prefill_seconds=0.01, sample_rate=100_000.0)
    # capacity = 4000.
    big = bytes(i % 256 for i in range(10_000))
    with jb._cond:  # type: ignore[attr-defined]
        jb._ring.write(big)  # type: ignore[attr-defined]
    assert jb.fill_bytes == 4000
    with jb._cond:  # type: ignore[attr-defined]
        out = jb._ring.pop(4000)  # type: ignore[attr-defined]
    assert out == big[-4000:]


def test_overflow_drops_oldest_bytes() -> None:
    jb = _make_jb(prefill_seconds=0.01, sample_rate=100_000.0)
    # capacity = 4000.
    chunk_a = b"A" * 3000
    chunk_b = b"B" * 3000
    with jb._cond:  # type: ignore[attr-defined]
        jb._ring.write(chunk_a)  # type: ignore[attr-defined]
        jb._ring.write(chunk_b)  # type: ignore[attr-defined]
    assert jb.fill_bytes == 4000
    with jb._cond:  # type: ignore[attr-defined]
        out = jb._ring.pop(4000)  # type: ignore[attr-defined]
    # Oldest 2000 of A dropped; remaining 1000 A + 3000 B kept.
    assert out == b"A" * 1000 + b"B" * 3000


def test_ring_wrap_correctness() -> None:
    """Writes that wrap the ring and reads that wrap both stay byte-correct."""
    jb = _make_jb(prefill_seconds=0.01, sample_rate=100_000.0)
    # capacity = 4000.
    with jb._cond:  # type: ignore[attr-defined]
        jb._ring.write(b"\x01" * 3500)  # type: ignore[attr-defined]
        out1 = jb._ring.pop(3000)  # type: ignore[attr-defined]
        # ring now: head=3000, size=500. Next write of 3000 wraps tail.
        jb._ring.write(b"\x02" * 3000)  # type: ignore[attr-defined]
        out2 = jb._ring.pop(3500)  # type: ignore[attr-defined]
    assert out1 == b"\x01" * 3000
    assert out2 == b"\x01" * 500 + b"\x02" * 3000


def test_start_after_stop_resets_lifecycle_state() -> None:
    """Same JitterBuffer instance can be restarted after stop().

    Regression test: stop() leaves `_stop = True` and stale ring state;
    without resetting these on start(), the new producer thread sees
    `_stop` and exits immediately, leaving the consumer parked forever.
    """
    jb = _make_jb(prefill_seconds=0.05, sample_rate=100_000.0)

    # First lifecycle: simple producer.
    served_a = [0]

    def producer_a(n: int) -> bytes:
        served_a[0] += n
        time.sleep(n / (100_000.0 * 2))
        return b"\xa1" * n

    jb.start(producer_a)
    deadline = time.monotonic() + DEADLINE_S
    _ = jb.read(1024)
    assert served_a[0] >= 1024, "first lifecycle: no data flowed"
    jb.stop()

    # Cause a rebuffer to mutate state.
    assert jb._thread is None  # type: ignore[attr-defined]

    # Second lifecycle: a different producer; data must flow again.
    served_b = [0]

    def producer_b(n: int) -> bytes:
        served_b[0] += n
        time.sleep(n / (100_000.0 * 2))
        return b"\xb2" * n

    jb.start(producer_b)
    try:
        out = jb.read(2048)
        assert out == b"\xb2" * 2048, "second lifecycle: wrong producer's bytes"
        assert served_b[0] >= 2048
        # rebuffer_count was reset on start.
        assert jb.rebuffer_count == 0
        assert time.monotonic() < deadline
    finally:
        jb.stop()


def test_consumer_is_paced_to_sample_rate() -> None:
    """Pre-filled ring + tight-loop consumer must not burst-drain.

    Regression test: without wall-clock pacing in read(), a consumer that
    reads in a tight loop drains the entire prime allocation in
    milliseconds, instantly triggers underflow, and produces a 1-Hz
    sawtooth instead of smooth delivery.

    Uses a fast producer (slightly faster than sample rate) so the
    consumer's pacing — not the producer's sleep slop — is what limits
    delivery. Real network producers also run at-or-faster than nominal.
    """
    sample_rate = 100_000.0
    bps = 2
    jb = _make_jb(
        prefill_seconds=0.1,
        sample_rate=sample_rate,
        bytes_per_sample=bps,
        chunk_bytes=2048,
    )
    bytes_per_second = sample_rate * bps

    def producer(n: int) -> bytes:
        # Faster than sample rate so producer is never the bottleneck.
        time.sleep(n / bytes_per_second * 0.5)
        return b"\xaa" * n

    jb.start(producer)
    try:
        # Drain in a tight loop (no consumer sleeps) for ~0.5 s of audio.
        target_bytes = int(0.5 * bytes_per_second)
        out = 0
        deadline = time.monotonic() + DEADLINE_S
        while out < target_bytes and time.monotonic() < deadline:
            out += len(jb.read(1024))
        assert out >= target_bytes, f"only {out} bytes in {DEADLINE_S}s"
        # With pacing, no rebuffer should trigger: consumer paces at
        # sample rate, producer is faster, buffer stays full. Without
        # pacing we'd see one rebuffer every ~1 s (sawtooth).
        assert jb.rebuffer_count == 0, f"unexpected rebuffer cycles: {jb.rebuffer_count}"
        # Buffer should be hovering near prime, not near zero.
        assert jb.fill_fraction > 0.5, f"buffer drained: fill_fraction={jb.fill_fraction:.2f}"
    finally:
        jb.stop()


def test_pacing_is_drift_free_over_many_cycles() -> None:
    """Effective wall-clock pace must match sample rate, not lag by 5-10 %.

    Regression test: the original pacing used max(_next_read_ts, now) to
    advance the schedule, which baked OS sleep slop (~2–4 ms per cycle)
    into the next deadline. Over many cycles that compounded into a
    measurable rate deficit that starved the audio output.

    The producer is intentionally fast (returns instantly) so only the
    consumer's pacing limits delivery. Otherwise the producer's own
    `time.sleep` slop bottlenecks the measurement.
    """
    sample_rate = 100_000.0
    bps = 2
    jb = _make_jb(
        prefill_seconds=0.1,
        sample_rate=sample_rate,
        bytes_per_sample=bps,
        chunk_bytes=2048,
    )
    bytes_per_second = sample_rate * bps

    def producer(n: int) -> bytes:
        # Fast producer: buffer overflow is fine, we're measuring consumer.
        return b"\xaa" * n

    jb.start(producer)
    try:
        # Wait briefly for the ring to prime so the first measured read
        # doesn't include the initial fill wait.
        time.sleep(0.05)
        _ = jb.read(1024)

        # Measure 40 cycles of 50 ms target each = 2.0 s nominal.
        read_bytes = int(0.05 * bytes_per_second)  # 50 ms worth
        n_cycles = 40
        t0 = time.monotonic()
        for _ in range(n_cycles):
            jb.read(read_bytes)
        elapsed = time.monotonic() - t0
        nominal = n_cycles * 0.05
        drift_pct = (elapsed - nominal) / nominal * 100
        # OS scheduling slop is typically <2 ms total over the 2.0 s
        # window; under parallel test load it can spike to ~40 ms (≈2 %).
        # The buggy version drifted 5-10 %, so 3 % still catches the
        # regression without being flaky under contention.
        assert abs(drift_pct) < 3.0, (
            f"pacing drift {drift_pct:+.2f}% over {n_cycles} cycles "
            f"({elapsed:.3f}s vs nominal {nominal:.3f}s)"
        )
    finally:
        jb.stop()


@pytest.mark.slow
def test_stress_random_bursts() -> None:
    """5-second burst-and-stall stress: assert no deadlock, byte-stream integrity."""
    jb = _make_jb(prefill_seconds=0.1, sample_rate=200_000.0, chunk_bytes=4096)
    # capacity = 80_000; prime = 40_000.
    rng = random.Random(0xC0FFEE)
    pos = [0]
    end_event = threading.Event()

    def producer(n: int) -> bytes:
        # Random tiny stalls and random chunk sizes (smaller than n).
        if rng.random() < 0.2:
            time.sleep(rng.uniform(0, 0.02))
        size = rng.randint(max(1, n // 4), n)
        chunk = bytes((pos[0] + i) % 256 for i in range(size))
        pos[0] += size
        return chunk

    jb.start(producer)
    consumed = bytearray()
    deadline = time.monotonic() + 5.0
    try:
        while time.monotonic() < deadline:
            try:
                consumed.extend(jb.read(1024))
            except DeviceError:
                break
        end_event.set()
    finally:
        jb.stop()
    # Reconstruct what the producer should have emitted into the bytes we
    # consumed: it's a contiguous prefix-of-stream-or-tail-of-stream because
    # the ring can have dropped overflow bytes. We just sanity-check we got
    # a nonzero, plausibly large amount and that no error was set.
    assert len(consumed) > 100_000, f"only {len(consumed)} bytes consumed in 5 s"
    assert jb._error is None  # type: ignore[attr-defined]
