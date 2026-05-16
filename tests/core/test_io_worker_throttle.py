import math
import queue
import threading
import time
from unittest.mock import MagicMock

from tsdr.core.events.bus import EventBus
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.core.sdr.workers.io_worker import CONFIG_APPLY_MIN_INTERVAL, IOWorker
from tsdr.core.workers import WorkerContext, WorkerLifecycle


class _CountingDevice:
    """Minimal SDRDevice stand-in that records every set_frequency call."""

    def __init__(self) -> None:
        self.freq_calls: list[float] = []
        self.supports_bias_tee = False
        self._sample_rate = 0.0

    def set_frequency(self, freq: float) -> None:
        self.freq_calls.append(freq)

    @property
    def frequency_range(self) -> tuple[float, float] | None:
        return None

    def set_sample_rate(self, rate: float) -> None:
        self._sample_rate = rate

    @property
    def actual_sample_rate(self) -> float:
        return self._sample_rate

    def set_gain(self, gain: float) -> None:
        pass

    def set_auto_gain(self, enable: bool) -> None:
        pass

    def set_bias_tee(self, enable: bool) -> None:
        pass

    def read_samples(self, count: int) -> bytes:
        # Tight read loop (~200 Hz) so the worker checks the queue often;
        # this is the worst case for throttling: without it, the device
        # would see one set_frequency per loop iteration.
        time.sleep(0.005)
        return b"\x80" * count


def test_io_worker_throttles_rapid_config_updates():
    device = _CountingDevice()
    initial_config = DeviceConfig(center_frequency=100e6)

    control_queue: queue.Queue = queue.Queue()
    sample_queue: queue.Queue = queue.Queue(maxsize=64)

    device_context = MagicMock()
    device_context.device = device
    device_context.device_id = "test"
    device_context.config = initial_config
    device_context.control_queue = control_queue
    device_context.sample_queue = sample_queue
    device_context.total_samples_read = 0
    device_context.dropped_samples = 0

    worker = IOWorker(device_context)
    worker.sample_format = SampleFormat.UINT8_IQ

    lifecycle = WorkerLifecycle()
    lifecycle.mark_running()
    ctx = WorkerContext(worker_id="test", event_bus=EventBus(), lifecycle=lifecycle)

    # Drain sample_queue so put() never blocks.
    drain_stop = threading.Event()

    def _drain():
        while not drain_stop.is_set():
            try:
                sample_queue.get(timeout=0.05)
            except queue.Empty:
                continue

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    runner = threading.Thread(target=worker.run, args=(ctx,), daemon=True)
    runner.start()

    # Spam 200 distinct frequencies over ~1 s (much faster than the throttle window).
    final_freq = 100e6
    start = time.perf_counter()
    for i in range(200):
        final_freq = 100e6 + (i + 1) * 1_000.0
        control_queue.put(initial_config.with_changes(center_frequency=final_freq))
        time.sleep(0.005)

    # Allow one more throttle window for the trailing apply.
    time.sleep(CONFIG_APPLY_MIN_INTERVAL + 0.05)
    elapsed = time.perf_counter() - start

    lifecycle.request_stop()
    runner.join(timeout=2.0)
    drain_stop.set()
    drainer.join(timeout=1.0)

    # Bound: at most ⌈elapsed / interval⌉ + 1 (the trailing apply may straddle the window).
    max_calls = math.ceil(elapsed / CONFIG_APPLY_MIN_INTERVAL) + 1
    assert len(device.freq_calls) <= max_calls, (
        f"throttle leaked: {len(device.freq_calls)} calls in {elapsed:.3f}s (allowed {max_calls})"
    )
    # Sanity: the throttle actually engaged (would be 200 without it).
    assert len(device.freq_calls) < 50

    # Final scrolled value must always land.
    assert device.freq_calls[-1] == final_freq
