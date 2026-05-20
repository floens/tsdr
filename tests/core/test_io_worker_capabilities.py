import queue
import threading
import time
from unittest.mock import MagicMock

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import DeviceCapabilitiesChangedEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.core.sdr.workers.io_worker import IOWorker
from tsdr.core.workers import WorkerContext, WorkerLifecycle
from tsdr.devices.base import DeviceCapabilities, DeviceIdentity

_IDENTITY = DeviceIdentity(type_label="MutableTest", serial=None)


def _make_caps(*, gain_supported: bool) -> DeviceCapabilities:
    return DeviceCapabilities(
        frequency_range=(24e6, 1766e6),
        frequency_controllable=True,
        sample_rates=None,
        gain_supported=gain_supported,
        gain_range=(0.0, 49.6),
        gain_step=1.0,
        gain_unit="dB",
        bias_tee_supported=False,
    )


class _MutableCapsDevice:
    def __init__(self) -> None:
        self._sample_rate = 0.0
        self.identity = _IDENTITY
        self.capabilities = _make_caps(gain_supported=True)

    def set_frequency(self, freq: float) -> None:
        pass

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
        time.sleep(0.005)
        return b"\x80" * count


def _drive_worker(device, event_bus, run_for_s: float = 0.4) -> list:
    captured: list = []
    event_bus.subscribe(
        DeviceCapabilitiesChangedEvent,
        lambda e: captured.append(e),  # type: ignore[arg-type]
    )

    control_queue: queue.Queue = queue.Queue()
    sample_queue: queue.Queue = queue.Queue(maxsize=64)

    device_context = MagicMock()
    device_context.device = device
    device_context.device_id = "test"
    device_context.config = DeviceConfig()
    device_context.control_queue = control_queue
    device_context.sample_queue = sample_queue
    device_context.total_samples_read = 0
    device_context.dropped_samples = 0

    worker = IOWorker(device_context)
    worker.sample_format = SampleFormat.UINT8_IQ

    lifecycle = WorkerLifecycle()
    lifecycle.mark_running()
    ctx = WorkerContext(worker_id="test", event_bus=event_bus, lifecycle=lifecycle)

    drain_stop = threading.Event()

    def _drain() -> None:
        while not drain_stop.is_set():
            try:
                sample_queue.get(timeout=0.05)
            except queue.Empty:
                continue

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    runner = threading.Thread(target=worker.run, args=(ctx,), daemon=True)
    runner.start()

    try:
        time.sleep(run_for_s)
    finally:
        lifecycle.request_stop()
        runner.join(timeout=2.0)
        drain_stop.set()
        drainer.join(timeout=1.0)

    return captured


def test_initial_capabilities_event_fires_then_stable():
    device = _MutableCapsDevice()
    bus = EventBus()
    captured = _drive_worker(device, bus, run_for_s=0.2)
    assert len(captured) == 1
    assert captured[0].capabilities.gain_supported is True


def test_event_published_on_capability_flip():
    device = _MutableCapsDevice()
    bus = EventBus()

    flipped = threading.Event()

    def _flip() -> None:
        time.sleep(0.1)
        device.capabilities = _make_caps(gain_supported=False)
        flipped.set()

    flipper = threading.Thread(target=_flip, daemon=True)
    flipper.start()

    captured = _drive_worker(device, bus, run_for_s=0.4)
    flipper.join(timeout=1.0)
    assert flipped.is_set()

    assert len(captured) == 2
    assert captured[0].capabilities.gain_supported is True
    assert captured[1].capabilities.gain_supported is False
