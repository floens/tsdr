"""Config snapshots must never be lost on their way to the workers.

The engine pushes full `DeviceConfig` / `SDRConfig` snapshots onto bounded
queues. A burst of updates (held arrow key, scroll wheel, bandwidth sweep)
can outrun the workers; the newest snapshot of each type must still arrive,
otherwise hardware, demodulator, or FFT state sticks at a stale value until
some unrelated setting changes.
"""

from __future__ import annotations

import threading
import time
from queue import Queue

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr import device_context
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.device_context import DeviceState, _offer_config
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.pipeline.stages.demodulator_stage import DemodulatorStage
from tsdr.core.sdr.workers.pipeline_worker import PipelineWorker
from tsdr.devices import MockParams
from tsdr.devices.mock import MockSDRDevice

BURST = 40


def _drain(q) -> list:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def _engine_with_idle_running_device() -> SDREngine:
    """A device marked running whose workers never drain (deterministic queues)."""
    engine = SDREngine()
    engine.add_device("m", "mock", MockParams(), DeviceConfig())
    engine.devices["m"].state = DeviceState.RUNNING
    return engine


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_control_queue_overflow_keeps_newest_config() -> None:
    engine = _engine_with_idle_running_device()
    ctx = engine.devices["m"]
    final = 100e6 + BURST * 1e3

    for i in range(1, BURST + 1):
        engine.update_device_config("m", tuned_frequency=100e6 + i * 1e3)

    pending = _drain(ctx.control_queue)
    assert pending, "control queue must hold the pending snapshot"
    assert pending[-1].center_frequency == final


def test_pipeline_control_queue_overflow_keeps_latest_of_each_type() -> None:
    engine = _engine_with_idle_running_device()
    ctx = engine.devices["m"]
    final = 100e6 + BURST * 1e3

    engine.update_global_config(audio_volume=0.3)
    for i in range(1, BURST + 1):
        engine.update_device_config("m", tuned_frequency=100e6 + i * 1e3)

    pending = _drain(ctx.pipeline_control_queue)
    device_configs = [c for c in pending if isinstance(c, DeviceConfig)]
    sdr_configs = [c for c in pending if isinstance(c, SDRConfig)]
    assert device_configs[-1].tuned_frequency == final
    # An SDRConfig queued before the burst is not evicted by DeviceConfigs.
    assert sdr_configs and sdr_configs[-1].audio_volume == 0.3


class _SpyStage:
    def __init__(self) -> None:
        self.seen: list[object] = []

    def on_config_change(self, config) -> None:
        self.seen.append(config)


def test_pipeline_worker_applies_only_latest_pending_configs() -> None:
    engine = _engine_with_idle_running_device()
    ctx = engine.devices["m"]
    spy = _SpyStage()
    ctx.pipelines["visualization"].stages = [spy]  # type: ignore[list-item]
    worker = PipelineWorker(ctx)
    worker.setup(_FakeWorkerContext(engine))

    configs = [ctx.config.with_changes(tuned_frequency=100e6 + i) for i in range(5)]
    for c in configs[:3]:
        ctx.pipeline_control_queue.put(c)
    ctx.pipeline_control_queue.put(SDRConfig(audio_volume=0.1))
    ctx.pipeline_control_queue.put(SDRConfig(audio_volume=0.2))
    for c in configs[3:]:
        ctx.pipeline_control_queue.put(c)

    worker._apply_pending_configs()

    assert ctx.pipeline_control_queue.empty()
    assert configs[-1] in spy.seen
    assert not any(c in spy.seen for c in configs[:-1])
    assert [c.audio_volume for c in spy.seen if isinstance(c, SDRConfig)] == [0.2]
    assert worker.pipeline_context is not None
    assert worker.pipeline_context.config.audio_volume == 0.2


class _FakeWorkerContext:
    def __init__(self, engine: SDREngine) -> None:
        self.event_bus = engine.event_bus


def test_hardware_follows_dial_after_tuning_burst(monkeypatch) -> None:
    applied: list[float] = []
    original = MockSDRDevice.set_frequency

    def record(self, freq: float) -> None:
        applied.append(freq)
        original(self, freq)

    monkeypatch.setattr(MockSDRDevice, "set_frequency", record)

    engine = SDREngine()
    engine.add_device("m", "mock", MockParams(), DeviceConfig())
    engine.start_device("m")
    try:
        assert _wait_until(lambda: len(applied) >= 1)
        final = 100e6 + BURST * 1e3
        for i in range(1, BURST + 1):
            engine.update_device_config("m", tuned_frequency=100e6 + i * 1e3)
        assert _wait_until(lambda: applied[-1] == final), (
            f"hardware stuck at {applied[-1]:.0f} Hz, dial at {final:.0f} Hz"
        )
    finally:
        engine.shutdown(timeout=2.0)


def test_demodulator_follows_bandwidth_burst() -> None:
    engine = SDREngine()
    engine.add_device("m", "mock", MockParams(), DeviceConfig(sample_rate=240e3))
    engine.start_device("m")
    try:
        engine.set_audio_demod("m", DemodSpec(mode="NFM"))
        ctx = engine.devices["m"]
        stage = next(s for s in ctx.pipelines["audio"].stages if isinstance(s, DemodulatorStage))
        for bw in range(5_000, 25_000, 250):
            engine.update_device_config("m", channel_bandwidth=float(bw))
        final = ctx.config.channel_bandwidth
        assert _wait_until(lambda: stage.demodulator.channel_bandwidth == final), (
            f"demodulator at {stage.demodulator.channel_bandwidth} Hz, config at {final} Hz"
        )
    finally:
        engine.shutdown(timeout=2.0)


def test_overflow_puts_back_the_newest_snapshots_not_the_drained_ones() -> None:
    """The pipeline worker drains concurrently: it can hold a newer global
    config than the one left in the queue, so the collapse must not put the
    older one back behind it."""
    old_global, new_global = SDRConfig(audio_volume=0.1), SDRConfig(audio_volume=0.9)
    old_device, new_device = DeviceConfig(rf_gain=10.0), DeviceConfig(rf_gain=20.0)
    q: Queue = Queue(maxsize=2)
    q.put(old_global)
    q.put(old_device)
    _offer_config(q, new_device, {SDRConfig: new_global, DeviceConfig: new_device})
    left = [q.get_nowait() for _ in range(q.qsize())]
    assert sorted(left, key=lambda c: type(c).__name__) == [new_device, new_global]


def test_concurrent_updates_reach_the_worker_in_install_order(monkeypatch) -> None:
    """Producer A installs gain 10 and stalls before queuing it; producer B
    installs gain 20. The hardware must end on 20, the public config."""
    engine = _engine_with_idle_running_device()
    ctx = engine.devices["m"]
    real_offer = device_context._offer_config
    a_queuing, release_a = threading.Event(), threading.Event()

    def offer(q, config, newest):
        if q is ctx.control_queue and config.rf_gain == 10.0 and not release_a.is_set():
            a_queuing.set()
            release_a.wait(1.0)
        real_offer(q, config, newest)

    monkeypatch.setattr(device_context, "_offer_config", offer)
    a = threading.Thread(target=ctx.update_config, kwargs={"rf_gain": 10.0})
    a.start()
    assert a_queuing.wait(1.0)
    b = threading.Thread(target=ctx.update_config, kwargs={"rf_gain": 20.0})
    b.start()
    time.sleep(0.05)
    release_a.set()
    a.join()
    b.join()

    queued = [ctx.control_queue.get_nowait() for _ in range(ctx.control_queue.qsize())]
    assert ctx.config.rf_gain == 20.0
    assert queued[-1].rf_gain == 20.0
