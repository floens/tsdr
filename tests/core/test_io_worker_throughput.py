"""The I/O worker reports a device that delivers well below its sample rate.

A network PlutoSDR asked for more than its link carries (~15 MS/s here)
loses samples without its driver reporting an overflow; audio then stutters
with no explanation. Measured over a window, the shortfall is reported.
"""

from __future__ import annotations

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import DeviceErrorEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.workers import io_worker as io_worker_module
from tsdr.core.sdr.workers.io_worker import IOWorker
from tsdr.core.workers import WorkerContext, WorkerLifecycle
from tsdr.devices import MockParams

RATE = 15e6
WINDOW = io_worker_module.THROUGHPUT_WINDOW_SECONDS


def _worker() -> tuple[IOWorker, WorkerContext, list[str]]:
    engine = SDREngine()
    engine.add_device("p", "mock", MockParams(), DeviceConfig())
    bus = EventBus()
    errors: list[str] = []
    bus.subscribe(DeviceErrorEvent, lambda e: errors.append(e.error))  # type: ignore[attr-defined]
    context = WorkerContext(worker_id="io_p", event_bus=bus, lifecycle=WorkerLifecycle())
    return IOWorker(engine.devices["p"]), context, errors


def _feed(
    worker: IOWorker, context: WorkerContext, delivered_rate: float, seconds: float, t0: float
) -> float:
    """Simulate 50 ms reads arriving at `delivered_rate` for `seconds`; return end time."""
    batch = int(RATE / 20)
    t = t0
    end = t0 + seconds
    while t < end:
        t += batch / delivered_rate
        worker._check_throughput(context, t, batch, RATE)
    return t


def test_full_rate_device_is_not_reported() -> None:
    worker, context, errors = _worker()
    _feed(worker, context, RATE, 3 * WINDOW, 0.0)
    assert errors == []


def test_shortfall_is_reported_after_the_warmup_window() -> None:
    worker, context, errors = _worker()
    t = _feed(worker, context, 14.4e6, WINDOW + 0.5, 0.0)
    assert errors == []  # first window is treated as startup transient
    _feed(worker, context, 14.4e6, WINDOW + 0.5, t)
    assert len(errors) == 1
    assert "14.40 of 15.00 MS/s" in errors[0]


def test_shortfall_report_is_rate_limited() -> None:
    worker, context, errors = _worker()
    _feed(worker, context, 12e6, 5 * WINDOW, 0.0)
    assert len(errors) == 1


def test_rate_change_restarts_the_warmup() -> None:
    worker, context, errors = _worker()
    t = _feed(worker, context, RATE, 2 * WINDOW, 0.0)
    worker._reset_throughput_window()
    _feed(worker, context, 12e6, WINDOW + 0.5, t)
    assert errors == []
