"""Worker framework: thread management for long-running background work.

Workers implement the BaseWorker protocol (setup/run/teardown). The runner owns
the thread, drives the lifecycle, and exposes a handle for shutdown.
"""

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tsdr.core.events.bus import EventBus
    from tsdr.core.events.events import Event

logger = logging.getLogger(__name__)


class WorkerLifecycle:
    """Tracks whether a worker should keep running.

    The runner calls mark_running() after setup, request_stop() to signal
    graceful shutdown, and mark_stopped() after teardown. Workers check
    is_running() in their main loop condition.
    """

    def __init__(self) -> None:
        self._running = False
        self._stop_requested = False
        self._lock = threading.Lock()

    def mark_running(self) -> None:
        with self._lock:
            self._running = True

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._running = False

    def mark_stopped(self) -> None:
        with self._lock:
            self._running = False

    def is_running(self) -> bool:
        with self._lock:
            return self._running and not self._stop_requested


@dataclass
class WorkerContext:
    """Worker execution context.

    Passed to each lifecycle method (setup/run/teardown). Workers use
    should_continue() in their main loop and emit_event() to publish events.
    """

    worker_id: str
    event_bus: EventBus
    lifecycle: WorkerLifecycle

    def should_continue(self) -> bool:
        return self.lifecycle.is_running()

    def emit_event(self, event: Event) -> None:
        self.event_bus.publish(event)


class BaseWorker(Protocol):
    """Protocol for all workers.

    Workers implement three lifecycle methods:
    - setup(): Called once before run() starts
    - run(): Main worker loop (should check context.should_continue())
    - teardown(): Called once after run() completes (always runs, even on error)
    """

    def setup(self, context: WorkerContext) -> None: ...

    def run(self, context: WorkerContext) -> None: ...

    def teardown(self, context: WorkerContext) -> None: ...


@dataclass
class WorkerHandle:
    """Handle for a running worker."""

    worker_id: str
    thread: threading.Thread
    lifecycle: WorkerLifecycle


class WorkerRunner:
    """Manages worker thread lifecycle.

    Wraps worker execution so that:
    - setup() runs before run()
    - teardown() always runs, even if setup/run raised
    - lifecycle state reflects the worker's actual status
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._workers: dict[str, WorkerHandle] = {}
        self._lock = threading.Lock()

    def start_worker(
        self,
        worker_id: str,
        worker: BaseWorker,
        daemon: bool = False,
    ) -> WorkerHandle:
        """Start a worker thread."""
        with self._lock:
            if worker_id in self._workers:
                raise ValueError(f"Worker {worker_id} already exists")

            lifecycle = WorkerLifecycle()
            context = WorkerContext(
                worker_id=worker_id,
                event_bus=self._event_bus,
                lifecycle=lifecycle,
            )

            thread = threading.Thread(
                target=self._worker_wrapper,
                args=(worker, context),
                name=worker_id,
                daemon=daemon,
            )

            handle = WorkerHandle(worker_id=worker_id, thread=thread, lifecycle=lifecycle)
            self._workers[worker_id] = handle
            thread.start()

            logger.info(f"Started worker {worker_id}")
            return handle

    def stop_worker(self, worker_id: str, timeout: float = 5.0) -> bool:
        """Stop a worker gracefully with timeout.

        Returns True if the worker stopped within the timeout, False
        otherwise. In both cases the worker is deregistered: leaving a
        timed-out handle in the dict would block any future `start_worker`
        for the same id (the only mechanism to recover would be to restart
        the app). If the orphaned thread eventually exits on its own, the
        self-deregister in `_worker_wrapper` is a no-op.
        """
        with self._lock:
            handle = self._workers.get(worker_id)
        if handle is None:
            # Already self-deregistered on thread exit, or never started.
            return True

        logger.info(f"Stopping worker {worker_id} (timeout={timeout}s)")
        handle.lifecycle.request_stop()
        handle.thread.join(timeout=timeout)

        with self._lock:
            self._workers.pop(worker_id, None)

        if handle.thread.is_alive():
            logger.warning(
                f"Worker {worker_id} did not stop within {timeout}s "
                "(thread abandoned, registration freed for retry)"
            )
            return False

        logger.info(f"Worker {worker_id} stopped gracefully")
        return True

    def stop_all_workers(self, timeout: float = 5.0) -> None:
        with self._lock:
            worker_ids = list(self._workers.keys())

        for worker_id in worker_ids:
            try:
                self.stop_worker(worker_id, timeout=timeout)
            except KeyError as e:
                logger.error(f"Error stopping worker {worker_id}: {e}")

    def _worker_wrapper(self, worker: BaseWorker, context: WorkerContext) -> None:
        try:
            worker.setup(context)
            context.lifecycle.mark_running()
            worker.run(context)
        except Exception:
            logger.error(f"Worker {context.worker_id} error in setup/run", exc_info=True)
        finally:
            try:
                worker.teardown(context)
            except Exception:
                logger.error(f"Worker {context.worker_id} error in teardown", exc_info=True)
            context.lifecycle.mark_stopped()
            # Self-deregister so a future start_worker can reuse the slot
            # even when nobody calls stop_worker (e.g. setup failed).
            with self._lock:
                self._workers.pop(context.worker_id, None)
            logger.info(f"Worker {context.worker_id} thread exiting")
