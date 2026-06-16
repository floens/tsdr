import logging
import queue
import threading
from collections.abc import Callable
from enum import Enum
from queue import Queue
from typing import TYPE_CHECKING, Unpack

from tsdr.core.sdr.config import DeviceConfig, DeviceConfigChanges, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import ProcessingPipeline
from tsdr.core.sdr.pipeline.stage_factory import create_stage, stage_type_of
from tsdr.core.sdr.pipeline.stages.demodulator_stage import DemodulatorStage
from tsdr.core.sdr.workers.io_worker import IOWorker
from tsdr.core.sdr.workers.pipeline_worker import PipelineWorker
from tsdr.core.tracing import traced
from tsdr.radio.registry import demod_profile as build_profile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tsdr.core.sdr.datatypes import DemodProfile, DemodStatus
    from tsdr.core.workers import WorkerHandle
    from tsdr.devices import DeviceParams, SDRDevice


def _close_stage(stage: object) -> None:
    close = getattr(stage, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - stages can fail in arbitrary ways during teardown
        logger.exception("stage_close_failed stage=%s", type(stage).__name__)


class DeviceState(Enum):
    """Device state machine."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class SDRDeviceContext:
    """Complete context for a single SDR device.

    Encapsulates all state, configuration, and resources for one SDR device.
    Maintains the invariant that self.pipelines always reflects self.config.pipelines.
    """

    def __init__(
        self,
        device_id: str,
        device_type: str,
        params: DeviceParams,
        device: SDRDevice,
        device_config: DeviceConfig,
        sample_queue: Queue,
        control_queue: Queue,
        pipeline_control_queue: Queue,
        audio_queue: Queue,
        get_sdr_config: Callable[[], SDRConfig],
    ) -> None:
        self.device_id = device_id
        self.device_type = device_type
        self.params = params
        self.device = device

        self._device_config = device_config
        self._config_lock = threading.Lock()
        self.get_sdr_config = get_sdr_config

        self.pipelines: dict[str, ProcessingPipeline] = {}

        self.io_worker: WorkerHandle | None = None
        self.pipeline_worker: WorkerHandle | None = None

        self.sample_queue = sample_queue
        self.control_queue = control_queue
        self.pipeline_control_queue = pipeline_control_queue
        self.audio_queue = audio_queue

        self.state = DeviceState.STOPPED
        self.error_message: str | None = None

        # Runtime counters (written by I/O worker, read by stats stage)
        self.total_samples_read = 0
        self.dropped_samples = 0
        self.queue_high_water_mark = 0
        self.total_queue_drops = 0

        # Stereo status (set by demodulators)
        self.stereo: bool | None = None

        # Initial materialization
        self._materialize_pipelines()

    @property
    def config(self) -> DeviceConfig:
        """Thread-safe: callable from any thread."""
        with self._config_lock:
            return self._device_config

    @property
    def demod_profile(self) -> DemodProfile | None:
        """Structural (desired-state) profile of the active demod, from the spec.

        None when no audio demod is configured (RAW). Thread-safe.
        """
        mode = self.active_mode
        if mode == "RAW":
            return None
        return build_profile(mode, self.config.channel_bandwidth)

    @property
    def demod_status(self) -> DemodStatus | None:
        """Live (actual-state) status of the running demodulator instance.

        Thread-safe: delegates to the demodulator's `status()`, which must
        itself be thread-safe.
        """
        for pipeline in self.pipelines.values():
            for stage in pipeline.stages:
                if isinstance(stage, DemodulatorStage):
                    return stage.demodulator.status()
        return None

    @property
    def active_mode(self) -> str:
        """Active demodulation mode name derived from config."""
        for pc in self.config.pipelines.values():
            if pc.audio_spec is not None:
                return pc.audio_spec.mode
        return "RAW"

    def start(self, worker_runner) -> None:
        """Start I/O and processing workers for this device."""
        if self.state != DeviceState.STOPPED:
            raise RuntimeError(f"Cannot start device in state {self.state}")

        logger.info("device_starting device=%s type=%s", self.device_id, self.device_type)
        self.state = DeviceState.STARTING

        logger.debug("io_worker_starting device=%s", self.device_id)
        io_worker_instance = IOWorker(device_context=self)
        self.io_worker = worker_runner.start_worker(
            worker_id=f"io_{self.device_id}", worker=io_worker_instance, daemon=False
        )

        if self.pipelines:
            logger.debug("pipeline_worker_launching device=%s", self.device_id)
            pipeline_worker_instance = PipelineWorker(device_context=self)
            self.pipeline_worker = worker_runner.start_worker(
                worker_id=f"pipeline_{self.device_id}",
                worker=pipeline_worker_instance,
                daemon=False,
            )

        self.state = DeviceState.RUNNING
        logger.info("device_started device=%s", self.device_id)

    def stop(self, worker_runner, timeout: float = 5.0) -> None:
        """Stop workers; the I/O worker's teardown closes the device.

        Ordering matters when the I/O worker is parked in a blocking
        device read (e.g. socket.recv via the jitter buffer): plain
        request_stop won't unblock it, so we call device.interrupt()
        to break pending I/O without freeing resources, then join.
        Resource cleanup is the I/O worker's job.
        """
        if self.state == DeviceState.STOPPED:
            return

        self.state = DeviceState.STOPPING
        logger.info("device_stopping device=%s", self.device_id)

        # Step 1: signal workers to stop iterating.
        if self.io_worker:
            self.io_worker.lifecycle.request_stop()
        if self.pipeline_worker:
            self.pipeline_worker.lifecycle.request_stop()

        # Step 2: unblock any in-flight read so the worker can observe
        # the stop flag. Doesn't free resources — close() does that, from
        # the I/O worker's teardown.
        self.device.interrupt()

        # Step 3: join workers.
        if self.io_worker:
            logger.debug("io_worker_stopping device=%s", self.device_id)
            try:
                worker_runner.stop_worker(f"io_{self.device_id}", timeout=timeout)
            except Exception as e:  # noqa: BLE001 - cleanup must not fail
                logger.warning("io_worker_stop_failed device=%s error=%r", self.device_id, e)
            self.io_worker = None

        if self.pipeline_worker:
            logger.debug("pipeline_worker_stopping device=%s", self.device_id)
            try:
                worker_runner.stop_worker(f"pipeline_{self.device_id}", timeout=timeout)
            except Exception as e:  # noqa: BLE001 - cleanup must not fail
                logger.warning("pipeline_worker_stop_failed device=%s error=%r", self.device_id, e)
            self.pipeline_worker = None

        self.state = DeviceState.STOPPED
        logger.info("device_stopped device=%s", self.device_id)

    @traced("ctx.update_config")
    def update_config(self, **changes: Unpack[DeviceConfigChanges]) -> None:
        """Update device configuration (triggers hardware reconfiguration).

        Thread-safe: protects config reference swap with a lock.
        If pipelines config changed, re-materializes pipeline stages.
        """
        with self._config_lock:
            old_config = self._device_config
            new_config = old_config.with_changes(**changes)
            new_config.validate()
            self._device_config = new_config

        # Re-materialize pipelines if the pipeline config changed
        if new_config.pipelines is not old_config.pipelines:
            self._materialize_pipelines()

        if self.state != DeviceState.STOPPED:
            try:
                self.control_queue.put(new_config, block=False)
            except queue.Full:
                logger.warning("control_queue_full device=%s", self.device_id)
            try:
                self.pipeline_control_queue.put(new_config, block=False)
            except queue.Full:
                pass

    def _materialize_pipelines(self) -> None:
        """Sync self.pipelines with self.config.pipelines.

        Positional diff: reuse stage instances at matching positions
        to preserve accumulated state (FFT buffers, phase accumulators, etc.).
        """
        sdr_config = self.get_sdr_config()
        device_config = self.config
        pipeline_configs = device_config.pipelines

        # Update or create pipelines
        for name, pc in pipeline_configs.items():
            old_pipeline = self.pipelines.get(name)
            old_stages = old_pipeline.stages if old_pipeline else []

            new_stages = []
            for i, stage_type in enumerate(pc.stages):
                if i < len(old_stages):
                    try:
                        old_type = stage_type_of(old_stages[i])
                    except ValueError:
                        old_type = None
                    if old_type == stage_type:
                        new_stages.append(old_stages[i])
                        continue

                new_stages.append(
                    create_stage(
                        stage_type,
                        pc,
                        sdr_config,
                        device_config,
                        self.device_id,
                        name,
                    )
                )

            if old_pipeline is not None:
                # Close any old stages that weren't reused in new_stages
                for old_stage in old_stages:
                    if old_stage in new_stages:
                        continue
                    _close_stage(old_stage)
                old_pipeline.stages = new_stages
            else:
                self.pipelines[name] = ProcessingPipeline(
                    pipeline_id=f"{self.device_id}_{name}",
                    device_id=self.device_id,
                    stages=new_stages,
                )

        # Remove pipelines no longer in config
        for name in list(self.pipelines):
            if name not in pipeline_configs:
                for stage in self.pipelines[name].stages:
                    _close_stage(stage)
                del self.pipelines[name]

    def notify_global_config_change(self, config: SDRConfig) -> None:
        """Notify pipeline worker of a global config change."""
        if self.state != DeviceState.STOPPED:
            try:
                self.pipeline_control_queue.put(config, block=False)
            except queue.Full:
                pass

    def __str__(self) -> str:
        return (
            f"SDRDeviceContext({self.device_id}, type={self.device_type}, state={self.state.value})"
        )

    def __repr__(self) -> str:
        return self.__str__()
