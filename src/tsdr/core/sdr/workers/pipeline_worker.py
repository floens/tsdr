"""Pipeline worker thread for flexible stage-based processing.

It:
    1. Dequeues SamplesBatch from sample_queue
    2. Converts raw samples to IQ array
    3. Executes the device's pipelines

Pipelines are materialized by DeviceContext (main thread). The worker
only executes stages and applies on_config_change() updates.
"""

import logging
import queue
from typing import TYPE_CHECKING

from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.tracing import span
from tsdr.core.workers import WorkerContext

if TYPE_CHECKING:
    from tsdr.core.sdr.device_context import SDRDeviceContext

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Pipeline worker for flexible stage-based processing.

    Thread Safety:
        - Reads from sample_queue (thread-safe Queue)
        - Executes pipeline stages (isolated per device)
        - Emits events via EventBus (thread-safe)
    """

    def __init__(self, device_context: SDRDeviceContext) -> None:
        self.device_context = device_context
        self.pipeline_context: PipelineContext | None = None

    def setup(self, context: WorkerContext) -> None:
        device_id = self.device_context.device_id
        logger.info("pipeline_worker_starting device=%s", device_id)

        self.pipeline_context = PipelineContext(
            device_context=self.device_context,
            event_bus=context.event_bus,
            audio_queue=self.device_context.audio_queue,
            config=self.device_context.get_sdr_config(),
        )

        logger.debug("pipeline_worker_initialized device=%s", device_id)

    def run(self, context: WorkerContext) -> None:
        """Main pipeline loop."""
        device_id = self.device_context.device_id
        sample_queue = self.device_context.sample_queue
        pipeline_control_queue = self.device_context.pipeline_control_queue

        assert self.device_context.pipelines, "Pipelines must be configured"
        assert self.pipeline_context is not None, "Pipeline context must be initialized in setup()"

        while context.should_continue():
            with span("pipeline_worker"):
                try:
                    new_config = pipeline_control_queue.get(block=False)
                    if isinstance(new_config, SDRConfig):
                        self.pipeline_context.config = new_config
                    self._apply_config_to_stages(new_config)
                except queue.Empty:
                    pass

                with span("queue_wait"):
                    try:
                        batch = sample_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                with span("iq_convert"):
                    try:
                        iq_samples = batch.to_iq_array()
                    except ValueError as e:
                        logger.error("iq_conversion_failed device=%s error=%r", device_id, e)
                        continue

                pipeline_data = batch.with_changes(
                    iq_samples=iq_samples,
                    raw_samples=None,
                    stage_name="input",
                )

                # Snapshot to avoid modification during iteration
                current_pipelines = list(self.device_context.pipelines.items())
                for pipeline_name, pipeline in current_pipelines:
                    with span(pipeline_name):
                        try:
                            pipeline.execute(pipeline_data, self.pipeline_context)
                        except Exception as e:  # noqa: BLE001 - pipeline stages can fail in various ways
                            logger.error(
                                "pipeline_stage_crash device=%s pipeline=%s error=%r",
                                device_id,
                                pipeline_name,
                                e,
                                exc_info=True,
                            )

    def teardown(self, context: WorkerContext) -> None:
        pass

    def _apply_config_to_stages(self, config: SDRConfig | DeviceConfig) -> None:
        """Apply configuration update to all pipeline stages."""
        if not self.device_context.pipelines:
            return

        device_id = self.device_context.device_id
        logger.debug("pipeline_config_update_applying device=%s", device_id)

        current_pipelines = list(self.device_context.pipelines.items())
        for pipeline_name, pipeline in current_pipelines:
            for stage in pipeline.stages:
                try:
                    stage.on_config_change(config)
                except Exception as e:  # noqa: BLE001 - isolate stage errors
                    logger.warning(
                        "stage_config_update_failed device=%s pipeline=%s stage=%s error=%r",
                        device_id,
                        pipeline_name,
                        type(stage).__name__,
                        e,
                    )
