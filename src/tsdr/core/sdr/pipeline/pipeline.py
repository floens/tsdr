import logging
from dataclasses import dataclass, field
from queue import Queue
from typing import TYPE_CHECKING, Any

from tsdr.core.events.events import PipelineErrorEvent

if TYPE_CHECKING:
    from tsdr.core.events.bus import EventBus
    from tsdr.core.sdr.config import SDRConfig
    from tsdr.core.sdr.device_context import SDRDeviceContext
    from tsdr.core.sdr.pipeline.stage import PipelineStage
    from tsdr.core.sdr.samples_batch import SamplesBatch

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Per-call context passed to every stage's `process()`."""

    device_context: SDRDeviceContext
    event_bus: EventBus
    config: SDRConfig
    audio_queue: Queue[Any] | None = None


@dataclass
class ProcessingPipeline:
    """An ordered list of processing stages executed in sequence.

    Each stage's output becomes the next stage's input; if a stage returns
    None the pipeline stops for that batch.
    """

    pipeline_id: str
    device_id: str
    stages: list[PipelineStage] = field(default_factory=list)

    def execute(self, data: SamplesBatch, context: PipelineContext) -> None:
        # Snapshot stages to avoid modification during iteration
        current_stages = list(self.stages)
        current_data: SamplesBatch | None = data

        for stage in current_stages:
            try:
                if current_data is None:
                    break
                current_data = stage.process(current_data, context)
            except Exception as e:  # noqa: BLE001 - stages can fail in arbitrary ways
                logger.error(
                    "pipeline_stage_error device=%s pipeline=%s stage=%s error=%r",
                    self.device_id,
                    self.pipeline_id,
                    type(stage).__name__,
                    e,
                    exc_info=True,
                )
                context.event_bus.publish(
                    PipelineErrorEvent(
                        source_id=f"pipeline_{self.pipeline_id}",
                        device_id=self.device_id,
                        pipeline_id=self.pipeline_id,
                        stage_name=stage.__class__.__name__,
                        error=str(e),
                    )
                )
                return

    def __str__(self) -> str:
        stages_str = " -> ".join(stage.__class__.__name__ for stage in self.stages)
        return f"ProcessingPipeline({self.pipeline_id}: [{stages_str}])"
