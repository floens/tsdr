from typing import Any, Protocol

from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch


class PipelineStage(Protocol):
    """Protocol for pipeline processing stages.

    Each stage receives data, processes it, and can:
    - Emit processed data to next stage (return data)
    - Emit side effects (UI messages, audio samples)
    - Consume data (return None to end pipeline)

    All stages must implement:
    - process(): Main processing function
    - on_config_change(): Handle configuration updates
    - reset(): Reset internal state
    """

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        """Process data through this stage."""
        ...

    def on_config_change(self, config: Any) -> None:
        """Called when configuration changes.

        Receives either SDRConfig (global) or DeviceConfig (per-device).
        Stages should check the type and extract relevant fields.
        """
        ...

    def reset(self) -> None:
        """Reset internal state to initial conditions."""
        ...
