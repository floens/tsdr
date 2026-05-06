"""Thin wrapper Messages for Textual integration.

Each wrapper holds a reference to a core Event - no field copying.
This keeps core independent of Textual while enabling type-specific message routing.
"""

from textual.message import Message

from tsdr.core.events.events import (
    AudioOutputErrorEvent,
    BandplanChangedEvent,
    ConfigChangedEvent,
    ConstellationUpdateEvent,
    DecoderOutputEvent,
    DeviceErrorEvent,
    DeviceStateChangedEvent,
    FFTUpdateEvent,
    MemoriesChangedEvent,
    PipelineChangedEvent,
    PipelineErrorEvent,
    RecordingFinishedEvent,
    SamplesDroppedEvent,
    SignalInfoEvent,
    StatsUpdateEvent,
)


class DeviceError(Message):
    """Wrapper for DeviceErrorEvent."""

    def __init__(self, event: DeviceErrorEvent) -> None:
        super().__init__()
        self.event = event


class SamplesDropped(Message):
    """Wrapper for SamplesDroppedEvent."""

    def __init__(self, event: SamplesDroppedEvent) -> None:
        super().__init__()
        self.event = event


class FFTUpdate(Message):
    """Wrapper for FFTUpdateEvent (consumed by spectrum and waterfall widgets)."""

    def __init__(self, event: FFTUpdateEvent) -> None:
        super().__init__()
        self.event = event


class StatsUpdate(Message):
    """Wrapper for StatsUpdateEvent."""

    def __init__(self, event: StatsUpdateEvent) -> None:
        super().__init__()
        self.event = event


class PipelineError(Message):
    """Wrapper for PipelineErrorEvent."""

    def __init__(self, event: PipelineErrorEvent) -> None:
        super().__init__()
        self.event = event


class AudioOutputError(Message):
    """Wrapper for AudioOutputErrorEvent."""

    def __init__(self, event: AudioOutputErrorEvent) -> None:
        super().__init__()
        self.event = event


class ConfigChanged(Message):
    """Wrapper for ConfigChangedEvent."""

    def __init__(self, event: ConfigChangedEvent) -> None:
        super().__init__()
        self.event = event


class DeviceStateChanged(Message):
    """Wrapper for DeviceStateChangedEvent."""

    def __init__(self, event: DeviceStateChangedEvent) -> None:
        super().__init__()
        self.event = event


class PipelineChanged(Message):
    """Wrapper for PipelineChangedEvent."""

    def __init__(self, event: PipelineChangedEvent) -> None:
        super().__init__()
        self.event = event


class SignalInfoUpdate(Message):
    """Wrapper for SignalInfoEvent."""

    def __init__(self, event: SignalInfoEvent) -> None:
        super().__init__()
        self.event = event


class DecoderOutput(Message):
    """Wrapper for DecoderOutputEvent."""

    def __init__(self, event: DecoderOutputEvent) -> None:
        super().__init__()
        self.event = event


class ConstellationUpdate(Message):
    """Wrapper for ConstellationUpdateEvent."""

    def __init__(self, event: ConstellationUpdateEvent) -> None:
        super().__init__()
        self.event = event


class MemoriesChanged(Message):
    """Wrapper for MemoriesChangedEvent."""

    def __init__(self, event: MemoriesChangedEvent) -> None:
        super().__init__()
        self.event = event


class BandplanChanged(Message):
    """Wrapper for BandplanChangedEvent."""

    def __init__(self, event: BandplanChangedEvent) -> None:
        super().__init__()
        self.event = event


class RecordingFinished(Message):
    """Wrapper for RecordingFinishedEvent."""

    def __init__(self, event: RecordingFinishedEvent) -> None:
        super().__init__()
        self.event = event
