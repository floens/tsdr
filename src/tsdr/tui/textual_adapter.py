import logging
import time
from typing import TYPE_CHECKING

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import (
    AudioOutputErrorEvent,
    BandplanChangedEvent,
    BandStackChangedEvent,
    ConfigChangedEvent,
    ConstellationUpdateEvent,
    DecoderOutputEvent,
    DeviceAddedEvent,
    DeviceCapabilitiesChangedEvent,
    DeviceErrorEvent,
    DeviceRemovedEvent,
    DeviceStateChangedEvent,
    Event,
    FFTUpdateEvent,
    FocusChangedEvent,
    JitterBufferUpdateEvent,
    MemoriesChangedEvent,
    PipelineChangedEvent,
    PipelineErrorEvent,
    RecordingFinishedEvent,
    SamplesDroppedEvent,
    SignalInfoEvent,
    StatsUpdateEvent,
    TuningStateChangedEvent,
)
from tsdr.core.events.subscription import Subscription
from tsdr.tui.messages import (
    AudioOutputError,
    BandplanChanged,
    BandStackChanged,
    ConfigChanged,
    ConstellationUpdate,
    DecoderOutput,
    DeviceAdded,
    DeviceCapabilitiesChanged,
    DeviceError,
    DeviceRemoved,
    DeviceStateChanged,
    FFTUpdate,
    FocusChanged,
    JitterBufferUpdate,
    MemoriesChanged,
    PipelineChanged,
    PipelineError,
    RecordingFinished,
    SamplesDropped,
    SignalInfoUpdate,
    StatsUpdate,
    TuningStateChanged,
)

if TYPE_CHECKING:
    from textual.app import App
    from textual.timer import Timer

logger = logging.getLogger(__name__)

# Coalesce ConfigChanged deliveries to the UI message loop. Rapid config
# changes (e.g. tuner scroll) trigger SpectrumWidget.update_config which is
# ~2 ms per call; without coalescing the UI thread stalls for seconds.
# Leading-edge fire keeps single clicks snappy; the trailing pending slot
# guarantees the user's final value always renders.
CONFIG_REFRESH_MIN_INTERVAL = 0.050


class TextualEventAdapter:
    """Subscribes to EventBus events and re-posts them as Textual Messages."""

    def __init__(self, app: App, event_bus: EventBus) -> None:
        self.app = app
        self.event_bus = event_bus
        self._subscriptions: list[Subscription] = []
        self._last_config_post_ts: float = 0.0
        self._pending_config_event: ConfigChangedEvent | None = None
        self._pending_config_timer: Timer | None = None

    def start(self) -> None:
        logger.info("textual_adapter_starting")

        # Subscribe to all event types
        self._subscriptions.append(
            self.event_bus.subscribe(DeviceErrorEvent, self._on_device_error)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(SamplesDroppedEvent, self._on_samples_dropped)
        )
        self._subscriptions.append(self.event_bus.subscribe(FFTUpdateEvent, self._on_fft_update))
        self._subscriptions.append(
            self.event_bus.subscribe(StatsUpdateEvent, self._on_stats_update)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(JitterBufferUpdateEvent, self._on_jitter_buffer_update)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(PipelineErrorEvent, self._on_pipeline_error)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(AudioOutputErrorEvent, self._on_audio_output_error)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(ConfigChangedEvent, self._on_config_changed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(DeviceStateChangedEvent, self._on_device_state_changed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(DeviceAddedEvent, self._on_device_added)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(DeviceRemovedEvent, self._on_device_removed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(FocusChangedEvent, self._on_focus_changed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(
                DeviceCapabilitiesChangedEvent, self._on_device_capabilities_changed
            )
        )
        self._subscriptions.append(
            self.event_bus.subscribe(PipelineChangedEvent, self._on_pipeline_changed)
        )
        self._subscriptions.append(self.event_bus.subscribe(SignalInfoEvent, self._on_signal_info))
        self._subscriptions.append(
            self.event_bus.subscribe(DecoderOutputEvent, self._on_decoder_output)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(ConstellationUpdateEvent, self._on_constellation_update)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(MemoriesChangedEvent, self._on_memories_changed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(BandplanChangedEvent, self._on_bandplan_changed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(RecordingFinishedEvent, self._on_recording_finished)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(BandStackChangedEvent, self._on_band_stack_changed)
        )
        self._subscriptions.append(
            self.event_bus.subscribe(TuningStateChangedEvent, self._on_tuning_state_changed)
        )

        logger.info("textual_adapter_started subscriptions=%d", len(self._subscriptions))

    def stop(self) -> None:
        logger.info("textual_adapter_stopping")

        for subscription in self._subscriptions:
            self.event_bus.unsubscribe(subscription)

        self._subscriptions.clear()

        if self._pending_config_timer is not None:
            self._pending_config_timer.stop()
            self._pending_config_timer = None
        self._pending_config_event = None

        logger.info("textual_adapter_stopped")

    def _on_device_error(self, event: Event) -> None:
        if not isinstance(event, DeviceErrorEvent):
            return
        self.app.post_message(DeviceError(event))

    def _on_samples_dropped(self, event: Event) -> None:
        if not isinstance(event, SamplesDroppedEvent):
            return
        self.app.post_message(SamplesDropped(event))

    def _on_fft_update(self, event: Event) -> None:
        if not isinstance(event, FFTUpdateEvent):
            return
        self.app.post_message(FFTUpdate(event))

    def _on_stats_update(self, event: Event) -> None:
        if not isinstance(event, StatsUpdateEvent):
            return
        self.app.post_message(StatsUpdate(event))

    def _on_jitter_buffer_update(self, event: Event) -> None:
        if not isinstance(event, JitterBufferUpdateEvent):
            return
        self.app.post_message(JitterBufferUpdate(event))

    def _on_pipeline_error(self, event: Event) -> None:
        if not isinstance(event, PipelineErrorEvent):
            return
        self.app.post_message(PipelineError(event))

    def _on_audio_output_error(self, event: Event) -> None:
        if not isinstance(event, AudioOutputErrorEvent):
            return
        self.app.post_message(AudioOutputError(event))

    def _on_config_changed(self, event: Event) -> None:
        if not isinstance(event, ConfigChangedEvent):
            return
        elapsed = time.perf_counter() - self._last_config_post_ts
        if elapsed >= CONFIG_REFRESH_MIN_INTERVAL:
            self.app.post_message(ConfigChanged(event))
            self._last_config_post_ts = time.perf_counter()
            return
        self._pending_config_event = event
        if self._pending_config_timer is None:
            self._pending_config_timer = self.app.set_timer(
                CONFIG_REFRESH_MIN_INTERVAL - elapsed, self._flush_pending_config
            )

    def _flush_pending_config(self) -> None:
        self._pending_config_timer = None
        event = self._pending_config_event
        if event is None:
            return
        self._pending_config_event = None
        self.app.post_message(ConfigChanged(event))
        self._last_config_post_ts = time.perf_counter()

    def _on_device_state_changed(self, event: Event) -> None:
        if not isinstance(event, DeviceStateChangedEvent):
            return
        self.app.post_message(DeviceStateChanged(event))

    def _on_device_added(self, event: Event) -> None:
        if not isinstance(event, DeviceAddedEvent):
            return
        self.app.post_message(DeviceAdded(event))

    def _on_device_removed(self, event: Event) -> None:
        if not isinstance(event, DeviceRemovedEvent):
            return
        self.app.post_message(DeviceRemoved(event))

    def _on_focus_changed(self, event: Event) -> None:
        if not isinstance(event, FocusChangedEvent):
            return
        self.app.post_message(FocusChanged(event))

    def _on_device_capabilities_changed(self, event: Event) -> None:
        if not isinstance(event, DeviceCapabilitiesChangedEvent):
            return
        self.app.post_message(DeviceCapabilitiesChanged(event))

    def _on_pipeline_changed(self, event: Event) -> None:
        if not isinstance(event, PipelineChangedEvent):
            return
        self.app.post_message(PipelineChanged(event))

    def _on_signal_info(self, event: Event) -> None:
        if not isinstance(event, SignalInfoEvent):
            return
        self.app.post_message(SignalInfoUpdate(event))

    def _on_decoder_output(self, event: Event) -> None:
        if not isinstance(event, DecoderOutputEvent):
            return
        self.app.post_message(DecoderOutput(event))

    def _on_constellation_update(self, event: Event) -> None:
        if not isinstance(event, ConstellationUpdateEvent):
            return
        self.app.post_message(ConstellationUpdate(event))

    def _on_memories_changed(self, event: Event) -> None:
        if not isinstance(event, MemoriesChangedEvent):
            return
        self.app.post_message(MemoriesChanged(event))

    def _on_bandplan_changed(self, event: Event) -> None:
        if not isinstance(event, BandplanChangedEvent):
            return
        self.app.post_message(BandplanChanged(event))

    def _on_recording_finished(self, event: Event) -> None:
        if not isinstance(event, RecordingFinishedEvent):
            return
        self.app.post_message(RecordingFinished(event))

    def _on_band_stack_changed(self, event: Event) -> None:
        if not isinstance(event, BandStackChangedEvent):
            return
        self.app.post_message(BandStackChanged(event))

    def _on_tuning_state_changed(self, event: Event) -> None:
        if not isinstance(event, TuningStateChangedEvent):
            return
        self.app.post_message(TuningStateChanged(event))
