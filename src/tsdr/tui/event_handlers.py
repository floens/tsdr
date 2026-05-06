from __future__ import annotations

import logging

from textual import on
from textual.css.query import NoMatches

from tsdr.core.sdr.engine import get_engine
from tsdr.core.tracing import span
from tsdr.tui._mixin_base import MixinBase
from tsdr.tui.console.widget import ConsoleWidget
from tsdr.tui.messages import (
    AudioOutputError,
    BandplanChanged,
    ConfigChanged,
    ConstellationUpdate,
    DecoderOutput,
    DeviceError,
    DeviceStateChanged,
    FFTUpdate,
    MemoriesChanged,
    PipelineChanged,
    PipelineError,
    RecordingFinished,
    SamplesDropped,
    SignalInfoUpdate,
    StatsUpdate,
)
from tsdr.tui.widgets import (
    ADSBWidget,
    ConstellationWidget,
    DABWidget,
    DecoderOutputWidget,
    DMRWidget,
    PerformanceWidget,
    RDSWidget,
    SpectrumWidget,
    StatsWidget,
    TETRAWidget,
    TunerWidget,
    WaterfallWidget,
)

logger = logging.getLogger(__name__)


class EventHandlerMixin(MixinBase):
    """Routes SDR engine events to the appropriate widgets."""

    def _forward(self, widget_type: type, method: str, event: object = None) -> None:
        try:
            widget = self.query_one(widget_type)
        except NoMatches:
            return
        if event is None:
            getattr(widget, method)()
        else:
            getattr(widget, method)(event)

    @on(FFTUpdate)
    def handle_fft_update(self, message: FFTUpdate) -> None:
        self._forward(SpectrumWidget, "update_spectrum", message.event)
        self._forward(WaterfallWidget, "update_waterfall", message.event)

    @on(StatsUpdate)
    def handle_stats_update(self, message: StatsUpdate) -> None:
        self._forward(StatsWidget, "update_stats", message.event)
        self._forward(PerformanceWidget, "update_stats", message.event)
        self._forward(TunerWidget, "update_stats", message.event)
        self._forward(TETRAWidget, "update_stats", message.event)
        self._forward(DMRWidget, "update_stats", message.event)

    @on(DeviceStateChanged)
    def handle_device_state_changed(self, message: DeviceStateChanged) -> None:
        self._forward(TunerWidget, "update_running_state", message.event)

    @on(ConfigChanged)
    def handle_config_changed(self, message: ConfigChanged) -> None:
        with span("ui.handle_config_changed"):
            with span("stats.update_config"):
                try:
                    self.query_one(StatsWidget).update_config()
                except NoMatches:
                    pass
            with span("spectrum.update_config"):
                try:
                    self.query_one(SpectrumWidget).update_config()
                except NoMatches:
                    pass
            with span("tuner.update_config"):
                try:
                    self.query_one(TunerWidget).update_config()
                except NoMatches:
                    pass
            with span("console.sync_prompt"):
                self._forward(ConsoleWidget, "sync_prompt")

    @on(DeviceError)
    def handle_device_error(self, message: DeviceError) -> None:
        self._show_error(f"Device error ({message.event.device_id}): {message.event.error}")

    @on(AudioOutputError)
    def handle_audio_error(self, message: AudioOutputError) -> None:
        self._show_error(f"Audio error ({message.event.source_id}): {message.event.error}")

    @on(SamplesDropped)
    def handle_samples_dropped(self, message: SamplesDropped) -> None:
        self._show_error(
            f"Samples dropped ({message.event.device_id}): {message.event.count} samples lost"
        )

    @on(PipelineError)
    def handle_pipeline_error(self, message: PipelineError) -> None:
        logger.error("Pipeline error %s", message)
        self._show_error(
            f"Pipeline error ({message.event.device_id}): "
            f"{message.event.stage_name}: {message.event.error}"
        )

    @on(RecordingFinished)
    def handle_recording_finished(self, message: RecordingFinished) -> None:
        event = message.event
        notice = f"recording finished → {event.path} ({event.samples_written} samples)"
        self.show_status(notice)
        try:
            self.query_one(ConsoleWidget).write_info(notice)
        except NoMatches:
            pass

    @on(SignalInfoUpdate)
    def handle_signal_info(self, message: SignalInfoUpdate) -> None:
        self._forward(TunerWidget, "update_signal_info", message.event)
        self._forward(StatsWidget, "update_signal_info", message.event)

    @on(PipelineChanged)
    def handle_pipeline_changed(self, message: PipelineChanged) -> None:
        """Show/hide widgets based on pipeline lifecycle and signal capabilities."""
        self._forward(ConsoleWidget, "sync_prompt")
        self._update_constellation_config()
        event = message.event
        if event.pipeline_name == "audio":
            message_type = None
            if event.active:
                engine = get_engine()
                context = engine.get_device(event.device_id)
                signal_info = context.active_demod_info
                if signal_info:
                    message_type = signal_info.message_type

            try:
                rds = self.query_one(RDSWidget)
                if message_type == "rds":
                    rds.display = True
                else:
                    rds.display = False
                    rds.clear()
            except NoMatches:
                pass

            try:
                dab = self.query_one(DABWidget)
                if message_type == "dab":
                    dab.display = True
                else:
                    dab.display = False
                    dab.clear()
            except NoMatches:
                pass

            try:
                adsb = self.query_one(ADSBWidget)
                if message_type == "adsb":
                    adsb.display = True
                else:
                    adsb.display = False
                    adsb.clear()
            except NoMatches:
                pass

            try:
                tetra = self.query_one(TETRAWidget)
                if message_type == "tetra":
                    tetra.display = True
                else:
                    tetra.display = False
                    tetra.clear()
            except NoMatches:
                pass

            try:
                dmr = self.query_one(DMRWidget)
                if message_type == "dmr":
                    dmr.display = True
                else:
                    dmr.display = False
                    dmr.clear()
            except NoMatches:
                pass

            try:
                decoder = self.query_one(DecoderOutputWidget)
                if message_type == "text":
                    decoder.show()
                else:
                    decoder.hide()
                    decoder.clear()
            except NoMatches:
                pass

    @on(MemoriesChanged)
    def handle_memories_changed(self, message: MemoriesChanged) -> None:
        self._forward(SpectrumWidget, "update_memories", message.event)

    @on(BandplanChanged)
    def handle_bandplan_changed(self, message: BandplanChanged) -> None:
        self._forward(SpectrumWidget, "update_bandplan", message.event)

    @on(ConstellationUpdate)
    def handle_constellation_update(self, message: ConstellationUpdate) -> None:
        self._forward(ConstellationWidget, "update_constellation", message.event)

    @on(DecoderOutput)
    def handle_decoder_output(self, message: DecoderOutput) -> None:
        self._forward(RDSWidget, "update_messages", message.event)
        self._forward(DABWidget, "update_messages", message.event)
        self._forward(ADSBWidget, "update_messages", message.event)
        self._forward(TETRAWidget, "update_messages", message.event)
        self._forward(DMRWidget, "update_messages", message.event)
        self._forward(DecoderOutputWidget, "update_decoder", message.event)
