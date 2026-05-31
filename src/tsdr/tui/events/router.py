"""EventRouter — single inbound funnel for engine events.

Two patterns:
  - Structural events mutate UIStore (which triggers a reconcile via subscriber)
  - Stream events push directly to widget instances looked up by reconciler.get(key)

Stream events targeting an unmounted widget are silently dropped — acceptable
race window between unmount and the next structural reconcile.
"""

from __future__ import annotations

import logging

from textual import on

from tsdr.core.sdr.datatypes import SignalInfo
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.tracing import span
from tsdr.tui._mixin_base import MixinBase
from tsdr.tui.events.engine_prefs_sync import EnginePrefsSync
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
from tsdr.tui.model import DecoderKind, DeviceUIState
from tsdr.tui.model.store import UIStore
from tsdr.tui.view.reconciler import Reconciler

logger = logging.getLogger(__name__)


def _decoder_kind(info: SignalInfo | None) -> DecoderKind | None:
    if info is None:
        return None
    kind = info.message_type
    if kind in ("rds", "dab", "adsb", "tetra", "dmr", "text", "sstv"):
        return kind  # type: ignore[return-value]
    return None


class EventRouter(MixinBase):
    """Mixin on TSDRApp; receives Textual messages relayed from the engine event bus."""

    # These attributes are wired in TSDRApp.__init__.
    _store: UIStore
    _reconciler: Reconciler
    _engine: SDREngine
    _engine_prefs_sync: EnginePrefsSync

    def seed_from_engine(self) -> None:
        """Build initial devices tuple in the store from current engine state.

        Atomic: devices and focused_device_id update in a single commit so
        subscribers never observe an intermediate state where focus points at
        a device not yet in the tuple (or vice versa).
        """
        devices = tuple(
            DeviceUIState(
                device_id=ctx.device_id,
                has_audio_pipeline="audio" in ctx.config.pipelines,
                active_decoder_kind=_decoder_kind(ctx.active_demod_info),
            )
            for ctx in self._engine.devices.values()
        )
        focused = self._engine.get_focused_device()
        self._store.update(
            devices=devices,
            focused_device_id=focused.device_id if focused is not None else None,
        )

    @on(FFTUpdate)
    def handle_fft_update(self, message: FFTUpdate) -> None:
        # Spectrum/waterfall are single instances bound to the focused device;
        # drop FFT frames from other devices to avoid interleaved displays.
        focused = self._store.model.focused_device_id
        if focused is not None and message.event.device_id != focused:
            return
        spectrum = self._reconciler.get("spectrum")
        if spectrum is not None:
            spectrum.update_spectrum(message.event)  # type: ignore[attr-defined]
        waterfall = self._reconciler.get("waterfall")
        if waterfall is not None:
            waterfall.update_waterfall(message.event)  # type: ignore[attr-defined]

    @on(StatsUpdate)
    def handle_stats_update(self, message: StatsUpdate) -> None:
        focused = self._store.model.focused_device_id
        # Tuner/stats/performance are bound to the focused device; ignore stats
        # from other devices.
        if focused is None or message.event.device_id == focused:
            for key in ("tuner", "stats", "performance"):
                w = self._reconciler.get(key)
                if w is not None:
                    w.update_stats(message.event)  # type: ignore[attr-defined]
        # tetra/dmr decoder widgets read SNR from stats; route by the event's
        # own device_id so multi-device setups don't paint A's SNR onto B's UI.
        for kind in ("tetra", "dmr"):
            w = self._reconciler.get(f"decoder:{message.event.device_id}:{kind}")
            if w is not None:
                w.update_stats(message.event)  # type: ignore[attr-defined]

    @on(JitterBufferUpdate)
    def handle_jitter_buffer_update(self, message: JitterBufferUpdate) -> None:
        stats = self._reconciler.get("stats")
        if stats is not None:
            stats.update_jitter_buffer(message.event)  # type: ignore[attr-defined]
        status = self._reconciler.get("status-bar")
        if status is not None:
            status.update_jitter_buffer(message.event)  # type: ignore[attr-defined]

    @on(DeviceStateChanged)
    def handle_device_state_changed(self, message: DeviceStateChanged) -> None:
        tuner = self._reconciler.get("tuner")
        if tuner is not None:
            tuner.update_running_state(message.event)  # type: ignore[attr-defined]
        stats = self._reconciler.get("stats")
        if stats is not None:
            stats.update_config()  # type: ignore[attr-defined]

    @on(DeviceCapabilitiesChanged)
    def handle_device_capabilities_changed(self, _message: DeviceCapabilitiesChanged) -> None:
        for key in ("tuner", "stats"):
            w = self._reconciler.get(key)
            if w is not None:
                w.update_config()  # type: ignore[attr-defined]

    @on(ConfigChanged)
    def handle_config_changed(self, _message: ConfigChanged) -> None:
        with span("ui.handle_config_changed"):
            for key in ("stats", "spectrum", "tuner"):
                w = self._reconciler.get(key)
                if w is not None:
                    w.update_config()  # type: ignore[attr-defined]
            console = self._reconciler.get("console")
            if console is not None:
                console.sync_prompt()  # type: ignore[attr-defined]
            self._engine_prefs_sync.mark_dirty()

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
        logger.error("ui_pipeline_error message=%r", message)
        self._show_error(
            f"Pipeline error ({message.event.device_id}): "
            f"{message.event.stage_name}: {message.event.error}"
        )

    @on(RecordingFinished)
    def handle_recording_finished(self, message: RecordingFinished) -> None:
        event = message.event
        notice = f"recording finished → {event.path} ({event.samples_written} samples)"
        self.show_status(notice)
        console = self._reconciler.get("console")
        if console is not None:
            console.write_info(notice)  # type: ignore[attr-defined]

    @on(SignalInfoUpdate)
    def handle_signal_info(self, message: SignalInfoUpdate) -> None:
        for key in ("tuner", "stats"):
            w = self._reconciler.get(key)
            if w is not None:
                w.update_signal_info(message.event)  # type: ignore[attr-defined]

    @on(MemoriesChanged)
    def handle_memories_changed(self, message: MemoriesChanged) -> None:
        spectrum = self._reconciler.get("spectrum")
        if spectrum is not None:
            spectrum.update_memories(message.event)  # type: ignore[attr-defined]

    @on(BandplanChanged)
    def handle_bandplan_changed(self, message: BandplanChanged) -> None:
        spectrum = self._reconciler.get("spectrum")
        if spectrum is not None:
            spectrum.update_bandplan(message.event)  # type: ignore[attr-defined]

    @on(BandStackChanged)
    def handle_band_stack_changed(self, _message: BandStackChanged) -> None:
        tuner = self._reconciler.get("tuner")
        if tuner is not None:
            tuner.update_config()  # type: ignore[attr-defined]

    @on(TuningStateChanged)
    def handle_tuning_state_changed(self, _message: TuningStateChanged) -> None:
        tuner = self._reconciler.get("tuner")
        if tuner is not None:
            tuner.update_config()  # type: ignore[attr-defined]

    @on(ConstellationUpdate)
    def handle_constellation_update(self, message: ConstellationUpdate) -> None:
        constellation = self._reconciler.get("constellation")
        if constellation is not None:
            constellation.update_constellation(message.event)  # type: ignore[attr-defined]

    @on(DecoderOutput)
    def handle_decoder_output(self, message: DecoderOutput) -> None:
        # The event's `protocol` carries the uppercase mode (WFM/DAB/…), which
        # doesn't map directly to widget kinds (WFM→rds, DAB→dab, CW→text, …).
        # The widget kind is already known to the store via
        # DeviceUIState.active_decoder_kind, derived from SignalInfo.message_type
        # — the same value that built the widget's key in derive_tree.
        device_id = message.event.device_id
        device = next(
            (d for d in self._store.model.devices if d.device_id == device_id),
            None,
        )
        if device is None or device.active_decoder_kind is None:
            return
        kind = device.active_decoder_kind
        w = self._reconciler.get(f"decoder:{device_id}:{kind}")
        if w is None:
            return
        if kind == "text":
            w.update_decoder(message.event)  # type: ignore[attr-defined]
        else:
            w.update_messages(message.event)  # type: ignore[attr-defined]

    @on(PipelineChanged)
    def handle_pipeline_changed(self, _message: PipelineChanged) -> None:
        console = self._reconciler.get("console")
        if console is not None:
            console.sync_prompt()  # type: ignore[attr-defined]
        self.seed_from_engine()
        self._engine_prefs_sync.mark_dirty()

    @on(DeviceAdded)
    def handle_device_added(self, _message: DeviceAdded) -> None:
        self.seed_from_engine()
        self._engine_prefs_sync.mark_dirty()

    @on(DeviceRemoved)
    def handle_device_removed(self, _message: DeviceRemoved) -> None:
        self.seed_from_engine()
        self._engine_prefs_sync.mark_dirty()

    @on(FocusChanged)
    def handle_focus_changed(self, _message: FocusChanged) -> None:
        # Re-seed (focused_device_id flows to widgets via reactive props),
        # then nudge widgets that read engine config directly rather than
        # taking it as a reactive prop.
        self.seed_from_engine()
        for key in ("tuner", "spectrum"):
            w = self._reconciler.get(key)
            if w is not None:
                w.update_config()  # type: ignore[attr-defined]
        console = self._reconciler.get("console")
        if console is not None:
            console.sync_prompt()  # type: ignore[attr-defined]
        self._engine_prefs_sync.mark_dirty()
