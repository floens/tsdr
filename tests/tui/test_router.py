"""Focused router tests.

The EventRouter is a Textual MessagePump mixin, so we can't trivially
instantiate it in isolation. Instead we bind the same handler functions to a
plain object that provides _store, _reconciler, _engine, and the show_status /
_show_error stubs. This exercises the routing logic without spinning up
Textual's message loop.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from tsdr.core.events.events import (
    ConfigChangedEvent,
    DecodedMessage,
    DecoderOutputEvent,
    DeviceAddedEvent,
    DeviceRemovedEvent,
    FFTUpdateEvent,
    FocusChangedEvent,
    JitterBufferUpdateEvent,
    MemoriesChangedEvent,
    PipelineChangedEvent,
)
from tsdr.tui.events.router import EventRouter
from tsdr.tui.messages import (
    ConfigChanged,
    DecoderOutput,
    DeviceAdded,
    DeviceRemoved,
    FFTUpdate,
    FocusChanged,
    JitterBufferUpdate,
    MemoriesChanged,
    PipelineChanged,
)
from tsdr.tui.model import DeviceUIState, UIModel
from tsdr.tui.model.store import UIStore


@dataclass
class _FakeEngine:
    devices: dict[str, Any] = field(default_factory=dict)
    _focused_id: str | None = None

    def get_device(self, device_id: str):
        return self.devices[device_id]

    def get_focused_device(self):
        return self.devices.get(self._focused_id) if self._focused_id else None


class _FakeReconciler:
    def __init__(self) -> None:
        self.widgets: dict[str, MagicMock] = {}

    def get(self, key: str) -> MagicMock | None:
        return self.widgets.get(key)

    def install(self, key: str) -> MagicMock:
        w = MagicMock()
        self.widgets[key] = w
        return w


def _make_router(store: UIStore, reconciler: _FakeReconciler, engine: _FakeEngine):
    """Bind a fresh router-method-bag onto a plain object so we can call handlers."""

    obj = types.SimpleNamespace()
    obj._store = store
    obj._reconciler = reconciler
    obj._engine = engine
    # show_status / _show_error are bound from MixinBase contract; stub them.
    obj.show_status = MagicMock()
    obj._show_error = MagicMock()
    # EnginePrefsSync needs mark_dirty(); the router calls it from config /
    # pipeline / device / focus handlers. Stub it so we can assert it fired.
    obj._engine_prefs_sync = MagicMock()

    # Bind every handler method off EventRouter as a free function call.
    for name in (
        "handle_fft_update",
        "handle_stats_update",
        "handle_jitter_buffer_update",
        "handle_device_state_changed",
        "handle_device_capabilities_changed",
        "handle_config_changed",
        "handle_device_error",
        "handle_audio_error",
        "handle_samples_dropped",
        "handle_pipeline_error",
        "handle_recording_finished",
        "handle_signal_info",
        "handle_memories_changed",
        "handle_bandplan_changed",
        "handle_band_stack_changed",
        "handle_tuning_state_changed",
        "handle_constellation_update",
        "handle_decoder_output",
        "handle_pipeline_changed",
        "handle_device_added",
        "handle_device_removed",
        "handle_focus_changed",
        "seed_from_engine",
    ):
        method = getattr(EventRouter, name)
        setattr(obj, name, method.__get__(obj, type(obj)))
    return obj


def _fft_event(device_id: str = "rtl0") -> FFTUpdateEvent:
    spectrum = np.zeros(8, dtype=np.float32)
    return FFTUpdateEvent(
        device_id=device_id,
        spectrum=spectrum,
        frequencies=spectrum,
        center_frequency=100e6,
        sample_rate=2.4e6,
    )


# --- stream routing ---------------------------------------------------------


def test_fft_update_pushes_to_spectrum_and_waterfall() -> None:
    store = UIStore(UIModel())
    rec = _FakeReconciler()
    spectrum = rec.install("spectrum")
    waterfall = rec.install("waterfall")
    router = _make_router(store, rec, _FakeEngine())

    event = _fft_event()
    router.handle_fft_update(FFTUpdate(event))
    spectrum.update_spectrum.assert_called_once_with(event)
    waterfall.update_waterfall.assert_called_once_with(event)


def test_fft_update_silently_drops_when_widgets_absent() -> None:
    """Stream events arriving between unmount and next structural event must
    not raise. Reconciler.get returns None; router skips."""
    router = _make_router(UIStore(UIModel()), _FakeReconciler(), _FakeEngine())
    router.handle_fft_update(FFTUpdate(_fft_event()))


def test_jitter_buffer_routes_to_stats_and_status_bar() -> None:
    rec = _FakeReconciler()
    stats = rec.install("stats")
    status = rec.install("status-bar")
    router = _make_router(UIStore(UIModel()), rec, _FakeEngine())
    event = JitterBufferUpdateEvent(
        device_id="rtl0",
        target_seconds=1.0,
        fill_seconds=0.5,
        fill_fraction=0.5,
        rebuffer_count=0,
        rebuffering=False,
    )
    router.handle_jitter_buffer_update(JitterBufferUpdate(event))
    stats.update_jitter_buffer.assert_called_once_with(event)
    status.update_jitter_buffer.assert_called_once_with(event)


def test_memories_changed_routes_to_spectrum_only() -> None:
    rec = _FakeReconciler()
    spectrum = rec.install("spectrum")
    router = _make_router(UIStore(UIModel()), rec, _FakeEngine())
    event = MemoriesChangedEvent(memories=())
    router.handle_memories_changed(MemoriesChanged(event))
    spectrum.update_memories.assert_called_once_with(event)


def test_decoder_output_routes_by_device_active_kind() -> None:
    """Routing uses the store's active_decoder_kind (lowercase, matches the
    widget key built in derive_tree). The event's `protocol` field, which
    carries the uppercase mode (e.g. 'WFM' for RDS), is intentionally ignored."""
    rec = _FakeReconciler()
    rds_widget = rec.install("decoder:rtl0:rds")
    store = UIStore(UIModel(devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="rds"),)))
    router = _make_router(store, rec, _FakeEngine())
    event = DecoderOutputEvent(
        device_id="rtl0",
        protocol="WFM",  # uppercase mode — what production actually emits
        messages=(DecodedMessage(text="x", timestamp=0.0),),
    )
    router.handle_decoder_output(DecoderOutput(event))
    rds_widget.update_messages.assert_called_once_with(event)


def test_decoder_output_text_kind_calls_update_decoder() -> None:
    rec = _FakeReconciler()
    text_widget = rec.install("decoder:rtl0:text")
    store = UIStore(UIModel(devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="text"),)))
    router = _make_router(store, rec, _FakeEngine())
    event = DecoderOutputEvent(
        device_id="rtl0",
        protocol="CW",
        messages=(DecodedMessage(text="hi", timestamp=0.0),),
    )
    router.handle_decoder_output(DecoderOutput(event))
    text_widget.update_decoder.assert_called_once_with(event)


def test_decoder_output_dropped_when_device_unknown() -> None:
    """If the store has no entry for the event's device_id, route nothing."""
    router = _make_router(UIStore(UIModel()), _FakeReconciler(), _FakeEngine())
    event = DecoderOutputEvent(device_id="rtl0", protocol="WFM", messages=())
    router.handle_decoder_output(DecoderOutput(event))  # no crash


def test_decoder_output_dropped_when_widget_absent() -> None:
    """Device is in the store but the widget isn't mounted — silent drop."""
    store = UIStore(UIModel(devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="rds"),)))
    router = _make_router(store, _FakeReconciler(), _FakeEngine())
    event = DecoderOutputEvent(device_id="rtl0", protocol="WFM", messages=())
    router.handle_decoder_output(DecoderOutput(event))


# --- structural routing -----------------------------------------------------


def test_pipeline_changed_seeds_devices_from_engine() -> None:
    """Any PipelineChanged event triggers a full re-seed from engine state.

    This catches the cases where command paths (add, restore_devices) populate
    engine.devices but skip the store; the first PipelineChanged for the new
    device's visualization pipeline then syncs the store.
    """
    store = UIStore(UIModel())
    engine = _FakeEngine()
    demod = types.SimpleNamespace(message_type="rds")
    engine.devices["rtl0"] = types.SimpleNamespace(
        device_id="rtl0",
        config=types.SimpleNamespace(pipelines={"audio": object(), "visualization": object()}),
        active_demod_info=demod,
    )
    engine._focused_id = "rtl0"
    router = _make_router(store, _FakeReconciler(), engine)

    event = PipelineChangedEvent(device_id="rtl0", pipeline_name="audio", active=True, mode="WFM")
    router.handle_pipeline_changed(PipelineChanged(event))

    assert store.model.devices == (
        DeviceUIState(device_id="rtl0", has_audio_pipeline=True, active_decoder_kind="rds"),
    )
    assert store.model.focused_device_id == "rtl0"


def test_pipeline_changed_visualization_still_seeds_device() -> None:
    """Visualization pipelines also seed — that's the first signal a new
    device exists for paths that bypass the store (sdr add, restore_devices)."""
    store = UIStore(UIModel())
    engine = _FakeEngine()
    engine.devices["rtl0"] = types.SimpleNamespace(
        device_id="rtl0",
        config=types.SimpleNamespace(pipelines={"visualization": object()}),
        active_demod_info=None,
    )
    engine._focused_id = "rtl0"
    router = _make_router(store, _FakeReconciler(), engine)
    event = PipelineChangedEvent(device_id="rtl0", pipeline_name="visualization", active=True)
    router.handle_pipeline_changed(PipelineChanged(event))
    assert store.model.devices == (
        DeviceUIState(device_id="rtl0", has_audio_pipeline=False, active_decoder_kind=None),
    )
    assert store.model.focused_device_id == "rtl0"


def test_pipeline_changed_removed_device_drops_from_store() -> None:
    """When a device disappears from the engine, seed_from_engine evicts it."""
    store = UIStore(
        UIModel(
            devices=(DeviceUIState(device_id="rtl0", active_decoder_kind="dab"),),
            focused_device_id="rtl0",
        )
    )
    router = _make_router(store, _FakeReconciler(), _FakeEngine())
    event = PipelineChangedEvent(device_id="rtl0", pipeline_name="audio", active=False)
    router.handle_pipeline_changed(PipelineChanged(event))
    assert store.model.devices == ()
    assert store.model.focused_device_id is None


# --- seed_from_engine -------------------------------------------------------


def test_seed_from_engine_populates_store() -> None:
    store = UIStore(UIModel())
    engine = _FakeEngine()
    ctx = types.SimpleNamespace(
        device_id="rtl0",
        config=types.SimpleNamespace(pipelines={"audio": object(), "visualization": object()}),
        active_demod_info=types.SimpleNamespace(message_type="rds"),
    )
    engine.devices["rtl0"] = ctx
    engine._focused_id = "rtl0"
    router = _make_router(store, _FakeReconciler(), engine)

    router.seed_from_engine()
    assert store.model.devices == (
        DeviceUIState(device_id="rtl0", has_audio_pipeline=True, active_decoder_kind="rds"),
    )
    assert store.model.focused_device_id == "rtl0"


def test_seed_from_engine_no_focused_clears_focus() -> None:
    store = UIStore(UIModel(focused_device_id="rtl0"))
    router = _make_router(store, _FakeReconciler(), _FakeEngine())
    router.seed_from_engine()
    assert store.model.focused_device_id is None


# --- DeviceAdded / DeviceRemoved / FocusChanged ----------------------------


def test_device_added_seeds_store_from_engine() -> None:
    store = UIStore(UIModel())
    engine = _FakeEngine()
    engine.devices["rtl0"] = types.SimpleNamespace(
        device_id="rtl0",
        config=types.SimpleNamespace(pipelines={"visualization": object()}),
        active_demod_info=None,
    )
    engine._focused_id = "rtl0"
    router = _make_router(store, _FakeReconciler(), engine)
    router.handle_device_added(DeviceAdded(DeviceAddedEvent(device_id="rtl0")))
    assert store.model.devices == (
        DeviceUIState(device_id="rtl0", has_audio_pipeline=False, active_decoder_kind=None),
    )
    assert store.model.focused_device_id == "rtl0"


def test_device_removed_seeds_store_from_engine() -> None:
    store = UIStore(
        UIModel(
            devices=(DeviceUIState(device_id="rtl0"),),
            focused_device_id="rtl0",
        )
    )
    router = _make_router(store, _FakeReconciler(), _FakeEngine())
    router.handle_device_removed(DeviceRemoved(DeviceRemovedEvent(device_id="rtl0")))
    assert store.model.devices == ()
    assert store.model.focused_device_id is None


def test_focus_changed_reseeds_and_nudges_widgets() -> None:
    rec = _FakeReconciler()
    tuner = rec.install("tuner")
    spectrum = rec.install("spectrum")
    console = rec.install("console")
    engine = _FakeEngine()
    engine.devices["rtl1"] = types.SimpleNamespace(
        device_id="rtl1",
        config=types.SimpleNamespace(pipelines={"visualization": object()}),
        active_demod_info=None,
    )
    engine._focused_id = "rtl1"
    store = UIStore(
        UIModel(
            devices=(DeviceUIState(device_id="rtl1"),),
            focused_device_id="rtl0",
        )
    )
    router = _make_router(store, rec, engine)
    router.handle_focus_changed(FocusChanged(FocusChangedEvent(focused_device_id="rtl1")))
    assert store.model.focused_device_id == "rtl1"
    tuner.update_config.assert_called_once_with()
    spectrum.update_config.assert_called_once_with()
    console.sync_prompt.assert_called_once_with()


# --- engine prefs persistence ----------------------------------------------


def test_config_changed_marks_prefs_dirty() -> None:
    """Every ConfigChangedEvent triggers a debounced prefs flush via EnginePrefsSync.
    Replaces the per-mutation save_device() calls that used to be sprinkled across
    commands and keyboard handlers."""
    router = _make_router(UIStore(UIModel()), _FakeReconciler(), _FakeEngine())
    router.handle_config_changed(ConfigChanged(ConfigChangedEvent(device_id="rtl0")))
    router._engine_prefs_sync.mark_dirty.assert_called_once_with()


def test_pipeline_changed_marks_prefs_dirty() -> None:
    store = UIStore(UIModel())
    router = _make_router(store, _FakeReconciler(), _FakeEngine())
    router.handle_pipeline_changed(
        PipelineChanged(PipelineChangedEvent(device_id="rtl0", pipeline_name="audio", active=True))
    )
    router._engine_prefs_sync.mark_dirty.assert_called_once_with()


def test_device_added_marks_prefs_dirty() -> None:
    router = _make_router(UIStore(UIModel()), _FakeReconciler(), _FakeEngine())
    router.handle_device_added(DeviceAdded(DeviceAddedEvent(device_id="rtl0")))
    router._engine_prefs_sync.mark_dirty.assert_called_once_with()


def test_device_removed_marks_prefs_dirty() -> None:
    router = _make_router(UIStore(UIModel()), _FakeReconciler(), _FakeEngine())
    router.handle_device_removed(DeviceRemoved(DeviceRemovedEvent(device_id="rtl0")))
    router._engine_prefs_sync.mark_dirty.assert_called_once_with()


def test_focus_changed_marks_prefs_dirty() -> None:
    rec = _FakeReconciler()
    rec.install("tuner")
    rec.install("spectrum")
    rec.install("console")
    router = _make_router(UIStore(UIModel()), rec, _FakeEngine())
    router.handle_focus_changed(FocusChanged(FocusChangedEvent(focused_device_id="rtl0")))
    router._engine_prefs_sync.mark_dirty.assert_called_once_with()
