from __future__ import annotations

from typing import assert_never

from rich.text import Text
from textual import events
from textual.timer import Timer

from tsdr.core.bandplans import find_band_at, get_bandplan_store
from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import MemoriesChangedEvent
from tsdr.core.memories import Memory, get_memory_store
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.tuning import current_spec_or_default
from tsdr.radio.dsp.rnnoise import rnnoise_available
from tsdr.tui._mixin_base import MixinBase
from tsdr.tui.console import ConsoleWidget, TerminalInput
from tsdr.tui.inline_edit import InlineEditor
from tsdr.tui.keybindings import (
    DEMOD_CHORD_MODES,
    Action,
    Context,
    build_lookup,
    demod_chord_prompt,
)
from tsdr.tui.model import Edge, adjusted_db_max, adjusted_db_min
from tsdr.tui.model.store import get_ui_store
from tsdr.tui.widgets import SpectrumWidget

_LOOKUP = build_lookup()


class KeyboardMixin(MixinBase):
    """Handles keyboard shortcuts for autocomplete and frequency/bandwidth tuning."""

    _pending_delete: Memory | None = None
    _pending_delete_timer: Timer | None = None
    _pending_demod_timer: Timer | None = None
    active_inline_editor: InlineEditor | None = None

    def on_key(self, event: events.Key) -> None:
        editor = self.active_inline_editor
        if editor is not None and editor.active:
            editor.handle_key(event)
            return

        cmd_input = self.query_one("#command-input", TerminalInput)
        context = Context.CONSOLE if cmd_input.active else Context.GLOBAL

        # Pending confirmations consume every key, so they must precede the table lookup.
        if context is Context.GLOBAL:
            if self._pending_delete is not None:
                if event.key == "y":
                    self._confirm_pending_delete()
                else:
                    self._cancel_pending_delete()
                event.prevent_default()
                event.stop()
                return

            if self._pending_demod_timer is not None:
                mode = DEMOD_CHORD_MODES.get(event.key)
                self._clear_demod_chord()
                if mode is not None:
                    self._apply_demod_chord(mode)
                else:
                    self.show_status("Demod: cancelled")
                event.prevent_default()
                event.stop()
                return

        action = _LOOKUP.get((context, event.key))
        if action is not None:
            if self._run_action(action):
                event.prevent_default()
                event.stop()
            return

        if context is Context.CONSOLE:
            return

        match event.key.split("+"):
            case [d] if len(d) == 1 and d.isdigit():
                store = get_ui_store()
                panel_id = next(
                    (pid for hk, pid in store.model.layout.hotkeys if hk == int(d)), None
                )
                if panel_id is not None:
                    store.toggle_panel(panel_id)
            case ["ctrl", d] if len(d) == 1 and d.isdigit():
                if d == "0":
                    self._swap_ab()
                else:
                    self._recall_band(int(d))
            case ["alt", d] if len(d) == 1 and d.isdigit():
                self._cycle_panel_edge(int(d))
            case _:
                return
        event.prevent_default()
        event.stop()

    def _run_action(self, action: Action) -> bool:
        """Return False to leave the key unconsumed."""
        match action:
            case Action.TUNE_DOWN:
                self._tune(-1)
            case Action.TUNE_UP:
                self._tune(1)
            case Action.TUNE_DOWN_COARSE:
                self._tune(-1, coarse=True)
            case Action.TUNE_UP_COARSE:
                self._tune(1, coarse=True)
            case Action.TUNE_DOWN_FINE:
                self._tune(-1, fine=True)
            case Action.TUNE_UP_FINE:
                self._tune(1, fine=True)
            case Action.TUNE_TARGET_PREV:
                self._jump_target(-1)
            case Action.TUNE_TARGET_NEXT:
                self._jump_target(1)
            case Action.TUNE_STEP_NEXT:
                self._cycle_step(True)
            case Action.TUNE_STEP_PREV:
                self._cycle_step(False)
            case Action.TUNE_MODE_TOGGLE:
                self._toggle_center_tuning()
            case Action.BANDWIDTH_UP:
                self._adjust_channel_bandwidth(1)
            case Action.BANDWIDTH_DOWN:
                self._adjust_channel_bandwidth(-1)
            case Action.BANDWIDTH_UP_FINE:
                self._adjust_channel_bandwidth(1, fine=True)
            case Action.BANDWIDTH_DOWN_FINE:
                self._adjust_channel_bandwidth(-1, fine=True)
            case Action.GAIN_DOWN:
                self._adjust_gain(-1)
            case Action.GAIN_UP:
                self._adjust_gain(1)
            case Action.GAIN_AGC:
                self._toggle_agc()
            case Action.SQUELCH_DOWN:
                self._adjust_squelch(-1)
            case Action.SQUELCH_UP:
                self._adjust_squelch(1)
            case Action.SQUELCH_OFF:
                self._disable_squelch()
            case Action.VOLUME_UP:
                self._adjust_volume(1)
            case Action.VOLUME_DOWN:
                self._adjust_volume(-1)
            case Action.AUDIO_DENOISE:
                self._toggle_denoise()
            case Action.DEVICE_TOGGLE:
                self._toggle_device_running()
            case Action.DEMOD_CHORD:
                self._start_demod_chord()
            case Action.DISPLAY_IMAGE:
                self._toggle_image_mode()
            case Action.DISPLAY_SPAN_IN:
                self._adjust_spectrum_span(1)
            case Action.DISPLAY_SPAN_OUT:
                self._adjust_spectrum_span(-1)
            case Action.DISPLAY_FLOOR_UP:
                self._adjust_db_floor(1)
            case Action.DISPLAY_FLOOR_DOWN:
                self._adjust_db_floor(-1)
            case Action.DISPLAY_CEILING_UP:
                self._adjust_db_ceiling(1)
            case Action.DISPLAY_CEILING_DOWN:
                self._adjust_db_ceiling(-1)
            case Action.DISPLAY_CENTER_ON_DIAL:
                self._center_view_on_dial()
            case Action.MEMORY_ADD:
                self._quick_add_memory()
            case Action.MEMORY_EDIT:
                self._quick_edit_memory()
            case Action.MEMORY_REMOVE:
                self._quick_remove_memory()
            case Action.CONSOLE_FOCUS:
                self._focus_command_input()
            case Action.CONSOLE_LEAVE:
                self._console_leave()
            case Action.CONSOLE_TAB:
                self._console_tab()
            case Action.CONSOLE_TAB_BACK:
                return self._console_shift_tab()
            case Action.CONSOLE_ESCAPE:
                self._console_escape()
            case Action.CONSOLE_HISTORY_PREV:
                self.query_one("#command-input", TerminalInput).history_up()
            case Action.CONSOLE_HISTORY_NEXT:
                self.query_one("#command-input", TerminalInput).history_down()
            case Action.CONSOLE_SEARCH:
                self.query_one("#command-input", TerminalInput).enter_search()
            case Action.CONSOLE_CLEAR:
                self._clear_console()
            case _:
                assert_never(action)
        return True

    def _console_leave(self) -> None:
        self._clear_preview()
        self._blur_command_input()

    def _console_tab(self) -> None:
        if get_ui_store().model.console.autocomplete_visible:
            self._cycle_preview(1)
        else:
            self._open_autocomplete()

    def _console_shift_tab(self) -> bool:
        if not get_ui_store().model.console.autocomplete_visible:
            return False
        self._cycle_preview(-1)
        return True

    def _console_escape(self) -> None:
        if get_ui_store().model.console.autocomplete_visible:
            self._dismiss_autocomplete()
        else:
            self._blur_command_input()

    def _toggle_image_mode(self) -> None:
        store = get_ui_store()
        new_mode = not store.model.image_mode
        store.update(image_mode=new_mode)
        self.show_status(f"Image mode: {'on' if new_mode else 'off'}")

    def _adjust_db_floor(self, direction: int) -> None:
        store = get_ui_store()
        m = store.model
        store.update(db_min=adjusted_db_min(m.db_min, m.db_max, direction))

    def _adjust_db_ceiling(self, direction: int) -> None:
        store = get_ui_store()
        m = store.model
        store.update(db_max=adjusted_db_max(m.db_max, m.db_min, direction))

    def _start_demod_chord(self) -> None:
        self._pending_demod_timer = self.set_timer(2.0, self._cancel_demod_chord)
        self.show_status(demod_chord_prompt())

    def _clear_console(self) -> None:
        self.query_one(ConsoleWidget).clear_history()

    def _cycle_panel_edge(self, digit: int) -> None:
        store = get_ui_store()
        layout = store.model.layout
        panel_id = next((pid for d, pid in layout.hotkeys if d == digit), None)
        if panel_id is None:
            return
        ring: tuple[Edge, ...] = ("left", "bottom", "right")
        current = next((e for e in ring if panel_id in getattr(layout, e).panels), None)
        start = ring.index(current) if current is not None else -1
        target = ring[(start + 1) % len(ring)]
        store.move_panel(panel_id, target)
        store.set_panel_active(target, panel_id)

    def _toggle_device_running(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            self._show_error("No device")
            return

        try:
            if device.state == DeviceState.RUNNING:
                engine.stop_device(device.device_id)
                self.show_status("Stopped")
            else:
                engine.start_device(device.device_id)
                self.show_status("Started")
        except SDRException as e:
            self._show_error(str(e))
            return

    def _gain_controllable_device(self):
        device = get_engine().get_focused_device()
        if device is None:
            return None
        if not device.device.capabilities.gain_supported:
            self._show_error("Gain locked by device")
            return None
        return device

    def _adjust_gain(self, direction: int) -> None:
        device = self._gain_controllable_device()
        if device is None:
            return

        caps = device.device.capabilities
        lo, hi = caps.gain_range
        new_gain = device.config.rf_gain + direction
        new_gain = max(lo, min(new_gain, hi))
        if new_gain == device.config.rf_gain:
            return

        get_engine().update_device_config(device.device_id, rf_gain=new_gain, enable_agc=False)
        self.show_status(f"Gain: {new_gain:.0f} {caps.gain_unit}")

    def _toggle_agc(self) -> None:
        device = self._gain_controllable_device()
        if device is None:
            return

        new_agc = not device.config.enable_agc
        get_engine().update_device_config(device.device_id, enable_agc=new_agc)
        self.show_status(f"AGC {'on' if new_agc else 'off'}")

    def _adjust_volume(self, direction: int) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        new_vol = engine.config.audio_volume + direction * 0.05
        new_vol = max(0.0, min(1.0, new_vol))

        engine.update_global_config(audio_volume=new_vol)
        self.show_status(f"Volume: {new_vol:.0%}")

    def _toggle_denoise(self) -> None:
        engine = get_engine()
        want = not engine.config.denoise
        if want and not rnnoise_available():
            self._show_error("Denoise unavailable (RNNoise not supported on this platform)")
            return
        engine.update_global_config(denoise=want)
        self.show_status(f"Denoise {'on' if want else 'off'}")

    def _adjust_squelch(self, direction: int) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        audio_pc = device.config.pipelines.get("audio")
        if audio_pc is None:
            self._show_error("No audio demodulator active")
            return

        step_db = 3.0
        new_threshold = audio_pc.squelch_threshold_db + direction * step_db
        new_threshold = max(-100.0, min(0.0, new_threshold))
        engine.update_squelch(
            device.device_id,
            "audio",
            enabled=True,
            threshold_db=new_threshold,
        )
        self.show_status(f"Squelch on, threshold: {new_threshold:.1f} dB")

    def _disable_squelch(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        audio_pc = device.config.pipelines.get("audio")
        if audio_pc is None:
            self._show_error("No audio demodulator active")
            return

        engine.update_squelch(device.device_id, "audio", enabled=False)
        self.show_status("Squelch off")

    def _quick_add_memory(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        freq = int(device.config.tuned_frequency)
        mode = device.active_mode
        profile = device.demod_profile
        status = device.demod_status
        bandwidth = int(profile.channel_bandwidth) if profile else 12500

        desc = status.description if status and status.description else ""
        desc = Text.from_markup(desc).plain if desc else ""

        if not desc:
            bp = get_bandplan_store().active
            if bp is not None:
                band = find_band_at(bp, freq)
                if band is not None:
                    desc = band.name

        name = f"{desc} {mode}" if desc else f"{freq / 1e6:.3f} {mode}"

        store = get_memory_store()
        spec = current_spec_or_default(device, override_mode=mode)
        memory = store.add(frequency=freq, name=name, spec=spec, bandwidth=bandwidth)
        engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))
        self.show_status(f"Memory saved: {memory.name} [{memory.id}]")

    def _quick_edit_memory(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        freq = int(device.config.tuned_frequency)
        profile = device.demod_profile
        max_dist = int(profile.channel_bandwidth) if profile else 12500

        store = get_memory_store()
        memory = store.find_nearest(freq, max_dist)
        if memory is None:
            self._show_error("No memory near current frequency")
            return

        spectrum = self.query_one(SpectrumWidget)
        spectrum.start_edit(memory)

    def _quick_remove_memory(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        freq = int(device.config.tuned_frequency)
        profile = device.demod_profile
        max_dist = int(profile.channel_bandwidth) if profile else 12500

        store = get_memory_store()
        memory = store.find_nearest(freq, max_dist)
        if memory is None:
            self._show_error("No memory near current frequency")
            return

        self._pending_delete = memory
        self._pending_delete_timer = self.set_timer(2.0, self._cancel_pending_delete)
        self.show_status(f"Delete '{memory.name}'? Press y to confirm")

    def _confirm_pending_delete(self) -> None:
        memory = self._pending_delete
        self._clear_pending_delete()
        if memory is None:
            return
        store = get_memory_store()
        store.remove(memory.id)
        engine = get_engine()
        engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))
        self.show_status(f"Removed memory: {memory.name} [{memory.id}]")

    def _cancel_pending_delete(self) -> None:
        self._clear_pending_delete()
        self.show_status("Cancelled")

    def _clear_pending_delete(self) -> None:
        self._pending_delete = None
        if self._pending_delete_timer is not None:
            self._pending_delete_timer.stop()
            self._pending_delete_timer = None

    def _apply_demod_chord(self, mode: str) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            self._show_error("No device focused")
            return

        if mode == "OFF":
            engine.stop_audio_output(device.device_id)
            engine.remove_pipeline(device.device_id, "audio")
            self.show_status("Demod: off")
            return

        try:
            engine.set_audio_demod(device.device_id, DemodSpec(mode=mode))
        except SDRException as e:
            self._show_error(str(e))
            return

        self.notify_demod_changed()
        self.show_status(f"Demod: {mode}")

    def _cancel_demod_chord(self) -> None:
        self._clear_demod_chord()
        self.show_status("Demod: cancelled")

    def _clear_demod_chord(self) -> None:
        if self._pending_demod_timer is not None:
            self._pending_demod_timer.stop()
            self._pending_demod_timer = None
