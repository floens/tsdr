from __future__ import annotations

from rich.text import Text
from textual import events
from textual.timer import Timer

from tsdr.core.bandplans import find_band_at, get_bandplan_store
from tsdr.core.events.events import MemoriesChangedEvent
from tsdr.core.memories import Memory, get_memory_store
from tsdr.core.preferences import save_device, save_engine_config, save_ui_state
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.tui._mixin_base import MixinBase
from tsdr.tui.console import ConsoleWidget, TerminalInput
from tsdr.tui.widgets import SpectrumWidget, WaterfallWidget

_DEMOD_CHORD_MODES = {
    "w": "WFM",
    "n": "NFM",
    "a": "AM",
    "u": "USB",
    "l": "LSB",
    "c": "CW",
    "o": "OFF",
}


class KeyboardMixin(MixinBase):
    """Handles keyboard shortcuts for autocomplete and frequency/bandwidth tuning."""

    _pending_delete: Memory | None = None
    _pending_delete_timer: Timer | None = None
    _pending_demod_timer: Timer | None = None

    def on_key(self, event: events.Key) -> None:
        spectrum = self.query_one(SpectrumWidget)
        if spectrum.is_editing:
            spectrum.handle_edit_key(event)
            return

        cmd_input = self.query_one("#command-input", TerminalInput)
        focused = cmd_input.active

        if focused:
            menu_open = self.ui_state.autocomplete_visible
            if event.key == "grave_accent":
                self._clear_preview()
                self._blur_command_input()
                event.prevent_default()
                event.stop()
            elif event.key == "tab":
                if menu_open:
                    self._cycle_preview(1)
                else:
                    self._open_autocomplete()
                event.prevent_default()
                event.stop()
            elif event.key == "shift+tab":
                if menu_open:
                    self._cycle_preview(-1)
                    event.prevent_default()
                    event.stop()
            elif event.key == "escape":
                if menu_open:
                    self._dismiss_autocomplete()
                else:
                    self._blur_command_input()
                event.prevent_default()
                event.stop()
            elif event.key in ("up", "ctrl+p"):
                cmd_input.history_up()
                event.prevent_default()
                event.stop()
            elif event.key in ("down", "ctrl+n"):
                cmd_input.history_down()
                event.prevent_default()
                event.stop()
            elif event.key == "ctrl+r":
                cmd_input.enter_search()
                event.prevent_default()
                event.stop()
            elif event.key == "ctrl+l":
                self._clear_console()
                event.prevent_default()
                event.stop()
        else:
            # Pending delete confirmation
            if self._pending_delete is not None:
                if event.key == "y":
                    self._confirm_pending_delete()
                else:
                    self._cancel_pending_delete()
                event.prevent_default()
                event.stop()
                return

            if self._pending_demod_timer is not None:
                mode = _DEMOD_CHORD_MODES.get(event.key)
                self._clear_demod_chord()
                if mode is not None:
                    self._apply_demod_chord(mode)
                else:
                    self.show_status("Demod: cancelled")
                event.prevent_default()
                event.stop()
                return

            # Unfocused mode: direct shortcuts
            if event.key == "grave_accent":
                self._focus_command_input()
                event.prevent_default()
                event.stop()
            elif event.key == "left":
                self._tune(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "right":
                self._tune(1)
                event.prevent_default()
                event.stop()
            elif event.key == "shift+left":
                self._tune(-1, coarse=True)
                event.prevent_default()
                event.stop()
            elif event.key == "shift+right":
                self._tune(1, coarse=True)
                event.prevent_default()
                event.stop()
            elif event.key in ("alt+left", "ctrl+left"):
                self._tune(-1, fine=True)
                event.prevent_default()
                event.stop()
            elif event.key in ("alt+right", "ctrl+right"):
                self._tune(1, fine=True)
                event.prevent_default()
                event.stop()
            elif event.key == "up":
                self._adjust_channel_bandwidth(1)
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                self._adjust_channel_bandwidth(-1)
                event.prevent_default()
                event.stop()
            elif event.key in ("alt+up", "ctrl+up"):
                self._adjust_channel_bandwidth(1, fine=True)
                event.prevent_default()
                event.stop()
            elif event.key in ("alt+down", "ctrl+down"):
                self._adjust_channel_bandwidth(-1, fine=True)
                event.prevent_default()
                event.stop()
            elif event.key == "left_square_bracket":
                self._jump_target(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "right_square_bracket":
                self._jump_target(1)
                event.prevent_default()
                event.stop()
            elif event.key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
                self._recall_band(int(event.key))
                event.prevent_default()
                event.stop()
            elif event.key == "0":
                self._swap_ab()
                event.prevent_default()
                event.stop()
            elif event.key == "s":
                self._cycle_step(True)
                event.prevent_default()
                event.stop()
            elif event.key == "S":
                self._cycle_step(False)
                event.prevent_default()
                event.stop()
            elif event.key == "g":
                self._adjust_gain(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "G":
                self._adjust_gain(1)
                event.prevent_default()
                event.stop()
            elif event.key == "q":
                self._adjust_squelch(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "Q":
                self._adjust_squelch(1)
                event.prevent_default()
                event.stop()
            elif event.key == "ctrl+q":
                self._disable_squelch()
                event.prevent_default()
                event.stop()
            elif event.key == "ctrl+g":
                self._toggle_agc()
                event.prevent_default()
                event.stop()
            elif event.key == "shift+up":
                self._adjust_volume(1)
                event.prevent_default()
                event.stop()
            elif event.key == "shift+down":
                self._adjust_volume(-1)
                event.prevent_default()
                event.stop()
            elif event.key == "ctrl+s":
                self._toggle_panel("stats")
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "ctrl+p":
                self._toggle_panel("performance")
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "i":
                self.ui_state.image_mode = not self.ui_state.image_mode
                status = "on" if self.ui_state.image_mode else "off"
                self.show_status(f"Image mode: {status}")
                self._notify_image_mode_changed()
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "k":
                self.ui_state.adjust_zoom(1)
                self.query_one(WaterfallWidget).invalidate_text_buffer()
                self.query_one(SpectrumWidget).invalidate_frame_buffer()
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "j":
                self.ui_state.adjust_zoom(-1)
                self.query_one(WaterfallWidget).invalidate_text_buffer()
                self.query_one(SpectrumWidget).invalidate_frame_buffer()
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "h":
                self.ui_state.adjust_db_min(1)
                self.query_one(WaterfallWidget).invalidate_text_buffer()
                self.query_one(SpectrumWidget).invalidate_frame_buffer()
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "l":
                self.ui_state.adjust_db_min(-1)
                self.query_one(WaterfallWidget).invalidate_text_buffer()
                self.query_one(SpectrumWidget).invalidate_frame_buffer()
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "H":
                self.ui_state.adjust_db_max(1)
                self.query_one(WaterfallWidget).invalidate_text_buffer()
                self.query_one(SpectrumWidget).invalidate_frame_buffer()
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "L":
                self.ui_state.adjust_db_max(-1)
                self.query_one(WaterfallWidget).invalidate_text_buffer()
                self.query_one(SpectrumWidget).invalidate_frame_buffer()
                save_ui_state(self.ui_state)
                event.prevent_default()
                event.stop()
            elif event.key == "m":
                self._quick_add_memory()
                event.prevent_default()
                event.stop()
            elif event.key == "M":
                self._quick_edit_memory()
                event.prevent_default()
                event.stop()
            elif event.key == "ctrl+m":
                self._quick_remove_memory()
                event.prevent_default()
                event.stop()
            elif event.key == "space":
                self._toggle_device_running()
                event.prevent_default()
                event.stop()
            elif event.key == "d":
                self._pending_demod_timer = self.set_timer(2.0, self._cancel_demod_chord)
                self.show_status(
                    "Demod: [b]w[/]fm [b]n[/]fm [b]a[/]m [b]u[/]sb [b]l[/]sb [b]c[/]w [b]o[/]ff"
                )
                event.prevent_default()
                event.stop()

    def _clear_console(self) -> None:
        self.query_one(ConsoleWidget).clear_history()

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

        save_device(engine)

    def _adjust_gain(self, direction: int) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        new_gain = device.config.rf_gain + direction
        new_gain = max(0, min(new_gain, 50))
        if new_gain == device.config.rf_gain:
            return

        engine.update_device_config(device.device_id, rf_gain=new_gain, enable_agc=False)
        save_device(engine)
        self.show_status(f"Gain: {new_gain:.0f} dB")

    def _toggle_agc(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        new_agc = not device.config.enable_agc
        engine.update_device_config(device.device_id, enable_agc=new_agc)
        self.show_status(f"AGC {'on' if new_agc else 'off'}")

    def _adjust_volume(self, direction: int) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        new_vol = engine.config.audio_volume + direction * 0.05
        new_vol = max(0.0, min(1.0, new_vol))

        engine.update_global_config(audio_volume=new_vol)
        save_engine_config(engine)
        self.show_status(f"Volume: {new_vol:.0%}")

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

        freq = int(device.config.center_frequency)
        mode = device.active_mode
        demod_info = device.active_demod_info
        bandwidth = int(demod_info.channel_bandwidth) if demod_info else 12500

        desc = demod_info.description if demod_info and demod_info.description else ""
        desc = Text.from_markup(desc).plain if desc else ""

        if not desc:
            bp = get_bandplan_store().active
            if bp is not None:
                band = find_band_at(bp, freq)
                if band is not None:
                    desc = band.name

        name = f"{desc} {mode}" if desc else f"{freq / 1e6:.3f} {mode}"

        store = get_memory_store()
        memory = store.add(frequency=freq, name=name, mode=mode, bandwidth=bandwidth)
        engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))
        self.show_status(f"Memory saved: {memory.name} [{memory.id}]")

    def _quick_edit_memory(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        freq = int(device.config.center_frequency)
        demod_info = device.active_demod_info
        max_dist = int(demod_info.channel_bandwidth) if demod_info else 12500

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

        freq = int(device.config.center_frequency)
        demod_info = device.active_demod_info
        max_dist = int(demod_info.channel_bandwidth) if demod_info else 12500

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
            save_device(engine)
            self.show_status("Demod: off")
            return

        if device.state != DeviceState.RUNNING:
            self._show_error("Device must be running")
            return

        try:
            engine.set_audio_demod(device.device_id, mode)
        except SDRException as e:
            self._show_error(str(e))
            return

        save_device(engine)
        self.notify_demod_changed()
        self.show_status(f"Demod: {mode}")

    def _cancel_demod_chord(self) -> None:
        self._clear_demod_chord()
        self.show_status("Demod: cancelled")

    def _clear_demod_chord(self) -> None:
        if self._pending_demod_timer is not None:
            self._pending_demod_timer.stop()
            self._pending_demod_timer = None
