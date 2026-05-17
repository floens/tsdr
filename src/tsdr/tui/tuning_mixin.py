from __future__ import annotations

import logging

from textual import on

from tsdr.core.band_stack import (
    REGISTERS_PER_BAND,
    BandRegister,
    get_band_stack,
    suspended_writeback,
)
from tsdr.core.bandplans import get_bandplan_store
from tsdr.core.events.events import (
    BandStackChangedEvent,
    TuningStateChangedEvent,
)
from tsdr.core.landmarks import next_target
from tsdr.core.memories import get_memory_store
from tsdr.core.preferences import save_device, save_tuning_state
from tsdr.core.sdr.device_context import SDRDeviceContext
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.tuning import (
    STEP_LADDER,
    bandwidth_step,
    current_channel_bandwidth,
    default_bandwidth,
    resolve_auto_step,
    save_previous_tune_state,
    snap_to_grid,
)
from tsdr.core.tuning_state import get_tuning_state
from tsdr.core.units import format_hz
from tsdr.radio.registry import DEMODULATORS
from tsdr.tui._mixin_base import MixinBase
from tsdr.tui.messages import DeviceStateChanged, FFTUpdate

logger = logging.getLogger(__name__)


class TuningMixin(MixinBase):
    """Frequency, bandwidth, band-stack, step-ladder, A/B, landmark actions."""

    @on(FFTUpdate)
    def _tuning_on_fft_update(self, message: FFTUpdate) -> None:
        self._latest_fft_by_device[message.event.device_id] = message.event

    @on(DeviceStateChanged)
    def _tuning_on_device_state_changed(self, message: DeviceStateChanged) -> None:
        if not message.event.running:
            self._latest_fft_by_device.pop(message.event.device_id, None)

    def _focused(self) -> SDRDeviceContext | None:
        return get_engine().get_focused_device()

    def _resolve_step(self, device: SDRDeviceContext) -> float:
        ts = get_tuning_state()
        if ts.step is not None:
            return ts.step
        return resolve_auto_step(device.active_mode, device.config.center_frequency)

    def _clamp_to_range(self, device: SDRDeviceContext, freq: float) -> float:
        rng = device.device.frequency_range
        if rng is None:
            return freq
        lo, hi = rng
        return max(lo, min(freq, hi))

    def _publish_tuning_state_changed(self) -> None:
        ts = get_tuning_state()
        save_tuning_state(ts)
        get_engine().event_bus.publish(TuningStateChangedEvent(state=ts))

    def _apply_freq(self, device: SDRDeviceContext, new_freq: float) -> None:
        new_freq = self._clamp_to_range(device, new_freq)
        if new_freq == device.config.center_frequency:
            return
        get_engine().update_device_config(device.device_id, center_frequency=new_freq)
        save_device(get_engine())

    def _tune(self, direction: int, *, coarse: bool = False, fine: bool = False) -> None:
        device = self._focused()
        if device is None:
            return
        step = self._resolve_step(device)
        if coarse:
            step *= 10
        if fine:
            step /= 10
        current = float(device.config.center_frequency)
        if fine:
            new_freq = current + direction * step
        else:
            new_freq = snap_to_grid(current, step, direction)
        self._apply_freq(device, new_freq)
        self.show_status(f"Freq: {format_hz(new_freq, interval=step, long_suffix=True)}")

    def _adjust_channel_bandwidth(self, direction: int, *, fine: bool = False) -> None:
        device = self._focused()
        if device is None:
            return
        if device.active_demod_info is None:
            self._show_error("No active channel")
            return
        current_bw = int(current_channel_bandwidth(device))
        step = bandwidth_step(current_bw)
        if fine:
            step = max(1, step // 10)
            new_bw = current_bw + direction * step
        else:
            new_bw = int(snap_to_grid(current_bw, step, direction))
        upper = int(device.config.sample_rate / 2)
        new_bw = max(step, min(new_bw, upper))
        if new_bw == current_bw:
            return
        get_engine().update_device_config(device.device_id, channel_bandwidth=new_bw)
        save_device(get_engine())
        self.show_status(f"Bandwidth: {format_hz(new_bw, decimals=6, long_suffix=True)}")

    def _cycle_step(self, forward: bool) -> None:
        ts = get_tuning_state()
        try:
            idx = STEP_LADDER.index(ts.step)
        except ValueError:
            idx = 0
        idx = (idx + (1 if forward else -1)) % len(STEP_LADDER)
        ts.step = STEP_LADDER[idx]
        self._publish_tuning_state_changed()

        device = self._focused()
        if ts.step is None and device is not None:
            resolved = resolve_auto_step(device.active_mode, device.config.center_frequency)
            self.show_status(f"Step: auto ({format_hz(resolved, decimals=1, long_suffix=True)})")
        elif ts.step is not None:
            self.show_status(f"Step: {format_hz(ts.step, decimals=1, long_suffix=True)}")

    def _reset_step_to_auto(self) -> None:
        ts = get_tuning_state()
        if ts.step is not None:
            ts.step = None
            self._publish_tuning_state_changed()

    def _jump_target(self, direction: int) -> None:
        device = self._focused()
        if device is None:
            return
        memories = get_memory_store().all()
        bandplan = get_bandplan_store().active
        fft = self._latest_fft_by_device.get(device.device_id)
        target = next_target(
            direction,
            float(device.config.center_frequency),
            fft,
            memories,
            bandplan,
            device.device.frequency_range,
        )
        if target is None:
            self.show_status("No landmark or peak")
            return
        save_previous_tune_state(device)
        self._apply_freq(device, target)
        self.show_status(f"Jump: {format_hz(target, interval=1.0, long_suffix=True)}")

    def _swap_ab(self) -> None:
        device = self._focused()
        if device is None:
            return
        ts = get_tuning_state()
        if ts.previous is None:
            self.show_status("No previous tune state")
            return
        captured = (
            float(device.config.center_frequency),
            device.active_mode,
            current_channel_bandwidth(device),
        )
        prev_freq, prev_mode, prev_bw = ts.previous
        if captured == ts.previous:
            return
        ts.previous = captured

        engine = get_engine()
        if prev_mode and prev_mode != device.active_mode and prev_mode != "RAW":
            try:
                engine.set_audio_demod(device.device_id, prev_mode)
            except SDRException as e:
                self._show_error(str(e))
                return
        engine.update_device_config(
            device.device_id,
            center_frequency=prev_freq,
            channel_bandwidth=int(prev_bw),
        )
        self._reset_step_to_auto()
        self._publish_tuning_state_changed()
        save_device(engine)
        self.show_status(f"A/B: {format_hz(prev_freq, interval=1.0, long_suffix=True)} {prev_mode}")

    def _recall_band(self, key: int) -> None:
        device = self._focused()
        if device is None:
            return
        store = get_band_stack()
        stack = store.get_by_key(key)
        if stack is None:
            self._show_error(f"No band on key {key}")
            return

        cur_freq = float(device.config.center_frequency)
        ts = get_tuning_state()
        on_this_band = stack.band.start <= cur_freq <= stack.band.end and ts.current_band_key == key
        new_idx = (
            (stack.current_idx + 1) % REGISTERS_PER_BAND if on_this_band else stack.current_idx
        )

        reg = store.get_register(key, new_idx)
        if reg is None:
            # First time on this slot — seed with band midpoint + current mode/bw.
            seed_mode = device.active_mode if device.active_mode != "RAW" else "AM"
            seed_bw = int(current_channel_bandwidth(device)) or default_bandwidth(seed_mode)
            reg = BandRegister(
                slot=new_idx,
                frequency=(stack.band.start + stack.band.end) // 2,
                mode=seed_mode,
                bandwidth=int(seed_bw),
            )
            store.set_register(key, new_idx, reg)

        save_previous_tune_state(device)

        engine = get_engine()
        with suspended_writeback():
            if reg.mode != device.active_mode and reg.mode in DEMODULATORS:
                try:
                    engine.set_audio_demod(device.device_id, reg.mode)
                except SDRException as e:
                    self._show_error(str(e))
                    return
            engine.update_device_config(
                device.device_id,
                center_frequency=float(reg.frequency),
                channel_bandwidth=int(reg.bandwidth),
            )
        store.set_current_idx(key, new_idx)
        ts.current_band_key = key
        ts.step = None
        self._publish_tuning_state_changed()
        get_engine().event_bus.publish(BandStackChangedEvent(band_stack=store))
        save_device(engine)
        self.show_status(f"BAND {stack.band.name} REG {new_idx + 1}/{REGISTERS_PER_BAND}")

    def notify_demod_changed(self) -> None:
        """Called when mode changes via demod chord — reset step ladder."""
        self._reset_step_to_auto()
