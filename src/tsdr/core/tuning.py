from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from tsdr.core.band_stack import (
    BandRegister,
    get_band_stack,
    is_writeback_suspended,
)
from tsdr.core.events.events import (
    ConfigChangedEvent,
    Event,
    FocusChangedEvent,
    TuningStateChangedEvent,
)
from tsdr.core.preferences import save_tuning_state
from tsdr.core.sdr.engine import SDREngine, get_engine
from tsdr.core.tuning_state import get_tuning_state

if TYPE_CHECKING:
    from tsdr.core.sdr.device_context import SDRDeviceContext

logger = logging.getLogger(__name__)

STEP_LADDER: tuple[float | None, ...] = (
    None,
    10,
    100,
    1_000,
    3_000,
    5_000,
    12_500,
    25_000,
    100_000,
    1_000_000,
)

_MODE_STEP_MAP: dict[str, float] = {
    "CW": 50,
    "USB": 100,
    "LSB": 100,
    "AM": 1_000,
    "NFM": 12_500,
    "WFM": 100_000,
}

# (upper_bound_hz, step_hz) — first row whose bound > freq wins.
_RAW_STEP_TABLE: tuple[tuple[float, float], ...] = (
    (500_000, 100),  # LF: NDB, time signals
    (30_000_000, 1_000),  # MW + SW: broadcast / SSB
    (88_000_000, 12_500),  # VHF low: business / military NFM
    (108_000_000, 100_000),  # FM broadcast
    (137_000_000, 25_000),  # air band AM
    (1_000_000_000, 12_500),  # VHF/UHF NFM-dominant
    (3_000_000_000, 100_000),  # L/S band
    (float("inf"), 1_000_000),  # microwave
)

# Default channel bandwidth per mode (mirrors each demodulator's
# DEFAULT_CHANNEL_BANDWIDTH; kept here so tuning helpers don't reach into
# radio.demodulators just to read a constant).
_MODE_DEFAULT_BANDWIDTH: dict[str, int] = {
    "WFM": 200_000,
    "NFM": 12_500,
    "AM": 10_000,
    "USB": 3_000,
    "LSB": 3_000,
    "CW": 200,
}


def resolve_auto_step(mode: str, freq_hz: float) -> float:
    step = _MODE_STEP_MAP.get(mode)
    if step is not None:
        return step
    for upper, raw_step in _RAW_STEP_TABLE:
        if freq_hz < upper:
            return raw_step
    return 1_000_000


def snap_to_grid(current_hz: float, step_hz: float, direction: int) -> float:
    """Step-aligned move. First press off-grid tidies onto the grid."""
    return step_hz * (round(current_hz / step_hz) + direction)


def snap_step_value(target_hz: float) -> int:
    """Snap to largest 1/2/5×10ⁿ that is ≤ target."""
    if target_hz <= 0:
        return 1
    exponent = math.floor(math.log10(target_hz))
    base = 10**exponent
    for mult in (10, 5, 2, 1):
        if base * mult <= target_hz:
            return int(base * mult)
    return int(base)


def bandwidth_step(current_bw: float) -> int:
    """Bandwidth step ~20% of current, snapped to 1/2/5×10ⁿ. Floor at 10 Hz."""
    return snap_step_value(max(current_bw * 0.20, 10.0))


def default_bandwidth(mode: str) -> int:
    return _MODE_DEFAULT_BANDWIDTH.get(mode, 12_500)


def current_channel_bandwidth(context: SDRDeviceContext) -> float:
    """Effective channel bandwidth for the device: config → demod_info → mode default.

    Config wins over demod_info because the demodulator's applied value lags
    update_device_config (config change is delivered via pipeline_control_queue).
    Reading demod_info first would let rapid bandwidth adjustments compute the
    next step from a stale applied value.
    """
    if context.config.channel_bandwidth is not None:
        return float(context.config.channel_bandwidth)
    info = context.active_demod_info
    if info is not None and info.channel_bandwidth:
        return float(info.channel_bandwidth)
    return float(default_bandwidth(context.active_mode))


def save_previous_tune_state(context: SDRDeviceContext) -> None:
    """Snapshot (freq, mode, bw) into TuningState.previous and persist+publish."""
    ts = get_tuning_state()
    ts.previous = (
        float(context.config.center_frequency),
        context.active_mode,
        current_channel_bandwidth(context),
    )
    save_tuning_state(ts)
    get_engine().event_bus.publish(TuningStateChangedEvent(state=ts))


# Holding a tune key fires config events at ~30 Hz; without throttling we'd
# rewrite the band-stack TOML on each event. In-memory updates are immediate;
# disk writes are throttled and flushed by `flush_band_stack_writeback` on app
# exit (see app.on_unmount).
_WRITEBACK_SAVE_MIN_INTERVAL = 0.25
_writeback_last_save_ts = 0.0
_writeback_pending = False


def _on_writeback_trigger(event: Event) -> None:
    """Mirror focused-device config into the active band-stack register.

    Fires on:
      - ConfigChangedEvent for the focused device — freq/mode/bw changed
        while on a band.
      - FocusChangedEvent — focus moved to another device; the active band's
        register should reflect the newly-focused device's tuning, and if the
        new device is tuned outside the current band, `current_band_key` must
        be cleared so subsequent ConfigChangedEvents don't corrupt that band's
        register.

    Suppressed during band recall (see `suspended_writeback`).
    """
    if is_writeback_suspended():
        return
    engine = get_engine()
    focused_id = engine.focused_device
    if focused_id is None:
        return
    # ConfigChangedEvent for a non-focused device is irrelevant; FocusChangedEvent
    # always applies (focus just moved to focused_id).
    if isinstance(event, ConfigChangedEvent) and event.device_id != focused_id:
        return
    ts = get_tuning_state()
    if ts.current_band_key is None:
        return
    device = engine.devices.get(focused_id)
    if device is None:
        return
    store = get_band_stack()
    stack = store.get_by_key(ts.current_band_key)
    if stack is None:
        ts.current_band_key = None
        save_tuning_state(ts)
        engine.event_bus.publish(TuningStateChangedEvent(state=ts))
        return
    freq = float(device.config.center_frequency)
    if not (stack.band.start <= freq <= stack.band.end):
        ts.current_band_key = None
        save_tuning_state(ts)
        engine.event_bus.publish(TuningStateChangedEvent(state=ts))
        return
    reg = BandRegister(
        slot=stack.current_idx,
        frequency=int(freq),
        mode=device.active_mode,
        bandwidth=int(current_channel_bandwidth(device)),
    )
    store.update_register(stack.band.key, stack.current_idx, reg)

    global _writeback_last_save_ts, _writeback_pending
    now = time.monotonic()
    if now - _writeback_last_save_ts >= _WRITEBACK_SAVE_MIN_INTERVAL:
        store.save()
        _writeback_last_save_ts = now
        _writeback_pending = False
    else:
        _writeback_pending = True


def flush_band_stack_writeback() -> None:
    """Flush any pending throttled band-stack write (call on app exit)."""
    global _writeback_last_save_ts, _writeback_pending
    if _writeback_pending:
        get_band_stack().save()
        _writeback_last_save_ts = time.monotonic()
        _writeback_pending = False


def subscribe_band_stack_writeback(engine: SDREngine) -> None:
    """Wire the band-stack writeback subscriber to the engine's event bus."""
    engine.event_bus.subscribe(ConfigChangedEvent, _on_writeback_trigger)
    engine.event_bus.subscribe(FocusChangedEvent, _on_writeback_trigger)
