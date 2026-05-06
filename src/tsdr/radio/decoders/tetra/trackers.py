"""Pure mutator functions that update TetraState.

All state changes in the TETRA decoder go through one of the `record_*`
functions here. Each function takes the state plus its specific input and
mutates the state in place; no return value unless the caller needs a boolean
decision (e.g. `record_sync_failure` returns whether the watchdog just tripped).

    1. _active_call_id / _call_encrypted desync   -> ActiveCall is atomic
    2. one-shot sysinfo_emitted without retry    -> identity compare
    3. watchdog off-by-one                        -> explicit threshold check
    4. deferred freq-offset init                  -> None sentinel handled here
    5. normal_2 bursts treated as traffic         -> usage label is AACH-gated
                                                      *and* burst-type-gated
    6. lifetime CRC%                              -> windowed deque
    7. hidden state mutation in process methods   -> visible at every call site
    8. TN drift across normal bursts              -> anchor_tdma/advance_tdma
"""

from __future__ import annotations

import logging

from tsdr.radio.decoders.tetra.mac import (
    AACHInfo,
    ChannelAllocation,
    CmceEvent,
    MacResult,
    SB1Info,
    SysInfo,
    carrier_to_freq,
    format_services,
)
from tsdr.radio.decoders.tetra.state import (
    MCCH_MATCH_TOLERANCE_HZ,
    QUALITY_WINDOW_SECONDS,
    SLOT_LABEL_WINDOW_SECONDS,
    SLOT_TRAFFIC_PROMOTE_RATIO,
    SLOT_USAGE_CONTROL,
    SLOT_USAGE_IDLE,
    SLOT_USAGE_SYNC,
    SLOT_USAGE_TRAFFIC,
    SLOT_USAGE_UNKNOWN,
    SYNC_STATE_LOCKED,
    SYNC_STATE_UNLOCKED,
    SYNC_STATE_UNLOCKING,
    ActiveCall,
    AllocationLogEntry,
    CellInfo,
    NetworkIdentity,
    SignalQualityWindow,
    SlotState,
    TetraState,
)

logger = logging.getLogger(__name__)


_ALGO_NAMES = {0: "clear", 1: "TEA1", 2: "TEA2", 3: "TEA3"}


def _algo_name(encryption_type: int) -> str:
    """Map the MAC-RESOURCE 2-bit encryption field to a display label."""
    return _ALGO_NAMES.get(encryption_type, f"TEA{encryption_type}")


# TDMA counter


def anchor_tdma(state: TetraState, sb1_timeslot_field: int) -> None:
    """Set the TDMA counter from a CRC-validated SB1 timeslot field (0..3)."""
    state.tdma.current_tn = (sb1_timeslot_field & 0x3) + 1


def advance_tdma(state: TetraState) -> None:
    """Move the TDMA counter to the next slot (modulo 4). No-op if unanchored."""
    if state.tdma.current_tn is not None:
        state.tdma.current_tn = (state.tdma.current_tn % 4) + 1


def reset_tdma(state: TetraState) -> None:
    """Drop the TDMA anchor (called on sync loss / decoder reset)."""
    state.tdma.current_tn = None


# network / cell


def record_sb1(state: TetraState, sb1: SB1Info, ts: float) -> bool:
    """Record a CRC-validated SB1. Returns True when the network identity changed."""
    new_identity = NetworkIdentity(
        mcc=sb1.mcc,
        mnc=sb1.mnc,
        colour_code=sb1.colour_code,
        scramble_init=sb1.scramble_init,
    )
    changed = state.network != new_identity
    if changed:
        state.network = new_identity
        state.mark_dirty()

    anchor_tdma(state, sb1.timeslot)

    q = state.quality
    q.last_sb1_ts = ts
    q.consecutive_sb1_failures = 0
    if q.sync_state != SYNC_STATE_LOCKED:
        q.sync_state = SYNC_STATE_LOCKED
        state.mark_dirty()
    return changed


def record_sysinfo(state: TetraState, si: SysInfo, ts: float) -> bool:
    """Record a parsed SYSINFO. Returns True when cell info changed materially."""
    services = format_services(si.bs_service_details)
    new_cell = CellInfo(
        main_carrier=si.main_carrier,
        freq_band=si.freq_band,
        freq_offset=si.freq_offset,
        duplex_spacing=si.duplex_spacing,
        reverse_operation=si.reverse_operation,
        dl_freq_hz=si.dl_freq_hz,
        ul_freq_hz=si.ul_freq_hz,
        location_area=si.location_area,
        services=services,
        has_air_encryption="air_encryption" in services,
        sysinfo_ts=ts,
    )

    changed = (
        state.cell is None
        or state.cell.main_carrier != new_cell.main_carrier
        or state.cell.dl_freq_hz != new_cell.dl_freq_hz
        or state.cell.location_area != new_cell.location_area
        or state.cell.services != new_cell.services
    )
    if changed:
        state.cell = new_cell
        state.mark_dirty()
    elif state.cell is not None:
        # Same cell content, just refresh the timestamp so consumers can tell
        # how recently we saw a SYSINFO. No dirty flag bump.
        state.cell.sysinfo_ts = ts
    return changed


# signal quality


def _trim_window(q: SignalQualityWindow, now: float) -> None:
    cutoff = now - QUALITY_WINDOW_SECONDS
    while q.crc_events and q.crc_events[0][0] < cutoff:
        _, ok = q.crc_events.popleft()
        if ok:
            q._crc_ok_count -= 1
    while q.speech_events and q.speech_events[0][0] < cutoff:
        _, bfi = q.speech_events.popleft()
        if bfi:
            q._speech_bad_count -= 1


def _refresh_crc_pct(q: SignalQualityWindow) -> None:
    n = len(q.crc_events)
    q._crc_pct = 100.0 * q._crc_ok_count / n if n else 0.0


def _refresh_bfi_pct(q: SignalQualityWindow) -> None:
    n = len(q.speech_events)
    q._bfi_pct = 100.0 * q._speech_bad_count / n if n else None


def record_quality(state: TetraState, crc_ok: bool, ts: float) -> None:
    q = state.quality
    q.crc_events.append((ts, crc_ok))
    if crc_ok:
        q._crc_ok_count += 1
    _trim_window(q, ts)
    _refresh_crc_pct(q)
    # bfi depends on the same window cutoff; refresh in case _trim_window evicted speech events.
    _refresh_bfi_pct(q)
    # Quality is derived on snapshot() so we don't need to mark dirty every burst.


def record_speech_frame(state: TetraState, bfi: bool, ts: float) -> None:
    q = state.quality
    q.speech_events.append((ts, bfi))
    if bfi:
        q._speech_bad_count += 1
    _trim_window(q, ts)
    _refresh_bfi_pct(q)
    _refresh_crc_pct(q)


def record_freq_offset(state: TetraState, hz: float) -> None:
    q = state.quality
    prev = q.freq_offset_hz
    if prev is None:
        q.freq_offset_hz = hz
    else:
        # EMA smoothing, alpha=0.1.
        q.freq_offset_hz = prev + 0.1 * (hz - prev)


# burst + AACH


def record_burst(state: TetraState, burst_type: str, ts: float) -> None:
    """Record that *some* burst arrived. Bumps lifetime counter + per-slot total."""
    state.quality.lifetime_bursts += 1
    tn = state.tdma.current_tn
    if tn is not None:
        slot = state.slots[tn]
        slot.total_bursts += 1
        slot.last_update_ts = ts


def record_aach(
    state: TetraState,
    aach: AACHInfo,
    burst_type: str,
    ts: float,
) -> None:
    """Record an AACH observation for the current slot.

    The raw per-burst label is a function of *both* the AACH DL usage field and
    the burst type -- a normal_2 (dual half-slot) burst is never voice traffic
    regardless of what field1 says, and a sync burst is always "sync".

    The *displayed* slot.usage_label is derived from a rolling window of raw
    labels (last SLOT_LABEL_WINDOW_SECONDS) so MCCH slots don't flap between
    sync/control/idle on every burst.
    """
    tn = state.tdma.current_tn
    if tn is None:
        return

    slot = state.slots[tn]
    slot.last_aach_header = aach.header
    slot.last_aach_field1 = aach.field1
    slot.last_update_ts = ts

    raw_label = _usage_label(aach, burst_type)
    if raw_label == SLOT_USAGE_TRAFFIC:
        slot.traffic_bursts += 1

    slot.label_events.append((ts, raw_label))
    _trim_slot_window(slot, ts)
    new_label = _dominant_label(slot)

    if slot.usage_label != new_label:
        slot.usage_label = new_label
        state.mark_dirty()


def _trim_slot_window(slot: SlotState, now: float) -> None:
    cutoff = now - SLOT_LABEL_WINDOW_SECONDS
    while slot.label_events and slot.label_events[0][0] < cutoff:
        slot.label_events.popleft()


def _dominant_label(slot: SlotState) -> str:
    """Pick a stable label from the rolling window.

    Rules:
    * If no events, keep "unknown".
    * If the fraction of 'traffic' events >= SLOT_TRAFFIC_PROMOTE_RATIO, return
      "traffic" -- voice presence wins even as a minority because "this slot is
      carrying voice calls right now" is the most operationally important fact.
    * Otherwise, return the mode of (sync, control, idle). Ties are broken in
      the order control > sync > idle (control is the most informative label
      for a signaling slot).
    """
    events = slot.label_events
    if not events:
        return SLOT_USAGE_UNKNOWN

    counts: dict[str, int] = {}
    for _, lab in events:
        counts[lab] = counts.get(lab, 0) + 1
    total = len(events)

    traffic = counts.get(SLOT_USAGE_TRAFFIC, 0)
    if total > 0 and traffic / total >= SLOT_TRAFFIC_PROMOTE_RATIO:
        return SLOT_USAGE_TRAFFIC

    # Priority-ordered list for mode + tiebreak.
    priority = (
        SLOT_USAGE_CONTROL,
        SLOT_USAGE_SYNC,
        SLOT_USAGE_IDLE,
        SLOT_USAGE_UNKNOWN,
    )
    best_label = SLOT_USAGE_UNKNOWN
    best_count = -1
    for label in priority:
        count = counts.get(label, 0)
        if count > best_count:
            best_count = count
            best_label = label
    return best_label


def _usage_label(aach: AACHInfo, burst_type: str) -> str:
    if burst_type == "sync":
        return SLOT_USAGE_SYNC
    if burst_type == "normal_2":
        # Dual half-slot; not a TCH slot. We still track the AACH header/field1
        # for diagnostics but the label never becomes "traffic".
        return SLOT_USAGE_CONTROL
    # burst_type == "normal_1" (SCH/F) or "unknown": fall back on AACH field1.
    if aach.header >= 1 and aach.field1 > 3:
        return SLOT_USAGE_TRAFFIC
    if aach.header >= 1 and aach.field1 in (1, 2):
        return SLOT_USAGE_CONTROL
    return SLOT_USAGE_IDLE


# MAC PDU / CMCE


def record_mac_pdu(state: TetraState, result: MacResult, ts: float) -> None:
    if result.cmce is not None:
        record_cmce(state, result.cmce, ts)


# CMCE events that create or refresh an ActiveCall entry. D-SETUP / D-CONNECT
# are the canonical call-setup signals, but we also learn about calls that
# were already in progress when we tuned in via the PTT floor-control events
# (D-TX-GRANTED / D-TX-CEASED) which carry the same call identifier.
_CALL_REFRESH_MSG_TYPES = frozenset({"D-SETUP", "D-CONNECT", "D-TX-GRANTED", "D-TX-CEASED"})

# How long an ActiveCall survives without a refreshing CMCE event before we
# assume the call ended and we missed the D-RELEASE. Picked to comfortably
# span the quiet gap between PTT holds on a trunked voice call.
ACTIVE_CALL_TTL_SECONDS = 10.0


def record_cmce(state: TetraState, cmce: CmceEvent, ts: float) -> None:
    """Update active_calls based on a CMCE signaling event.

    Creates/updates/removes ActiveCall entries atomically so that call_id and
    encryption_algo can never desync.
    """
    call_id = cmce.call_id
    algo = _algo_name(cmce.encryption_type)

    if cmce.msg_type in _CALL_REFRESH_MSG_TYPES:
        if call_id is None:
            return
        existing = state.active_calls.get(call_id)
        if existing is None:
            call = ActiveCall(
                call_id=call_id,
                encryption_algo=algo,
                setup_ts=ts,
                last_seen_ts=ts,
            )
        else:
            call = existing
            # Only D-SETUP / D-CONNECT are authoritative for encryption state;
            # the PTT events don't carry encryption info.
            if cmce.msg_type in ("D-SETUP", "D-CONNECT"):
                call.encryption_algo = algo
            call.last_seen_ts = ts
        _apply_channel_allocation_to_call(call, cmce.channel_allocation, state)
        state.active_calls[call_id] = call
        state.mark_dirty()
    elif cmce.msg_type in ("D-DISCONNECT", "D-RELEASE"):
        if call_id is not None and call_id in state.active_calls:
            del state.active_calls[call_id]
            state.mark_dirty()

    if cmce.channel_allocation is not None:
        record_channel_allocation(state, cmce.channel_allocation, cmce, ts)


def expire_stale_calls(state: TetraState, ts: float) -> None:
    """Drop ActiveCalls whose last CMCE event is older than the TTL.

    Called from the decoder's periodic snapshot path so calls that end
    without a visible D-RELEASE (e.g. the sample ran out, or the D-RELEASE
    burst CRC'd out) eventually fall off the widget instead of accumulating.
    """
    stale = [
        cid
        for cid, call in state.active_calls.items()
        if ts - call.last_seen_ts > ACTIVE_CALL_TTL_SECONDS
    ]
    if stale:
        for cid in stale:
            del state.active_calls[cid]
        state.mark_dirty()


def _apply_channel_allocation_to_call(
    call: ActiveCall,
    ca: ChannelAllocation | None,
    state: TetraState,
) -> None:
    if ca is None:
        return
    call.assigned_slot = ca.timeslot
    call.assigned_carrier = ca.carrier_number
    call.assigned_dl_freq_hz = resolve_allocation_dl_hz(ca, state)


def resolve_allocation_dl_hz(ca: ChannelAllocation, state: TetraState) -> int | None:
    """Return the absolute DL frequency of an allocation.

    If the allocation IE included extended carrier numbering, use its computed
    `dl_freq_hz`. Otherwise fall back on our stored SYSINFO to reuse the cell's
    freq band / duplex spacing for the new carrier number.
    """
    if ca.dl_freq_hz is not None:
        return ca.dl_freq_hz
    if state.cell is None:
        return None
    dl, _ = carrier_to_freq(
        ca.carrier_number,
        state.cell.freq_band,
        state.cell.freq_offset,
        state.cell.duplex_spacing,
        state.cell.reverse_operation,
    )
    return dl


def record_channel_allocation(
    state: TetraState,
    ca: ChannelAllocation,
    cmce: CmceEvent,
    ts: float,
) -> None:
    """Append an entry to the allocation log and flag multi-carrier if off-carrier."""
    dl_hz = resolve_allocation_dl_hz(ca, state)
    is_off = (
        state.cell is not None
        and dl_hz is not None
        and abs(dl_hz - state.cell.dl_freq_hz) > MCCH_MATCH_TOLERANCE_HZ
    )
    entry = AllocationLogEntry(
        timestamp=ts,
        call_id=cmce.call_id,
        msg_type=cmce.msg_type,
        timeslot=ca.timeslot,
        carrier_number=ca.carrier_number,
        dl_freq_hz=dl_hz,
        is_offcarrier=bool(is_off),
        encryption_algo=_algo_name(cmce.encryption_type),
    )
    state.allocation_log.append(entry)
    if is_off and dl_hz is not None:
        state.alt_dl_frequencies.add(int(dl_hz))
    state.mark_dirty()


# sync watchdog


def record_sync_failure(state: TetraState, max_failures: int) -> bool:
    """Increment the consecutive-SB1-failure counter.

    Returns True when the counter has reached `max_failures` so the caller
    should trigger a full decoder reset. The threshold is inclusive: the
    N-th failure trips the watchdog, not the N+1-th.
    """
    q = state.quality
    q.consecutive_sb1_failures += 1
    if q.sync_state == SYNC_STATE_LOCKED:
        q.sync_state = SYNC_STATE_UNLOCKING
        state.mark_dirty()
    if q.consecutive_sb1_failures >= max_failures:
        q.sync_state = SYNC_STATE_UNLOCKED
        state.mark_dirty()
        return True
    return False


def record_sync_recovery(state: TetraState) -> None:
    state.quality.consecutive_sb1_failures = 0


# MAC fragmentation counters


def record_fragment_started(state: TetraState) -> None:
    """Bump the fragmentation-started counter (a MAC-RESOURCE with LI=0x3F)."""
    state.quality.fragments_started += 1


def record_fragment_completed(state: TetraState) -> None:
    """Bump the fragmentation-completed counter (a MAC-END finalised a chain)."""
    state.quality.fragments_completed += 1
    state.mark_dirty()


# external triggers from decoder


def set_tuned_frequency(state: TetraState, tuned_hz: int | None) -> None:
    if state.tuned_hz != tuned_hz:
        state.tuned_hz = tuned_hz
        state.mark_dirty()


def add_sds_text(state: TetraState, text: str) -> None:
    state.recent_sds.append(text)
    state.mark_dirty()
