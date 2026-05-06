from tsdr.radio.decoders.tetra import trackers
from tsdr.radio.decoders.tetra.mac import (
    AACHInfo,
    ChannelAllocation,
    CmceEvent,
    SB1Info,
    SysInfo,
)
from tsdr.radio.decoders.tetra.state import (
    CARRIER_ROLE_MULTI,
    CARRIER_ROLE_SINGLE,
    CARRIER_ROLE_TCH,
    CARRIER_ROLE_UNKNOWN,
    SLOT_USAGE_CONTROL,
    SLOT_USAGE_IDLE,
    SLOT_USAGE_SYNC,
    SLOT_USAGE_TRAFFIC,
    SYNC_STATE_LOCKED,
    SYNC_STATE_UNLOCKED,
    SYNC_STATE_UNLOCKING,
    TetraState,
)


def _make_sb1(timeslot: int = 0, mcc: int = 204, mnc: int = 500, cc: int = 2) -> SB1Info:
    return SB1Info(
        system_code=0,
        colour_code=cc,
        timeslot=timeslot,
        frame_number=1,
        multiframe_number=1,
        sharing_mode=0,
        ts_reserved=0,
        u_plane_dtx=0,
        frame18_ext=0,
        mcc=mcc,
        mnc=mnc,
        scramble_init=0xDEADBEEF,
    )


def _make_sysinfo(dl_freq_hz: int, main_carrier: int = 5) -> SysInfo:
    return SysInfo(
        main_carrier=main_carrier,
        freq_band=4,
        freq_offset=0,
        duplex_spacing=3,
        reverse_operation=0,
        num_csch=0,
        ms_txpwr_max=0,
        rxlev_access_min=0,
        access_parameter=0,
        radio_dl_timeout=0,
        cck_valid=0,
        cck_id_or_hyper_frame=0,
        location_area=1,
        subscriber_class=0,
        bs_service_details=0b100010,  # air_encryption + voice
        dl_freq_hz=dl_freq_hz,
        ul_freq_hz=dl_freq_hz - 10_000_000,
    )


# TDMA counter


def test_anchor_and_advance_tdma():
    state = TetraState()
    assert state.tdma.current_tn is None

    trackers.anchor_tdma(state, 0)  # SB1 timeslot 0 -> TN 1
    assert state.tdma.current_tn == 1

    trackers.advance_tdma(state)
    assert state.tdma.current_tn == 2
    trackers.advance_tdma(state)
    assert state.tdma.current_tn == 3
    trackers.advance_tdma(state)
    assert state.tdma.current_tn == 4
    trackers.advance_tdma(state)
    assert state.tdma.current_tn == 1  # wraps


def test_reset_tdma():
    state = TetraState()
    trackers.anchor_tdma(state, 2)
    assert state.tdma.current_tn == 3
    trackers.reset_tdma(state)
    assert state.tdma.current_tn is None
    # advance is a no-op when unanchored
    trackers.advance_tdma(state)
    assert state.tdma.current_tn is None


# network lock


def test_record_sb1_first_lock_marks_dirty():
    state = TetraState()
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    assert state.network is not None
    assert state.network.mcc == 204
    assert state.dirty
    assert state.quality.sync_state == SYNC_STATE_LOCKED
    assert state.tdma.current_tn == 1


def test_record_sb1_same_identity_does_not_redirty():
    state = TetraState()
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    state.dirty = False
    trackers.record_sb1(state, _make_sb1(), ts=1.0)
    assert not state.dirty


def test_record_sb1_different_identity_redirties():
    state = TetraState()
    trackers.record_sb1(state, _make_sb1(mcc=204, mnc=500), ts=0.0)
    state.dirty = False
    trackers.record_sb1(state, _make_sb1(mcc=204, mnc=641), ts=1.0)
    assert state.dirty
    assert state.network.mnc == 641


def test_record_sysinfo_derives_services():
    state = TetraState()
    si = _make_sysinfo(dl_freq_hz=425_206_250)
    trackers.record_sysinfo(state, si, ts=0.0)
    assert state.cell is not None
    assert state.cell.dl_freq_hz == 425_206_250
    assert "voice" in state.cell.services
    assert state.cell.has_air_encryption is True


def test_record_sysinfo_unchanged_skips_dirty():
    state = TetraState()
    si = _make_sysinfo(dl_freq_hz=425_206_250)
    trackers.record_sysinfo(state, si, ts=0.0)
    state.dirty = False
    trackers.record_sysinfo(state, si, ts=1.0)
    assert not state.dirty
    assert state.cell.sysinfo_ts == 1.0  # still refreshed


# AACH slot usage


def test_record_aach_normal_1_traffic_flag_sets_traffic_label():
    state = TetraState()
    trackers.anchor_tdma(state, 0)  # TN 1
    trackers.advance_tdma(state)  # TN 2
    aach = AACHInfo(header=1, field1=10, field2=0)  # field1 > 3 -> traffic
    trackers.record_aach(state, aach, "normal_1", ts=0.0)
    assert state.slots[2].usage_label == SLOT_USAGE_TRAFFIC
    assert state.slots[2].traffic_bursts == 1


def test_record_aach_normal_2_never_traffic():
    """normal_2 (dual half-slot) is never voice no matter what AACH says."""
    state = TetraState()
    trackers.anchor_tdma(state, 1)  # TN 2
    aach = AACHInfo(header=1, field1=10, field2=0)
    trackers.record_aach(state, aach, "normal_2", ts=0.0)
    assert state.slots[2].usage_label == SLOT_USAGE_CONTROL
    assert state.slots[2].traffic_bursts == 0


def test_record_aach_sync_burst_always_sync_label():
    state = TetraState()
    trackers.anchor_tdma(state, 0)  # TN 1
    aach = AACHInfo(header=1, field1=10, field2=0)  # would be traffic on normal_1
    trackers.record_aach(state, aach, "sync", ts=0.0)
    assert state.slots[1].usage_label == SLOT_USAGE_SYNC
    assert state.slots[1].traffic_bursts == 0


def test_record_aach_control_field1():
    state = TetraState()
    trackers.anchor_tdma(state, 2)  # TN 3
    aach = AACHInfo(header=1, field1=1, field2=0)
    trackers.record_aach(state, aach, "normal_1", ts=0.0)
    assert state.slots[3].usage_label == SLOT_USAGE_CONTROL


def test_record_aach_idle_field1():
    state = TetraState()
    trackers.anchor_tdma(state, 0)
    aach = AACHInfo(header=0, field1=0, field2=0)
    trackers.record_aach(state, aach, "normal_1", ts=0.0)
    assert state.slots[1].usage_label == SLOT_USAGE_IDLE


def test_record_aach_without_tdma_is_noop():
    state = TetraState()
    aach = AACHInfo(header=1, field1=10, field2=0)
    trackers.record_aach(state, aach, "normal_1", ts=0.0)
    assert all(slot.total_bursts == 0 for slot in state.slots.values())


def test_slot_label_smoothed_across_window():
    """Alternating control/idle bursts in a window should stabilise on control.

    Mirrors the TS1 (MCCH) situation the user hit: real bursts legitimately
    flip between 'control' (SCH/F frames 1-17) and 'sync' (frame 18) and
    occasionally 'idle', but the displayed label should be stable.
    """
    state = TetraState()
    trackers.anchor_tdma(state, 0)  # TN 1
    # Hand-roll: 10 CTRL, 3 SYNC, 2 IDLE in a 1-second window
    aach_ctrl = AACHInfo(header=1, field1=1, field2=0)
    aach_idle = AACHInfo(header=0, field1=0, field2=0)
    for i in range(10):
        trackers.record_aach(state, aach_ctrl, "normal_1", ts=i * 0.05)
    for i in range(3):
        trackers.record_aach(state, aach_ctrl, "sync", ts=0.5 + i * 0.05)
    for i in range(2):
        trackers.record_aach(state, aach_idle, "normal_1", ts=0.7 + i * 0.05)

    # record_aach on sync fires against TN currently held; we're on TN 1 for
    # every call above because advance_tdma isn't invoked. After 10 ctrl bursts
    # the dominant label is CTRL and must stay CTRL after the few sync and
    # idle samples. This is the fix for the flapping report.
    assert state.slots[1].usage_label == SLOT_USAGE_CONTROL


def test_slot_label_traffic_promoted_from_minority():
    """Traffic wins even as a minority because a voice call is the signal."""
    state = TetraState()
    trackers.anchor_tdma(state, 0)  # TN 1
    aach_ctrl = AACHInfo(header=1, field1=1, field2=0)
    aach_traffic = AACHInfo(header=1, field1=10, field2=0)
    # 17 control bursts then 3 traffic bursts -> 3/20 = 15% -> promoted
    for i in range(17):
        trackers.record_aach(state, aach_ctrl, "normal_1", ts=i * 0.05)
    for i in range(3):
        trackers.record_aach(state, aach_traffic, "normal_1", ts=0.85 + i * 0.05)
    assert state.slots[1].usage_label == SLOT_USAGE_TRAFFIC


def test_slot_label_window_trims_old_events():
    """Labels older than the window shouldn't keep influencing the display."""
    state = TetraState()
    trackers.anchor_tdma(state, 0)  # TN 1
    aach_traffic = AACHInfo(header=1, field1=10, field2=0)
    aach_ctrl = AACHInfo(header=1, field1=1, field2=0)
    # Ancient traffic burst
    trackers.record_aach(state, aach_traffic, "normal_1", ts=0.0)
    # ...and many recent control bursts far outside the ~2.5s window
    for i in range(20):
        trackers.record_aach(state, aach_ctrl, "normal_1", ts=100.0 + i * 0.05)
    assert state.slots[1].usage_label == SLOT_USAGE_CONTROL


# CMCE atomic call state


def test_record_cmce_d_setup_creates_active_call():
    state = TetraState()
    cmce = CmceEvent(msg_type="D-SETUP", call_id=1234, encryption_type=0)
    trackers.record_cmce(state, cmce, ts=5.0)
    assert 1234 in state.active_calls
    call = state.active_calls[1234]
    assert call.encryption_algo == "clear"


def test_record_cmce_d_release_removes_call():
    state = TetraState()
    trackers.record_cmce(
        state, CmceEvent(msg_type="D-SETUP", call_id=1234, encryption_type=0), ts=0.0
    )
    trackers.record_cmce(state, CmceEvent(msg_type="D-RELEASE", call_id=1234), ts=5.0)
    assert 1234 not in state.active_calls


def test_record_cmce_encryption_cannot_desync():
    """A D-CONNECT that flips encryption updates the existing call atomically."""
    state = TetraState()
    trackers.record_cmce(
        state, CmceEvent(msg_type="D-SETUP", call_id=1234, encryption_type=0), ts=0.0
    )
    trackers.record_cmce(
        state, CmceEvent(msg_type="D-CONNECT", call_id=1234, encryption_type=1), ts=1.0
    )
    call = state.active_calls[1234]
    assert call.encryption_algo == "TEA1"


# channel allocation / carrier role


def test_channel_allocation_on_carrier_keeps_role_single():
    state = TetraState()
    trackers.record_sysinfo(state, _make_sysinfo(dl_freq_hz=425_206_250), ts=0.0)
    trackers.set_tuned_frequency(state, 425_206_250)

    ca = ChannelAllocation(
        allocation_type=0,
        timeslot=2,
        ul_dl_type=0,
        clch_permission=0,
        cell_change_flag=0,
        carrier_number=5,
        extended=True,
        freq_band=4,
        freq_offset=0,
        duplex_spacing=3,
        reverse_operation=0,
        dl_freq_hz=425_206_250,
        ul_freq_hz=415_206_250,
    )
    cmce = CmceEvent(msg_type="D-SETUP", call_id=1, encryption_type=0, channel_allocation=ca)
    trackers.record_cmce(state, cmce, ts=0.0)

    assert not state.alt_dl_frequencies
    assert state.allocation_log[-1].is_offcarrier is False

    # Force some traffic so carrier_role can say single
    trackers.anchor_tdma(state, 0)
    trackers.advance_tdma(state)
    trackers.record_aach(state, AACHInfo(header=1, field1=10, field2=0), "normal_1", ts=0.1)
    assert state.carrier_role() == CARRIER_ROLE_SINGLE


def test_channel_allocation_off_carrier_flips_role_multi():
    state = TetraState()
    trackers.record_sysinfo(state, _make_sysinfo(dl_freq_hz=425_206_250), ts=0.0)
    trackers.set_tuned_frequency(state, 425_206_250)

    ca = ChannelAllocation(
        allocation_type=0,
        timeslot=2,
        ul_dl_type=0,
        clch_permission=0,
        cell_change_flag=0,
        carrier_number=99,
        extended=True,
        freq_band=4,
        freq_offset=0,
        duplex_spacing=3,
        reverse_operation=0,
        dl_freq_hz=425_475_000,
        ul_freq_hz=415_475_000,
    )
    cmce = CmceEvent(msg_type="D-SETUP", call_id=1, encryption_type=0, channel_allocation=ca)
    trackers.record_cmce(state, cmce, ts=0.0)

    assert state.allocation_log[-1].is_offcarrier is True
    assert 425_475_000 in state.alt_dl_frequencies
    assert state.carrier_role() == CARRIER_ROLE_MULTI


def test_carrier_role_tch_when_tuned_away():
    state = TetraState()
    trackers.record_sysinfo(state, _make_sysinfo(dl_freq_hz=425_206_250), ts=0.0)
    trackers.set_tuned_frequency(state, 425_106_250)  # different carrier
    assert state.carrier_role() == CARRIER_ROLE_TCH


def test_carrier_role_unknown_without_sysinfo():
    state = TetraState()
    trackers.set_tuned_frequency(state, 425_206_250)
    assert state.carrier_role() == CARRIER_ROLE_UNKNOWN


# sync watchdog


def test_watchdog_trips_at_exact_threshold():
    state = TetraState()
    # Put state in LOCKED so failures escalate to UNLOCKING first.
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    assert state.quality.sync_state == SYNC_STATE_LOCKED

    tripped = [trackers.record_sync_failure(state, max_failures=3) for _ in range(3)]
    assert tripped == [False, False, True]
    assert state.quality.sync_state == SYNC_STATE_UNLOCKED


def test_watchdog_unlocking_state_on_first_failure():
    state = TetraState()
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    trackers.record_sync_failure(state, max_failures=5)
    assert state.quality.sync_state == SYNC_STATE_UNLOCKING


def test_watchdog_recovery_resets_counter():
    state = TetraState()
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    trackers.record_sync_failure(state, max_failures=10)
    trackers.record_sync_failure(state, max_failures=10)
    assert state.quality.consecutive_sb1_failures == 2
    trackers.record_sync_recovery(state)
    assert state.quality.consecutive_sb1_failures == 0


# quality windowing


def test_crc_window_trims_stale_entries():
    state = TetraState()
    for i in range(5):
        trackers.record_quality(state, crc_ok=True, ts=float(i))
    assert state.quality.crc_pct() == 100.0

    # Jump forward past the window; only events newer than (now - 30s) survive.
    trackers.record_quality(state, crc_ok=False, ts=100.0)
    assert len(state.quality.crc_events) == 1
    assert state.quality.crc_pct() == 0.0


def test_bfi_window():
    state = TetraState()
    trackers.record_speech_frame(state, bfi=False, ts=0.0)
    trackers.record_speech_frame(state, bfi=True, ts=1.0)
    assert state.quality.bfi_pct() == 50.0


def test_freq_offset_ema():
    state = TetraState()
    trackers.record_freq_offset(state, 100.0)
    assert state.quality.freq_offset_hz == 100.0  # first sample seeds the EMA
    trackers.record_freq_offset(state, 200.0)
    # alpha=0.1 -> 100 + 0.1*(200-100) = 110
    assert abs(state.quality.freq_offset_hz - 110.0) < 1e-6
