"""Minimal render test for TETRAWidget.

Builds a hand-crafted TetraSnapshot and feeds it through the widget's rendering
functions. We can't easily exercise the full Textual event loop in a unit test
so we call the render helpers directly.
"""

from __future__ import annotations

from tsdr.radio.decoders.tetra.state import (
    CARRIER_ROLE_MULTI,
    CARRIER_ROLE_SINGLE,
    CARRIER_ROLE_TCH,
    AllocationLogEntry,
    CallSnapshot,
    CellInfo,
    NetworkIdentity,
    QualitySnapshot,
    SlotSnapshot,
    TetraSnapshot,
)
from tsdr.tui.widgets.tetra_widget import TETRAWidget


def _snap(
    carrier_role: str,
    alloc_entries: tuple[AllocationLogEntry, ...] = (),
    calls: tuple[CallSnapshot, ...] = (),
) -> TetraSnapshot:
    network = NetworkIdentity(mcc=204, mnc=500, colour_code=2, scramble_init=0xDEAD)
    cell = CellInfo(
        main_carrier=5,
        freq_band=4,
        freq_offset=0,
        duplex_spacing=3,
        reverse_operation=0,
        dl_freq_hz=425_206_250,
        ul_freq_hz=415_206_250,
        location_area=1,
        services=("voice", "sndcp", "air_encryption"),
        has_air_encryption=True,
        sysinfo_ts=0.0,
    )
    slots = (
        SlotSnapshot(
            tn=1,
            usage_label="sync",
            traffic_ratio=0.0,
            total_bursts=10,
            traffic_bursts=0,
            active_call_id=None,
        ),
        SlotSnapshot(
            tn=2,
            usage_label="traffic",
            traffic_ratio=0.7,
            total_bursts=10,
            traffic_bursts=7,
            active_call_id=1234,
        ),
        SlotSnapshot(
            tn=3,
            usage_label="control",
            traffic_ratio=0.0,
            total_bursts=10,
            traffic_bursts=0,
            active_call_id=None,
        ),
        SlotSnapshot(
            tn=4,
            usage_label="idle",
            traffic_ratio=0.0,
            total_bursts=10,
            traffic_bursts=0,
            active_call_id=None,
        ),
    )
    quality = QualitySnapshot(
        crc_pct=95.0,
        bfi_pct=12.0,
        freq_offset_hz=5.0,
        sync_state="locked",
        burst_count=12_847,
    )
    return TetraSnapshot(
        network=network,
        cell=cell,
        slots=slots,
        active_calls=calls,
        allocation_log=alloc_entries,
        alt_dl_frequencies=(),
        carrier_role=carrier_role,
        quality=quality,
        tuned_hz=425_206_250,
    )


def test_render_network_single_carrier():
    snap = _snap(carrier_role=CARRIER_ROLE_SINGLE)
    text = TETRAWidget._render_network(snap)
    # "TETRA" used to be part of the column content but is now the widget's
    # border_title, so we only check the fields rendered into this column.
    assert "MCC 204 MNC 500" in text
    assert "425.2063 MHz" in text  # 425_206_250 formatted with .4f rounds up
    assert "SINGLE" in text


def test_render_network_multi_carrier_shows_magenta_label():
    snap = _snap(carrier_role=CARRIER_ROLE_MULTI)
    text = TETRAWidget._render_network(snap)
    assert "MULTI" in text


def test_render_network_tch():
    snap = _snap(carrier_role=CARRIER_ROLE_TCH)
    text = TETRAWidget._render_network(snap)
    assert "TCH" in text


def test_render_slots_labels_each_timeslot():
    snap = _snap(carrier_role=CARRIER_ROLE_SINGLE)
    text = TETRAWidget._render_slots(snap)
    assert "TS1" in text and "TS2" in text and "TS3" in text and "TS4" in text
    assert "VOICE" in text  # TS2 is traffic in the fixture
    assert "SYNC" in text  # TS1 is sync


def test_render_calls_active_and_allocations():
    alloc_entries = (
        AllocationLogEntry(
            timestamp=0.0,
            call_id=8521,
            msg_type="D-SETUP",
            timeslot=3,
            carrier_number=10,
            dl_freq_hz=425_106_250,
            is_offcarrier=True,
            encryption_algo="clear",
        ),
    )
    calls = (
        CallSnapshot(
            call_id=7604,
            encryption_algo="clear",
            assigned_slot=2,
            assigned_dl_freq_hz=425_206_250,
            is_offcarrier=False,
            setup_ts=0.0,
        ),
    )
    snap = _snap(carrier_role=CARRIER_ROLE_SINGLE, alloc_entries=alloc_entries, calls=calls)
    text = TETRAWidget._render_calls(snap)
    assert "7604" in text
    assert "8521" in text
    assert "ACTIVE" in text
    assert "ALLOCS" in text


def test_render_quality_contains_all_metrics():
    widget = TETRAWidget()
    widget._snr_db = 22.1
    snap = _snap(carrier_role=CARRIER_ROLE_SINGLE)
    text = widget._render_quality(snap)
    assert "LOCKED" in text
    assert "CRC" in text and "95%" in text
    assert "BFI" in text and "12%" in text
    assert "SNR" in text and "22.1dB" in text
    assert "12847 bursts" in text.replace(" ", "") or "12847" in text.replace(" ", "")
    assert "ACELP" in text
