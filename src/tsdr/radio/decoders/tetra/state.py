"""TETRA decoder state and immutable snapshot.

The decoder's mutable working state lives in `TetraState` (and its sub-dataclasses).
A `TetraSnapshot` is an immutable view produced on demand by `TetraState.to_snapshot()`
and shipped to the TUI through the existing `DecoderOutputEvent` path, the same way
`RDSData` is shipped from the RDS decoder.

Design rules:
- Only this module knows the layout of the state; trackers and the decoder mutate
  it through well-named functions in `trackers.py`.
- Snapshots are frozen and structurally compared so the stage/widget can dedupe.
- No numpy types leak into the snapshot; fields are plain ints/strings/tuples for
  safe equality and serialization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Carrier-role verdict strings are a closed set.
CARRIER_ROLE_SINGLE = "single"
CARRIER_ROLE_MULTI = "multi"
CARRIER_ROLE_TCH = "tch"
CARRIER_ROLE_UNKNOWN = "unknown"

# Slot usage labels are a closed set.
SLOT_USAGE_UNKNOWN = "unknown"
SLOT_USAGE_IDLE = "idle"
SLOT_USAGE_SYNC = "sync"
SLOT_USAGE_CONTROL = "control"
SLOT_USAGE_TRAFFIC = "traffic"

# Sync state strings.
SYNC_STATE_UNLOCKED = "unlocked"
SYNC_STATE_UNLOCKING = "unlocking"
SYNC_STATE_LOCKED = "locked"

# How close the tuned frequency must be to the SYSINFO DL frequency to count
# as "on the MCCH". Matches the old TunerWidget tolerance.
MCCH_MATCH_TOLERANCE_HZ = 10_000

# Quality window length (seconds). CRC/BFI stats are computed over this window.
QUALITY_WINDOW_SECONDS = 30.0

# Rolling window length for per-slot usage labels. Shorter than the quality
# window so the slot column reacts to call start/stop reasonably quickly, but
# long enough to cover ~2 full TETRA multiframes (~2.16 s) so per-frame label
# churn (sync bursts only appear in frame 18, etc.) is smoothed out.
SLOT_LABEL_WINDOW_SECONDS = 2.5

# Minimum fraction of bursts in the window that must carry a given label for
# it to win over a lower-priority but more frequent label. Used only for the
# "traffic" priority promotion: even a minority of traffic bursts means the
# slot is being used for voice right now.
SLOT_TRAFFIC_PROMOTE_RATIO = 0.15


# mutable operational state (decoder-owned)


@dataclass
class NetworkIdentity:
    """MCC/MNC/CC triple identifying a TETRA network + the derived scramble init."""

    mcc: int
    mnc: int
    colour_code: int
    scramble_init: int


@dataclass
class CellInfo:
    """Cell-level parameters decoded from SYSINFO (BROADCAST PDU in SB2)."""

    main_carrier: int
    freq_band: int
    freq_offset: int
    duplex_spacing: int
    reverse_operation: int
    dl_freq_hz: int
    ul_freq_hz: int
    location_area: int
    services: tuple[str, ...]
    has_air_encryption: bool
    sysinfo_ts: float


@dataclass
class SlotState:
    """Rolling state for one TDMA slot (TN 1-4).

    The decoder calls `record_aach(..., burst_type, ...)` on every burst whose
    TN we know. That function classifies the burst into a raw label (sync,
    control, traffic, idle, unknown) and appends it to `label_events`. The
    *displayed* `usage_label` is the dominant label over the last
    SLOT_LABEL_WINDOW_SECONDS, not the last-seen label -- this avoids the
    label flapping that happens on MCCH slots where consecutive bursts
    legitimately carry different traffic (SB1 in frame 18, SCH/F in others).
    """

    tn: int
    usage_label: str = SLOT_USAGE_UNKNOWN
    last_aach_header: int | None = None
    last_aach_field1: int | None = None
    last_update_ts: float = 0.0
    total_bursts: int = 0
    traffic_bursts: int = 0
    last_active_call_id: int | None = None
    label_events: deque[tuple[float, str]] = field(default_factory=deque)


@dataclass
class ActiveCall:
    """A single active call, atomically created from a D-SETUP/D-CONNECT event."""

    call_id: int
    encryption_algo: str  # "clear" | "TEA1" | "TEA2" | "TEA3"
    setup_ts: float
    last_seen_ts: float
    assigned_slot: int | None = None
    assigned_carrier: int | None = None
    assigned_dl_freq_hz: int | None = None
    ssi: int | None = None


@dataclass
class AllocationLogEntry:
    """One channel-allocation observation. Stored in a bounded deque."""

    timestamp: float
    call_id: int | None
    msg_type: str
    timeslot: int
    carrier_number: int
    dl_freq_hz: int | None
    is_offcarrier: bool
    encryption_algo: str


@dataclass
class SignalQualityWindow:
    """Windowed quality metrics.

    `crc_events` / `speech_events` are deques of (timestamp, ok) tuples
    from the last QUALITY_WINDOW_SECONDS. The decoder thread owns mutation
    via trackers; the UI thread only reads `crc_pct` / `bfi_pct`, which
    are plain floats maintained incrementally so readers never iterate
    the deques. (The free-threaded build has no GIL to make iteration
    concurrent-safe.)

    `_crc_ok_count` / `_speech_bad_count` cache the running sums so the
    cached percentages can be updated in O(1) on append/trim.

    Thread-safety: `record_quality` / `record_speech_frame` / `_trim_window`
    must only be called from the decoder thread.
    """

    freq_offset_hz: float | None = None
    sync_state: str = SYNC_STATE_UNLOCKED
    consecutive_sb1_failures: int = 0
    last_sb1_ts: float = 0.0
    lifetime_bursts: int = 0
    fragments_started: int = 0
    fragments_completed: int = 0
    crc_events: deque[tuple[float, bool]] = field(default_factory=deque)
    speech_events: deque[tuple[float, bool]] = field(default_factory=deque)
    _crc_pct: float = 0.0
    _bfi_pct: float | None = None
    _crc_ok_count: int = 0
    _speech_bad_count: int = 0

    def crc_pct(self) -> float:
        """Thread-safe: returns a cached scalar updated by the decoder thread."""
        return self._crc_pct

    def bfi_pct(self) -> float | None:
        """Thread-safe: returns a cached scalar updated by the decoder thread."""
        return self._bfi_pct


@dataclass
class TdmaCounter:
    """TN of the next burst to be processed. `None` until we have anchored on an SB1."""

    current_tn: int | None = None


@dataclass
class TetraState:
    """Top-level mutable state of the TETRA decoder.

    All mutation goes through functions in `trackers.py`. `dirty` is set by any
    tracker that materially changes the state and cleared by the decoder when it
    emits a snapshot.
    """

    network: NetworkIdentity | None = None
    cell: CellInfo | None = None
    slots: dict[int, SlotState] = field(
        default_factory=lambda: {n: SlotState(tn=n) for n in (1, 2, 3, 4)}
    )
    active_calls: dict[int, ActiveCall] = field(default_factory=dict)
    allocation_log: deque[AllocationLogEntry] = field(default_factory=lambda: deque(maxlen=20))
    alt_dl_frequencies: set[int] = field(default_factory=set)
    quality: SignalQualityWindow = field(default_factory=SignalQualityWindow)
    tdma: TdmaCounter = field(default_factory=TdmaCounter)
    recent_sds: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    tuned_hz: int | None = None
    dirty: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def carrier_role(self) -> str:
        """Derive the single/multi/tch/unknown verdict from current state."""
        if self.cell is None or self.tuned_hz is None:
            return CARRIER_ROLE_UNKNOWN

        delta = abs(int(self.tuned_hz) - int(self.cell.dl_freq_hz))
        if delta > MCCH_MATCH_TOLERANCE_HZ:
            return CARRIER_ROLE_TCH

        if self.alt_dl_frequencies:
            return CARRIER_ROLE_MULTI

        any_traffic = any(slot.traffic_bursts > 0 for slot in self.slots.values())
        if any_traffic:
            return CARRIER_ROLE_SINGLE

        return CARRIER_ROLE_UNKNOWN

    def to_snapshot(self) -> TetraSnapshot:
        """Produce an immutable view of the current state for the TUI."""
        q = self.quality
        quality_snap = QualitySnapshot(
            crc_pct=q.crc_pct(),
            bfi_pct=q.bfi_pct(),
            freq_offset_hz=q.freq_offset_hz,
            sync_state=q.sync_state,
            burst_count=q.lifetime_bursts,
            fragments_started=q.fragments_started,
            fragments_completed=q.fragments_completed,
        )

        def _snap_slot(tn: int) -> SlotSnapshot:
            slot = self.slots[tn]
            return SlotSnapshot(
                tn=slot.tn,
                usage_label=slot.usage_label,
                traffic_ratio=(
                    slot.traffic_bursts / slot.total_bursts if slot.total_bursts > 0 else 0.0
                ),
                total_bursts=slot.total_bursts,
                traffic_bursts=slot.traffic_bursts,
                active_call_id=slot.last_active_call_id,
            )

        slot_snaps: tuple[SlotSnapshot, SlotSnapshot, SlotSnapshot, SlotSnapshot] = (
            _snap_slot(1),
            _snap_slot(2),
            _snap_slot(3),
            _snap_slot(4),
        )

        call_snaps = tuple(
            CallSnapshot(
                call_id=c.call_id,
                encryption_algo=c.encryption_algo,
                assigned_slot=c.assigned_slot,
                assigned_dl_freq_hz=c.assigned_dl_freq_hz,
                is_offcarrier=(
                    c.assigned_dl_freq_hz is not None
                    and self.cell is not None
                    and c.assigned_dl_freq_hz != self.cell.dl_freq_hz
                ),
                setup_ts=c.setup_ts,
            )
            for c in sorted(self.active_calls.values(), key=lambda x: x.setup_ts)
        )

        return TetraSnapshot(
            network=self.network,
            cell=self.cell,
            slots=slot_snaps,
            active_calls=call_snaps,
            allocation_log=tuple(self.allocation_log),
            alt_dl_frequencies=tuple(sorted(self.alt_dl_frequencies)),
            carrier_role=self.carrier_role(),
            quality=quality_snap,
            recent_sds=tuple(self.recent_sds),
            tuned_hz=self.tuned_hz,
        )


# immutable snapshot (widget-facing)


@dataclass(frozen=True)
class SlotSnapshot:
    tn: int
    usage_label: str
    traffic_ratio: float
    total_bursts: int
    traffic_bursts: int
    active_call_id: int | None


@dataclass(frozen=True)
class CallSnapshot:
    call_id: int
    encryption_algo: str
    assigned_slot: int | None
    assigned_dl_freq_hz: int | None
    is_offcarrier: bool
    setup_ts: float


@dataclass(frozen=True)
class QualitySnapshot:
    crc_pct: float
    bfi_pct: float | None
    freq_offset_hz: float | None
    sync_state: str
    burst_count: int
    fragments_started: int = 0
    fragments_completed: int = 0


@dataclass(frozen=True)
class TetraSnapshot:
    """Immutable snapshot of TETRA decoder state shipped to the TUI.

    Frozen and structurally compared so the stage/widget can dedupe.
    """

    network: NetworkIdentity | None
    cell: CellInfo | None
    slots: tuple[SlotSnapshot, SlotSnapshot, SlotSnapshot, SlotSnapshot]
    active_calls: tuple[CallSnapshot, ...]
    allocation_log: tuple[AllocationLogEntry, ...]
    alt_dl_frequencies: tuple[int, ...]
    carrier_role: str
    quality: QualitySnapshot
    recent_sds: tuple[str, ...] = ()
    tuned_hz: int | None = None
    voice_codec: str = "ACELP"
