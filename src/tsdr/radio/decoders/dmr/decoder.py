"""DMR protocol decoder.

IQ -> decimate -> FM discriminator -> M&M timing -> 4-level slicer -> dibit stream -> sync search -> burst parse

Reference: ETSI TS 102 361-1
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, DemodStatus
from tsdr.radio.decoders.dmr.constants import (
    BURST_DIBITS,
    CACH_START,
    DEVIATION,
    FIRST_HALF_DIBITS,
    SECOND_HALF_DIBITS,
    SLOT_TYPE_DIBITS,
    ST1_START,
    ST2_START,
    SYMBOL_RATE,
    SYNC_DIBITS,
    SYNC_MAX_ERRORS,
    SYNC_PATTERNS,
    VOICE_SYNC_TYPES,
    DataType,
    DecoderState,
    SyncType,
)
from tsdr.radio.decoders.dmr.fec import decode_cach, decode_slot_type, extract_voice_frames
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import FMDiscriminator, MuellerMuller, StreamingFilter, firwin
from tsdr.radio.vocoder.ambe import AmbePlus2Decoder

logger = logging.getLogger(__name__)

_TARGET_RATE = SYMBOL_RATE * 10
_DMR_PCM_RATE = 8000
_VOICE_TIMEOUT_SEC = 0.5
# Require this many consecutive voice bursts before decoding audio.
# Filters out false VOICE sync matches on noise.
_VOICE_CONFIDENCE_BURSTS = 3
_SLOT_ACTIVE_WINDOW_SEC = 2.0
_SLOT_SWITCH_HOLD_SEC = 0.5
# DMR superframe = 6 bursts (A has sync, B-E have EMB, F has sync).
# Allow up to 5 continuation bursts without a fresh sync match before
# forcing a re-sync.
_MAX_CONTINUATION_BURSTS = 5


@dataclass
class TimeslotState:
    """Per-timeslot voice call tracking."""

    in_voice_call: bool = False
    confirmed: bool = False  # True after VOICE_LC_HEADER or enough consecutive bursts
    voice_burst_count: int = 0
    last_voice_ts: float = 0.0
    voice_sync_type: SyncType = field(default=SyncType.BS_VOICE)


@dataclass(frozen=True)
class SlotSnapshot:
    """Immutable per-timeslot state for the DMR widget."""

    timeslot: int
    in_voice_call: bool
    voice_burst_count: int


@dataclass(frozen=True)
class QualitySnapshot:
    """Immutable quality metrics for the DMR widget."""

    locked: bool
    lock_pct: float
    cach_pct: float
    slot_type_pct: float
    burst_count: int


@dataclass(frozen=True)
class DMRSnapshot:
    """Immutable decoder state snapshot for the DMR widget."""

    color_code: int | None
    slots: tuple[SlotSnapshot, SlotSnapshot]
    quality: QualitySnapshot


def _slice_dibit(symbol: float) -> int:
    """Map a normalized FM symbol to a DMR dibit."""
    return 1 if symbol > 0 else 3


def _slice_dibit_4level(symbol: float) -> int:
    """Map FM symbol to 4-level dibit for data extraction.

    4FSK levels (ETSI TS 102 361-1 S4.2.2):
        +3 (+1944 Hz, sym > +0.5) -> dibit 1
        +1  (+648 Hz, sym > 0)    -> dibit 0
        -1  (-648 Hz, sym > -0.5) -> dibit 2
        -3 (-1944 Hz, sym <= -0.5) -> dibit 3
    """
    if symbol > 0.5:
        return 1
    elif symbol > 0:
        return 0
    elif symbol > -0.5:
        return 2
    else:
        return 3


def _match_sync(dibits: np.ndarray) -> tuple[SyncType, int] | None:
    """Match 24 dibits against all DMR sync patterns."""
    best_type: SyncType | None = None
    best_errors = SYNC_DIBITS

    for sync_type, pattern in SYNC_PATTERNS.items():
        errors = 0
        for i in range(SYNC_DIBITS):
            if dibits[i] != pattern[i]:
                errors += 1
                if errors > SYNC_MAX_ERRORS:
                    break
        if errors <= SYNC_MAX_ERRORS and errors < best_errors:
            best_errors = errors
            best_type = sync_type

    if best_type is not None:
        return best_type, best_errors
    return None


class DMRDecoder(Demodulator):
    """DMR protocol decoder with voice output.

    Processes raw IQ samples through:
    decimate -> FM discriminator -> M&M timing -> 4-level slicer -> sync detection -> burst parse

    Voice bursts are decoded through AMBE+2 vocoder to 8 kHz PCM audio.
    """

    HAS_AUDIO = True
    LABEL = "DMR"
    MODULATION = "4FSK"
    MESSAGE_TYPE = "dmr"
    HAS_TEXT = True
    FIXED_CHANNEL_BANDWIDTH = 12_500.0

    @property
    def audio_prebuffer_seconds(self) -> float:
        return 0.25

    def __init__(self, sample_rate: float = 250_000):
        super().__init__()
        self.sample_rate = sample_rate

        # Decimation to ~48 kHz (10 samples/symbol)
        self._decimation = max(1, round(sample_rate / _TARGET_RATE))
        self._decimated_rate = sample_rate / self._decimation
        self._sps = self._decimated_rate / SYMBOL_RATE

        # Anti-alias filter
        cutoff = min(self._decimated_rate * 0.45, DEVIATION * 3)
        self._antialias = StreamingFilter(
            firwin(101, cutoff, fs=sample_rate),
            [1.0],
            dtype=np.complex64,
        )
        self._decim_phase = 0

        # FM discriminator (output normalized: +/-1.0 = +/-DEVIATION Hz)
        self._fm = FMDiscriminator(self._decimated_rate, DEVIATION)

        # Mueller-Muller symbol timing recovery
        self._mm = MuellerMuller(self._sps, gain=0.01)

        # Constellation delay line: one symbol period of FM samples
        self._constellation_delay = round(self._sps)
        self._fm_tail = np.array([], dtype=np.float64)

        # Ring buffers for sync search and data extraction
        self._sync_buf = np.zeros(BURST_DIBITS, dtype=np.uint8)
        self._data_buf = np.zeros(BURST_DIBITS, dtype=np.uint8)
        self._sync_pos = 0
        self._dibit_count = 0

        # Burst collection state machine
        self._state = DecoderState.SEARCHING
        self._burst_buf = np.zeros(BURST_DIBITS, dtype=np.uint8)
        self._burst_second_half = np.zeros(SECOND_HALF_DIBITS, dtype=np.uint8)
        self._burst_collect_pos = 0
        self._burst_sync_type = SyncType.BS_DATA
        self._lock_remaining = 0
        self._continuation_count = 0  # bursts since last real sync match

        # Voice call state per timeslot (DMR has 2: TS0, TS1)
        self._ts_state: dict[int, TimeslotState] = {0: TimeslotState(), 1: TimeslotState()}
        self._last_slot = -1

        # AMBE+2 vocoders -- one per timeslot, state must not be shared
        self._vocoders: dict[int, AmbePlus2Decoder] = {
            0: AmbePlus2Decoder(),
            1: AmbePlus2Decoder(),
        }
        self._pcm_per_ts: dict[int, list[np.ndarray]] = {0: [], 1: []}
        self._selected_ts: int | None = None
        self._candidate_ts: int | None = None
        self._candidate_since: float = 0.0
        self._audio_batches: list[AudioBatch] = []

        # Statistics
        self._syncs_found = 0
        self._bursts_decoded = 0
        self._cach_ok = 0
        self._cach_fail = 0
        self._slot_type_ok = 0
        self._slot_type_fail = 0
        self._lock_hits = 0
        self._lock_misses = 0
        self._color_code_counts: dict[int, int] = {}
        self._data_type_counts: dict[int, int] = {}
        self._sync_type_counts: dict[SyncType, int] = {}
        self._dominant_cc: int | None = None
        self._dominant_cc_count: int = 0

        # Constellation
        self._constellation_points: np.ndarray | None = None

        # Output
        self._pending_messages: list[DecodedMessage] = []
        self._dirty = False
        self._last_snapshot_ts: float = 0.0

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        filtered = self._antialias.process(iq_samples)
        n_before = len(filtered)
        decimated = filtered[self._decim_phase :: self._decimation]
        n_used = n_before - self._decim_phase
        self._decim_phase = (self._decimation - n_used % self._decimation) % self._decimation

        fm = self._fm.process(decimated)

        # Transition constellation: plot each FM sample
        # against the sample one symbol period earlier. Shows a 4x4 grid
        # of clusters for the 16 possible dibit transitions.
        delay = self._constellation_delay
        if len(self._fm_tail) > 0:
            fm_full = np.concatenate([self._fm_tail, fm])
        else:
            fm_full = fm
        if len(fm_full) > delay:
            current = fm_full[delay:]
            delayed = fm_full[: len(current)]
            self._constellation_points = (current + 1j * delayed).astype(np.complex64)
        self._fm_tail = fm[-delay:].copy() if len(fm) >= delay else fm.copy()

        symbols = self._mm.process(fm)
        if len(symbols) < 2:
            return

        for sym in symbols:
            sym_f = float(sym)
            sync_dibit = _slice_dibit(sym_f)
            data_dibit = _slice_dibit_4level(sym_f)

            pos = self._sync_pos % BURST_DIBITS
            self._sync_buf[pos] = sync_dibit
            self._data_buf[pos] = data_dibit
            self._sync_pos = (self._sync_pos + 1) % BURST_DIBITS
            self._dibit_count += 1

            match self._state:
                case DecoderState.SEARCHING:
                    self._do_sync_search(capture_utc_s)
                case DecoderState.COLLECTING:
                    self._do_collect(capture_utc_s)
                case DecoderState.LOCKED:
                    self._do_locked(capture_utc_s)

        self._flush_selected_audio(capture_utc_s)
        self._maybe_emit_snapshot(capture_utc_s)

    # Burst framing state machine

    def _do_sync_search(self, timestamp: float) -> None:
        if self._dibit_count < SYNC_DIBITS:
            return

        sync_dibits = self._get_ring_history(self._sync_buf, SYNC_DIBITS)
        result = _match_sync(sync_dibits)
        if result is None:
            return

        sync_type, _errors = result
        self._syncs_found += 1
        self._burst_sync_type = sync_type

        if self._dibit_count < FIRST_HALF_DIBITS + SYNC_DIBITS:
            return

        self._extract_first_half()
        self._begin_collecting()

    def _do_collect(self, timestamp: float) -> None:
        prev_pos = (self._sync_pos - 1) % BURST_DIBITS
        self._burst_second_half[self._burst_collect_pos] = self._data_buf[prev_pos]
        self._burst_collect_pos += 1

        if self._burst_collect_pos >= SECOND_HALF_DIBITS:
            self._burst_buf[FIRST_HALF_DIBITS : FIRST_HALF_DIBITS + SYNC_DIBITS] = 0
            self._burst_buf[FIRST_HALF_DIBITS + SYNC_DIBITS :] = self._burst_second_half

            valid = self._process_burst(self._burst_buf, self._burst_sync_type, timestamp)
            if valid:
                self._state = DecoderState.LOCKED
                self._lock_remaining = FIRST_HALF_DIBITS + SYNC_DIBITS
            else:
                self._drop_lock()

    def _do_locked(self, timestamp: float) -> None:
        """Frame-locked tracking with voice continuation support.

        Voice continuation bursts (B-E in a superframe) have EMB instead of
        sync. Trust frame timing only within a superframe window
        (_MAX_CONTINUATION_BURSTS). After that, force a re-sync to
        prevent locking on noise indefinitely.
        """
        self._lock_remaining -= 1
        if self._lock_remaining > 0:
            return

        sync_dibits = self._get_ring_history(self._sync_buf, SYNC_DIBITS)
        result = _match_sync(sync_dibits)

        if result is not None:
            sync_type, _errors = result
            self._lock_hits += 1
            self._syncs_found += 1
            self._continuation_count = 0
            self._burst_sync_type = sync_type
            self._extract_first_half()
            self._begin_collecting()
        elif self._voice_continuation_expected(timestamp):
            self._lock_hits += 1
            self._continuation_count += 1
            expected_slot = 1 - self._last_slot
            self._burst_sync_type = self._ts_state[expected_slot].voice_sync_type
            self._extract_first_half()
            self._begin_collecting()
        else:
            self._lock_misses += 1
            self._drop_lock()

    def _drop_lock(self) -> None:
        """Return to SEARCHING and reset all transient state so the UI
        doesn't show stale info (quality percentages, voice call status)."""
        self._state = DecoderState.SEARCHING
        self._lock_hits = 0
        self._lock_misses = 0
        self._cach_ok = 0
        self._cach_fail = 0
        self._slot_type_ok = 0
        self._slot_type_fail = 0
        for ts in self._ts_state.values():
            ts.in_voice_call = False
            ts.confirmed = False
            ts.voice_burst_count = 0
        self._dirty = True

    def _voice_continuation_expected(self, timestamp: float) -> bool:
        if self._last_slot < 0:
            return False
        if self._continuation_count >= _MAX_CONTINUATION_BURSTS:
            return False
        expected_slot = 1 - self._last_slot
        ts = self._ts_state.get(expected_slot)
        if ts is None or not ts.in_voice_call:
            return False
        return (timestamp - ts.last_voice_ts) < _VOICE_TIMEOUT_SEC

    def _extract_first_half(self) -> None:
        full_history = FIRST_HALF_DIBITS + SYNC_DIBITS
        history = self._get_ring_history(self._data_buf, full_history)
        self._burst_buf[:FIRST_HALF_DIBITS] = history[:FIRST_HALF_DIBITS]

    def _begin_collecting(self) -> None:
        self._state = DecoderState.COLLECTING
        self._burst_collect_pos = 0

    def _get_ring_history(self, buf: np.ndarray, n: int) -> np.ndarray:
        start = (self._sync_pos - n) % BURST_DIBITS
        if start < self._sync_pos:
            return buf[start : self._sync_pos].copy()
        return np.concatenate([buf[start:], buf[: self._sync_pos]])

    # Burst processing

    def _process_burst(self, burst: np.ndarray, sync_type: SyncType, timestamp: float) -> bool:
        """Parse a complete 144-dibit burst. Returns True to stay locked."""
        self._bursts_decoded += 1
        self._sync_type_counts[sync_type] = self._sync_type_counts.get(sync_type, 0) + 1

        cach = decode_cach(burst[CACH_START : CACH_START + 12])
        if cach is not None:
            self._cach_ok += 1
        else:
            self._cach_fail += 1

        # CACH must decode for any burst to be valid -- this is the primary
        # guard against locking on noise.
        if cach is None:
            return False

        timeslot = cach.timeslot
        self._last_slot = timeslot

        if sync_type in VOICE_SYNC_TYPES:
            return self._process_voice_burst(burst, sync_type, timeslot, timestamp)
        return self._process_data_burst(burst, sync_type, timeslot, True, timestamp)

    def _process_voice_burst(
        self, burst: np.ndarray, sync_type: SyncType, timeslot: int, timestamp: float
    ) -> bool:
        """Handle a voice burst -- extract AMBE frames, decode to PCM.

        Only decodes audio after confidence is established (VOICE_LC_HEADER
        seen, or enough consecutive voice bursts at locked timing). This
        prevents false VOICE sync matches on noise from producing audio.
        """
        ts = self._ts_state[timeslot]
        if not ts.in_voice_call:
            ts.in_voice_call = True
            ts.voice_burst_count = 0
            ts.voice_sync_type = sync_type
            self._dirty = True
            logger.debug("dmr_voice_call_started slot=%d sync_type=%s", timeslot, sync_type)

        ts.voice_burst_count += 1
        ts.last_voice_ts = timestamp

        if not ts.confirmed and ts.voice_burst_count >= _VOICE_CONFIDENCE_BURSTS:
            ts.confirmed = True
            logger.debug(
                "dmr_voice_call_confirmed slot=%d bursts=%d", timeslot, ts.voice_burst_count
            )

        if ts.confirmed:
            for frame_bytes in extract_voice_frames(burst):
                pcm = self._vocoders[timeslot].decode_frame(frame_bytes)
                self._pcm_per_ts[timeslot].append(pcm)

        if ts.voice_burst_count <= 3 or ts.voice_burst_count % 100 == 0:
            self._pending_messages.append(
                DecodedMessage(
                    text=f"{sync_type} slot={timeslot} voice burst#{ts.voice_burst_count}",
                    timestamp=timestamp,
                )
            )

        return True

    def _process_data_burst(
        self,
        burst: np.ndarray,
        sync_type: SyncType,
        timeslot: int,
        cach_ok: bool,
        timestamp: float,
    ) -> bool:
        """Handle a data burst -- decode Slot Type, track voice call lifecycle."""
        st1 = burst[ST1_START : ST1_START + SLOT_TYPE_DIBITS // 2]
        st2 = burst[ST2_START : ST2_START + SLOT_TYPE_DIBITS // 2]
        slot_type = decode_slot_type(st1, st2)

        if slot_type is None:
            self._slot_type_fail += 1
            text = f"{sync_type} slot={timeslot} [SlotType err]"
            if not cach_ok:
                text += " [CACH err]"
            self._pending_messages.append(DecodedMessage(text=text, timestamp=timestamp))
            return False

        self._slot_type_ok += 1
        cc = slot_type.color_code
        new_cc_count = self._color_code_counts.get(cc, 0) + 1
        self._color_code_counts[cc] = new_cc_count
        if new_cc_count > self._dominant_cc_count:
            self._dominant_cc = cc
            self._dominant_cc_count = new_cc_count
        self._data_type_counts[slot_type.data_type] = (
            self._data_type_counts.get(slot_type.data_type, 0) + 1
        )

        # Voice call lifecycle
        ts = self._ts_state[timeslot]
        if slot_type.data_type == DataType.VOICE_LC_HEADER:
            ts.in_voice_call = True
            ts.confirmed = True  # header from Slot Type = high confidence
            ts.voice_burst_count = 0
            ts.last_voice_ts = timestamp
            self._dirty = True
        elif slot_type.data_type == DataType.TERMINATOR_WITH_LC and ts.in_voice_call:
            logger.debug("dmr_voice_call_ended slot=%d bursts=%d", timeslot, ts.voice_burst_count)
            ts.in_voice_call = False
            ts.confirmed = False
            self._dirty = True
            self._vocoders[timeslot] = AmbePlus2Decoder()

        try:
            dt_name = DataType(slot_type.data_type).name
        except ValueError:
            dt_name = f"UNKNOWN({slot_type.data_type})"

        if self._bursts_decoded <= 5 or self._bursts_decoded % 100 == 0:
            logger.debug(
                "dmr_burst index=%d sync_type=%s slot=%d cc=%d type=%s",
                self._bursts_decoded,
                sync_type,
                timeslot,
                slot_type.color_code,
                dt_name,
            )

        if slot_type.data_type != DataType.IDLE:
            text = f"{sync_type} slot={timeslot} cc={slot_type.color_code} type={dt_name}"
            if not cach_ok:
                text += " [CACH err]"
            self._pending_messages.append(DecodedMessage(text=text, timestamp=timestamp))

        return True

    # Snapshot emission

    _SNAPSHOT_INTERVAL = 0.25  # ~4 Hz

    def _maybe_emit_snapshot(self, timestamp: float) -> None:
        due = (timestamp - self._last_snapshot_ts) >= self._SNAPSHOT_INTERVAL
        if not (self._dirty or due):
            return
        self._dirty = False
        self._last_snapshot_ts = timestamp
        snap = self._to_snapshot()
        self._pending_messages.append(DecodedMessage(text="", timestamp=timestamp, data=snap))

    def _to_snapshot(self) -> DMRSnapshot:
        cc = self._dominant_cc

        slots = tuple(
            SlotSnapshot(
                timeslot=ts,
                in_voice_call=state.in_voice_call,
                voice_burst_count=state.voice_burst_count,
            )
            for ts, state in sorted(self._ts_state.items())
        )

        total_lock = self._lock_hits + self._lock_misses
        total_cach = self._cach_ok + self._cach_fail
        total_st = self._slot_type_ok + self._slot_type_fail

        quality = QualitySnapshot(
            locked=self._state != DecoderState.SEARCHING,
            lock_pct=self._lock_hits / total_lock * 100 if total_lock > 0 else 0.0,
            cach_pct=self._cach_ok / total_cach * 100 if total_cach > 0 else 0.0,
            slot_type_pct=self._slot_type_ok / total_st * 100 if total_st > 0 else 0.0,
            burst_count=self._bursts_decoded,
        )

        return DMRSnapshot(color_code=cc, slots=slots, quality=quality)  # type: ignore[arg-type]

    # Audio output

    def _select_active_ts(self, now: float) -> int | None:
        """Pick timeslot with freshest speech, with hysteresis to prevent flapping."""
        fresh = [
            (slot, state.last_voice_ts)
            for slot, state in self._ts_state.items()
            if state.last_voice_ts > 0.0 and now - state.last_voice_ts <= _SLOT_ACTIVE_WINDOW_SEC
        ]
        if not fresh:
            self._selected_ts = None
            self._candidate_ts = None
            return None

        newest_ts = max(fresh, key=lambda x: x[1])[0]
        fresh_set = {ts for ts, _ in fresh}

        if self._selected_ts is None or self._selected_ts not in fresh_set:
            self._selected_ts = newest_ts
            self._candidate_ts = None
            return newest_ts

        if newest_ts == self._selected_ts:
            self._candidate_ts = None
            return self._selected_ts

        if self._candidate_ts != newest_ts:
            self._candidate_ts = newest_ts
            self._candidate_since = now
            return self._selected_ts
        if now - self._candidate_since >= _SLOT_SWITCH_HOLD_SEC:
            self._selected_ts = newest_ts
            self._candidate_ts = None
        return self._selected_ts

    def _flush_selected_audio(self, timestamp: float) -> None:
        """Package the selected slot's buffered PCM into AudioBatch."""
        selected = self._select_active_ts(timestamp)

        # Discard non-selected slot's PCM immediately to prevent backlog
        # buildup when both slots have active voice calls
        for slot in self._pcm_per_ts:
            if slot != selected:
                self._pcm_per_ts[slot].clear()

        if selected is None:
            return
        chunks = self._pcm_per_ts[selected]
        if not chunks:
            return
        pcm_int16 = np.concatenate(chunks)
        self._pcm_per_ts[selected] = []

        samples = pcm_int16.astype(np.float32) / 32768.0 * 8.0  # +18 dB
        self._audio_batches.append(
            AudioBatch(
                samples=np.clip(samples, -1.0, 1.0),
                sample_rate=_DMR_PCM_RATE,
                stereo=False,
                prebuffer_seconds=self.audio_prebuffer_seconds,
            )
        )

    # Demodulator interface

    def get_audio(self) -> list[AudioBatch]:
        batches = self._audio_batches
        self._audio_batches = []
        return batches

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._pending_messages
        self._pending_messages = []
        return messages

    def get_constellation(self) -> tuple[np.ndarray, str] | None:
        points = self._constellation_points
        self._constellation_points = None
        if points is None:
            return None
        return points, "4FSK"

    def status(self) -> DemodStatus:
        """Thread-safe: callable from any thread. Reads scalar counters only."""
        quality = None
        quality_label = None
        total_lock = self._lock_hits + self._lock_misses
        if total_lock > 0:
            quality = self._lock_hits / total_lock
            quality_label = f"Lock {quality * 100:.0f}%"
        description = None
        cc = self._dominant_cc
        if cc is not None:
            description = f"CC{cc}"

        return DemodStatus(
            quality_label=quality_label,
            quality=quality,
            description=description,
        )

    def reset(self) -> None:
        self._fm.reset()
        self._mm.reset()
        self._fm_tail = np.array([], dtype=np.float64)
        self._sync_buf[:] = 0
        self._data_buf[:] = 0
        self._sync_pos = 0
        self._dibit_count = 0
        self._state = DecoderState.SEARCHING
        self._burst_buf[:] = 0
        self._burst_collect_pos = 0
        self._lock_remaining = 0
        self._continuation_count = 0
        self._last_slot = -1
        for ts in self._ts_state.values():
            ts.in_voice_call = False
            ts.confirmed = False
            ts.voice_burst_count = 0
            ts.last_voice_ts = 0.0
        self._vocoders = {ts: AmbePlus2Decoder() for ts in self._vocoders}
        for buf in self._pcm_per_ts.values():
            buf.clear()
        self._selected_ts = None
        self._candidate_ts = None
        self._audio_batches.clear()
        self._syncs_found = 0
        self._bursts_decoded = 0
        self._cach_ok = 0
        self._cach_fail = 0
        self._slot_type_ok = 0
        self._slot_type_fail = 0
        self._lock_hits = 0
        self._lock_misses = 0
        self._color_code_counts.clear()
        self._data_type_counts.clear()
        self._sync_type_counts.clear()
        self._dominant_cc = None
        self._dominant_cc_count = 0
        self._pending_messages.clear()
        self._antialias.reset()
        self._decim_phase = 0
