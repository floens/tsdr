"""TETRA protocol decoder -- thin orchestrator over TetraState + trackers.

All mutable state lives in `self._state: TetraState`. All mutation happens
through functions in `trackers.py`. This class only:

1. Parses raw IQ -> symbols via `TetraDemod`.
2. Extracts bursts via `SyncDetector`.
3. For each burst, calls `decode_block` + `parse_*` to turn bits into typed
   dataclasses, then hands them to `trackers.record_*`.
4. Emits human-readable text messages for the decoder-output console.
5. Emits a `TetraSnapshot` (attached to a `DecodedMessage.data` payload) for
   the `TETRAWidget` whenever the state is dirty or a periodic refresh is due.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, SignalInfo
from tsdr.radio.decoders.tetra import trackers
from tsdr.radio.decoders.tetra.bit_reader import BitReader
from tsdr.radio.decoders.tetra.burst import (
    NormalBurst,
    extract_normal_burst,
    extract_schf,
    extract_sync_burst,
)
from tsdr.radio.decoders.tetra.channel import decode_block, rm3014_decode
from tsdr.radio.decoders.tetra.demod import TetraDemod, estimate_freq_offset
from tsdr.radio.decoders.tetra.mac import (
    ChannelAllocation,
    CmceEvent,
    MacFragmentContinue,
    MacFragmentEnd,
    MacFragmentStart,
    MacResult,
    format_mac_resource_summary,
    format_sysinfo,
    parse_aach,
    parse_llc_and_mle,
    parse_mac_pdu,
    parse_sb1,
    parse_sysinfo,
    strip_fill_bits,
)
from tsdr.radio.decoders.tetra.scramble import SCRAMB_INIT
from tsdr.radio.decoders.tetra.speech_channel import bits_to_bytes, decode_speech
from tsdr.radio.decoders.tetra.state import (
    CARRIER_ROLE_MULTI,
    CARRIER_ROLE_SINGLE,
    CARRIER_ROLE_TCH,
    CARRIER_ROLE_UNKNOWN,
    TetraState,
)
from tsdr.radio.decoders.tetra.sync import BurstResult, SyncDetector
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.vocoder.acelp import TetraAcelpVocoder

logger = logging.getLogger(__name__)

# SSI broadcast address -- RESOURCE PDUs to this SSI are idle channel markers
_SSI_BROADCAST = 16777215

# Watchdog: this many consecutive SB1 CRC failures triggers a full reset
# (~0.4 s of sync bursts at the nominal rate).
_MAX_SB1_FAILURES = 30

# Emit a state snapshot at least this often even when nothing material changed,
# so the widget's quality column stays live.
_SNAPSHOT_INTERVAL_SEC = 1.0

# ACELP native PCM rate; the audio worker resamples to its target rate.
_TETRA_PCM_RATE = 8_000
# A slot is "active" for audio selection if it produced speech within this window.
_SLOT_ACTIVE_WINDOW_SEC = 2.0
# Hysteresis: how long a new candidate TN must be freshest before we switch.
_SLOT_SWITCH_HOLD_SEC = 0.5

# How long an in-flight MAC fragmentation chain survives without a refreshing
# MAC-FRAG / MAC-END before we drop it. Spans several multiframes so a
# few CRC-failed continuation bursts don't kill an otherwise valid chain.
_FRAG_TIMEOUT_SEC = 2.0


@dataclass
class _FragCache:
    """Last MAC-RESOURCE snapshot per logical channel, plus a cached_ts for expiry."""

    snapshot: MacFragmentStart
    cached_ts: float


@dataclass
class _FragChain:
    """In-flight TM-SDU reassembly seeded from a cached MAC-RESOURCE snapshot."""

    snapshot: MacFragmentStart
    extra_pieces: list[np.ndarray]
    started_ts: float
    last_update_ts: float
    source_label: str = field(default="")


class TETRADecoder(Demodulator):
    """TETRA digital trunked radio decoder."""

    @property
    def audio_prebuffer_seconds(self) -> float:
        return 0.25

    def __init__(self, sample_rate: float = 2_048_000):
        super().__init__()
        self._sample_rate = sample_rate
        self._demod = TetraDemod(sample_rate)
        self._sync = SyncDetector()
        self._state = TetraState()
        self._pending_messages: list[DecodedMessage] = []
        self._last_snapshot_ts: float = 0.0

        # Per-channel snapshot of the most recent MAC-RESOURCE's TM-SDU
        # bits + addressing. Becomes the implicit start of a fragmentation
        # chain if MAC-FRAG / MAC-END arrives later on the same channel.
        # Real-world encoders frequently fragment without LI=0x3F.
        self._frag_cache: dict[tuple[int, str], _FragCache] = {}
        # In-flight chains keyed by (TN, logical-channel). A chain begins
        # when a MAC-FRAG promotes a cache entry; MAC-END finalises it.
        self._frag_chains: dict[tuple[int, str], _FragChain] = {}

        # One ACELP vocoder per TDMA slot -- LSP/excitation history cannot be
        # shared across slots without scrambling the predictor state.
        self._vocoders: dict[int, TetraAcelpVocoder] = {
            tn: TetraAcelpVocoder() for tn in (1, 2, 3, 4)
        }
        self._pcm_per_tn: dict[int, list[np.ndarray]] = {tn: [] for tn in (1, 2, 3, 4)}
        self._last_speech_ts: dict[int, float] = dict.fromkeys((1, 2, 3, 4), 0.0)
        self._selected_tn: int | None = None
        self._candidate_tn: int | None = None
        self._candidate_since: float = 0.0
        self._audio_batches: list[AudioBatch] = []

    # Demodulator hooks

    def set_tuned_frequency(self, frequency_hz: int) -> None:
        trackers.set_tuned_frequency(self._state, int(frequency_hz))

    def demodulate(self, iq_samples: np.ndarray, timestamp: float) -> None:
        symbols = self._demod.process_symbols(iq_samples)
        if len(symbols) == 0:
            return

        for burst in self._sync.process(symbols):
            trackers.record_burst(self._state, burst.burst_type, timestamp)

            if burst.burst_type == "sync":
                self._handle_sync_burst(burst, timestamp)
            elif burst.burst_type in ("normal_1", "normal_2"):
                self._handle_normal_burst(burst, timestamp)

            trackers.advance_tdma(self._state)

        self._flush_selected_audio(timestamp)
        self._maybe_emit_snapshot(timestamp)

    # burst handlers

    def _handle_sync_burst(self, burst: BurstResult, timestamp: float) -> None:
        sb = extract_sync_burst(burst.soft_bits)

        freq_offset = estimate_freq_offset(burst.diff_symbols)

        type1, crc_ok = decode_block(sb.sb1, "SB1", SCRAMB_INIT)
        trackers.record_quality(self._state, crc_ok, timestamp)
        if not crc_ok:
            tripped = trackers.record_sync_failure(self._state, _MAX_SB1_FAILURES)
            if tripped:
                logger.info("tetra_decoder_reset reason=sb1_crc_failures")
                self.reset()
            return

        trackers.record_sync_recovery(self._state)
        trackers.record_freq_offset(self._state, freq_offset)
        if self._state.quality.freq_offset_hz is not None:
            self._demod.apply_freq_correction(self._state.quality.freq_offset_hz)

        info = parse_sb1(type1)

        identity_changed = trackers.record_sb1(self._state, info, timestamp)
        if identity_changed:
            logger.info(
                "tetra_locked mcc=%d mnc=%d cc=%d scramble_init=0x%08X",
                info.mcc,
                info.mnc,
                info.colour_code,
                info.scramble_init,
            )
            self._emit(
                f"Network: MCC={info.mcc} MNC={info.mnc} CC={info.colour_code}",
                timestamp,
                data=info,
            )

        bbk_info = rm3014_decode(sb.bbk)
        aach = parse_aach(bbk_info)
        trackers.record_aach(self._state, aach, "sync", timestamp)

        if self._state.network is None:
            return

        scramble_init = self._state.network.scramble_init
        sb2_type1, sb2_ok = decode_block(sb.sb2, "SB2", scramble_init)
        trackers.record_quality(self._state, sb2_ok, timestamp)
        if not sb2_ok:
            logger.debug("tetra_sb2_crc_failed")
            return

        si = parse_sysinfo(sb2_type1)
        if si is not None:
            cell_changed = trackers.record_sysinfo(self._state, si, timestamp)
            if cell_changed:
                self._emit(format_sysinfo(si), timestamp)
        else:
            self._process_mac_pdu(sb2_type1, "SB2", timestamp)

    def _handle_normal_burst(self, burst: BurstResult, timestamp: float) -> None:
        if self._state.network is None:
            return
        scramble_init = self._state.network.scramble_init

        ndb = extract_normal_burst(burst.soft_bits)
        bbk_info = rm3014_decode(ndb.bbk)
        aach = parse_aach(bbk_info)
        trackers.record_aach(self._state, aach, burst.burst_type, timestamp)

        # Speech traffic can only live in normal_1 (SCH/F) bursts. normal_2 is
        # the dual-half-slot format and is never a TCH. Decode every traffic
        # burst unconditionally -- gating on known ActiveCalls means we decode
        # nothing when tuned directly to a TCH (no CMCE signaling is carried
        # there). Encryption cannot be detected from AACH alone, so garbage
        # audio on an encrypted TCH is accepted as the documented behavior.
        is_traffic = burst.burst_type == "normal_1" and aach.header >= 1 and aach.field1 > 3

        if is_traffic:
            self._process_traffic_burst(ndb, timestamp)
            return

        if burst.burst_type == "normal_1":
            schf = extract_schf(ndb)
            schf_type1, ok = decode_block(schf, "SCH_F", scramble_init)
            trackers.record_quality(self._state, ok, timestamp)
            if ok:
                self._process_mac_pdu(schf_type1, "SCH/F", timestamp)
            else:
                logger.debug("tetra_schf_crc_failed")
        else:
            for label, blk in [("BKN1", ndb.bkn1), ("BKN2", ndb.bkn2)]:
                blk_type1, ok = decode_block(blk, "NDB", scramble_init)
                trackers.record_quality(self._state, ok, timestamp)
                if ok:
                    self._process_mac_pdu(blk_type1, label, timestamp)
                else:
                    logger.debug("tetra_block_crc_failed label=%s", label)

    def _process_traffic_burst(self, ndb: NormalBurst, timestamp: float) -> None:
        """Decode a traffic burst into two ACELP frames and buffer PCM per TN."""
        assert self._state.network is not None
        schf = extract_schf(ndb)
        frame1, frame2, bfi1, bfi2 = decode_speech(schf, self._state.network.scramble_init)
        trackers.record_speech_frame(self._state, bool(bfi1 or bfi2), timestamp)

        tn = self._state.tdma.current_tn
        if tn is None:
            return  # TDMA not anchored yet -- drop silently

        voc = self._vocoders[tn]
        pcm1 = voc.decode_frame(bits_to_bytes(frame1), bfi=bool(bfi1))
        pcm2 = voc.decode_frame(bits_to_bytes(frame2), bfi=bool(bfi2))
        self._pcm_per_tn[tn].append(pcm1)
        self._pcm_per_tn[tn].append(pcm2)
        self._last_speech_ts[tn] = timestamp

    def _select_active_tn(self, now: float) -> int | None:
        """Choose which TN feeds the audio output: newest-wins with hysteresis.

        Picks the TN with the freshest speech burst within
        _SLOT_ACTIVE_WINDOW_SEC. When another TN becomes the freshest, hold
        the previous selection for _SLOT_SWITCH_HOLD_SEC before switching to
        avoid flapping between concurrent calls on different slots.
        """
        fresh = [
            (tn, ts)
            for tn, ts in self._last_speech_ts.items()
            if ts > 0.0 and now - ts <= _SLOT_ACTIVE_WINDOW_SEC
        ]
        if not fresh:
            self._selected_tn = None
            self._candidate_tn = None
            return None

        newest_tn = max(fresh, key=lambda x: x[1])[0]
        fresh_tns = {tn for tn, _ in fresh}

        # First selection or current selection aged out -> snap to newest.
        if self._selected_tn is None or self._selected_tn not in fresh_tns:
            self._selected_tn = newest_tn
            self._candidate_tn = None
            return newest_tn

        # Same slot still winning -> keep it.
        if newest_tn == self._selected_tn:
            self._candidate_tn = None
            return self._selected_tn

        # A different slot is now freshest: start/extend the candidate hold.
        if self._candidate_tn != newest_tn:
            self._candidate_tn = newest_tn
            self._candidate_since = now
            return self._selected_tn
        if now - self._candidate_since >= _SLOT_SWITCH_HOLD_SEC:
            self._selected_tn = newest_tn
            self._candidate_tn = None
        return self._selected_tn

    def _flush_selected_audio(self, timestamp: float) -> None:
        """Package the selected slot's buffered PCM into one AudioBatch."""
        selected = self._select_active_tn(timestamp)

        # Age-out stale buffers on non-selected slots so a once-loud slot
        # doesn't leak old audio into the next time it becomes active.
        for tn, ts in self._last_speech_ts.items():
            if tn != selected and (timestamp - ts) > _SLOT_ACTIVE_WINDOW_SEC:
                self._pcm_per_tn[tn].clear()

        if selected is None:
            return
        chunks = self._pcm_per_tn[selected]
        if not chunks:
            return
        pcm_int16 = np.concatenate(chunks)
        self._pcm_per_tn[selected] = []

        samples = pcm_int16.astype(np.float32) / 32768.0 * 8.0  # +18 dB
        self._audio_batches.append(
            AudioBatch(
                samples=np.clip(samples, -1.0, 1.0),
                sample_rate=_TETRA_PCM_RATE,
                timestamp=timestamp,
                stereo=False,
                prebuffer_seconds=self.audio_prebuffer_seconds,
            )
        )

    # MAC PDU processing

    def _process_mac_pdu(self, type1: np.ndarray, source: str, timestamp: float) -> None:
        """Parse MAC PDU, emit signaling events, let trackers update call state."""
        outcome = parse_mac_pdu(type1)
        if outcome is None:
            return

        if isinstance(outcome, MacFragmentContinue):
            self._fragment_continue(outcome, source, timestamp)
            return
        if isinstance(outcome, MacFragmentEnd):
            self._fragment_end(outcome, source, timestamp)
            return

        # A complete MAC-RESOURCE on this channel: cache its TM-SDU snapshot
        # so a later MAC-FRAG/END can reassemble retrospectively. A second
        # MAC-RESOURCE replaces both the cache entry and any in-flight chain
        # on the same channel (the encoder moved on). BKN1 and BKN2 share a
        # TN but are independent half-slot logical channels, so the source
        # label is part of the key.
        tn = self._state.tdma.current_tn
        if tn is not None and outcome.fragment_start is not None:
            key = (tn, source)
            self._frag_cache[key] = _FragCache(snapshot=outcome.fragment_start, cached_ts=timestamp)
            if key in self._frag_chains:
                logger.debug("tetra_mac_frag_dropped tn=%d source=%s reason=preempted", tn, source)
                del self._frag_chains[key]

        self._emit_mac_result(outcome, timestamp)

    def _emit_mac_result(self, result: MacResult, timestamp: float) -> None:
        summary = result.summary
        if result.cmce is not None and result.cmce.channel_allocation is not None:
            summary = f"{summary} -> {self._format_allocation(result.cmce.channel_allocation)}"

        trackers.record_mac_pdu(self._state, result, timestamp)

        has_signaling = (
            "D-" in summary
            or "SDS" in summary
            or ("SSI=" in summary and "SSI=16777215" not in summary)
        )
        if has_signaling:
            self._emit(summary, timestamp)

    # MAC fragmentation FSM

    def _seed_chain(self, tn: int, source: str, timestamp: float) -> _FragChain | None:
        """Promote a cached MAC-RESOURCE snapshot to an in-flight chain.

        Returns the new chain on success, or `None` if no cache entry exists
        for this channel (orphan FRAG / END). The cache entry is consumed.
        """
        cache = self._frag_cache.pop((tn, source), None)
        if cache is None:
            return None
        chain = _FragChain(
            snapshot=cache.snapshot,
            extra_pieces=[],
            started_ts=cache.cached_ts,
            last_update_ts=timestamp,
            source_label=source,
        )
        self._frag_chains[(tn, source)] = chain
        trackers.record_fragment_started(self._state)
        logger.debug(
            "tetra_mac_frag_start tn=%d source=%s seed_bits=%d",
            tn,
            source,
            len(cache.snapshot.tm_sdu_bits),
        )
        return chain

    def _fragment_continue(self, frag: MacFragmentContinue, source: str, timestamp: float) -> None:
        tn = self._state.tdma.current_tn
        if tn is None:
            return
        chain = self._frag_chains.get((tn, source)) or self._seed_chain(tn, source, timestamp)
        if chain is None:
            logger.debug("tetra_mac_frag_orphan tn=%d source=%s reason=no_resource", tn, source)
            return
        chain.extra_pieces.append(frag.tm_sdu_bits)
        chain.last_update_ts = timestamp
        logger.debug(
            "tetra_mac_frag_cont tn=%d source=%s bits=%d pieces=%d",
            tn,
            source,
            len(frag.tm_sdu_bits),
            1 + len(chain.extra_pieces),
        )

    def _fragment_end(self, frag: MacFragmentEnd, source: str, timestamp: float) -> None:
        tn = self._state.tdma.current_tn
        if tn is None:
            return
        chain = self._frag_chains.pop((tn, source), None)
        if chain is None:
            # Two-PDU chain: MAC-RESOURCE → MAC-END (no MAC-FRAG between).
            chain = self._seed_chain(tn, source, timestamp)
            if chain is None:
                logger.debug(
                    "tetra_mac_frag_orphan_end tn=%d source=%s reason=no_resource",
                    tn,
                    source,
                )
                return
            # The seed promoted into _frag_chains; pop it back out for finalisation.
            self._frag_chains.pop((tn, source), None)
        chain.extra_pieces.append(frag.tm_sdu_bits)
        # MAC-END's allocation IE wins when present; otherwise keep what
        # came in on the originating MAC-RESOURCE.
        allocation = frag.channel_allocation or chain.snapshot.channel_allocation
        self._finalize_fragment_chain(chain, allocation, timestamp)

    def _finalize_fragment_chain(
        self,
        chain: _FragChain,
        channel_allocation: ChannelAllocation | None,
        timestamp: float,
    ) -> None:
        snap = chain.snapshot
        # Strip fill bits from the seed only at finalisation, so the per-burst
        # parse_mac_resource path doesn't pay for a copy that's almost always
        # discarded.
        bits = np.concatenate(
            [strip_fill_bits(snap.tm_sdu_bits, snap.has_fill), *chain.extra_pieces]
        )
        logger.debug(
            "tetra_mac_frag_end source=%s pieces=%d total_bits=%d",
            chain.source_label,
            1 + len(chain.extra_pieces),
            len(bits),
        )
        trackers.record_fragment_completed(self._state)

        if len(bits) < 5:
            return  # Not enough bits to even read an LLC type.

        upper_info, cmce = parse_llc_and_mle(BitReader(bits))

        if cmce is not None and (snap.encryption or channel_allocation is not None):
            cmce = CmceEvent(
                msg_type=cmce.msg_type,
                call_id=cmce.call_id,
                encryption_type=snap.encryption,
                channel_allocation=channel_allocation,
            )

        # Tag reassembled PDUs so the operator can tell at a glance when a
        # signaling event came from a fragment chain vs a single block.
        summary = format_mac_resource_summary(
            "RESOURCE+frag", snap.encryption, snap.ssi, upper_info
        )
        self._emit_mac_result(MacResult(summary=summary, cmce=cmce), timestamp)

    def _expire_fragments(self, timestamp: float) -> None:
        stale_chains = [
            key
            for key, chain in self._frag_chains.items()
            if timestamp - chain.last_update_ts > _FRAG_TIMEOUT_SEC
        ]
        for key in stale_chains:
            tn, source = key
            logger.debug("tetra_mac_frag_expired tn=%d source=%s", tn, source)
            del self._frag_chains[key]
        stale_cache = [
            key
            for key, entry in self._frag_cache.items()
            if timestamp - entry.cached_ts > _FRAG_TIMEOUT_SEC
        ]
        for key in stale_cache:
            del self._frag_cache[key]

    def _format_allocation(self, ca) -> str:
        """Format a ChannelAllocation as 'TS<n> carr#<num> [freq MHz]' for console text."""
        parts = [f"TS{ca.timeslot}", f"carr#{ca.carrier_number}"]
        dl_hz = trackers.resolve_allocation_dl_hz(ca, self._state)
        if dl_hz is not None:
            parts.append(f"{dl_hz / 1e6:.4f}MHz")
        return " ".join(parts)

    # snapshot emission

    def _maybe_emit_snapshot(self, timestamp: float) -> None:
        due = (timestamp - self._last_snapshot_ts) >= _SNAPSHOT_INTERVAL_SEC
        if not (self._state.dirty or due):
            return
        trackers.expire_stale_calls(self._state, timestamp)
        self._expire_fragments(timestamp)
        self._last_snapshot_ts = timestamp
        self._state.dirty = False
        snapshot = self._state.to_snapshot()
        # The TUI widget reads `data`; the text is a one-line summary for the
        # decoder output console (same convention as RDSData).
        self._pending_messages.append(
            DecodedMessage(
                text=self._snapshot_text(snapshot),
                timestamp=timestamp,
                data=snapshot,
            )
        )

    @staticmethod
    def _snapshot_text(snapshot) -> str:
        net = snapshot.network
        if net is None:
            return "TETRA unlocked"
        role = snapshot.carrier_role
        return (
            f"TETRA {net.mcc}/{net.mnc} CC{net.colour_code} {role} "
            f"CRC {snapshot.quality.crc_pct:.0f}%"
        )

    # emit helper

    def _emit(self, text: str, timestamp: float, *, data: object = None) -> None:
        self._pending_messages.append(DecodedMessage(text=text, timestamp=timestamp, data=data))

    # Demodulator API

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread.

        Reads cached scalars (crc_pct, carrier_role inputs) only. Does not
        iterate crc_events / speech_events / slots: those would race with
        the decoder thread's tracker mutations under free-threaded Python.
        """
        quality_label = None
        quality = None
        crc_pct = self._state.quality.crc_pct()
        if crc_pct > 0 or self._state.quality.crc_events:
            quality = crc_pct / 100.0
            quality_label = f"CRC {crc_pct:.0f}%"

        description = self._format_description()

        return SignalInfo(
            label="TETRA",
            channel_bandwidth=25000.0,
            modulation="pi/4-DQPSK",
            sample_rate=self._sample_rate,
            has_audio=True,
            has_text=True,
            message_type="tetra",
            quality_label=quality_label,
            quality=quality,
            description=description,
        )

    def _format_description(self) -> str | None:
        """Compose the compact TunerWidget description line.

        Examples (24 chars max):
            "204/500 [cyan]MCCH[/cyan]"
            "204/5947 [magenta]MCCH![/magenta]"
            "204/5947 [yellow]TCH[/yellow]"
            "204/500"
        """
        net = self._state.network
        if net is None:
            return None
        base = f"{net.mcc}/{net.mnc}"
        role = self._state.carrier_role()
        if role == CARRIER_ROLE_SINGLE:
            return f"{base} [cyan]MCCH[/cyan]"
        if role == CARRIER_ROLE_MULTI:
            return f"{base} [magenta]MCCH![/magenta]"
        if role == CARRIER_ROLE_TCH:
            return f"{base} [yellow]TCH[/yellow]"
        if role == CARRIER_ROLE_UNKNOWN:
            return base
        return base

    def get_messages(self) -> list[DecodedMessage]:
        msgs = self._pending_messages
        self._pending_messages = []
        return msgs

    def get_audio(self) -> list[AudioBatch]:
        batches = self._audio_batches
        self._audio_batches = []
        return batches

    def get_constellation(self) -> tuple[np.ndarray, str] | None:
        points = self._demod.get_constellation()
        if points is None:
            return None
        return points, "π/4-DQPSK"

    def reset(self) -> None:
        self._demod.reset()
        self._sync.reset()
        self._pending_messages.clear()
        tuned = self._state.tuned_hz
        self._state = TetraState()
        self._state.tuned_hz = tuned
        self._last_snapshot_ts = 0.0

        self._vocoders = {tn: TetraAcelpVocoder() for tn in (1, 2, 3, 4)}
        self._pcm_per_tn = {tn: [] for tn in (1, 2, 3, 4)}
        self._last_speech_ts = dict.fromkeys((1, 2, 3, 4), 0.0)
        self._selected_tn = None
        self._candidate_tn = None
        self._candidate_since = 0.0
        self._audio_batches.clear()
        self._frag_chains.clear()
        self._frag_cache.clear()
