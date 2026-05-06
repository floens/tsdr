"""Tests for MAC fragmentation reassembly.

Two layers:

1. Parser-level: `parse_mac_pdu` returns the right outcome type
   (`MacFragmentStart` / `Continue` / `End`) and exposes the right TM-SDU bits.
2. Decoder-level: feeding a sequence of synthetic RESOURCE+FRAG+END PDUs
   into a `TETRADecoder` reassembles the TM-SDU and emits the same signaling
   message that a non-fragmented PDU carrying the same TM-SDU would produce.
"""

from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.tetra.decoder import TETRADecoder
from tsdr.radio.decoders.tetra.mac import (
    MacFragmentContinue,
    MacFragmentEnd,
    MacResult,
    parse_mac_pdu,
)


def _bits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s], dtype=np.uint8)


def _pack(parts: list[tuple[int, int]]) -> str:
    return "".join(f"{val:0{n}b}" for n, val in parts)


def _bits_from_parts(parts: list[tuple[int, int]]) -> np.ndarray:
    return _bits(_pack(parts))


# PDU builders


def _mac_resource_frag_start(
    *,
    ssi: int,
    encryption: int = 0,
    has_fill: bool = False,
    tm_sdu: np.ndarray,
    chan_flag: int = 0,
) -> np.ndarray:
    """Build a MAC-RESOURCE carrying `tm_sdu`.

    The decoder caches the post-flags TM-SDU on every MAC-RESOURCE; the
    cache is promoted to a chain when a MAC-FRAG / MAC-END follows on the
    same channel, so the LI value is irrelevant for fragmentation
    detection. We use LI=0 here (any value works).
    """
    prefix = _bits_from_parts(
        [
            (2, 0b00),  # PDU type RESOURCE
            (1, int(has_fill)),
            (1, 0),  # position of grant
            (2, encryption),
            (1, 0),  # random access flag
            (6, 0),  # length indication (informational only)
            (3, 0b001),  # address type SSI
            (24, ssi),
            (1, 0),  # power_control_flag
            (1, 0),  # slot_granting_flag
            (1, chan_flag),  # chan_flag
        ]
    )
    return np.concatenate([prefix, tm_sdu])


def _mac_frag(*, has_fill: bool = False, tm_sdu: np.ndarray) -> np.ndarray:
    prefix = _bits_from_parts(
        [
            (2, 0b01),  # PDU type FRAG_END
            (1, int(has_fill)),
            (1, 0),  # sub-type 0 = MAC-FRAG
        ]
    )
    return np.concatenate([prefix, tm_sdu])


def _mac_end(
    *,
    has_fill: bool = False,
    tm_sdu: np.ndarray,
    chan_flag: int = 0,
    chan_alloc_bits: np.ndarray | None = None,
) -> np.ndarray:
    """Build a MAC-END PDU. If `chan_flag=1`, `chan_alloc_bits` are inserted."""
    head = _bits_from_parts(
        [
            (2, 0b01),  # PDU type FRAG_END
            (1, int(has_fill)),
            (1, 1),  # sub-type 1 = MAC-END
            (6, 0),  # length indication (informational)
            (1, chan_flag),
        ]
    )
    if chan_flag and chan_alloc_bits is not None:
        return np.concatenate([head, chan_alloc_bits, tm_sdu])
    return np.concatenate([head, tm_sdu])


def _channel_allocation_bits(*, timeslot: int, carrier_number: int) -> np.ndarray:
    """Minimum (non-extended) 21-bit channel allocation IE."""
    return _bits_from_parts(
        [
            (2, 0b00),  # allocation_type
            (2, timeslot - 1),  # timeslot field (0..3 -> TN 1..4)
            (2, 0b00),  # ul_dl_type
            (1, 0),  # clch_permission
            (1, 0),  # cell_change_flag
            (12, carrier_number),
            (1, 0),  # extended flag (0 -> no extension)
            (2, 0b00),  # monitoring_pattern
        ]
    )


def _build_sds_tm_sdu(*, from_ssi: int, protocol: int, text: str) -> np.ndarray:
    """A complete TM-SDU = LLC BL-UDATA + MLE CMCE + D-SDS-DATA(text)."""
    parts: list[tuple[int, int]] = [
        (4, 0b0010),  # LLC BL-UDATA (header is just the type byte)
        (3, 0b010),  # MLE discriminator: CMCE
        (5, 0x0F),  # CMCE: D-SDS-DATA
        (2, 0b01),  # CPTI: SSI addressing
        (24, from_ssi),
        (2, 0b11),  # SDTI: variable-length data with protocol id
        (8, protocol),
    ]
    parts.extend((8, ord(c)) for c in text)
    parts.append((8, 0))  # null terminator
    return _bits_from_parts(parts)


def _build_d_setup_tm_sdu(*, call_id: int) -> np.ndarray:
    """A complete TM-SDU = LLC BL-UDATA + MLE CMCE + D-SETUP(call_id)."""
    return _bits_from_parts(
        [
            (4, 0b0010),  # BL-UDATA
            (3, 0b010),  # CMCE
            (5, 0x07),  # D-SETUP
            (14, call_id),
        ]
    )


# Parser-level tests


def test_parse_mac_resource_carries_fragment_start_snapshot():
    """Every MAC-RESOURCE returns a `MacResult` with a `fragment_start` snapshot.

    The decoder caches the snapshot per logical channel and promotes it to a
    fragmentation chain only when a MAC-FRAG / MAC-END follows. Real-world
    encoders frequently fragment without setting LI=0x3F, so the decoder
    can't rely on an explicit start marker and must cache eagerly.
    """
    payload = _bits_from_parts([(8, 0xAB), (8, 0xCD)])
    pdu = _mac_resource_frag_start(ssi=1019905, tm_sdu=payload)
    out = parse_mac_pdu(pdu)
    assert isinstance(out, MacResult)
    assert out.fragment_start is not None
    assert out.fragment_start.encryption == 0
    assert out.fragment_start.ssi == 1019905
    assert out.fragment_start.channel_allocation is None
    # The post-flags bits are exactly our TM-SDU payload (no fill, no chan alloc).
    assert np.array_equal(out.fragment_start.tm_sdu_bits, payload)


def test_parse_mac_frag_returns_continue():
    payload = _bits_from_parts([(16, 0xBEEF)])
    pdu = _mac_frag(tm_sdu=payload)
    out = parse_mac_pdu(pdu)
    assert isinstance(out, MacFragmentContinue)
    assert np.array_equal(out.tm_sdu_bits, payload)


def test_parse_mac_end_returns_end():
    payload = _bits_from_parts([(16, 0xCAFE)])
    pdu = _mac_end(tm_sdu=payload)
    out = parse_mac_pdu(pdu)
    assert isinstance(out, MacFragmentEnd)
    assert out.channel_allocation is None
    assert np.array_equal(out.tm_sdu_bits, payload)


def test_parse_mac_end_with_channel_allocation():
    payload = _bits_from_parts([(8, 0x42)])
    alloc = _channel_allocation_bits(timeslot=2, carrier_number=1234)
    pdu = _mac_end(tm_sdu=payload, chan_flag=1, chan_alloc_bits=alloc)
    out = parse_mac_pdu(pdu)
    assert isinstance(out, MacFragmentEnd)
    assert out.channel_allocation is not None
    assert out.channel_allocation.timeslot == 2
    assert out.channel_allocation.carrier_number == 1234
    assert np.array_equal(out.tm_sdu_bits, payload)


def test_fill_bit_stripping():
    """Fill bit indication set: trailing `1 0*` is removed before reassembly."""
    real = _bits_from_parts([(8, 0xA5), (4, 0b1010)])  # 12 real bits
    # Fill marker: leading `1` then trailing zeros (here, 5 zeros pad to 18 bits).
    fill = _bits_from_parts([(1, 1), (5, 0)])
    payload = np.concatenate([real, fill])
    pdu = _mac_frag(has_fill=True, tm_sdu=payload)
    out = parse_mac_pdu(pdu)
    assert isinstance(out, MacFragmentContinue)
    assert np.array_equal(out.tm_sdu_bits, real)


def test_self_contained_mac_resource_still_returns_result():
    """A MAC-RESOURCE with LI != 0x3F flows through the existing path."""
    # Reuse the standard CMCE vector: LI=9, returns a MacResult.
    bits = _bits(
        "0010000001001001000011111001000000000001000001001001001000000111"
        "011110010010000111110001000011110111110111111110000001111000"
    )
    out = parse_mac_pdu(bits)
    assert isinstance(out, MacResult)
    assert out.summary == "RESOURCE clear SSI=1019905 BL-UDATA D-TX-CEASED call=239"


# Decoder-level reassembly tests


def _new_anchored_decoder(tn: int = 1) -> TETRADecoder:
    decoder = TETRADecoder(sample_rate=2_048_000)
    # Skip the SB1 path; pin the TDMA cursor directly so fragments key on TN.
    decoder._state.tdma.current_tn = tn
    return decoder


def test_decoder_reassembles_sds_across_three_pdus():
    text = "Hi everyone, this is fragmented."
    tm_sdu = _build_sds_tm_sdu(from_ssi=0x123456, protocol=0x82, text=text)

    # Two splits -> three pieces -> RESOURCE + FRAG + END.
    s1, s2 = 60, 140
    p1, p2, p3 = tm_sdu[:s1], tm_sdu[s1:s2], tm_sdu[s2:]

    decoder = _new_anchored_decoder()
    decoder._process_mac_pdu(_mac_resource_frag_start(ssi=0x123456, tm_sdu=p1), "SCH/F", 0.0)
    decoder._process_mac_pdu(_mac_frag(tm_sdu=p2), "SCH/F", 0.05)
    decoder._process_mac_pdu(_mac_end(tm_sdu=p3), "SCH/F", 0.10)

    msgs = decoder.get_messages()
    sds = [m for m in msgs if f'"{text}"' in m.text and "RESOURCE+frag" in m.text]
    assert sds, f"no reassembled SDS in messages: {[m.text for m in msgs]}"
    assert decoder._state.quality.fragments_started == 1
    assert decoder._state.quality.fragments_completed == 1


def test_decoder_reassembles_two_pdu_chain_resource_to_end():
    """RESOURCE → END (no MAC-FRAG between) is the most common real-world pattern.

    On the captured 393.663 MHz network all 44 MAC-END events arrive
    without an intermediate MAC-FRAG, so this path must work even when the
    cache hasn't been promoted to an in-flight chain by a MAC-FRAG yet.
    """
    text = "two-piece chain"
    tm_sdu = _build_sds_tm_sdu(from_ssi=0x123456, protocol=0x82, text=text)
    s1 = 60
    p1, p2 = tm_sdu[:s1], tm_sdu[s1:]

    decoder = _new_anchored_decoder()
    decoder._process_mac_pdu(_mac_resource_frag_start(ssi=0x123456, tm_sdu=p1), "SCH/F", 0.0)
    decoder._process_mac_pdu(_mac_end(tm_sdu=p2), "SCH/F", 0.05)

    msgs = decoder.get_messages()
    sds = [m for m in msgs if f'"{text}"' in m.text and "RESOURCE+frag" in m.text]
    assert sds, f"no reassembled 2-PDU SDS: {[m.text for m in msgs]}"
    assert decoder._state.quality.fragments_started == 1
    assert decoder._state.quality.fragments_completed == 1


def test_decoder_reassembles_d_setup_with_allocation_in_mac_end():
    """Channel allocation in MAC-END is propagated onto the synthesised CmceEvent."""
    tm_sdu = _build_d_setup_tm_sdu(call_id=4242)
    s1 = 8  # split right after the LLC type byte
    p1, p2 = tm_sdu[:s1], tm_sdu[s1:]

    alloc_bits = _channel_allocation_bits(timeslot=3, carrier_number=999)
    decoder = _new_anchored_decoder()
    decoder._process_mac_pdu(_mac_resource_frag_start(ssi=0xAA0011, tm_sdu=p1), "SCH/F", 0.0)
    decoder._process_mac_pdu(
        _mac_end(tm_sdu=p2, chan_flag=1, chan_alloc_bits=alloc_bits),
        "SCH/F",
        0.05,
    )

    msgs = decoder.get_messages()
    setups = [m for m in msgs if "D-SETUP call=4242" in m.text and "RESOURCE+frag" in m.text]
    assert setups, f"no reassembled D-SETUP in messages: {[m.text for m in msgs]}"
    assert "TS3 carr#999" in setups[0].text
    assert 4242 in decoder._state.active_calls
    call = decoder._state.active_calls[4242]
    assert call.assigned_slot == 3
    assert call.assigned_carrier == 999


def test_decoder_drops_orphan_mac_frag_without_crash():
    decoder = _new_anchored_decoder()
    # MAC-FRAG / MAC-END arriving with no preceding MAC-RESOURCE on this
    # channel: nothing in the cache, silently dropped.
    decoder._process_mac_pdu(_mac_frag(tm_sdu=_bits_from_parts([(8, 0xFF)])), "SCH/F", 0.0)
    decoder._process_mac_pdu(_mac_end(tm_sdu=_bits_from_parts([(8, 0xFF)])), "SCH/F", 0.05)
    assert decoder.get_messages() == []
    assert decoder._state.quality.fragments_started == 0
    assert decoder._state.quality.fragments_completed == 0


def test_decoder_drops_chain_when_preempted_by_new_resource():
    """A new MAC-RESOURCE on the same channel preempts a chain that's already
    in flight (MAC-FRAG arrived but END hasn't yet)."""
    decoder = _new_anchored_decoder()
    # First MAC-RESOURCE seeds the cache.
    tm_sdu = _build_sds_tm_sdu(from_ssi=0x111111, protocol=0x82, text="abandoned")
    decoder._process_mac_pdu(
        _mac_resource_frag_start(ssi=0x111111, tm_sdu=tm_sdu[:40]), "SCH/F", 0.0
    )
    # MAC-FRAG promotes cache to chain.
    decoder._process_mac_pdu(_mac_frag(tm_sdu=tm_sdu[40:80]), "SCH/F", 0.05)
    assert (1, "SCH/F") in decoder._frag_chains

    # A complete MAC-RESOURCE arrives: in-flight chain is preempted, cache
    # gets the new snapshot.
    bits = _bits(
        "0010000001001001000011111001000000000001000001001001001000000111"
        "011110010010000111110001000011110111110111111110000001111000"
    )
    decoder._process_mac_pdu(bits, "SCH/F", 0.10)
    assert (1, "SCH/F") not in decoder._frag_chains
    # The new RESOURCE still produced its own message.
    msgs = decoder.get_messages()
    assert any("D-TX-CEASED call=239" in m.text for m in msgs)


def test_bkn1_and_bkn2_chains_are_independent():
    """A BKN2 burst with a complete RESOURCE must NOT preempt a BKN1 chain.

    Within a single normal_2 burst, BKN1 and BKN2 are two SCH/HD half-slot
    logical channels that happen to share a TN. Each maintains its own
    fragmentation cache; cross-clobbering would lose ~half of all real
    fragmentation activity in practice.
    """
    decoder = _new_anchored_decoder()
    # BKN1 seeds a cache and then promotes to a chain via MAC-FRAG.
    text = "split across two halfslots ok"
    tm_sdu = _build_sds_tm_sdu(from_ssi=0xBEEF01, protocol=0x82, text=text)
    s1, s2 = 60, 120
    decoder._process_mac_pdu(
        _mac_resource_frag_start(ssi=0xBEEF01, tm_sdu=tm_sdu[:s1]), "BKN1", 0.0
    )
    decoder._process_mac_pdu(_mac_frag(tm_sdu=tm_sdu[s1:s2]), "BKN1", 0.0)
    assert (1, "BKN1") in decoder._frag_chains

    # BKN2 of the same burst is a complete (non-fragment) MAC-RESOURCE on
    # the same TN. Must NOT preempt the BKN1 chain.
    bkn2_complete = _bits(
        "0010000001001001000011111001000000000001000001001001001000000111"
        "011110010010000111110001000011110111110111111110000001111000"
    )
    decoder._process_mac_pdu(bkn2_complete, "BKN2", 0.0)
    assert (1, "BKN1") in decoder._frag_chains, "BKN1 chain was incorrectly preempted by BKN2"

    # The BKN1 chain finishes on the next BKN1 burst (next multiframe).
    decoder._process_mac_pdu(_mac_end(tm_sdu=tm_sdu[s2:]), "BKN1", 0.05)

    msgs = decoder.get_messages()
    assert any(f'"{text}"' in m.text and "RESOURCE+frag" in m.text for m in msgs), [
        m.text for m in msgs
    ]
    assert decoder._state.quality.fragments_completed == 1


def test_decoder_expires_stale_caches():
    decoder = _new_anchored_decoder()
    decoder._process_mac_pdu(
        _mac_resource_frag_start(ssi=0x222222, tm_sdu=_bits_from_parts([(8, 0xAA)])),
        "SCH/F",
        0.0,
    )
    assert (1, "SCH/F") in decoder._frag_cache
    # Far past the timeout (2 s): cache should be cleaned out so an
    # unrelated MAC-FRAG arriving much later doesn't accidentally seed a
    # bogus chain.
    decoder._expire_fragments(10.0)
    assert (1, "SCH/F") not in decoder._frag_cache


# Sample-based end-to-end test


# Local-only recording known to contain MAC fragmentation (44 MAC-END PDUs,
# SB2 BNCH frames with LI=0x3F). The file lives outside `tests/samples/`
# because it is too large to commit; the test skips if it is not present.
_FRAG_SAMPLE_PATH = Path("samples/freq=393.663M_sr=2400k_dur=30s_gain=17_20260426T2254.cu8.zst")
_FRAG_SAMPLE_RATE = 2_400_000


def test_real_sample_drives_fragment_reassembly():
    """Run a known-fragmenting capture through the decoder end-to-end.

    Asserts both that fragmentation actually fires (`fragments_completed > 0`)
    and that the FSM accounting stays self-consistent. This is the
    integration counterpart to the synthetic FSM tests above: it catches
    regressions in the parser-FSM seam that synthetic vectors can miss
    (e.g. a real LI value mis-decoded under CRC noise).
    """
    if not _FRAG_SAMPLE_PATH.exists():
        pytest.skip(f"sample not found: {_FRAG_SAMPLE_PATH}")

    iq = load_iq(_FRAG_SAMPLE_PATH)
    decoder = TETRADecoder(_FRAG_SAMPLE_RATE)

    chunk = int(0.5 * _FRAG_SAMPLE_RATE)
    total_msgs = 0
    n_chunks = (len(iq) + chunk - 1) // chunk
    for i in range(n_chunks):
        start = i * chunk
        end = start + chunk
        if end > len(iq):
            break
        decoder.demodulate(iq[start:end], i * 0.5)
        total_msgs += len(decoder.get_messages())

    q = decoder._state.quality
    print(
        f"\nfragments_started={q.fragments_started} "
        f"fragments_completed={q.fragments_completed} "
        f"messages={total_msgs} "
        f"bursts={q.lifetime_bursts}"
    )
    assert q.lifetime_bursts > 0
    assert q.fragments_completed > 0, "expected reassembly to fire on this sample"
    # Started chains never exceed completed by more than the in-flight cap (4 TNs).
    assert q.fragments_started - q.fragments_completed <= 4
