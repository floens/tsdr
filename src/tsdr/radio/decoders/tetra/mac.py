"""TETRA MAC layer PDU parsing.

Parses decoded type1 bits from SB1, SB2, BBK, and NDB blocks into
network parameters, system information, access assignments, and signaling.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from tsdr.radio.decoders.tetra.bit_reader import BitReader
from tsdr.radio.decoders.tetra.scramble import scramble_init
from tsdr.radio.decoders.tetra.tables import (
    AACH_HEADER_NAMES,
    ADDR_LENGTH_BITS,
    BS_SERVICE_FLAGS,
    DL_USAGE_NAMES,
    DUPLEX_SPACING_KHZ,
    FREQ_OFFSET_HZ,
    LLC_HEADER_BITS,
    AddressType,
    CmcePduType,
    LlcPduType,
    MacPduType,
    MleDiscriminator,
    MmPduType,
)


def _try_enum[E: IntEnum](cls: type[E], val: int) -> E | None:
    """Look up an enum member by value, returning None on unknown values."""
    try:
        return cls(val)
    except ValueError:
        return None


def _spec_name(member: IntEnum) -> str:
    """Return the spec-table display name for an enum member.

    Python identifiers can't carry hyphens, so enum members like
    `LlcPduType.BL_UDATA` and `CmcePduType.D_TX_GRANTED` use underscores.
    The spec (and every log / test vector) spells them with hyphens.
    """
    return member.name.replace("_", "-")


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelAllocation:
    """Parsed channel allocation element (ETSI EN 300 392-2 §21.5.2).

    Describes the traffic channel assigned to a call by D-SETUP/D-CONNECT.
    """

    allocation_type: int  # 2 bits
    timeslot: int  # assigned TN (1-4)
    ul_dl_type: int  # 2 bits
    clch_permission: int  # 1 bit
    cell_change_flag: int  # 1 bit
    carrier_number: int  # 12 bits (main carrier of assigned TCH)
    extended: bool  # extended carrier numbering flag
    freq_band: int | None = None  # 4 bits (only if extended)
    freq_offset: int | None = None  # 2 bits (only if extended)
    duplex_spacing: int | None = None  # 3 bits (only if extended)
    reverse_operation: int | None = None  # 1 bit (only if extended)
    dl_freq_hz: int | None = None  # Computed if extended fields present
    ul_freq_hz: int | None = None  # Computed if extended fields present


@dataclass(frozen=True)
class CmceEvent:
    """Structured CMCE signaling event for call tracking."""

    msg_type: str  # e.g. "D-SETUP", "D-CONNECT", "D-RELEASE"
    call_id: int | None = None
    encryption_type: int = 0  # 2-bit MAC-RESOURCE field: 0=clear, 1..3=TEA1..TEA3
    channel_allocation: ChannelAllocation | None = None


@dataclass(frozen=True)
class MacFragmentStart:
    """Snapshot of a MAC-RESOURCE's post-flags TM-SDU bits and addressing.

    Every parsed MAC-RESOURCE returns one of these alongside its `MacResult`.
    The decoder caches it per logical channel; if a MAC-FRAG / MAC-END
    arrives later on the same channel, the cached snapshot becomes the
    implicit start of a fragmentation chain. Real-world encoders don't
    always set LI=0x3F as the fragmentation marker.

    `has_fill` is the MAC-RESOURCE's fill-bit indication; the decoder
    strips trailing fill bits only when the snapshot is promoted to a
    chain, to avoid an unconditional copy on the per-burst hot path.
    """

    tm_sdu_bits: np.ndarray
    encryption: int
    ssi: int | None
    channel_allocation: ChannelAllocation | None
    has_fill: bool = False


@dataclass(frozen=True)
class MacResult:
    """Result of MAC PDU parsing.

    `fragment_start` is non-None for MAC-RESOURCE PDUs and carries the
    snapshot the decoder needs to reassemble retrospectively.
    """

    summary: str
    cmce: CmceEvent | None = None
    fragment_start: MacFragmentStart | None = None


@dataclass(frozen=True)
class MacFragmentContinue:
    """Continuation of an in-flight TM-SDU (MAC-FRAG, sub-type=0)."""

    tm_sdu_bits: np.ndarray


@dataclass(frozen=True)
class MacFragmentEnd:
    """Final fragment of a TM-SDU (MAC-END, sub-type=1).

    A MAC-END can carry its own channel-allocation IE; when the originating
    MAC-RESOURCE did not, this is the only place an allocation arrives.
    The decoder prefers MAC-END's allocation when both are present.
    """

    tm_sdu_bits: np.ndarray
    channel_allocation: ChannelAllocation | None


# Outcomes produced by `parse_mac_pdu`. A `MacResult` from a MAC-RESOURCE
# always carries the per-channel `fragment_start` snapshot the decoder
# caches for retrospective reassembly when MAC-FRAG/END follows.
type MacOutcome = MacResult | MacFragmentContinue | MacFragmentEnd


def strip_fill_bits(bits: np.ndarray, has_fill: bool) -> np.ndarray:
    """Drop a trailing fill marker (`1` followed by zero or more `0`s).

    ETSI §23.4.2.4. Without stripping, fill zeros from fragment N would
    bleed into the LLC parse of fragment N+1 once concatenated.
    """
    if not has_fill or len(bits) == 0:
        return bits
    end = len(bits) - 1
    while end >= 0 and bits[end] == 0:
        end -= 1
    # No `1` marker: malformed or already stripped -- leave the bits alone
    # rather than chopping the PDU to zero length.
    if end < 0 or bits[end] != 1:
        return bits
    return bits[:end]


# SB1 (BSCH)


@dataclass(frozen=True)
class SB1Info:
    """Parsed SB1 (BSCH) information."""

    system_code: int
    colour_code: int
    timeslot: int
    frame_number: int
    multiframe_number: int
    sharing_mode: int
    ts_reserved: int
    u_plane_dtx: int
    frame18_ext: int
    mcc: int
    mnc: int
    scramble_init: int


def parse_sb1(type1: np.ndarray) -> SB1Info:
    """Parse SB1 type1 bits (60 bits). ETSI EN 300 392-2 §21.4.4.1."""
    r = BitReader(type1)
    system_code = r.u(4)
    colour_code = r.u(6)
    timeslot = r.u(2)
    frame_number = r.u(5)
    multiframe_number = r.u(6)
    sharing_mode = r.u(2)
    ts_reserved = r.u(2)
    u_plane_dtx = r.u(2)
    frame18_ext = r.u(2)
    mcc = r.u(10)
    mnc = r.u(14)
    return SB1Info(
        system_code=system_code,
        colour_code=colour_code,
        timeslot=timeslot,
        frame_number=frame_number,
        multiframe_number=multiframe_number,
        sharing_mode=sharing_mode,
        ts_reserved=ts_reserved,
        u_plane_dtx=u_plane_dtx,
        frame18_ext=frame18_ext,
        mcc=mcc,
        mnc=mnc,
        scramble_init=scramble_init(mcc, mnc, colour_code),
    )


# AACH (Access Assignment Channel, from BBK)


@dataclass(frozen=True)
class AACHInfo:
    """Parsed AACH (Access Assignment Channel)."""

    header: int  # 2 bits
    field1: int  # 6 bits
    field2: int  # 6 bits


def parse_aach(info_bits: np.ndarray) -> AACHInfo:
    """Parse 14 AACH info bits from BBK RM(30,14) decode. ETSI §21.4.7."""
    r = BitReader(info_bits)
    return AACHInfo(header=r.u(2), field1=r.u(6), field2=r.u(6))


def format_aach(aach: AACHInfo) -> str:
    """Format AACH for logging."""
    hdr_name = AACH_HEADER_NAMES.get(aach.header, f"?{aach.header}")
    # DL usage from field1 when header indicates it
    if aach.header >= 1:
        dl_name = DL_USAGE_NAMES.get(aach.field1 & 0x3, f"traffic({aach.field1})")
        if aach.field1 > 3:
            dl_name = f"traffic({aach.field1})"
        return f"AACH {hdr_name} DL={dl_name} F2={aach.field2}"
    return f"AACH {hdr_name} F1={aach.field1} F2={aach.field2}"


# SYSINFO (from BROADCAST PDU in SB2)


def format_services(bs_service_details: int) -> tuple[str, ...]:
    """Return the list of service flag names set in SYSINFO.bs_service_details."""
    return tuple(name for bit, name in BS_SERVICE_FLAGS if (bs_service_details >> bit) & 1)


def carrier_to_freq(
    carrier_number: int,
    freq_band: int,
    freq_offset: int,
    duplex_spacing: int,
    reverse_operation: int,
) -> tuple[int, int]:
    """Compute (dl_freq_hz, ul_freq_hz) from TETRA carrier numbering fields.

    ETSI EN 300 392-2 §21.4.4:
        DL = freq_band * 100 MHz + carrier_number * 25 kHz + offset
    ETSI EN 300 392-2 §21.4.4.
    """
    offset_hz = FREQ_OFFSET_HZ.get(freq_offset, 0)
    dl = (freq_band & 0xF) * 100_000_000 + carrier_number * 25000 + offset_hz
    spacing_hz = DUPLEX_SPACING_KHZ[duplex_spacing & 7][freq_band & 0xF] * 1000
    if reverse_operation:
        ul = dl + spacing_hz
    else:
        ul = dl - spacing_hz
    return dl, ul


def parse_channel_allocation(r: BitReader) -> ChannelAllocation | None:
    """Parse channel allocation IE from the current cursor position.

    Advances the reader past the entire allocation element. Returns None
    (and leaves the cursor where it was) if the buffer runs out.

    ETSI EN 300 392-2 §21.5.2 / Table 21.42. Fields are read in spec
    table order; the 2-bit monitoring_pattern trailer is skipped, not
    parsed, because the frame18_monitoring_pattern follow-up is only
    present when frame==18 and callers already length-guard the TM-SDU.
    """
    # Minimum 21 bits, 31 bits if extended carrier numbering
    if r.remaining < 21:
        return None

    start = r.pos
    allocation_type = r.u(2)
    timeslot_field = r.u(2)  # 0..3 -- TN 1..4
    ul_dl_type = r.u(2)
    clch_permission = r.u(1)
    cell_change_flag = r.u(1)
    carrier_number = r.u(12)
    extended_flag = r.u(1)

    freq_band: int | None = None
    freq_offset: int | None = None
    duplex_spacing: int | None = None
    reverse_operation: int | None = None
    dl_freq_hz: int | None = None
    ul_freq_hz: int | None = None

    if extended_flag:
        if r.remaining < 10:
            r.pos = start
            return None
        freq_band = r.u(4)
        freq_offset = r.u(2)
        duplex_spacing = r.u(3)
        reverse_operation = r.u(1)
        dl_freq_hz, ul_freq_hz = carrier_to_freq(
            carrier_number, freq_band, freq_offset, duplex_spacing, reverse_operation
        )

    r.skip(2)  # monitoring_pattern (not parsed)

    return ChannelAllocation(
        allocation_type=allocation_type,
        timeslot=timeslot_field + 1,
        ul_dl_type=ul_dl_type,
        clch_permission=clch_permission,
        cell_change_flag=cell_change_flag,
        carrier_number=carrier_number,
        extended=bool(extended_flag),
        freq_band=freq_band,
        freq_offset=freq_offset,
        duplex_spacing=duplex_spacing,
        reverse_operation=reverse_operation,
        dl_freq_hz=dl_freq_hz,
        ul_freq_hz=ul_freq_hz,
    )


@dataclass(frozen=True)
class SysInfo:
    """Parsed SYSINFO from BROADCAST PDU."""

    main_carrier: int
    freq_band: int
    freq_offset: int
    duplex_spacing: int
    reverse_operation: int
    num_csch: int
    ms_txpwr_max: int
    rxlev_access_min: int
    access_parameter: int
    radio_dl_timeout: int
    cck_valid: int
    cck_id_or_hyper_frame: int
    location_area: int
    subscriber_class: int
    bs_service_details: int
    dl_freq_hz: int
    ul_freq_hz: int


def parse_sysinfo(type1: np.ndarray) -> SysInfo | None:
    """Parse SYSINFO from SB2/BNCH type1 bits (124 bits).

    SYSINFO is carried as a BROADCAST MAC PDU (type 0b10). ETSI EN 300 392-2
    §21.4.4.1 Table 21.26. Fields are read in spec-table order.
    """
    r = BitReader(type1)
    pdu_type = _try_enum(MacPduType, r.u(2))
    if pdu_type != MacPduType.BROADCAST:
        logger.debug("SB2 MAC PDU type %s (not BROADCAST)", pdu_type)
        return None
    r.skip(2)  # broadcast sub-type

    main_carrier = r.u(12)
    freq_band = r.u(4)
    freq_offset = r.u(2)
    duplex_spacing = r.u(3)
    reverse_op = r.u(1)
    num_csch = r.u(2)
    ms_txpwr = r.u(3)
    rxlev_min = r.u(4)
    access_param = r.u(4)
    radio_dl_to = r.u(4)
    cck_valid = r.u(1)
    cck_or_hf = r.u(16)
    r.skip(22)  # bits 60-81: optional elements (not used)
    location_area = r.u(14)
    subscriber_class = r.u(16)
    bs_service = r.u(12)

    dl_freq, ul_freq = carrier_to_freq(
        main_carrier, freq_band, freq_offset, duplex_spacing, reverse_op
    )
    return SysInfo(
        main_carrier=main_carrier,
        freq_band=freq_band,
        freq_offset=freq_offset,
        duplex_spacing=duplex_spacing,
        reverse_operation=reverse_op,
        num_csch=num_csch,
        ms_txpwr_max=ms_txpwr,
        rxlev_access_min=rxlev_min,
        access_parameter=access_param,
        radio_dl_timeout=radio_dl_to,
        cck_valid=cck_valid,
        cck_id_or_hyper_frame=cck_or_hf,
        location_area=location_area,
        subscriber_class=subscriber_class,
        bs_service_details=bs_service,
        dl_freq_hz=dl_freq,
        ul_freq_hz=ul_freq,
    )


def format_sysinfo(si: SysInfo) -> str:
    """Format SYSINFO for display."""
    services = format_services(si.bs_service_details)
    svc_str = ",".join(services) if services else "none"

    return (
        f"DL={si.dl_freq_hz / 1e6:.5f}MHz UL={si.ul_freq_hz / 1e6:.5f}MHz "
        f"LA={si.location_area} services=[{svc_str}]"
    )


# MAC PDU parsing (from NDB/SCH_F type1 bits)


def _read_address_ssi(r: BitReader, addr_type: AddressType) -> int | None:
    """Read a MAC-RESOURCE address field, returning the SSI if one is present.

    Advances the cursor past the full address length per `ADDR_LENGTH_BITS`.
    An SSI is only returned for address types that start with a 24-bit SSI
    (SSI, USSI, SMI, SSI_EVENT, SSI_USAGE). EVENT_LABEL and SMI_EVENT just
    consume their bits and return None. Returns None without advancing if
    the buffer runs short.
    """
    length = ADDR_LENGTH_BITS[addr_type]
    if length == 0 or r.remaining < length:
        return None
    carries_ssi = addr_type in (
        AddressType.SSI,
        AddressType.USSI,
        AddressType.SMI,
        AddressType.SSI_EVENT,
        AddressType.SSI_USAGE,
    )
    ssi = r.peek(24) if carries_ssi else None
    r.skip(length)
    return ssi


def parse_mac_resource(type1: np.ndarray) -> MacResult | None:
    """Parse MAC-RESOURCE PDU. ETSI EN 300 392-2 §21.4.3.1 / Table 21.4.

    Returns a `MacResult` that always carries a `MacFragmentStart` snapshot of
    the post-flags TM-SDU bits (`fragment_start` field). The decoder caches
    that snapshot per logical channel; when MAC-FRAG / MAC-END arrives later
    on the same channel, the cached snapshot becomes the implicit start of a
    fragmentation chain. This handles encoders that don't use LI=0x3F as the
    explicit fragmentation marker -- in real captures, MAC-FRAG/END routinely
    arrives without a preceding LI=0x3F. Fields are read in spec-table order
    via `BitReader`.
    """
    r = BitReader(type1)
    if r.remaining < 16:
        return None

    r.skip(2)  # PDU type (already matched by caller)
    has_fill = bool(r.u(1))  # fill bit indication
    r.skip(1)  # position of grant
    encrypted = r.u(2)
    r.skip(1)  # random access flag
    r.skip(6)  # length indication (informational; we always cache for retro reassembly)
    addr_type = _try_enum(AddressType, r.u(3))
    if addr_type is None:
        return None

    ssi = _read_address_ssi(r, addr_type)

    channel_allocation: ChannelAllocation | None = None

    # Optional flags block (§21.4.3.1): power control, slot granting,
    # channel allocation. Each is a 1-bit presence flag followed by the
    # fixed-size element body when set. Then the TM-SDU starts.
    if r.remaining >= 3:
        if r.u(1) and r.remaining >= 4:  # power_control_flag
            r.skip(4)
        if r.u(1) and r.remaining >= 8:  # slot_granting_flag
            r.skip(8)
        chan_flag = r.u(1)
        # Channel allocation element is opaque when encryption is on.
        if chan_flag and not encrypted:
            channel_allocation = parse_channel_allocation(r)

    # Snapshot the post-flags TM-SDU bits as a view (no copy) so the
    # decoder can reassemble retrospectively if a MAC-FRAG / MAC-END
    # follows. Fill-bit stripping is deferred until promotion to a chain.
    fragment_start = MacFragmentStart(
        tm_sdu_bits=r.bits[r.pos :],
        encryption=encrypted,
        ssi=ssi,
        channel_allocation=channel_allocation,
        has_fill=has_fill,
    )

    upper_info = ""
    cmce: CmceEvent | None = None
    if r.remaining >= 5:
        upper_info, cmce = parse_llc_and_mle(r)

    # Propagate encryption state and channel allocation onto any CMCE event.
    if cmce and (encrypted or channel_allocation is not None):
        cmce = CmceEvent(
            msg_type=cmce.msg_type,
            call_id=cmce.call_id,
            encryption_type=encrypted,
            channel_allocation=channel_allocation,
        )

    summary = format_mac_resource_summary("RESOURCE", encrypted, ssi, upper_info)
    return MacResult(summary=summary, cmce=cmce, fragment_start=fragment_start)


def format_mac_resource_summary(
    prefix: str, encryption: int, ssi: int | None, upper_info: str
) -> str:
    """Format the human-readable summary for a (possibly reassembled) MAC-RESOURCE."""
    enc_str = f"TEA{encryption}" if encryption else "clear"
    parts = [f"{prefix} {enc_str}"]
    if ssi is not None:
        parts.append(f"SSI={ssi}")
    if upper_info:
        parts.append(upper_info)
    return " ".join(parts)


def parse_mac_frag(type1: np.ndarray) -> MacFragmentContinue | None:
    """Parse a MAC-FRAG PDU (downlink, sub-type=0). ETSI Table 21.13.

    Header is 4 bits: PDU type (2) + fill bit indication (1) + sub-type (1).
    The remainder is TM-SDU continuation. Fill bits are stripped if the
    indication is set so concatenation across fragments doesn't pick up
    bogus zeros between pieces.
    """
    r = BitReader(type1)
    if r.remaining < 4:
        return None
    r.skip(2)  # PDU type (already matched by caller)
    has_fill = bool(r.u(1))
    r.skip(1)  # sub-type (already matched as 0)
    return MacFragmentContinue(tm_sdu_bits=strip_fill_bits(r.bits[r.pos :], has_fill))


def parse_mac_end(type1: np.ndarray) -> MacFragmentEnd | None:
    """Parse a MAC-END PDU (downlink, sub-type=1). ETSI Table 21.14.

    Header is 4 bits + 6-bit length indication + 1-bit channel allocation
    flag (+ optional channel allocation IE). Then TM-SDU final fragment.
    Like MAC-FRAG, trailing fill bits are stripped.
    """
    r = BitReader(type1)
    if r.remaining < 11:
        return None
    r.skip(2)  # PDU type
    has_fill = bool(r.u(1))
    r.skip(1)  # sub-type (already matched as 1)
    r.skip(6)  # length indication (informational; fill-bit stripping bounds the TM-SDU)
    chan_flag = r.u(1)

    channel_allocation: ChannelAllocation | None = None
    if chan_flag:
        channel_allocation = parse_channel_allocation(r)

    return MacFragmentEnd(
        tm_sdu_bits=strip_fill_bits(r.bits[r.pos :], has_fill),
        channel_allocation=channel_allocation,
    )


def parse_llc_and_mle(r: BitReader) -> tuple[str, CmceEvent | None]:
    """Parse TM-SDU: LLC PDU header, then MLE discriminator + upper-layer.

    The TM-SDU inside a MAC-RESOURCE is an LLC PDU, not the MLE PDU
    directly. The LLC header (variable length by PDU type -- see
    `LLC_HEADER_BITS`) must be consumed first; what remains is the
    TL-SDU == MLE PDU, which starts with a 3-bit MLE protocol
    discriminator.
    """
    if r.remaining < 4:
        return "", None
    llc_type = _try_enum(LlcPduType, r.u(4))
    if llc_type is None:
        return "", None
    # LLC_HEADER_BITS covers the full header including the 4-bit type
    # we just read, so advance by the remainder. FCS variants also carry
    # a 32-bit trailer at the end, but we only read from the start of the
    # TL-SDU so we don't need to strip it.
    header_remainder = LLC_HEADER_BITS[llc_type] - 4
    if r.remaining < header_remainder:
        return llc_type.name, None
    r.skip(header_remainder)

    llc_name = _spec_name(llc_type)
    if r.remaining < 3:
        return llc_name, None
    disc = _try_enum(MleDiscriminator, r.u(3))
    if disc == MleDiscriminator.CMCE:
        cmce_summary, cmce_event = _parse_cmce(r)
        return f"{llc_name} {cmce_summary}", cmce_event
    if disc == MleDiscriminator.MM:
        return f"{llc_name} {_parse_mm(r)}", None
    if disc is None:
        return llc_name, None
    return f"{llc_name} {_spec_name(disc)}", None


def _parse_cmce(r: BitReader) -> tuple[str, CmceEvent | None]:
    """Parse a CMCE PDU starting at the reader's current position."""
    if r.remaining < 5:
        return "CMCE", None
    pdu = _try_enum(CmcePduType, r.u(5))
    if pdu is None:
        return "CMCE-unknown", None
    name = _spec_name(pdu)

    # D-SETUP / D-CONNECT / D-RELEASE: 14-bit call identifier directly
    # after the PDU type. ETSI §14.8.{20,8,17}.
    if pdu in (CmcePduType.D_SETUP, CmcePduType.D_CONNECT, CmcePduType.D_RELEASE):
        if r.remaining < 14:
            return name, None
        call_id = r.u(14)
        return f"{name} call={call_id}", CmceEvent(msg_type=name, call_id=call_id)

    # D-TX-GRANTED / D-TX-CEASED: also begin with a 14-bit call identifier
    # directly after the PDU type. ETSI §14.8.32 / §14.8.30.
    if pdu in (CmcePduType.D_TX_GRANTED, CmcePduType.D_TX_CEASED):
        if r.remaining < 14:
            return name, None
        call_id = r.u(14)
        return f"{name} call={call_id}", CmceEvent(msg_type=name, call_id=call_id)

    # D-SDS-DATA has its own sub-parser.
    if pdu == CmcePduType.D_SDS_DATA:
        return _parse_sds_data(r), None

    # Any other known D-* PDU: emit the event with no call_id so upstream
    # signaling loggers still see it.
    return name, CmceEvent(msg_type=name)


def _parse_mm(r: BitReader) -> str:
    """Parse the MM PDU type byte. ETSI EN 300 392-2 §16.9, Table 16.1.

    Only the 4-bit PDU type is decoded -- the body (cipher params, location
    area, group identities, ...) is left opaque. The goal is just to make
    registration / location-update / authentication activity visible in the
    decoder console; full body parsing is a follow-up.
    """
    if r.remaining < 4:
        return "MM"
    pdu = _try_enum(MmPduType, r.u(4))
    if pdu is None:
        return "MM-unknown"
    return _spec_name(pdu)


def _parse_sds_data(r: BitReader) -> str:
    """Parse D-SDS-DATA PDU for text messages / short data."""
    if r.remaining < 2:
        return "SDS"

    cpti = r.u(2)
    ssi: int | None = None
    if cpti == 0:  # SNA (8 bits)
        if r.remaining < 8:
            return "SDS"
        r.skip(8)
    elif cpti == 1:  # SSI (24 bits)
        if r.remaining < 24:
            return "SDS"
        ssi = r.u(24)
    elif cpti == 2:  # TSI (48 bits; first 24 = SSI)
        if r.remaining < 48:
            return "SDS"
        ssi = r.u(24)
        r.skip(24)
    else:
        if r.remaining < 24:
            return "SDS"
        r.skip(24)

    prefix = f"SDS from={ssi}" if ssi else "SDS"
    if r.remaining < 2:
        return prefix

    sdti = r.u(2)
    if sdti == 3 and r.remaining >= 8:
        protocol_id = r.u(8)
        if protocol_id in (0x02, 0x82):
            text = _decode_sds_text(r)
            if text:
                return f'{prefix} "{text}"'
        payload = _read_hex(r, r.remaining)
        if payload:
            return f"{prefix} proto=0x{protocol_id:02X} data=0x{payload}"
        return f"{prefix} proto=0x{protocol_id:02X}"

    data_lens = {0: 16, 1: 32, 2: 64}
    data_len = data_lens.get(sdti, 0)
    if data_len and r.remaining >= data_len:
        return f"{prefix} data=0x{_read_hex(r, data_len)}"
    return prefix


def _read_hex(r: BitReader, n: int) -> str:
    """Read `n` bits and return them as upper-case hex.

    Reads in 32-bit chunks so 64-bit SDS payloads (and longer SDTI=3
    protocol bodies) survive without being silently truncated. The final
    chunk is zero-padded to ceil(bits/4) hex digits so leading zeros are
    preserved -- a 16-bit `0x0042` reads as `0042`, not `42`.
    """
    out: list[str] = []
    while n > 0:
        take = min(n, 32)
        out.append(f"{r.u(take):0{(take + 3) // 4}X}")
        n -= take
    return "".join(out)


def _decode_sds_text(r: BitReader) -> str:
    """Best-effort 8-bit ISO-8859-1 text decode from a bit stream."""
    chars: list[str] = []
    while r.remaining >= 8:
        c = r.u(8)
        if c == 0:
            break
        if 32 <= c < 127:
            chars.append(chr(c))
        else:
            chars.append("?")
    return "".join(chars)


def parse_mac_pdu(type1: np.ndarray) -> MacOutcome | None:
    """Parse a MAC PDU from NDB or SB2 type1 bits.

    Returns one of:
    * `MacResult` for a self-contained PDU (BROADCAST, SUPPLEMENT, or
      MAC-RESOURCE whose TM-SDU fits in this block);
    * `MacFragmentStart` for a MAC-RESOURCE whose LI signals fragmentation;
    * `MacFragmentContinue` for MAC-FRAG;
    * `MacFragmentEnd` for MAC-END;
    * `None` if the buffer is too short or the PDU type is unrecognised.

    Decoder is responsible for stitching fragments together and re-running
    the LLC/MLE parse on the assembled TM-SDU.
    """
    if len(type1) < 4:
        return None

    pdu = _try_enum(MacPduType, BitReader(type1).u(2))
    if pdu == MacPduType.RESOURCE:
        return parse_mac_resource(type1)
    if pdu == MacPduType.BROADCAST:
        si = parse_sysinfo(type1)
        if si:
            return MacResult(summary=f"BROADCAST {format_sysinfo(si)}")
        return MacResult(summary="BROADCAST (parse failed)")
    if pdu == MacPduType.FRAG_END:
        # Bit 3: sub-type (0=frag, 1=end). The fill-bit indication is at bit 2.
        sub = int(type1[3])
        if sub:
            return parse_mac_end(type1)
        return parse_mac_frag(type1)
    if pdu == MacPduType.SUPPLEMENT:
        return MacResult(summary="MAC-SUPPLEMENT")
    return None
