"""DAB Fast Information Group (FIG) parsing.

FIGs carry ensemble metadata, service-to-subchannel mapping, labels, and
user-application info inside FIBs (30 bytes each, after CRC strip). This
module dispatches each FIG to the correct extension parser and accumulates
results in `_FIGParserState`. Once enough FIGs are collected the caller
turns the state into an immutable `DABEnsemble` via `_build_ensemble`.

Spec catalog: see `tables.py`.
Field reader: see `cursor.py` -- per-extension parsers receive a `Cursor`
sliced to the FIG's advertised length, so a wrong-width read raises
`CursorTruncated` and the parent dispatch keeps going with the next FIG
instead of drifting into it.

Reference: ETSI EN 300 401.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .charset import decode_charset
from .constants import FIB_DATA_BYTES
from .cursor import Cursor, CursorTruncated, bits
from .tables import (
    ANNOUNCEMENT_ALARM_BIT,
    ANNOUNCEMENT_TYPES,
    FIG_END_MARKER,
    LABEL_BYTES,
    PROGRAMME_TYPES,
    UEP_TABLE,
    USER_APP_TYPES,
    Fig0Extension,
    Fig1Extension,
    FigType,
    SubchForm,
    TMId,
)

logger = logging.getLogger(__name__)

# Public output types


@dataclass(frozen=True)
class DABService:
    """A single DAB service component (one row in the UI)."""

    service_id: int
    label: str
    subchannel_id: int | None = None
    start_address: int | None = None
    subchannel_size: int | None = None
    protection_level: int | None = None
    eep_option: int = 0  # 0=EEP-A / N/A for UEP, 1=EEP-B
    is_audio: bool = True


@dataclass(frozen=True)
class DABEnsemble:
    """Decoded DAB ensemble metadata."""

    ensemble_id: int = 0
    label: str = ""
    services: tuple[DABService, ...] = ()


# Internal parser state


@dataclass(frozen=True)
class _SubchannelInfo:
    """One row of FIG 0/1 sub-channel organization."""

    start_address: int
    size: int
    protection: int
    eep_option: int  # 0=EEP-A / N/A for UEP, 1=EEP-B


@dataclass(frozen=True)
class _ServiceComponent:
    """One service component referenced from FIG 0/2."""

    subchannel_id: int
    is_audio: bool


@dataclass
class _FIGParserState:
    """Mutable accumulator across FIGs / FIBs."""

    ensemble_id: int = 0
    ensemble_label: str = ""
    service_labels: dict[int, str] = field(default_factory=dict)
    service_components: dict[int, list[_ServiceComponent]] = field(default_factory=dict)
    subchannels: dict[int, _SubchannelInfo] = field(default_factory=dict)
    user_app_types: dict[int, list[int]] = field(default_factory=dict)
    # Dedup keys for "log first occurrence" of FIG events that would otherwise
    # repeat in every FIB. Tuple-keyed; freeform per call site.
    logged: set[tuple] = field(default_factory=set)


def _log_once(state: _FIGParserState, key: tuple, level: int, msg: str, *args: object) -> None:
    """Log a FIG event the first time `key` is seen for this parser state."""
    if key in state.logged:
        return
    state.logged.add(key)
    logger.log(level, msg, *args)


# Top-level FIG dispatch


def _parse_figs(fib_bytes: bytes, state: _FIGParserState) -> None:
    """Parse all FIGs in a 30-byte FIB and update state in place."""
    cur = Cursor(fib_bytes[:FIB_DATA_BYTES])
    while cur.remaining > 0:
        header = cur.u8()
        if header == FIG_END_MARKER:
            return
        fig_length = bits(header, 4, 0)
        if fig_length == 0 or not cur.has(fig_length):
            return
        body = Cursor(cur.bytes(fig_length))
        try:
            fig_type = FigType(bits(header, 7, 5))
        except ValueError:
            continue
        try:
            if fig_type is FigType.MCI_SI:
                _dispatch_fig0(body, state)
            elif fig_type is FigType.LABELS:
                _dispatch_fig1(body, state)
            elif fig_type is FigType.XPAD:
                _dispatch_fig2(body, state)
        except CursorTruncated:
            continue


def _dispatch_fig0(cur: Cursor, state: _FIGParserState) -> None:
    """Route a FIG type 0 to its extension parser."""
    if not cur.has(1):
        return
    header = cur.u8()
    pd = bits(header, 5, 5)
    try:
        extension = Fig0Extension(bits(header, 4, 0))
    except ValueError:
        return
    if extension is Fig0Extension.ENSEMBLE_INFO:
        _parse_fig0_ensemble_info(cur, state)
    elif extension is Fig0Extension.SUBCHANNEL_ORG:
        _parse_fig0_subchannel_org(cur, state)
    elif extension is Fig0Extension.SERVICE_ORG:
        _parse_fig0_service_org(cur, state, pd)
    elif extension is Fig0Extension.SERVICE_COMP_PACKET:
        _parse_fig0_service_comp_packet(cur, state)
    elif extension is Fig0Extension.SERVICE_COMP_GLOBAL:
        _parse_fig0_service_comp_global(cur, state, pd)
    elif extension is Fig0Extension.DATE_TIME:
        _parse_fig0_date_time(cur, state)
    elif extension is Fig0Extension.USER_APP_INFO:
        _parse_fig0_user_app_info(cur, state, pd)
    elif extension is Fig0Extension.PROGRAMME_TYPE:
        _parse_fig0_programme_type(cur, state, pd)
    elif extension is Fig0Extension.ANNOUNCEMENT_SUPPORT:
        _parse_fig0_announcement_support(cur, state)
    elif extension is Fig0Extension.ANNOUNCEMENT_SWITCHING:
        _parse_fig0_announcement_switching(cur, state)
    elif extension is Fig0Extension.FREQ_INFO:
        _parse_fig0_freq_info(cur, state)
    elif extension is Fig0Extension.OE_SERVICES:
        _parse_fig0_oe_services(cur, state, pd)
    elif extension is Fig0Extension.OE_ANNOUNCEMENT_SUPPORT:
        _parse_fig0_oe_announcement_support(cur, state)
    elif extension is Fig0Extension.OE_ANNOUNCEMENT_SWITCHING:
        _parse_fig0_oe_announcement_switching(cur, state)


def _dispatch_fig1(cur: Cursor, state: _FIGParserState) -> None:
    """Route a FIG type 1 to its extension parser."""
    if not cur.has(1):
        return
    header = cur.u8()
    charset = bits(header, 7, 4)
    try:
        extension = Fig1Extension(bits(header, 2, 0))
    except ValueError:
        return
    if extension is Fig1Extension.ENSEMBLE_LABEL:
        _parse_fig1_ensemble_label(cur, state, charset)
    elif extension is Fig1Extension.SERVICE_LABEL:
        _parse_fig1_service_label(cur, state, charset)
    elif extension is Fig1Extension.SERVICE_COMPONENT_LABEL:
        _parse_fig1_service_component_label(cur, state, charset)
    elif extension is Fig1Extension.DATA_SERVICE_LABEL:
        _parse_fig1_data_service_label(cur, state, charset)
    elif extension is Fig1Extension.XPAD_USER_APP_LABEL:
        _parse_fig1_xpad_label(cur, state, charset)


def _dispatch_fig2(cur: Cursor, state: _FIGParserState) -> None:
    """Note presence of FIG 2 extended labels (UTF-8 / UCS-2, multi-segment).

    Full FIG 2 reassembly across segments is involved; for now we just log
    that the ensemble carries them so a future pass can wire it up.
    """
    _log_once(
        state,
        ("fig2",),
        logging.INFO,
        "dab_fig2_extended_labels_present",
    )
    cur.skip(cur.remaining)


# FIG 0/0: Ensemble information (§6.4)


def _parse_fig0_ensemble_info(cur: Cursor, state: _FIGParserState) -> None:
    if not cur.has(4):
        return
    state.ensemble_id = cur.u16()
    flags = cur.u8()
    cur.u8()  # CIF count low byte
    change = bits(flags, 7, 6)  # 00=no change, others=reconfig in next CIF
    al = bits(flags, 5, 5)  # 1=alarm announcement currently active in ensemble
    if change != 0:
        _log_once(
            state,
            ("fig0_0_change", state.ensemble_id, change),
            logging.INFO,
            "dab_fig0_0_reconfig change=%d ensemble=%#06x",
            change,
            state.ensemble_id,
        )
    if al:
        _log_once(
            state,
            ("fig0_0_al", state.ensemble_id),
            logging.WARNING,
            "dab_fig0_0_alarm ensemble=%#06x",
            state.ensemble_id,
        )


# FIG 0/1: Sub-channel organization (§6.2.1)


def _parse_fig0_subchannel_org(cur: Cursor, state: _FIGParserState) -> None:
    while cur.has(3):
        b0 = cur.u8()
        b1 = cur.u8()
        b2 = cur.u8()
        subchannel_id = bits(b0, 7, 2)
        start_address = (bits(b0, 1, 0) << 8) | b1
        form = SubchForm(bits(b2, 7, 7))
        if form is SubchForm.SHORT_UEP:
            # Table switch=1 is reserved; only table 8 is defined.
            if bits(b2, 6, 6) != 0:
                continue
            size, protection, _bitrate = UEP_TABLE[bits(b2, 5, 0)]
            state.subchannels[subchannel_id] = _SubchannelInfo(
                start_address=start_address,
                size=size,
                protection=protection,
                eep_option=0,
            )
        else:
            if not cur.has(1):
                return
            b3 = cur.u8()
            option = bits(b2, 6, 4)
            protection = bits(b2, 3, 2)
            size = (bits(b2, 1, 0) << 8) | b3
            state.subchannels[subchannel_id] = _SubchannelInfo(
                start_address=start_address,
                size=size,
                protection=protection,
                eep_option=option,
            )


# FIG 0/2: Service organization (§6.3.1)


def _parse_fig0_service_org(cur: Cursor, state: _FIGParserState, pd: int) -> None:
    sid_size = 2 if pd == 0 else 4
    while cur.has(sid_size + 1):
        service_id = _read_service_id(cur, pd)
        n_components = bits(cur.u8(), 3, 0)
        components: list[_ServiceComponent] = []
        for _ in range(n_components):
            if not cur.has(2):
                break
            b0 = cur.u8()
            b1 = cur.u8()
            tmid = TMId(bits(b0, 7, 6))
            # tmid 2 (packet) and 3 (FIC) are not handled; bytes still consumed.
            if tmid is TMId.MSC_STREAM_AUDIO:
                components.append(_ServiceComponent(bits(b1, 7, 2), is_audio=True))
            elif tmid is TMId.MSC_STREAM_DATA:
                components.append(_ServiceComponent(bits(b1, 7, 2), is_audio=False))
        if components:
            state.service_components[service_id] = components


# FIG 0/3: Service component in packet mode (§6.3.2)


def _parse_fig0_service_comp_packet(cur: Cursor, state: _FIGParserState) -> None:
    """Identify packet-mode components carried under one Service Component Id.

    The packet-mode subchannel is what carries TPEG, EPG, Journaline, MOT
    Broadcast Web Site, etc. We log first occurrence of each (SCId, SCTy) so
    operators can see "this ensemble carries packet-mode data".
    """
    while cur.has(5):
        b0 = cur.u8()
        b1 = cur.u8()
        scid = (b0 << 4) | bits(b1, 7, 4)  # 12-bit Service Component Id
        b2 = cur.u8()
        scty = bits(b2, 5, 0)  # 6-bit Service component type (DSCTy)
        cur.u8()  # SubChId (6) | Packet address high (2)
        cur.u8()  # Packet address low
        if cur.has(2):
            cur.u16()  # CAOrg (16) -- present when CAflag set; consume to stay aligned
        _log_once(
            state,
            ("fig0_3", scid),
            logging.INFO,
            "dab_fig0_3_packet_mode scid=%#x dscty=%d",
            scid,
            scty,
        )


# FIG 0/8: Service component global definition (§6.3.5)


def _parse_fig0_service_comp_global(cur: Cursor, state: _FIGParserState, pd: int) -> None:
    """Map secondary components (SCIdS) to their service.

    The first component for a service is the primary (audio) one; SCIdS>0
    means a secondary component -- alternate language tracks, music-only
    feeds, director's commentary, packet-mode data riding alongside audio.
    """
    sid_size = 2 if pd == 0 else 4
    while cur.has(sid_size + 2):
        service_id = _read_service_id(cur, pd)
        b = cur.u8()
        ext = bits(b, 7, 7)
        scids = bits(b, 3, 0)
        # 1-byte (LS flag=0) or 2-byte (LS flag=1) trailing component reference.
        ls = cur.u8()
        if bits(ls, 7, 7) == 0:
            pass  # subchannel-id form, already consumed
        elif cur.has(1):
            cur.u8()  # SCId low byte
        if ext and cur.has(1):
            cur.u8()  # Rfa extension byte
        if scids > 0:
            _log_once(
                state,
                ("fig0_8", service_id, scids),
                logging.INFO,
                "dab_fig0_8_secondary_component service=%#x scids=%d",
                service_id,
                scids,
            )


# FIG 0/10: Date and time (§8.1.3.1)


def _parse_fig0_date_time(cur: Cursor, state: _FIGParserState) -> None:
    """Decode the ensemble-broadcast UTC date/time.

    Wire format (short form, UTC flag=0, 4 bytes):
        bit  31: Rfu
        bits 30-14: MJD (17-bit Modified Julian Day)
        bit  13: LSI (leap-second indication)
        bit  12: Rfa
        bit  11: UTC flag
        bits 10-6: hours
        bits  5-0: minutes
    Long form (UTC flag=1) appends 2 more bytes for seconds/milliseconds; we
    capture them when present but only log to the minute.
    """
    if not cur.has(4):
        return
    word = (cur.u8() << 24) | (cur.u8() << 16) | (cur.u8() << 8) | cur.u8()
    mjd = (word >> 14) & 0x1FFFF
    utc_flag = (word >> 11) & 1
    hours = (word >> 6) & 0x1F
    minutes = word & 0x3F
    seconds = 0
    if utc_flag and cur.has(2):
        long_word = (cur.u8() << 8) | cur.u8()
        seconds = (long_word >> 10) & 0x3F
    # MJD -> Gregorian (Hatcher's algorithm, valid for MJD > 15078 = 1900-03-01).
    a = mjd + 2400001 + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + (m // 10)
    _log_once(
        state,
        ("fig0_10", mjd, hours, minutes),
        logging.INFO,
        "dab_fig0_10_utc year=%04d month=%02d day=%02d hour=%02d minute=%02d second=%02d",
        year,
        month,
        day,
        hours,
        minutes,
        seconds,
    )


# FIG 0/13: User application information (§8.1.20)


def _parse_fig0_user_app_info(cur: Cursor, state: _FIGParserState, pd: int) -> None:
    sid_size = 2 if pd == 0 else 4
    while cur.has(sid_size + 1):
        service_id = _read_service_id(cur, pd)
        n_apps = bits(cur.u8(), 3, 0)
        app_types: list[int] = []
        for _ in range(n_apps):
            if not cur.has(2):
                break
            b0 = cur.u8()
            b1 = cur.u8()
            app_type = ((b0 << 3) | (b1 >> 5)) & 0x7FF
            app_data_len = bits(b1, 4, 0)
            app_types.append(app_type)
            if not cur.has(app_data_len):
                break
            cur.skip(app_data_len)
            _log_once(
                state,
                ("fig0_13", service_id, app_type),
                logging.INFO,
                "dab_fig0_13_user_app service=%#x app=%s type=%#05x",
                service_id,
                USER_APP_TYPES.get(app_type, "Unknown"),
                app_type,
            )
        if app_types:
            state.user_app_types[service_id] = app_types


# FIG 0/17: Programme type (§8.1.5)


def _parse_fig0_programme_type(cur: Cursor, state: _FIGParserState, pd: int) -> None:
    """Per service: genre code from the international PTy table."""
    sid_size = 2 if pd == 0 else 4
    while cur.has(sid_size + 2):
        service_id = _read_service_id(cur, pd)
        b = cur.u8()
        sd = bits(b, 6, 6)  # static/dynamic
        # Per spec: this byte holds a Language code (8 bits) when L=1; we
        # simply step past it and read the next one for the PTy. Most
        # broadcasters set L=0 / NFC=0, leaving the code in the next byte.
        l_flag = bits(b, 5, 5)
        if l_flag and cur.has(1):
            cur.u8()  # Language byte
        if not cur.has(1):
            return
        pty = bits(cur.u8(), 4, 0)
        name = PROGRAMME_TYPES[pty] if pty < len(PROGRAMME_TYPES) else f"PTy{pty}"
        _log_once(
            state,
            ("fig0_17", service_id, pty, sd),
            logging.INFO,
            "dab_fig0_17_programme_type service=%#x type=%s code=%d kind=%s",
            service_id,
            name,
            pty,
            "dynamic" if sd else "static",
        )


# FIG 0/18: Announcement support (§8.1.6.1)


def _parse_fig0_announcement_support(cur: Cursor, state: _FIGParserState) -> None:
    """Per service, the set of announcement types it can carry."""
    while cur.has(5):
        service_id = cur.u16()
        asu = cur.u16()  # 16-bit ASu flag field
        n_clusters = bits(cur.u8(), 2, 0)
        if not cur.has(n_clusters):
            return
        cur.skip(n_clusters)  # cluster IDs (8 bits each); not surfaced
        if asu == 0:
            continue
        types = ", ".join(_announcement_flag_names(asu))
        _log_once(
            state,
            ("fig0_18", service_id, asu),
            logging.INFO,
            "dab_fig0_18_announcements service=%#x kinds=%s",
            service_id,
            types,
        )


# FIG 0/19: Announcement switching (§8.1.6.2)


def _parse_fig0_announcement_switching(cur: Cursor, state: _FIGParserState) -> None:
    """An announcement just went live on a cluster.

    Receivers tuned elsewhere (or paused) are supposed to interrupt and
    switch to the announcement subchannel for the duration. The Alarm bit
    (LSB) is the one that overrides user volume / forces playback, so log
    it at warning level.
    """
    while cur.has(4):
        cluster_id = cur.u8()
        asw = cur.u16()  # 16-bit ASw flag field (active types)
        flags = cur.u8()
        new_flag = bits(flags, 7, 7)
        region_flag = bits(flags, 6, 6)
        sub_ch_id = bits(flags, 5, 0)
        if region_flag:
            if not cur.has(1):
                return
            cur.u8()  # Region Id Lower Part
        if asw == 0:
            continue
        types = ", ".join(_announcement_flag_names(asw))
        is_alarm = (asw >> ANNOUNCEMENT_ALARM_BIT) & 1
        level = logging.WARNING if is_alarm else logging.INFO
        # Re-log on every (cluster, asw) transition rather than once-ever:
        # state changes are exactly what an operator wants to see.
        _log_once(
            state,
            ("fig0_19", cluster_id, asw, new_flag, sub_ch_id),
            level,
            "dab_fig0_19_announcement_active cluster=%d subchannel=%d new=%d kind=%s",
            cluster_id,
            sub_ch_id,
            new_flag,
            types,
        )


# FIG 0/21: Frequency information (§8.1.8)


def _parse_fig0_freq_info(cur: Cursor, state: _FIGParserState) -> None:
    """Service-following hints across DAB/FM/DRM/AMSS.

    The full FI list structure is non-trivial; we only need to *notice* it,
    so log first occurrence per ensemble and consume bytes by following the
    advertised lengths.
    """
    _log_once(
        state,
        ("fig0_21", state.ensemble_id),
        logging.INFO,
        "dab_fig0_21_service_following_present",
    )
    # Skip the rest of the FIG body; field-level FI parsing is out of scope.
    cur.skip(cur.remaining)


# FIG 0/24: OE services (§8.1.10.2)


def _parse_fig0_oe_services(cur: Cursor, state: _FIGParserState, pd: int) -> None:
    """Services available in *other* DAB ensembles -- service-following sibling."""
    sid_size = 2 if pd == 0 else 4
    while cur.has(sid_size + 1):
        service_id = _read_service_id(cur, pd)
        b = cur.u8()
        n_eids = bits(b, 3, 0)
        if not cur.has(2 * n_eids):
            return
        eids = [cur.u16() for _ in range(n_eids)]
        _log_once(
            state,
            ("fig0_24", service_id),
            logging.INFO,
            "dab_fig0_24_oe_service service=%#x ensembles=%s",
            service_id,
            ", ".join(f"{e:#06x}" for e in eids),
        )


# FIG 0/25: OE announcement support (§8.1.10.3)


def _parse_fig0_oe_announcement_support(cur: Cursor, state: _FIGParserState) -> None:
    while cur.has(5):
        service_id = cur.u16()
        asu = cur.u16()
        n_eids = bits(cur.u8(), 3, 0)
        if not cur.has(2 * n_eids):
            return
        cur.skip(2 * n_eids)
        if asu == 0:
            continue
        _log_once(
            state,
            ("fig0_25", service_id, asu),
            logging.INFO,
            "dab_fig0_25_oe_announcements service=%#x kinds=%s",
            service_id,
            ", ".join(_announcement_flag_names(asu)),
        )


# FIG 0/26: OE announcement switching (§8.1.10.4)


def _parse_fig0_oe_announcement_switching(cur: Cursor, state: _FIGParserState) -> None:
    while cur.has(7):
        cluster_id_current = cur.u8()
        asw = cur.u16()
        cur.u8()  # New flag | flags
        cur.u16()  # OE EId
        cluster_id_other = cur.u8()
        if asw == 0:
            continue
        is_alarm = (asw >> ANNOUNCEMENT_ALARM_BIT) & 1
        level = logging.WARNING if is_alarm else logging.INFO
        _log_once(
            state,
            ("fig0_26", cluster_id_current, cluster_id_other, asw),
            level,
            "dab_fig0_26_oe_announcement_active old_cluster=%d new_cluster=%d kind=%s",
            cluster_id_current,
            cluster_id_other,
            ", ".join(_announcement_flag_names(asw)),
        )


def _announcement_flag_names(flags: int) -> list[str]:
    return [name for i, name in enumerate(ANNOUNCEMENT_TYPES) if flags & (1 << i)]


# FIG 1/0: Ensemble label (§8.1.13)


def _parse_fig1_ensemble_label(cur: Cursor, state: _FIGParserState, charset: int) -> None:
    if not cur.has(2 + LABEL_BYTES):
        return
    cur.skip(2)  # EId already captured via FIG 0/0
    state.ensemble_label = _decode_label(cur.bytes(LABEL_BYTES), charset)


# FIG 1/1: Service label, 16-bit SId (§8.1.14)


def _parse_fig1_service_label(cur: Cursor, state: _FIGParserState, charset: int) -> None:
    if not cur.has(LABEL_BYTES + 4):
        return
    sid = cur.u16()
    state.service_labels[sid] = _decode_label(cur.bytes(LABEL_BYTES), charset)
    # Trailing 2 bytes (char flag field) ignored.


# FIG 1/4: Service component label (§8.1.14)


def _parse_fig1_service_component_label(cur: Cursor, state: _FIGParserState, charset: int) -> None:
    """Label for a secondary component (alt-language track, music-only feed)."""
    if not cur.has(1):
        return
    pd_byte = cur.u8()
    pd = bits(pd_byte, 7, 7)
    scids = bits(pd_byte, 3, 0)
    sid_size = 2 if pd == 0 else 4
    if not cur.has(sid_size + LABEL_BYTES + 2):
        return
    sid = cur.u16() if pd == 0 else cur.u32()
    label = _decode_label(cur.bytes(LABEL_BYTES), charset)
    cur.u16()  # char-flag field
    _log_once(
        state,
        ("fig1_4", sid, scids),
        logging.INFO,
        "dab_fig1_4_component_label service=%#x scids=%d label=%r",
        sid,
        scids,
        label,
    )


# FIG 1/5: Data service label, 32-bit SId (§8.1.14)


def _parse_fig1_data_service_label(cur: Cursor, state: _FIGParserState, charset: int) -> None:
    if not cur.has(LABEL_BYTES + 6):
        return
    sid = cur.u32()
    label = _decode_label(cur.bytes(LABEL_BYTES), charset)
    cur.u16()  # char-flag field
    _log_once(
        state,
        ("fig1_5", sid),
        logging.INFO,
        "dab_fig1_5_data_service_label service=%#010x label=%r",
        sid,
        label,
    )


# FIG 1/6: X-PAD user-application label (§8.1.14)


def _parse_fig1_xpad_label(cur: Cursor, state: _FIGParserState, charset: int) -> None:
    if not cur.has(1):
        return
    pd_byte = cur.u8()
    pd = bits(pd_byte, 7, 7)
    scids = bits(pd_byte, 3, 0)
    sid_size = 2 if pd == 0 else 4
    if not cur.has(sid_size + 1 + LABEL_BYTES + 2):
        return
    sid = cur.u16() if pd == 0 else cur.u32()
    xpad_app_type = bits(cur.u8(), 4, 0)
    label = _decode_label(cur.bytes(LABEL_BYTES), charset)
    cur.u16()  # char-flag field
    _log_once(
        state,
        ("fig1_6", sid, scids, xpad_app_type),
        logging.INFO,
        "dab_fig1_6_xpad_label service=%#x scids=%d xpad_app=%d label=%r",
        sid,
        scids,
        xpad_app_type,
        label,
    )


# Shared field readers


def _read_service_id(cur: Cursor, pd: int) -> int:
    """16-bit SId (pd=0) or 32-bit SId (pd=1)."""
    return cur.u16() if pd == 0 else cur.u32()


def _decode_label(label_bytes: bytes, charset: int) -> str:
    # Spec-mandated padding is 0x00; broadcasters often use 0x20 (space) too.
    return decode_charset(label_bytes, charset).rstrip(" \x00")


# Build immutable output


def _build_ensemble(state: _FIGParserState) -> DABEnsemble:
    """Build immutable DABEnsemble from parser state."""
    services: list[DABService] = []
    for sid, components in state.service_components.items():
        label = state.service_labels.get(sid, f"Service {sid:#06x}")
        for comp in components:
            sub = state.subchannels.get(comp.subchannel_id)
            services.append(
                DABService(
                    service_id=sid,
                    label=label,
                    subchannel_id=comp.subchannel_id,
                    start_address=sub.start_address if sub else None,
                    subchannel_size=sub.size if sub else None,
                    protection_level=sub.protection if sub else None,
                    eep_option=sub.eep_option if sub else 0,
                    is_audio=comp.is_audio,
                )
            )
            if not comp.is_audio:
                _log_once(
                    state,
                    ("data_service", sid, comp.subchannel_id),
                    logging.INFO,
                    "dab_data_service service=%#x label='%s' subchannel=%d",
                    sid,
                    label,
                    comp.subchannel_id,
                )
    return DABEnsemble(
        ensemble_id=state.ensemble_id,
        label=state.ensemble_label,
        services=tuple(services),
    )
