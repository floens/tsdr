"""Catalog of TETRA protocol elements the mac parser supports.

Every enum / mapping here corresponds to a table in ETSI EN 300 392-2.
If a wire value doesn't appear here, the parser doesn't handle it. Adding
support means extending this file first, parser second.

Manual alignment is load-bearing for readability here -- the whole point
of the file is to look like a printed spec table -- so the aligned blocks
are wrapped in `# fmt: off` / `# fmt: on` to keep ruff-format away.
"""

from collections.abc import Mapping
from enum import IntEnum

__all__ = [
    # MAC layer (§21.4)
    "MacPduType",
    "AddressType",
    "ADDR_LENGTH_BITS",
    # LLC layer (§23)
    "LlcPduType",
    "LLC_HEADER_BITS",
    # MLE layer (§18)
    "MleDiscriminator",
    # CMCE (§14)
    "CmcePduType",
    # MM (§16)
    "MmPduType",
    # AACH (§21.4.7)
    "AACH_HEADER_NAMES",
    "DL_USAGE_NAMES",
    # SYSINFO (§21.4.4)
    "BS_SERVICE_FLAGS",
    # Carrier numbering (§21.4.4)
    "FREQ_OFFSET_HZ",
    "DUPLEX_SPACING_KHZ",
]


# MAC layer


# fmt: off
class MacPduType(IntEnum):
    """2-bit MAC PDU type (ETSI EN 300 392-2 §21.4.3, Table 21.1)."""
    RESOURCE   = 0
    FRAG_END   = 1
    BROADCAST  = 2
    SUPPLEMENT = 3


class AddressType(IntEnum):
    """3-bit MAC address type (ETSI EN 300 392-2 §21.4.3.3, Table 21.7)."""
    NULL        = 0
    SSI         = 1
    EVENT_LABEL = 2
    USSI        = 3
    SMI         = 4
    SSI_EVENT   = 5
    SSI_USAGE   = 6
    SMI_EVENT   = 7


ADDR_LENGTH_BITS: Mapping[AddressType, int] = {
    AddressType.NULL:        0,
    AddressType.SSI:         24,
    AddressType.EVENT_LABEL: 10,
    AddressType.USSI:        24,
    AddressType.SMI:         24,
    AddressType.SSI_EVENT:   34,  # SSI(24) + event label(10)
    AddressType.SSI_USAGE:   30,  # SSI(24) + usage marker(6)
    AddressType.SMI_EVENT:   34,  # SMI(24) + event label(10)
}
# fmt: on


# LLC layer


# fmt: off
class LlcPduType(IntEnum):
    """4-bit LLC PDU type (ETSI EN 300 392-2 §23, Table 23.1).

    AL-* types (8-15) are not decoded -- they carry acknowledged /
    segmented upper-layer data that the current decoder doesn't need.
    """
    BL_ADATA     = 0b0000
    BL_DATA      = 0b0001
    BL_UDATA     = 0b0010
    BL_ACK       = 0b0011
    BL_ADATA_FCS = 0b0100
    BL_DATA_FCS  = 0b0101
    BL_UDATA_FCS = 0b0110
    BL_ACK_FCS   = 0b0111


# 4-bit type + optional N(R)/N(S). FCS variants share the header with their
# non-FCS cousin and append a 32-bit FCS at the tail (not stripped here --
# parsers only read from the start of the TL-SDU).
LLC_HEADER_BITS: Mapping[LlcPduType, int] = {
    LlcPduType.BL_ADATA:     4 + 1 + 1,  # type + N(R) + N(S)
    LlcPduType.BL_DATA:      4 + 1,      # type + N(S)
    LlcPduType.BL_UDATA:     4,          # type only
    LlcPduType.BL_ACK:       4 + 1,      # type + N(R)
    LlcPduType.BL_ADATA_FCS: 4 + 1 + 1,
    LlcPduType.BL_DATA_FCS:  4 + 1,
    LlcPduType.BL_UDATA_FCS: 4,
    LlcPduType.BL_ACK_FCS:   4 + 1,
}
# fmt: on


# MLE layer


# fmt: off
class MleDiscriminator(IntEnum):
    """3-bit MLE protocol discriminator (ETSI EN 300 392-2 §18.5.19)."""
    RESERVED_0 = 0
    MM         = 1
    CMCE       = 2
    RESERVED_3 = 3
    SNDCP      = 4
    MLE        = 5
    # 6, 7 reserved
# fmt: on


# CMCE


# fmt: off
class CmcePduType(IntEnum):
    """5-bit downlink CMCE PDU type (ETSI EN 300 392-2 §14.7, Table 14.1)."""
    D_ALERT           = 0x00
    D_CALL_PROCEEDING = 0x01
    D_CONNECT         = 0x02
    D_CONNECT_ACK     = 0x03
    D_DISCONNECT      = 0x04
    D_INFO            = 0x05
    D_RELEASE         = 0x06
    D_SETUP           = 0x07
    D_STATUS          = 0x08
    D_TX_CEASED       = 0x09
    D_TX_CONTINUE     = 0x0A
    D_TX_GRANTED      = 0x0B
    D_TX_WAIT         = 0x0C
    D_SDS_DATA        = 0x0F
    D_FACILITY        = 0x10
# fmt: on


# MM


# fmt: off
class MmPduType(IntEnum):
    """4-bit downlink MM PDU type (ETSI EN 300 392-2 §16.9, Table 16.1).

    Sent inside an MLE TL-SDU with discriminator = MM (1). The body of each
    PDU is not parsed today -- we only surface the PDU name so registration,
    location updates, and authentication exchanges become visible in the
    decoder console.
    """
    D_OTAR                       = 0x0
    D_AUTHENTICATION             = 0x1
    D_AUTHENTICATION_REJECT      = 0x2
    D_CK_CHANGE_DEMAND           = 0x3
    D_DISABLE                    = 0x4
    D_ENABLE                     = 0x5
    D_LOCATION_UPDATE_ACCEPT     = 0x6
    D_LOCATION_UPDATE_COMMAND    = 0x7
    D_LOCATION_UPDATE_REJECT     = 0x8
    D_LOCATION_UPDATE_PROCEEDING = 0xA
    D_ATTACH_DETACH_GROUP_IDENT  = 0xB
    D_ATTACH_DETACH_GROUP_ACK    = 0xC
    D_MM_STATUS                  = 0xD
    D_MM_FUNCTION_NOT_SUPPORTED  = 0xF
# fmt: on


# AACH

# fmt: off
AACH_HEADER_NAMES: Mapping[int, str] = {
    0: "DLCC+ULCO",
    1: "DLF1+ULCA",
    2: "DLF1+ULAO",
    3: "DLF1+ULF1",
}

DL_USAGE_NAMES: Mapping[int, str] = {
    0: "unalloc",
    1: "assigned_ctrl",
    2: "common_ctrl",
    3: "reserved",
}
# fmt: on


# SYSINFO service flags

# (bit_index, service_name). Bits not listed here are reserved.
# fmt: off
BS_SERVICE_FLAGS: tuple[tuple[int, str], ...] = (
    (0,  "advanced_link"),
    (1,  "air_encryption"),
    (2,  "sndcp"),
    (4,  "circuit_data"),
    (5,  "voice"),
    (6,  "system_wide"),
    (7,  "migration"),
    (8,  "never_minimum_mode"),
    (9,  "priority_cell"),
    (10, "dereg_mandatory"),
    (11, "reg_mandatory"),
)
# fmt: on


# Carrier numbering math

# Frequency offset encoding (ETSI EN 300 392-2 §21.4.4, Table 21.31).
# fmt: off
FREQ_OFFSET_HZ: Mapping[int, int] = {
    0:  0,
    1:  6250,
    2: -6250,
    3:  12500,
}

# [duplex_spacing][freq_band] in kHz. ETSI TS 100 392-15 Table 2.
# 0 = reserved (UL returned as 0).
DUPLEX_SPACING_KHZ: tuple[tuple[int, ...], ...] = (
    (0, 1600, 10000, 10000, 10000, 10000, 10000,     0,     0,     0, 0, 0, 0, 0, 0, 0),  # noqa: E501
    (0, 4500,     0, 36000,  7000,     0,     0,     0, 45000, 45000, 0, 0, 0, 0, 0, 0),  # noqa: E501
    (0,    0,     0,     0,     0,     0,     0,     0,     0,     0, 0, 0, 0, 0, 0, 0),  # noqa: E501
    (0,    0,     0,  8000,  8000,     0,     0,     0, 18000, 18000, 0, 0, 0, 0, 0, 0),  # noqa: E501
    (0,    0,     0, 18000,  5000,     0, 30000, 30000,     0, 39000, 0, 0, 0, 0, 0, 0),  # noqa: E501
    (0,    0,     0,     0,  9500,     0,     0,     0,     0,     0, 0, 0, 0, 0, 0, 0),  # noqa: E501
    (0,    0,     0,     0,     0,     0,     0,     0,     0,     0, 0, 0, 0, 0, 0, 0),  # noqa: E501
    (0,    0,     0,     0,     0,     0,     0,     0,     0,     0, 0, 0, 0, 0, 0, 0),  # noqa: E501
)
# fmt: on
