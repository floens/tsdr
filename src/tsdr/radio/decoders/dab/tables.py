"""Catalog of FIG types/extensions the DAB decoder supports.

Every enum / mapping here corresponds to a table or field in
ETSI EN 300 401. If a wire value doesn't appear here, the parser
doesn't handle it. Adding support means extending this file first,
parser second.

Manual alignment is load-bearing for readability here -- the whole
point of the file is to look like a printed spec table -- so the
aligned blocks are wrapped in `# fmt: off` / `# fmt: on` to keep
ruff-format away.
"""

from enum import IntEnum

__all__ = [
    # FIG framing (§5.2.2)
    "FigType",
    # FIG 0 / FIG 1 extensions (§6, §8)
    "Fig0Extension",
    "Fig1Extension",
    # Service component (§6.3.1)
    "TMId",
    # Sub-channel organization (§6.2.1)
    "SubchForm",
    "UEP_TABLE",
    # Announcement / user-app / programme-type lookups
    "ANNOUNCEMENT_TYPES",
    "USER_APP_TYPES",
    "ANNOUNCEMENT_ALARM_BIT",
    "PROGRAMME_TYPES",
    # Constants
    "LABEL_BYTES",
    "FIG_END_MARKER",
]


# FIG framing


# fmt: off
class FigType(IntEnum):
    """3-bit FIG type field (ETSI EN 300 401 §5.2.2, Table 2)."""
    MCI_SI = 0   # FIG 0 - MCI / service information
    LABELS = 1   # FIG 1 - labels (charset-encoded, 16-byte labels)
    XPAD   = 2   # FIG 2 - extended labels (UTF-8/UCS-2, multi-segment)
    # 3-7 reserved / not handled
# fmt: on


# FIG 0 extensions


# fmt: off
class Fig0Extension(IntEnum):
    """5-bit FIG 0 extension field (ETSI EN 300 401 §6, §8)."""
    ENSEMBLE_INFO            = 0   # 0/0  §6.4    ensemble id + Change/Al flags
    SUBCHANNEL_ORG           = 1   # 0/1  §6.2.1  subchannel start/size/protection
    SERVICE_ORG              = 2   # 0/2  §6.3.1  service -> component(s) mapping
    SERVICE_COMP_PACKET      = 3   # 0/3  §6.3.2  packet-mode component definition
    SERVICE_COMP_GLOBAL      = 8   # 0/8  §6.3.5  secondary-component (SCIdS) mapping
    DATE_TIME                = 10  # 0/10 §8.1.3.1 UTC date/time (MJD + h/m/s)
    USER_APP_INFO            = 13  # 0/13 §8.1.20 user-application types per service
    PROGRAMME_TYPE           = 17  # 0/17 §8.1.5  programme type (genre) per service
    ANNOUNCEMENT_SUPPORT     = 18  # 0/18 §8.1.6.1 services that can carry announcements
    ANNOUNCEMENT_SWITCHING   = 19  # 0/19 §8.1.6.2 announcement currently active on cluster
    FREQ_INFO                = 21  # 0/21 §8.1.8   service-following (DAB/FM/DRM/AMSS)
    OE_SERVICES              = 24  # 0/24 §8.1.10.2 services available in other ensembles
    OE_ANNOUNCEMENT_SUPPORT  = 25  # 0/25 §8.1.10.3 OE announcement support
    OE_ANNOUNCEMENT_SWITCHING= 26  # 0/26 §8.1.10.4 OE announcement switching
# fmt: on


# FIG 1 extensions


# fmt: off
class Fig1Extension(IntEnum):
    """3-bit FIG 1 extension field (ETSI EN 300 401 §8.1)."""
    ENSEMBLE_LABEL          = 0  # 1/0 §8.1.13 ensemble label (16 chars)
    SERVICE_LABEL           = 1  # 1/1 §8.1.14 service label, 16-bit SId
    SERVICE_COMPONENT_LABEL = 4  # 1/4 §8.1.14 service component label (SCIdS)
    DATA_SERVICE_LABEL      = 5  # 1/5 §8.1.14 data service label (32-bit SId)
    XPAD_USER_APP_LABEL     = 6  # 1/6 §8.1.14 X-PAD user-app label
# fmt: on


# Service component


# fmt: off
class TMId(IntEnum):
    """2-bit TM-Id - service component transport mechanism (§6.3.1)."""
    MSC_STREAM_AUDIO = 0
    MSC_STREAM_DATA  = 1
    MSC_PACKET_DATA  = 2  # not handled
    FIC_DATA         = 3  # not handled
# fmt: on


# Sub-channel organization


# fmt: off
class SubchForm(IntEnum):
    """1-bit short/long form for sub-channel organization (§6.2.1)."""
    SHORT_UEP = 0  # UEP, table-driven size
    LONG_EEP  = 1  # EEP, explicit size + protection option (A=0, B=1)
# fmt: on


# Short-form UEP profiles. Indexed by the 6-bit Table index (§6.2.1, Table 8):
# table_index -> (sub-channel size in CU, protection level 1-5, audio bit rate kbit/s).
# fmt: off
UEP_TABLE: tuple[tuple[int, int, int], ...] = (
    (16,  5, 32),  (21,  4, 32),  (24,  3, 32),  (29,  2, 32),  (35,  1, 32),
    (24,  5, 48),  (29,  4, 48),  (35,  3, 48),  (42,  2, 48),  (52,  1, 48),
    (29,  5, 56),  (35,  4, 56),  (42,  3, 56),  (52,  2, 56),
    (32,  5, 64),  (42,  4, 64),  (48,  3, 64),  (58,  2, 64),  (70,  1, 64),
    (40,  5, 80),  (52,  4, 80),  (58,  3, 80),  (70,  2, 80),  (84,  1, 80),
    (48,  5, 96),  (58,  4, 96),  (70,  3, 96),  (84,  2, 96),  (104, 1, 96),
    (58,  5, 112), (70,  4, 112), (84,  3, 112), (104, 2, 112),
    (64,  5, 128), (84,  4, 128), (96,  3, 128), (116, 2, 128), (140, 1, 128),
    (80,  5, 160), (104, 4, 160), (116, 3, 160), (140, 2, 160), (168, 1, 160),
    (96,  5, 192), (116, 4, 192), (140, 3, 192), (168, 2, 192), (208, 1, 192),
    (116, 5, 224), (140, 4, 224), (168, 3, 224), (208, 2, 224), (232, 1, 224),
    (128, 5, 256), (168, 4, 256), (192, 3, 256), (232, 2, 256), (280, 1, 256),
    (160, 5, 320), (208, 4, 320), (280, 2, 320),
    (192, 5, 384), (280, 3, 384), (416, 1, 384),
)
# fmt: on
assert len(UEP_TABLE) == 64


# Announcement types (ETSI EN 300 401 §8.1.6, Table 14)


# 16-bit ASu/ASw flag field; bit n (LSB=0) corresponds to row n below.
# fmt: off
ANNOUNCEMENT_TYPES: tuple[str, ...] = (
    "Alarm",                # bit 0 - overrides volume / forced un-mute
    "Road traffic flash",   # bit 1
    "Transport flash",      # bit 2
    "Warning/Service",      # bit 3
    "News flash",           # bit 4
    "Area weather flash",   # bit 5
    "Event announcement",   # bit 6
    "Special event",        # bit 7
    "Programme information",  # bit 8
    "Sport report",         # bit 9
    "Financial report",     # bit 10
    "Reserved (11)",
    "Reserved (12)",
    "Reserved (13)",
    "Reserved (14)",
    "Reserved (15)",
)
# fmt: on
ANNOUNCEMENT_ALARM_BIT = 0


# User-application types (ETSI TS 101 756 v2, Table 16)


# Codes the UI cares about by name; unknown codes fall back to "0xNNN".
# fmt: off
USER_APP_TYPES: dict[int, str] = {
    0x000: "Reserved",
    0x002: "MOT Slideshow",
    0x003: "MOT Broadcast Web Site",
    0x004: "TPEG (traffic)",
    0x005: "DGPS",
    0x006: "TMC",
    0x007: "SPI/EPG",
    0x008: "DAB Java",
    0x009: "DMB",
    0x00A: "IPDC",
    0x00B: "Voice applications",
    0x00C: "Middleware",
    0x00D: "Filecasting",
    0x044: "Journaline",
}
# fmt: on


# Programme type / genre (ETSI TS 101 756 v2, Table 12, "international" set)


# Indexed by 5-bit Int. Code; codes 0 and 32+ are "None"/reserved.
# fmt: off
PROGRAMME_TYPES: tuple[str, ...] = (
    "None",         "News",            "Current Affairs", "Information",
    "Sport",        "Education",       "Drama",           "Culture",
    "Science",      "Varied",          "Pop Music",       "Rock Music",
    "Easy Listening", "Light Classical", "Serious Classical", "Other Music",
    "Weather",      "Finance",         "Children's",      "Factual",
    "Religion",     "Phone In",        "Travel",          "Leisure",
    "Jazz",         "Country",         "National Music",  "Oldies",
    "Folk",         "Documentary",     "Reserved (30)",   "Reserved (31)",
)
# fmt: on


# Constants

LABEL_BYTES = 16  # FIG 1/x label is always 16 chars; charset signaled in header byte
FIG_END_MARKER = 0xFF  # padding byte that terminates a FIB
