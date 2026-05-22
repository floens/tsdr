"""FT8 / FT4 77-bit message unpacker.

The 10-byte payload layout (MSB-first, bit 0 = payload[0] high bit):

    bits  0..28   n29a = (n28_to << 1) | ipa     -- standard messages
    bits 29..57   n29b = (n28_de << 1) | ipb
    bit  58       ir (R-flag for grid/report)
    bits 59..73   igrid4 (15 bits: grid or report code)
    bits 74..76   i3
    bits 71..73   n3 (only when i3 == 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

CT_FULL = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"
CT_ALPHANUM_SPACE = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CT_LETTERS_SPACE = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CT_ALPHANUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CT_NUMERIC = "0123456789"

NTOKENS = 2063592
MAX22 = 4194304
MAXGRID4 = 32400


class MessageType(IntEnum):
    FREE_TEXT = 0
    DXPEDITION = 1
    EU_VHF = 2
    ARRL_FD = 3
    TELEMETRY = 4
    CONTESTING = 5
    STANDARD = 6
    ARRL_RTTY = 7
    NONSTD_CALL = 8
    WWROF = 9
    UNKNOWN = 10


@dataclass(frozen=True)
class DecodedFTMessage:
    """A decoded FT8/FT4 frame."""

    text: str
    msg_type: MessageType
    payload: bytes  # the 77-bit message packed into 10 bytes


def _bits(payload: bytes, start: int, length: int) -> int:
    """Extract a big-endian bit field of length ``length`` starting at bit ``start``."""
    total = int.from_bytes(payload, "big")
    shift = len(payload) * 8 - start - length
    return (total >> shift) & ((1 << length) - 1)


_N3_TO_TYPE: dict[int, MessageType] = {
    0: MessageType.FREE_TEXT,
    1: MessageType.DXPEDITION,
    2: MessageType.EU_VHF,
    3: MessageType.ARRL_FD,
    4: MessageType.ARRL_FD,
    5: MessageType.TELEMETRY,
    6: MessageType.CONTESTING,
}

_I3_TO_TYPE: dict[int, MessageType] = {
    1: MessageType.STANDARD,
    2: MessageType.STANDARD,
    3: MessageType.ARRL_RTTY,
    4: MessageType.NONSTD_CALL,
    5: MessageType.WWROF,
}


def get_message_type(payload: bytes) -> MessageType:
    """Return the FT8/FT4 message-type tag for a 10-byte payload."""
    i3 = (payload[9] >> 3) & 0x07
    if i3 == 0:
        n3 = ((payload[8] << 2) & 0x04) | ((payload[9] >> 6) & 0x03)
        return _N3_TO_TYPE.get(n3, MessageType.UNKNOWN)
    return _I3_TO_TYPE.get(i3, MessageType.UNKNOWN)


def _unpack28(n28: int, ip: int, i3: int) -> str | None:
    """Decode a 28-bit callsign or special token."""
    if n28 < NTOKENS:
        if n28 == 0:
            return "DE"
        if n28 == 1:
            return "QRZ"
        if n28 == 2:
            return "CQ"
        if n28 <= 1002:
            return f"CQ {n28 - 3:03d}"
        if n28 <= 532443:
            n = n28 - 1003
            chars = [""] * 4
            for k in range(3, -1, -1):
                chars[k] = CT_LETTERS_SPACE[n % 27]
                n //= 27
            return "CQ " + "".join(chars).lstrip(" ")
        return None

    n28 -= NTOKENS
    if n28 < MAX22:
        # 22-bit hashed callsign — no hash table => placeholder
        return "<...>"

    n = n28 - MAX22
    chars = [""] * 6
    chars[5] = CT_LETTERS_SPACE[n % 27]
    n //= 27
    chars[4] = CT_LETTERS_SPACE[n % 27]
    n //= 27
    chars[3] = CT_LETTERS_SPACE[n % 27]
    n //= 27
    chars[2] = CT_NUMERIC[n % 10]
    n //= 10
    chars[1] = CT_ALPHANUM[n % 36]
    n //= 36
    chars[0] = CT_ALPHANUM_SPACE[n % 37]
    callsign = "".join(chars)

    if callsign.startswith("3D0") and callsign[3] != " ":
        result = "3DA0" + callsign[3:].rstrip()
    elif callsign[0] == "Q" and callsign[1].isalpha():
        result = "3X" + callsign[1:].rstrip()
    else:
        result = callsign.strip()

    if len(result) < 3:
        return None
    if ip != 0:
        if i3 == 1:
            result += "/R"
        elif i3 == 2:
            result += "/P"
        else:
            return None
    return result


def _unpack_grid(igrid4: int, ir: int) -> str:
    """Decode the 15-bit grid/report field. Empty string means 'no extra'."""
    if igrid4 <= MAXGRID4:
        prefix = "R " if ir else ""
        n = igrid4
        d3 = n % 10
        n //= 10
        d2 = n % 10
        n //= 10
        d1 = n % 18
        n //= 18
        d0 = n % 18
        return f"{prefix}{chr(ord('A') + d0)}{chr(ord('A') + d1)}{d2}{d3}"

    irpt = igrid4 - MAXGRID4
    if irpt == 1:
        return ""
    if irpt == 2:
        return "RRR"
    if irpt == 3:
        return "RR73"
    if irpt == 4:
        return "73"
    val = irpt - 35
    prefix = "R" if ir else ""
    return f"{prefix}{val:+03d}"


def _decode_standard(payload: bytes) -> str | None:
    n29a = _bits(payload, 0, 29)
    n29b = _bits(payload, 29, 29)
    ir = _bits(payload, 58, 1)
    igrid4 = _bits(payload, 59, 15)
    i3 = _bits(payload, 74, 3)

    call_to = _unpack28(n29a >> 1, n29a & 1, i3)
    if call_to is None:
        return None
    call_de = _unpack28(n29b >> 1, n29b & 1, i3)
    if call_de is None:
        return None
    extra = _unpack_grid(igrid4, ir)

    fields = [call_to, call_de] + ([extra] if extra else [])
    return " ".join(fields)


def _decode_telemetry_bytes(payload: bytes) -> bytes:
    """Right-shift the 80-bit payload by 1 bit -> 9-byte big-endian buffer."""
    out = bytearray(9)
    carry = 0
    for i in range(9):
        out[i] = ((carry << 7) | (payload[i] >> 1)) & 0xFF
        carry = payload[i] & 0x01
    return bytes(out)


def _decode_free_text(payload: bytes) -> str:
    """Unpack 13-char free text from a 71-bit message."""
    b71 = bytearray(_decode_telemetry_bytes(payload))
    chars = [" "] * 13
    for idx in range(12, -1, -1):
        rem = 0
        for i in range(9):
            rem = (rem << 8) | b71[i]
            b71[i] = rem // 42
            rem = rem % 42
        chars[idx] = CT_FULL[rem]
    return "".join(chars).strip()


def _decode_telemetry_hex(payload: bytes) -> str:
    return _decode_telemetry_bytes(payload).hex().upper()


def decode_message(payload: bytes) -> DecodedFTMessage:
    """Decode a 10-byte (77-bit) FT8/FT4 message payload."""
    if len(payload) < 10:
        raise ValueError(f"payload must be at least 10 bytes, got {len(payload)}")
    payload = bytes(payload[:10])
    mtype = get_message_type(payload)

    match mtype:
        case MessageType.STANDARD:
            text = _decode_standard(payload)
        case MessageType.FREE_TEXT:
            text = _decode_free_text(payload)
        case MessageType.TELEMETRY:
            text = _decode_telemetry_hex(payload)
        case _:
            text = None

    if text is None:
        text = f"<type {mtype.name}>"

    return DecodedFTMessage(text=text, msg_type=mtype, payload=payload)


__all__ = [
    "DecodedFTMessage",
    "MessageType",
    "decode_message",
    "get_message_type",
]
