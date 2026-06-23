"""AX.25 UI frame parsing and the strict-validity gate.

Address field: dest(7) + src(7) + 0..8 digipeaters(7 each), each callsign byte
left-shifted by 1, terminated by the low bit of the SSID byte. APRS frames are
always UI (control 0x03, PID 0xF0); `parse_frame` reads a PID after the control
byte accordingly. `validate_strict` rejects CRC false-positives (random noise
that happens to satisfy the FCS) and gates the HDLC bit-fix retry.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsdr.radio.decoders.aprs.fec import fcs_ok

UI_CONTROL = 0x03
PID_NO_LAYER3 = 0xF0

_MIN_FRAME_LEN = 17  # dest(7) + src(7) + control(1) + FCS(2)
_MAX_ADDRESSES = 10  # dest + src + up to 8 digipeaters
_MAX_FIX_LEN = 64  # don't single-bit-fix frames longer than this (cost guard)


@dataclass(frozen=True)
class Address:
    call: str
    ssid: int
    has_been_repeated: bool  # H-bit; only meaningful on digipeater addresses

    def __str__(self) -> str:
        return self.call if self.ssid == 0 else f"{self.call}-{self.ssid}"


@dataclass(frozen=True)
class AX25Frame:
    dest: Address
    src: Address
    digis: tuple[Address, ...]
    control: int
    pid: int | None
    info: bytes

    def path_str(self) -> str:
        """`SRC>DEST,DIGI1*,DIGI2` with `*` on digipeaters that have repeated."""
        parts = [str(self.src), ">", str(self.dest)]
        for d in self.digis:
            parts.append("," + str(d) + ("*" if d.has_been_repeated else ""))
        return "".join(parts)


def _decode_address(raw: bytes) -> tuple[Address, bool]:
    call = "".join(chr(b >> 1) for b in raw[:6]).rstrip()
    ssid_byte = raw[6]
    ssid = (ssid_byte >> 1) & 0x0F
    return Address(call, ssid, bool(ssid_byte & 0x80)), bool(ssid_byte & 0x01)


def parse_frame(raw: bytes) -> AX25Frame | None:
    """Parse a raw AX.25 frame (including trailing FCS). Returns None if the FCS
    is bad or the structure is malformed."""
    if len(raw) < _MIN_FRAME_LEN or not fcs_ok(raw):
        return None
    body = raw[:-2]

    addrs: list[Address] = []
    i = 0
    while i + 7 <= len(body) and len(addrs) < _MAX_ADDRESSES:
        addr, is_last = _decode_address(body[i : i + 7])
        addrs.append(addr)
        i += 7
        if is_last:
            break
    else:
        return None  # ran out of bytes before the address field terminated

    if len(addrs) < 2 or i >= len(body):
        return None
    control = body[i]
    i += 1
    pid = body[i] if i < len(body) else None
    info = body[i + 1 :] if pid is not None else b""
    return AX25Frame(addrs[0], addrs[1], tuple(addrs[2:]), control, pid, info)


def recover(raw: bytes) -> AX25Frame | None:
    """Parse + validate a candidate frame, with a single-bit-fix retry on failure.

    On its own, a valid FCS could be a 1/65536 fluke; gating the fix on
    `validate_strict` (UI control, PID 0xF0, ASCII callsigns) drives false
    accepts to effectively zero. The flip is in the assembled-byte domain
    (stuffing already removed), so it recovers single bit errors that don't fall
    on a stuff bit. Frames over `_MAX_FIX_LEN` skip the O(8n) search.
    """
    frame = parse_frame(raw)
    if frame is not None and validate_strict(frame):
        return frame
    if len(raw) > _MAX_FIX_LEN:
        return None
    buf = bytearray(raw)
    for i in range(len(buf)):
        original = buf[i]
        for bit in range(8):
            buf[i] = original ^ (1 << bit)
            cand = parse_frame(bytes(buf))
            if cand is not None and validate_strict(cand):
                return cand
        buf[i] = original
    return None


def _valid_call(call: str) -> bool:
    return 1 <= len(call) <= 6 and all(c.isupper() or c.isdigit() for c in call)


def validate_strict(frame: AX25Frame) -> bool:
    """Reject CRC false-positives: APRS UI frame, ASCII callsigns, sane info.

    The control/PID/callsign constraints make a chance FCS pass on random bits
    astronomically unlikely; the printable-info check is a lenient backstop kept
    loose enough to admit Mic-E (whose info field carries sub-0x20 bytes).
    """
    if frame.control != UI_CONTROL or frame.pid != PID_NO_LAYER3:
        return False
    if not _valid_call(frame.dest.call) or not _valid_call(frame.src.call):
        return False
    if not all(_valid_call(d.call) for d in frame.digis):
        return False
    if frame.info:
        printable = sum(0x20 <= b < 0x7F for b in frame.info)
        if printable < 0.5 * len(frame.info):
            return False
    return True
