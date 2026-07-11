"""ACARS block framer: soft bits -> validated ARINC-618 blocks -> messages.

A rolling shift register hunts the post-preamble sync anchor `'+' '*' SYN SYN
SOH` (and its bitwise complement, which also recovers per-burst polarity); a match
sets the byte boundary and the decoder reads LSb-first bytes until `ETX`/`ETB`
+ 2 CRC bytes. Each block is checked for odd parity + CRC-16, repaired via the
syndrome FEC if needed, then parsed into fields.

A block that reaches a terminator but fails parity/CRC/FEC is still emitted when
`emit_partial` is set (default), marked `[unverified]` with every parity-failing
byte masked to '.' so the reader can see which positions are corrupt.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsdr.core.events.events import DecodedMessage
from tsdr.radio.decoders.acars import fec
from tsdr.radio.decoders.acars.constants import (
    DEL,
    ETB,
    ETX,
    MAX_PARITY_ERRORS,
    PLUS,
    SOH,
    STAR,
    SYN,
    TXT_MAX_LEN,
    TXT_MIN_LEN,
)
from tsdr.radio.decoders.acars.crc import crc16, update
from tsdr.radio.decoders.acars.labels import describe_label
from tsdr.radio.decoders.acars.oooi import Oooi, decode_oooi


def _build_sync() -> tuple[int, int]:
    bits = [(byte >> i) & 1 for byte in (PLUS, STAR, SYN, SYN, SOH) for i in range(8)]
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val, len(bits)


_SYNC, _SYNC_LEN = _build_sync()
_SYNC_MASK = (1 << _SYNC_LEN) - 1
_SYNC_INV = _SYNC ^ _SYNC_MASK
_SYNC_TOL = 4  # max bit errors in the 40-bit anchor; CRC rejects false syncs
_MASK = ord(".")  # stands in for a parity-failed byte in an unverified block


def _parity_ok(b: int) -> bool:
    return b.bit_count() & 1 == 1  # ACARS uses odd parity


@dataclass(frozen=True)
class AcarsBlock:
    mode: str
    reg: str
    ack: str
    label: str
    block_id: str
    seqno: str
    flight: str
    text: str
    timestamp: float
    errors: int
    verified: bool
    label_desc: str
    oooi: Oooi | None


def _e(s: str) -> str:
    """Escape Rich markup in decoder-supplied text ('[' is the only tag opener)."""
    return s.replace("[", r"\[")


def _format_oooi(o: Oooi) -> str:
    parts = []
    if o.dep or o.dest:
        parts.append(f"[yellow]{_e(o.dep) or '?'}→{_e(o.dest) or '?'}[/]")
    times = [
        f"{name} {_e(v)}"
        for name, v in (
            ("eta", o.eta),
            ("out", o.gate_out),
            ("off", o.wheels_off),
            ("on", o.wheels_on),
            ("in", o.gate_in),
        )
        if v
    ]
    if times:
        parts.append(f"[dim yellow]{' '.join(times)}[/]")
    return " ".join(parts)


def format_block(b: AcarsBlock) -> str:
    parts = [f"[bold cyan]{_e(b.reg) or '.......'}[/]"]
    lbl = f"[magenta]{_e(b.label)}[/]"
    if b.label_desc:
        lbl += f"[dim italic] {_e(b.label_desc)}[/]"
    parts.append(lbl)
    flight, seqno = b.flight.strip(), b.seqno.strip()
    if flight:
        parts.append(f"[bold green]{_e(flight)}[/]")
    meta = []
    if b.block_id and b.block_id != "\x00":
        meta.append("#" + _e(b.block_id))
    if seqno:
        meta.append(_e(seqno))
    if meta:
        parts.append(f"[dim]{' '.join(meta)}[/]")
    line = " ".join(parts)
    if b.text:
        line += "  " + _e(b.text)
    if b.oooi:
        line += "\n  " + _format_oooi(b.oooi)
    if not b.verified:
        line += (
            f"  [red]\\[unverified {b.errors} bad][/]" if b.errors else "  [red]\\[unverified][/]"
        )
    elif b.errors:
        line += f"  [dim yellow]\\[fixed {b.errors}][/]"
    return line


class AcarsFramer:
    def __init__(self, *, emit_partial: bool = True) -> None:
        self._emit_partial = emit_partial
        self._pending: list[DecodedMessage] = []
        self.reset()

    def reset(self) -> None:
        self._reg = 0
        self._collecting = False
        self._invert = False
        self._raw = bytearray()
        self._curbyte = 0
        self._bc = 0
        self._state = "text"
        self._crc0 = 0
        self._crc1 = 0
        self._terminated = False
        self._pending = []

    def process(self, soft, ts: float) -> None:
        for v in soft:
            bit = 1 if v > 0 else 0
            if self._collecting:
                self._collect_bit(bit, ts)
            else:
                self._reg = ((self._reg << 1) | bit) & _SYNC_MASK
                if (self._reg ^ _SYNC).bit_count() <= _SYNC_TOL:
                    self._begin(invert=False)
                elif (self._reg ^ _SYNC_INV).bit_count() <= _SYNC_TOL:
                    self._begin(invert=True)

    def drain(self) -> list[DecodedMessage]:
        out = self._pending
        self._pending = []
        return out

    def _begin(self, *, invert: bool) -> None:
        self._collecting = True
        self._invert = invert
        self._raw = bytearray()
        self._curbyte = 0
        self._bc = 0
        self._state = "text"

    def _abort(self) -> None:
        self._collecting = False
        self._reg = 0

    def _collect_bit(self, bit: int, ts: float) -> None:
        b = bit ^ (1 if self._invert else 0)
        self._curbyte |= b << self._bc
        self._bc += 1
        if self._bc == 8:
            byte = self._curbyte
            self._curbyte = 0
            self._bc = 0
            self._block_byte(byte, ts)

    def _block_byte(self, byte: int, ts: float) -> None:
        if self._state == "text":
            if len(self._raw) >= TXT_MAX_LEN:
                self._abort()
                return
            self._raw.append(byte)
            if byte in (ETX, ETB):
                self._terminated = True
                self._state = "crc0"
            elif byte == DEL and len(self._raw) >= TXT_MIN_LEN + 3:
                # missed text terminator: raw ends [..., crc0, crc1, DEL]
                self._crc0 = self._raw[-3]
                self._crc1 = self._raw[-2]
                del self._raw[-3:]
                self._terminated = False
                self._finalize(ts)
                self._abort()
        elif self._state == "crc0":
            self._crc0 = byte
            self._state = "crc1"
        else:  # crc1
            self._crc1 = byte
            self._finalize(ts)
            self._abort()

    def _finalize(self, ts: float) -> None:
        raw = self._raw
        if len(raw) < TXT_MIN_LEN:
            return
        block = bytearray(raw)
        pr = [i for i, b in enumerate(block) if not _parity_ok(b)]
        verified = False
        if len(pr) <= MAX_PARITY_ERRORS:
            residual = update(update(crc16(block), self._crc0), self._crc1)
            if pr:
                verified = fec.fix_parity_errors(block, residual, pr)
            else:
                verified = residual == 0 or fec.fix_double_error(block, residual)
            if verified and any(not _parity_ok(b) for b in block):
                verified = False
        if not verified and not self._emit_partial:
            return
        errors = len(pr) if verified else sum(1 for b in block if not _parity_ok(b))
        for i, b in enumerate(block):
            block[i] = (b & 0x7F) if _parity_ok(b) else _MASK
        msg = _parse(block, self._terminated, ts, errors, verified)
        self._pending.append(
            DecodedMessage(text=format_block(msg), timestamp=ts, data=msg, markup=True)
        )


def _parse(
    block: bytearray, terminated: bool, ts: float, errors: int, verified: bool
) -> AcarsBlock:
    mode = chr(block[0])
    reg = bytes(block[1:8]).decode("ascii", "replace").lstrip(".")
    reg = "".join(c for c in reg if c.isprintable()).strip()  # squitters have no tail
    ack = "!" if block[8] == 0x15 else chr(block[8])  # NAK is nonprintable
    label = bytes(block[9:11]).decode("ascii", "replace")
    if block[10] == 0x7F:
        label = label[0] + "d"
    bid = chr(block[11])
    body = bytes(block[13:-1] if terminated else block[13:])
    seqno = flight = ""
    if bid.isdigit() and len(body) >= 10:
        seqno = body[:4].decode("ascii", "replace")
        flight = body[4:10].decode("ascii", "replace")
        body = body[10:]
    raw_text = body.decode("ascii", "replace").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c if c.isprintable() or c == "\n" else "." for c in raw_text).rstrip()
    label_desc = ""
    oooi: Oooi | None = None
    if verified:  # only a verified block is trustworthy enough to interpret
        label_desc = describe_label(label)
        oooi = decode_oooi(label, raw_text)
    return AcarsBlock(
        mode, reg, ack, label, bid, seqno, flight, text, ts, errors, verified, label_desc, oooi
    )
