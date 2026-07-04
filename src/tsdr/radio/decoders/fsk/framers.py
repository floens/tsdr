"""Framers turn a per-baud soft-bit stream into decoded text messages.

`SitorBFramer` does CCIR-476 / SITOR-B character sync and FEC de-interleave;
`StartStopFramer` does asynchronous ITA2 start/stop framing. Both hold their
sync state across streaming chunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.radio.decoders.fsk.fec import FEC_DELAY, bytes_to_code, check_bits, resolve_char
from tsdr.radio.decoders.fsk.tables import ALPHABETS, CCIR_ALPHA, CCIR_REP, ShiftDecoder

_BUFFER_LEN = 100  # soft-bit ring: spans the 35-bit FEC delay + the 14-offset sync search
_SYNC_OFFSETS = 14  # 7 char-grid phases x 2 alpha/rep parities
_MIN_REPS = 3  # FEC rep matches required to accept a sync candidate
_MIN_SYNC_SCORE = 8
_MAX_ERRORS = 5  # accumulated errors before dropping sync
_MIN_FLUSH_CHARS = 20  # shortest in-progress message worth flushing on sync loss
_MAX_TEXT = 4000
_LINE_WIDTH = 80


class Framer(ABC):
    def __init__(self, alphabet: str) -> None:
        ltrs, figs, ltrs_code, figs_code = ALPHABETS[alphabet]
        self._shift = ShiftDecoder(ltrs, figs, ltrs_code, figs_code)
        self._ts = 0.0
        self._synced = False
        self._valid = 0  # characters that passed their framing check
        self._frames = 0  # characters attempted

    @property
    def synced(self) -> bool:
        return self._synced

    @property
    def fit(self) -> tuple[int, int]:
        """(valid, attempted) framing counts. The decoder scores acquisition candidates
        and watches for a wrong lock by the valid-to-attempted ratio."""
        return self._valid, self._frames

    def process(self, soft_bits: np.ndarray, ts: float) -> list[DecodedMessage]:
        self._ts = ts
        out: list[DecodedMessage] = []
        for value in soft_bits.tolist():
            self._handle_bit(value, out)
        return out

    def _emit(self, text: str, out: list[DecodedMessage]) -> None:
        text = text.strip()
        if text:
            out.append(DecodedMessage(text=text, timestamp=self._ts))

    @abstractmethod
    def _handle_bit(self, value: float, out: list[DecodedMessage]) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...


class SitorBFramer(Framer):
    """CCIR-476 / SITOR-B: soft bits -> characters -> NAVTEX messages."""

    def __init__(self, alphabet: str) -> None:
        super().__init__(alphabet)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(_BUFFER_LEN, dtype=np.float64)
        self._cursor = 0
        self._state = "sync"  # "sync" | "read"
        self._alpha_phase = True
        self._error_count = 0
        self._last_code = 0
        self._shift.reset()
        self._text = ""
        self._in_msg = False
        self._synced = False
        self._valid = 0
        self._frames = 0

    def _handle_bit(self, value: float, out: list[DecodedMessage]) -> None:
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = value
        if self._cursor > 0:
            self._cursor -= 1

        if self._state == "sync":
            offset = self._find_alpha()
            if offset >= 0:
                self._state = "read"
                self._cursor = offset
                self._alpha_phase = True
                self._synced = True
            else:
                self._error_count = 0
                self._shift.reset()

        if self._state == "read" and self._cursor < _BUFFER_LEN - 7:
            if self._alpha_phase:
                self._error_count -= self._process_char(self._cursor, out)
                if self._error_count > _MAX_ERRORS:
                    if self._in_msg and len(self._text) > _MIN_FLUSH_CHARS:
                        self._emit(self._text, out)
                    self._text = ""
                    self._in_msg = False
                    self._state = "sync"
                    self._error_count = 0
                    self._shift.reset()
                elif self._error_count < 0:
                    self._error_count = 0
            self._alpha_phase = not self._alpha_phase
            self._cursor += 7

    def _find_alpha(self) -> int:
        best_offset, best_score = -1, 0
        limit = _BUFFER_LEN - 7
        for offset in range(FEC_DELAY, FEC_DELAY + _SYNC_OFFSETS):
            score = reps = 0
            i = offset
            while i < limit:
                code = bytes_to_code(self._buf, i)
                if check_bits(code):
                    score += 1
                    rep = bytes_to_code(self._buf, i - FEC_DELAY)
                    if code == rep:
                        if code in (CCIR_ALPHA, CCIR_REP):
                            score = 0
                        else:
                            reps += 1
                    elif code == CCIR_ALPHA and bytes_to_code(self._buf, i - 7) == CCIR_REP:
                        reps += 1
                i += 7
            if reps >= _MIN_REPS and score + reps > best_score:
                best_score, best_offset = score + reps, offset
        return best_offset if best_score > _MIN_SYNC_SCORE else -1

    def _process_char(self, cursor: int, out: list[DecodedMessage]) -> int:
        rep_cursor = cursor - FEC_DELAY
        code, status = resolve_char(self._buf, cursor, rep_cursor if rep_cursor >= 0 else -1)
        self._frames += 1
        if code is not None:
            self._valid += 1
            if code == CCIR_REP:
                if self._last_code == CCIR_REP:
                    self._alpha_phase = False
            else:
                char = self._shift.decode(code)
                if char is not None:
                    self._append(char, out)
            self._last_code = code
        return status

    def _append(self, char: str, out: list[DecodedMessage]) -> None:
        if char == "\x07":  # bell
            return
        self._text += char
        # A fresh ZCZC flushes the previous message, since the NNNN terminator
        # frequently lands in an error burst; NNNN ends cleanly when it survives.
        if self._text.endswith("ZCZC"):
            if self._in_msg:
                self._emit(self._text[:-4], out)
            self._text = "ZCZC"
            self._in_msg = True
        elif self._in_msg and self._text.endswith("NNNN"):
            self._emit(self._text, out)
            self._text = ""
            self._in_msg = False
        elif len(self._text) > _MAX_TEXT:
            self._text = self._text[-_MAX_TEXT // 2 :] if self._in_msg else self._text[-8:]


class StartStopFramer(Framer):
    """Asynchronous ITA2 / Baudot: soft bits -> characters -> line messages."""

    def __init__(self, alphabet: str, data_bits: int) -> None:
        super().__init__(alphabet)
        self._data_bits = data_bits
        self.reset()

    def reset(self) -> None:
        self._fsm = "hunt"  # "hunt" | "data" | "stop"
        self._code = 0
        self._nbits = 0
        self._shift.reset()
        self._line = ""
        self._synced = False
        self._valid = 0
        self._frames = 0

    def _handle_bit(self, value: float, out: list[DecodedMessage]) -> None:
        mark = value > 0.0
        if self._fsm == "hunt":
            if not mark:  # space = start bit
                self._fsm = "data"
                self._code = 0
                self._nbits = 0
        elif self._fsm == "data":
            if mark:
                self._code |= 1 << self._nbits
            self._nbits += 1
            if self._nbits >= self._data_bits:
                self._fsm = "stop"
        elif self._fsm == "stop":
            self._frames += 1
            if mark:  # valid stop bit
                self._valid += 1
                self._synced = True
                char = self._shift.decode(self._code)
                if char is not None:
                    self._append(char, out)
            self._fsm = "hunt"

    def _append(self, char: str, out: list[DecodedMessage]) -> None:
        if char == "\x07":
            return
        if char in ("\r", "\n"):
            self._emit(self._line, out)
            self._line = ""
            return
        self._line += char
        if len(self._line) >= _LINE_WIDTH:
            self._emit(self._line, out)
            self._line = ""


def make_framer(framing: str, alphabet: str, data_bits: int) -> Framer:
    if framing == "sitor_b":
        return SitorBFramer(alphabet)
    return StartStopFramer(alphabet, data_bits)
