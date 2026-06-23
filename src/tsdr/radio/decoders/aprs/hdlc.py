"""Streaming NRZI decode + HDLC deframing.

NRZI: a data 1 is "no transition", a 0 is "a transition" (polarity-independent).
HDLC: frames are delimited by the flag 0x7E (01111110); inside a frame a 0 is
destuffed after five consecutive 1s. Both keep state across `process()` calls so
they work on the chunked IQ stream. The deframer is pure HDLC; FCS validation
and the bit-fix retry live one layer up in `ax25`.
"""

from __future__ import annotations

import numpy as np

_FLAG = 0x7E


class NRZIDecoder:
    """Tone-level bits -> data bits, carrying the last level across chunks."""

    def __init__(self) -> None:
        self._prev = np.uint8(1)

    def reset(self) -> None:
        self._prev = np.uint8(1)

    def process(self, levels: np.ndarray) -> np.ndarray:
        lv: np.ndarray = np.asarray(levels, dtype=np.uint8)
        if len(lv) == 0:
            return lv
        prev = np.empty(len(lv), dtype=np.uint8)
        prev[0] = self._prev
        prev[1:] = lv[:-1]
        out: np.ndarray = (lv == prev).astype(np.uint8)
        self._prev = lv[-1]
        return out


def _bits_to_bytes_lsb(bits: list[int]) -> bytes:
    n = len(bits) // 8
    return np.packbits(np.array(bits[: n * 8], dtype=np.uint8), bitorder="little").tobytes()


class HDLCDeframer:
    """Streaming flag/destuff deframer; emits candidate frame byte blobs (incl. FCS).

    `min_bits` rejects runt frames (default 136 = 17-byte minimum AX.25). The
    closing flag's six 1s would otherwise trip the destuffer's stuff logic; they
    are appended to the in-progress frame and dropped (the trailing 7 partial-flag
    bits) when the flag completes.
    """

    def __init__(self, min_bits: int = 136) -> None:
        self._min_bits = min_bits
        self._history = 0
        self._frame: list[int] = []
        self._ones = 0
        self._synced = False

    def reset(self) -> None:
        self._history = 0
        self._frame = []
        self._ones = 0
        self._synced = False

    def process(self, bits: np.ndarray) -> list[bytes]:
        out: list[bytes] = []
        for raw in bits:
            b = int(raw)
            self._history = ((self._history << 1) | b) & 0xFF
            if self._history == _FLAG:
                if self._synced and len(self._frame) >= self._min_bits + 7:
                    payload = self._frame[:-7]  # drop the 7 leading bits of this flag
                    if len(payload) % 8 == 0:
                        out.append(_bits_to_bytes_lsb(payload))
                self._frame = []
                self._ones = 0
                self._synced = True
                continue
            if not self._synced:
                continue
            if self._ones == 5:
                self._ones = 0
                if b == 0:
                    continue  # stuffed zero -> drop
                # sixth consecutive 1: part of an upcoming flag; fall through
            self._frame.append(b)
            self._ones = self._ones + 1 if b == 1 else 0
        return out
