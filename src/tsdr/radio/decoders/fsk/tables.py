"""Code alphabets for FSK teleprinter modes: CCIR-476 (NAVTEX) and ITA2 (RTTY).

Both are 7-/5-bit codes read LSB-first (bit i set when soft sample i > 0) with a
LETTERS/FIGURES shift state, decoded through the shared `ShiftDecoder`.
"""

from __future__ import annotations

# CCIR-476 (ITU-R M.476): 7-bit, exactly 4 mark bits
# fmt: off
_CCIR_LTRS = {
    0x17: "J", 0x1B: "F", 0x1D: "C", 0x1E: "K", 0x27: "W", 0x2B: "Y",
    0x2D: "P", 0x2E: "Q", 0x35: "G", 0x39: "M", 0x3A: "X", 0x3C: "V",
    0x47: "A", 0x4B: "S", 0x4D: "I", 0x4E: "U", 0x53: "D", 0x55: "R",
    0x56: "E", 0x59: "N", 0x5C: " ", 0x63: "Z", 0x65: "L", 0x69: "H",
    0x6C: "\n", 0x71: "O", 0x72: "B", 0x74: "T", 0x78: "\r",
}
_CCIR_FIGS = {
    0x17: "'", 0x1B: "!", 0x1D: ":", 0x1E: "(", 0x27: "2", 0x2B: "6",
    0x2D: "0", 0x2E: "1", 0x35: "&", 0x39: ".", 0x3A: "/", 0x3C: ";",
    0x47: "-", 0x4B: "\x07", 0x4D: "8", 0x4E: "7", 0x53: "$", 0x55: "4",
    0x56: "3", 0x59: ",", 0x5C: " ", 0x63: '"', 0x65: ")", 0x69: "#",
    0x6C: "\n", 0x71: "9", 0x72: "?", 0x74: "5", 0x78: "\r",
}
# fmt: on
CCIR_LTRS_CODE = 0x5A
CCIR_FIGS_CODE = 0x36
CCIR_ALPHA = 0x0F  # idle / phasing signal
CCIR_REP = 0x66  # signal repetition (FEC rep marker)

# ITA2 / Baudot (US-TTY figures), 5-bit
# fmt: off
_ITA2_LTRS = {
    0x01: "E", 0x02: "\n", 0x03: "A", 0x04: " ", 0x05: "S", 0x06: "I",
    0x07: "U", 0x08: "\r", 0x09: "D", 0x0A: "R", 0x0B: "J", 0x0C: "N",
    0x0D: "F", 0x0E: "C", 0x0F: "K", 0x10: "T", 0x11: "Z", 0x12: "L",
    0x13: "W", 0x14: "H", 0x15: "Y", 0x16: "P", 0x17: "Q", 0x18: "O",
    0x19: "B", 0x1A: "G", 0x1C: "M", 0x1D: "X", 0x1E: "V",
}
_ITA2_FIGS = {
    0x01: "3", 0x02: "\n", 0x03: "-", 0x04: " ", 0x05: "\x07", 0x06: "8",
    0x07: "7", 0x08: "\r", 0x09: "$", 0x0A: "4", 0x0B: "'", 0x0C: ",",
    0x0D: "!", 0x0E: ":", 0x0F: "(", 0x10: "5", 0x11: '"', 0x12: ")",
    0x13: "2", 0x14: "#", 0x15: "6", 0x16: "0", 0x17: "1", 0x18: "9",
    0x19: "?", 0x1A: "&", 0x1C: ".", 0x1D: "/", 0x1E: ";",
}
# fmt: on
ITA2_LTRS_CODE = 0x1F
ITA2_FIGS_CODE = 0x1B


def _to_table(mapping: dict[int, str], size: int) -> list[str | None]:
    table: list[str | None] = [None] * size
    for code, char in mapping.items():
        table[code] = char
    return table


CCIR_TO_LTRS = _to_table(_CCIR_LTRS, 128)
CCIR_TO_FIGS = _to_table(_CCIR_FIGS, 128)
ITA2_TO_LTRS = _to_table(_ITA2_LTRS, 32)
ITA2_TO_FIGS = _to_table(_ITA2_FIGS, 32)

ALPHABETS = {
    "ccir476": (CCIR_TO_LTRS, CCIR_TO_FIGS, CCIR_LTRS_CODE, CCIR_FIGS_CODE),
    "ita2": (ITA2_TO_LTRS, ITA2_TO_FIGS, ITA2_LTRS_CODE, ITA2_FIGS_CODE),
}


class ShiftDecoder:
    """LETTERS/FIGURES shift state machine over a code alphabet.

    `decode(code)` returns the character, or None for a shift/unmapped/control
    code. The FIGS/LTRS shift state persists until the next shift code.
    """

    def __init__(
        self,
        ltrs_table: list[str | None],
        figs_table: list[str | None],
        ltrs_code: int,
        figs_code: int,
    ) -> None:
        self._ltrs = ltrs_table
        self._figs = figs_table
        self._ltrs_code = ltrs_code
        self._figs_code = figs_code
        self._shift = False  # False = letters, True = figures

    def decode(self, code: int) -> str | None:
        if code == self._ltrs_code:
            self._shift = False
            return None
        if code == self._figs_code:
            self._shift = True
            return None
        table = self._figs if self._shift else self._ltrs
        if 0 <= code < len(table):
            return table[code]
        return None

    def reset(self) -> None:
        self._shift = False
