"""ACARS 2-char message-label descriptions.

`LABELS` holds only labels whose meaning is stable across operators. Labels that
carry OOOI/movement data (see `oooi.OOOI_LABELS`) get a generic description, since
their route and times render on their own line; everything else falls back to the
raw code. It covers a high-confidence subset of the ~100 ARINC-618 labels; extend
the dict as needed. A wrong description is worse than none, so unknowns stay as the
code.
"""

from __future__ import annotations

from tsdr.radio.decoders.acars.oooi import OOOI_LABELS

LABELS: dict[str, str] = {
    "B6": "ADS report",
    "H1": "message to/from terminal",
    "SQ": "squitter",
    "Q0": "link test",
    "5Z": "airline-designated downlink",
    "MA": "media advisory",
}


def describe_label(label: str) -> str:
    """Short English description for a 2-char label, or "" if unknown."""
    if len(label) == 2 and label[1] == "d":  # our synthetic tech-ack variant (block[10]==DEL)
        return "no-op / technical ack"
    if label in LABELS:
        return LABELS[label]
    if label in OOOI_LABELS:
        return "OOOI / movement report"
    return ""
