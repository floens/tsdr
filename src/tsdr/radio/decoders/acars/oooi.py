"""ACARS OOOI / route extraction.

A subset of ACARS labels carry Out/Off/On/In movement data with airport and ETA
fields at fixed byte offsets in the message text. This pulls those into an `Oooi`
(departure / destination ICAO, ETA, gate-out/in and wheels-off/on times). The
offsets and guard checks are fixed by the ACARS message format. Runs on the
post-seqno/flight message text.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Oooi:
    dep: str = ""
    dest: str = ""
    eta: str = ""
    gate_out: str = ""
    gate_in: str = ""
    wheels_off: str = ""
    wheels_on: str = ""


def _sl(t: str, start: int, n: int = 4) -> str:
    return t[start : start + n]


Fields = dict[str, str]

# Unguarded fixed-offset labels: field -> byte offset. Every field is a 4-char
# slice, so each label is pure data, kept in one table for auditability.
_FIXED: dict[str, list[tuple[str, int]]] = {
    "Q1": [
        ("dep", 0),
        ("gate_out", 4),
        ("wheels_off", 8),
        ("wheels_on", 12),
        ("gate_in", 16),
        ("dest", 24),
    ],
    "Q2": [("dep", 0), ("eta", 4)],
    "QA": [("dep", 0), ("gate_out", 4)],
    "QB": [("dep", 0), ("wheels_off", 4)],
    "QC": [("dep", 0), ("wheels_on", 4)],
    "QD": [("dep", 0), ("gate_in", 4)],
    "QE": [("dep", 0), ("gate_out", 4), ("dest", 8)],
    "QF": [("dep", 0), ("wheels_off", 4), ("dest", 8)],
    "QG": [("dep", 0), ("gate_out", 4), ("gate_in", 8)],
    "QH": [("dep", 0), ("gate_out", 4)],
    "QK": [("dep", 0), ("wheels_on", 4), ("dest", 8)],
    "QL": [("dest", 0), ("gate_in", 8), ("dep", 13)],
    "QM": [("dest", 0), ("dep", 8)],
    "QN": [("dest", 4), ("eta", 8)],
    "QP": [("dep", 0), ("dest", 4), ("gate_out", 8)],
    "QQ": [("dep", 0), ("dest", 4), ("wheels_off", 8)],
    "QR": [("dep", 0), ("dest", 4), ("wheels_on", 8)],
    "QS": [("dep", 0), ("dest", 4), ("gate_in", 8)],
    "QT": [("dep", 0), ("dest", 4), ("gate_out", 8), ("gate_in", 12)],
    "2Z": [("dest", 0)],
}


def _fixed(t: str, spec: list[tuple[str, int]]) -> Fields:
    return {name: _sl(t, off) for name, off in spec}


# Guarded labels: a prefix / delimiter layout must match before extraction.


def _dep_comma_dest(t: str) -> Fields | None:  # shared by labels 12, 1G, 83
    if t[4:5] != ",":
        return None
    return {"dep": _sl(t, 0), "dest": _sl(t, 5)}


def _10(t: str) -> Fields | None:
    if not t.startswith("ARR01"):
        return None
    return {"dest": _sl(t, 12), "eta": _sl(t, 16)}


def _11(t: str) -> Fields | None:
    if t[13:17] != "/DS ":
        return None
    if t[21:26] != "/ETA ":
        return None
    return {"dest": _sl(t, 17), "eta": _sl(t, 26)}


def _15(t: str) -> Fields | None:
    if not t.startswith("FST01"):
        return None
    return {"dep": _sl(t, 5), "dest": _sl(t, 9)}


def _17(t: str) -> Fields | None:
    if t[0:4] != "ETA ":
        return None
    if t[8:9] != "," or t[13:14] != ",":
        return None
    return {"eta": _sl(t, 4), "dep": _sl(t, 9), "dest": _sl(t, 14)}


def _20(t: str) -> Fields | None:
    if not t.startswith("RST"):
        return None
    return {"dep": _sl(t, 22), "dest": _sl(t, 26)}


def _21(t: str) -> Fields | None:
    if t[6:7] != "," or t[11:12] != ",":
        return None
    return {"dep": _sl(t, 7), "dest": _sl(t, 12)}


def _26(t: str) -> Fields | None:
    if not t.startswith("VER/077"):
        return None
    i = t.find("\n")
    if i < 0:
        return None
    p = t[i + 1 :]
    if not p.startswith("SCH/"):
        return None
    j = p.find("/", 4)
    if j < 0:
        return None
    res: Fields = {"dep": p[j + 1 : j + 5], "dest": p[j + 6 : j + 10]}
    k = p.find("\n", j)
    if k < 0:
        return res
    q = p[k + 1 :]
    if not q.startswith("ETA/"):
        return None
    res["eta"] = q[4:8]
    return res


def _2n(t: str) -> Fields | None:
    if not t.startswith("TKO01") or t[11:12] != "/":
        return None
    return {"dep": _sl(t, 20), "dest": _sl(t, 24)}


def _33(t: str) -> Fields | None:
    if t[0:1] != "," or t[20:21] != "," or t[25:26] != ",":
        return None
    return {"dep": _sl(t, 21), "dest": _sl(t, 26)}


def _39(t: str) -> Fields | None:
    if not t.startswith("GTA01") or t[15:16] != "/":
        return None
    return {"dep": _sl(t, 24), "dest": _sl(t, 28)}


def _44(t: str) -> Fields | None:
    if t[:1] == "0":
        if t[1:2] != "0":
            return None
        t = t[2:]
    if not (t.startswith("POS0") or t.startswith("ETA0")):
        return None
    if t[4:5] not in ("2", "3"):
        return None
    if any(t[i : i + 1] != "," for i in (23, 28, 33, 38, 43)):
        return None
    return {"dest": _sl(t, 24), "eta": _sl(t, 44)}


def _45(t: str) -> Fields | None:
    if t[0:1] != "A":
        return None
    return {"dest": _sl(t, 1)}


def _80(t: str) -> Fields | None:
    if t[6:11] != "/DEST":
        return None
    return {"dest": _sl(t, 12)}


def _8d(t: str) -> Fields | None:
    if t[4:5] != "," or t[35:36] != "," or t[40:41] != ",":
        return None
    return {"dep": _sl(t, 36), "dest": _sl(t, 41)}


def _8e(t: str) -> Fields | None:
    if t[4:5] != ",":
        return None
    return {"dest": _sl(t, 0), "eta": _sl(t, 5)}


_PARSERS: dict[str, Callable[[str], Fields | None]] = {
    "10": _10,
    "11": _11,
    "12": _dep_comma_dest,
    "15": _15,
    "17": _17,
    "1G": _dep_comma_dest,
    "20": _20,
    "21": _21,
    "26": _26,
    "2N": _2n,
    "33": _33,
    "39": _39,
    "44": _44,
    "45": _45,
    "80": _80,
    "83": _dep_comma_dest,
    "8D": _8d,
    "8E": _8e,
    "8S": _8e,  # 8S shares 8E's layout
    "RB": _26,  # RB shares 26's layout
}

OOOI_LABELS = frozenset(_FIXED) | frozenset(_PARSERS)


def decode_oooi(label: str, text: str) -> Oooi | None:
    """Extract OOOI/route fields for `label` from the message text, or None."""
    if (spec := _FIXED.get(label)) is not None:
        fields: Fields | None = _fixed(text, spec)
    elif (parser := _PARSERS.get(label)) is not None:
        fields = parser(text)
    else:
        return None
    if fields is None:
        return None
    kept = {k: s for k, v in fields.items() if (s := v.strip())}
    return Oooi(**kept) if kept else None
