"""Textual strips must not reference themselves (tsdr.tui.strip_cycles): a
self-referencing strip is cyclic garbage, and collecting it stalls the device
reads."""

import gc

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

import tsdr.tui.strip_cycles  # noqa: F401


def _reaches_itself(obj: object, depth: int = 6) -> bool:
    seen: set[int] = set()
    frontier = gc.get_referents(obj)
    for _ in range(depth):
        following = []
        for o in frontier:
            if o is obj:
                return True
            if id(o) not in seen:
                seen.add(id(o))
                following.extend(gc.get_referents(o))
        frontier = following
    return False


def _strip() -> Strip:
    return Strip([Segment("ab", Style(bold=True)), Segment("cd")], 4)


def test_whole_strip_operations_leave_no_cycle() -> None:
    strip = _strip()
    assert strip.divide([4]) == [strip]
    assert strip.divide([4, 9]) == [strip]
    assert strip.crop_extend(0, 4, None) is strip
    assert strip.crop_extend(-1, 4, None) is strip
    assert not _reaches_itself(strip)


def test_textual_cycle_is_what_the_patch_removes() -> None:
    strip = _strip()
    tsdr.tui.strip_cycles._textual_divide(strip, [4])
    assert _reaches_itself(strip)


def test_real_cuts_and_crops_match_textual() -> None:
    strip = _strip()
    assert [s.text for s in strip.divide([1, 4])] == ["a", "bcd"]
    assert strip.crop_extend(1, 6, None).text == "bcd  "
    assert strip.crop_extend(0, 3, None).text == "abc"
