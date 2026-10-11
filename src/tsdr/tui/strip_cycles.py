"""Keep Textual's `Strip` from caching itself.

`Strip.divide` caches `[self]` when the only cut is the strip's own length,
and `Strip.crop_extend` caches `self` when the crop changes nothing. Both are
the usual case for a widget that fills its region, so almost every line the
compositor renders ends up in a reference cycle that only the cyclic garbage
collector can free: ~30k objects/s with the spectrum and waterfall running.
Each collection then stops every thread for ~100 ms, which overruns the
driver buffers of an SDR streaming at 10 MS/s and loses samples. Returning
the strip without caching it gives the same result with no cycle.
"""

from collections.abc import Iterable, Sequence

from rich.style import Style
from textual.strip import Strip

_textual_divide = Strip.divide
_textual_crop_extend = Strip.crop_extend


def _divide(self: Strip, cuts: Iterable[int]) -> Sequence[Strip]:
    cuts = list(cuts)
    if [cut for cut in cuts if cut <= self.cell_length] == [self.cell_length]:
        return [self]
    strips: Sequence[Strip] = _textual_divide(self, cuts)
    return strips


def _crop_extend(self: Strip, start: int, end: int, style: Style | None) -> Strip:
    if start <= 0 and end == self.cell_length:
        return self
    return _textual_crop_extend(self, start, end, style)


Strip.divide = _divide  # type: ignore[method-assign]
Strip.crop_extend = _crop_extend  # type: ignore[method-assign]
