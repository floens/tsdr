"""Panel registry: panel_id → metadata used by derive_tree and the panel command.

Panels are edge-agnostic; the active edge for each is recorded in UIModel.layout.
The `kind` string references an entry in `tui/view/factory.py:FACTORY`.

The `demod` panel is a multiplexer: derive_tree picks the concrete decoder
widget (RDS/DAB/ADSB/TETRA/DMR) at render time based on the focused device's
`active_decoder_kind`, so `demod`'s `kind` here is purely a placeholder. Its
`title_of` resolver mirrors that, reporting the active decoder's name for the bar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from tsdr.tui.model import focused_device

if TYPE_CHECKING:
    from tsdr.tui.model import UIModel


@dataclass(frozen=True)
class PanelDef:
    panel_id: str
    title: str  # shown as-is after the hotkey digit in the bottom bar
    kind: str  # factory key used by Reconciler to construct the widget
    # Resolved from the UIModel only — keeps the bar a pure function of it
    # (derive_tree never reads widgets).
    title_of: Callable[[UIModel], str | None] | None = None


_DEMOD_TITLES: Final[dict[str, str]] = {
    "rds": "RDS",
    "dab": "DAB",
    "adsb": "ADSB",
    "tetra": "TETRA",
    "dmr": "DMR",
    "sstv": "SSTV",
}


def _demod_title(m: UIModel) -> str | None:
    """The active decoder's name for the focused device, or None when none."""
    focused = focused_device(m)
    if focused is None or focused.active_decoder_kind is None:
        return None
    return _DEMOD_TITLES.get(focused.active_decoder_kind)


PANELS: Final[dict[str, PanelDef]] = {
    "demod": PanelDef("demod", "Demod", "_demod_placeholder", title_of=_demod_title),
    "decoder-output": PanelDef("decoder-output", "Decoder", "decoder_text"),
    "stats": PanelDef("stats", "Stats", "stats"),
    "performance": PanelDef("performance", "Performance", "performance"),
    "directory": PanelDef("directory", "Directory", "directory"),
    "memories": PanelDef("memories", "Memories", "memories"),
}
