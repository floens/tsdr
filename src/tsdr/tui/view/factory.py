"""kind → Widget constructor map for the Reconciler.

Each entry must produce a widget whose reactive attrs match the prop schema
emitted by derive_tree for the same kind. Container kinds produce bare
Textual containers with no reactive props.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from textual.containers import Container, Vertical
from textual.widget import Widget

from tsdr.tui.console.widget import ConsoleWidget
from tsdr.tui.widgets.adsb_widget import ADSBWidget
from tsdr.tui.widgets.constellation_widget import ConstellationWidget
from tsdr.tui.widgets.dab_widget import DABWidget
from tsdr.tui.widgets.decoder_output_widget import DecoderOutputWidget
from tsdr.tui.widgets.dmr_widget import DMRWidget
from tsdr.tui.widgets.performance_widget import PerformanceWidget
from tsdr.tui.widgets.rds_widget import RDSWidget
from tsdr.tui.widgets.spectrum_widget import SpectrumWidget
from tsdr.tui.widgets.sstv_widget import SSTVWidget
from tsdr.tui.widgets.stats_widget import StatsWidget
from tsdr.tui.widgets.status_bar import StatusBar
from tsdr.tui.widgets.tetra_widget import TETRAWidget
from tsdr.tui.widgets.tuner_widget import TunerWidget
from tsdr.tui.widgets.waterfall_widget import WaterfallWidget

FACTORY: Final[dict[str, Callable[[], Widget]]] = {
    "main_container": Container,
    "viz_container": Vertical,
    "sidebar": Container,
    "tuner": TunerWidget,
    "status_bar": StatusBar,
    "console": ConsoleWidget,
    "spectrum": SpectrumWidget,
    "waterfall": WaterfallWidget,
    "stats": StatsWidget,
    "performance": PerformanceWidget,
    "constellation": ConstellationWidget,
    "decoder_rds": RDSWidget,
    "decoder_dab": DABWidget,
    "decoder_adsb": ADSBWidget,
    "decoder_tetra": TETRAWidget,
    "decoder_dmr": DMRWidget,
    "decoder_text": DecoderOutputWidget,
    "decoder_sstv": SSTVWidget,
}
