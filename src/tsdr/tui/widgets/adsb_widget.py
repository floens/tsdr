from __future__ import annotations

import time

from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Static

from tsdr.core.events.events import DecoderOutputEvent
from tsdr.radio.decoders.adsb import ADSBData
from tsdr.tui.markup import escape_forced
from tsdr.tui.widgets.panel import PanelWidget

# Styles

_STYLE_NONE = Style()
_STYLE_DIM = Style(dim=True)
_STYLE_HEADER = Style(bold=True, dim=True)
_STYLE_CYAN = Style(color="cyan")
_STYLE_WHITE_BOLD = Style(color="white", bold=True)
_STYLE_WHITE = Style(color="white")
_STYLE_DIM_WHITE = Style(color="white", dim=True)
_STYLE_GREEN = Style(color="green")
_STYLE_YELLOW = Style(color="yellow")
_STYLE_RED = Style(color="red")
_STYLE_DIM_RED = Style(color="red", dim=True)
_STYLE_BOLD_WHITE = Style(color="white", bold=True)
_STYLE_SCALE = Style(dim=True)

# Age thresholds (seconds)
_AGE_GREEN = 5
_AGE_YELLOW = 15
_AGE_RED = 30
_AGE_STALE = 60


def _age_style(age: float) -> Style:
    """Age-based Style for the map panel."""
    if age < _AGE_GREEN:
        return _STYLE_GREEN
    if age < _AGE_YELLOW:
        return _STYLE_YELLOW
    if age < _AGE_RED:
        return _STYLE_RED
    return _STYLE_DIM_RED


def _age_style_name(age: float) -> str:
    """Age-based color name for Rich Table cells."""
    if age < _AGE_GREEN:
        return "green"
    if age < _AGE_YELLOW:
        return "yellow"
    if age < _AGE_RED:
        return "red"
    return "dim red"


def _vr_style_name(vr: int | None) -> str:
    """VR color name for Rich Table cells."""
    if vr is not None and vr > 0:
        return "green"
    if vr is not None and vr < 0:
        return "yellow"
    return "dim"


def _heading_arrow(heading: float | None) -> str:
    if heading is None:
        return ""
    # 8-way compass: N, NE, E, SE, S, SW, W, NW
    arrows = "↑↗→↘↓↙←↖"
    idx = int((heading + 22.5) / 45) % 8
    return arrows[idx]


def _format_altitude(alt: int | None) -> str:
    if alt is None:
        return ""
    if alt >= 18000:
        return f"FL{alt // 100:>3d}"
    return str(alt)


def _format_vr(vr: int | None) -> str:
    if vr is None:
        return ""
    if vr > 0:
        return f"▲ {vr}"
    if vr < 0:
        return f"▼ {abs(vr)}"
    return ""


def _format_seen_text(age: float) -> str:
    if age < 60:
        return f"{age:.0f}s"
    return f"{age / 60:.0f}m"


# Braille map rendering

# Braille character has 2 columns × 4 rows = 8 dots
# Dot numbering (bit positions):
#   dot1(0x01) dot4(0x08)
#   dot2(0x02) dot5(0x10)
#   dot3(0x04) dot6(0x20)
#   dot7(0x40) dot8(0x80)
_BRAILLE_BASE = 0x2800

# Map from (row 0-3, col 0-1) to bit value
_BRAILLE_DOT_BITS = [
    [0x01, 0x08],  # row 0 (top)
    [0x02, 0x10],  # row 1
    [0x04, 0x20],  # row 2
    [0x40, 0x80],  # row 3 (bottom)
]


def _altitude_dot_row(alt: int | None) -> int:
    """Map altitude to braille dot vertical position within cell (0=top, 3=bottom)."""
    if alt is None:
        return 1  # middle-ish default
    if alt >= 30000:
        return 0  # high: top
    if alt >= 10000:
        return 1  # mid: upper-middle
    return 3  # low: bottom


def _nm_to_deg_lat(nm: float) -> float:
    return nm / 60.0


def _deg_lat_to_nm(deg: float) -> float:
    return deg * 60.0


class _ADSBMapPanel(Widget):
    """Braille dot-map of aircraft positions."""

    DEFAULT_CSS = """
    _ADSBMapPanel {
        width: 1fr;
        height: 100%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(id="adsb-map", **kwargs)
        self._data: ADSBData | None = None
        self._now: float = 0.0

    def update_data(self, data: ADSBData) -> None:
        self._data = data
        self._now = time.time()
        self.refresh()

    def refresh_ages(self) -> None:
        """Re-render with updated time (ages change, data stays the same)."""
        if self._data is not None:
            self._now = time.time()
            self.refresh()

    def clear_data(self) -> None:
        self._data = None
        self.refresh()

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        height = self.size.height
        base = self.rich_style
        if width <= 0 or height <= 0:
            return Strip.blank(width, base)

        data = self._data
        if data is None or not data.aircraft:
            if y == 0:
                msg = "No aircraft"
                pad = max(0, (width - len(msg)) // 2)
                return Strip([Segment(" " * pad, base), Segment(msg, base + _STYLE_DIM)], width)
            return Strip.blank(width, base)

        # Gather aircraft with positions, hide stale (>60s)
        positioned = [
            ac
            for ac in data.aircraft
            if ac.lat is not None and ac.lon is not None and self._now - ac.last_seen < _AGE_STALE
        ]
        if not positioned:
            if y == 0:
                msg = f"{len(data.aircraft)} aircraft, no positions"
                pad = max(0, (width - len(msg)) // 2)
                return Strip([Segment(" " * pad, base), Segment(msg, base + _STYLE_DIM)], width)
            return Strip.blank(width, base)

        # Compute bounding box with margin
        # Note: lat/lon guaranteed non-None by the positioned filter above
        lats: list[float] = [ac.lat for ac in positioned if ac.lat is not None]
        lons: list[float] = [ac.lon for ac in positioned if ac.lon is not None]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Ensure minimum span (~10 nm)
        lat_span = max(max_lat - min_lat, 0.2)
        lon_span = max(max_lon - min_lon, 0.3)

        # Add 15% margin
        lat_margin = lat_span * 0.15
        lon_margin = lon_span * 0.15
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon

        # Reserve last row for scale bar
        map_height = height - 1
        if map_height <= 0:
            return Strip.blank(width, base)

        # Scale bar on last row
        if y == height - 1:
            return self._render_scale_bar(width, lat_span, lon_span)

        if y >= map_height:
            return Strip.blank(width, base)

        # Build grid: each cell is 2 cols × 4 rows in braille
        # Map coordinates to cell (col, row within cell)
        cell_cols = width
        cell_rows = map_height
        braille_rows = cell_rows * 4
        braille_cols = cell_cols * 2

        # Place aircraft into grid
        # grid[cell_row][cell_col] = (braille_bits, style, label_info)
        grid_bits: dict[tuple[int, int], int] = {}
        grid_styles: dict[tuple[int, int], Style] = {}
        grid_count: dict[tuple[int, int], int] = {}
        labels: list[tuple[int, int, str, Style]] = []  # (cell_col, cell_row, text, style)

        for ac in positioned:
            ac_lat: float = ac.lat  # type: ignore[assignment]  # filtered above
            ac_lon: float = ac.lon  # type: ignore[assignment]  # filtered above
            age = self._now - ac.last_seen
            style = _age_style(age)

            # Map to braille sub-pixel coordinates
            # Latitude is inverted (higher lat = lower y)
            frac_y = 1.0 - (ac_lat - min_lat) / lat_span if lat_span > 0 else 0.5
            frac_x = (ac_lon - min_lon) / lon_span if lon_span > 0 else 0.5

            bx = int(frac_x * (braille_cols - 1))
            bx = max(0, min(braille_cols - 1, bx))

            # Use altitude for vertical dot position within cell
            alt_row = _altitude_dot_row(ac.altitude)
            # Map to braille row: use altitude-aware positioning
            by_base = int(frac_y * (cell_rows - 1)) * 4
            by = by_base + alt_row
            by = max(0, min(braille_rows - 1, by))

            cell_col = bx // 2
            cell_row = by // 4
            sub_col = bx % 2
            sub_row = by % 4

            key = (cell_row, cell_col)
            bit = _BRAILLE_DOT_BITS[sub_row][sub_col]
            grid_bits[key] = grid_bits.get(key, 0) | bit
            grid_count[key] = grid_count.get(key, 0) + 1
            # Keep freshest style
            if key not in grid_styles or age < (self._now - ac.last_seen):
                grid_styles[key] = style

            # Build label
            label_text = ac.callsign or ac.icao
            if ac.altitude is not None:
                alt_short = f"{ac.altitude // 100:03d}" if ac.altitude >= 100 else str(ac.altitude)
                label_text += f" {alt_short}"

            labels.append((cell_col, cell_row, label_text, style))

        # Resolve label positions (simple: place to right, skip on overlap)
        used_cols: dict[int, set[int]] = {}  # row -> set of occupied columns

        # Mark aircraft dot cells as used
        for (row, col), _ in grid_bits.items():
            used_cols.setdefault(row, set()).add(col)

        placed_labels: list[tuple[int, int, str, Style]] = []
        for col, row, text, style in labels:
            label_len = len(text)
            placed = False

            # Try right
            start = col + 1
            if start + label_len <= cell_cols:
                overlap = False
                row_used = used_cols.get(row, set())
                for c in range(start, start + label_len):
                    if c in row_used:
                        overlap = True
                        break
                if not overlap:
                    placed_labels.append((start, row, text, style))
                    for c in range(start, start + label_len):
                        used_cols.setdefault(row, set()).add(c)
                    placed = True

            # Try left
            if not placed:
                start = col - label_len
                if start >= 0:
                    overlap = False
                    row_used = used_cols.get(row, set())
                    for c in range(start, start + label_len):
                        if c in row_used:
                            overlap = True
                            break
                    if not overlap:
                        placed_labels.append((start, row, text, style))
                        for c in range(start, start + label_len):
                            used_cols.setdefault(row, set()).add(c)

        # Render this row
        # First, build character + style arrays
        chars = [" "] * cell_cols
        styles = [base] * cell_cols

        # Draw braille dots
        for (crow, ccol), bits in grid_bits.items():
            if crow != y:
                continue
            count = grid_count[(crow, ccol)]
            if count > 1:
                # Cluster: show count digit
                chars[ccol] = str(min(count, 9))
                styles[ccol] = base + _STYLE_BOLD_WHITE
            else:
                chars[ccol] = chr(_BRAILLE_BASE | bits)
                styles[ccol] = base + grid_styles.get((crow, ccol), _STYLE_GREEN)

        # Draw labels
        for lcol, lrow, text, style in placed_labels:
            if lrow != y:
                continue
            for i, ch in enumerate(text):
                c = lcol + i
                if 0 <= c < cell_cols:
                    chars[c] = ch
                    styles[c] = base + style

        # Build segments (run-length encode same styles)
        segments: list[Segment] = []
        if cell_cols > 0:
            run_start = 0
            run_style = styles[0]
            for i in range(1, cell_cols):
                if styles[i] is not run_style:
                    segments.append(Segment("".join(chars[run_start:i]), run_style))
                    run_start = i
                    run_style = styles[i]
            segments.append(Segment("".join(chars[run_start:]), run_style))

        return Strip(segments, width)

    def _render_scale_bar(self, width: int, lat_span: float, lon_span: float) -> Strip:
        range_nm = _deg_lat_to_nm(max(lat_span, lon_span))
        # Scale bar is ~1/4 of the width
        bar_len = max(3, width // 4)
        bar_nm = range_nm * bar_len / width
        # Round to nice number
        for nice in (1, 2, 5, 10, 20, 50, 100, 200, 500):
            if nice >= bar_nm * 0.7:
                bar_nm = nice
                bar_len = max(3, int(width * nice / range_nm))
                break

        base = self.rich_style
        bar = "├" + "─" * max(1, bar_len - 2) + "┤"
        label = f" {bar_nm:.0f} nm"
        text = bar + label
        pad = max(0, width - len(text))
        segments = [Segment(" " * pad, base), Segment(text, base + _STYLE_SCALE)]
        return Strip(segments, width)


class _ADSBTablePanel(Static):
    """Aircraft table using Rich Table for automatic column alignment."""

    DEFAULT_CSS = """
    _ADSBTablePanel {
        width: auto;
        height: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", id="adsb-table", **kwargs)
        self._data: ADSBData | None = None

    def update_data(self, data: ADSBData) -> None:
        self._data = data
        self._refresh_display()

    def refresh_ages(self) -> None:
        """Re-render to update Seen column and age-based styles."""
        if self._data is not None:
            self._refresh_display()

    def clear_data(self) -> None:
        self._data = None
        self.update("")

    def _refresh_display(self) -> None:
        data = self._data
        if data is None:
            self.update("")
            return

        now = time.time()

        table = Table(
            box=None,
            show_header=True,
            header_style="bold dim",
            padding=(0, 1),
            expand=False,
        )
        table.add_column("ICAO", style="cyan", no_wrap=True)
        table.add_column("Call", style="bold white", no_wrap=True)
        table.add_column("Alt", justify="right", no_wrap=True)
        table.add_column("Spd", justify="right", no_wrap=True)
        table.add_column("·", justify="center", width=1, no_wrap=True)
        table.add_column("VR", justify="right", no_wrap=True)
        table.add_column("Msgs", justify="right", style="dim", no_wrap=True)
        table.add_column("Seen", justify="right", no_wrap=True)

        for ac in data.aircraft:
            age = now - ac.last_seen
            row_style = "dim" if age > _AGE_STALE else ""
            table.add_row(
                ac.icao,
                escape_forced(ac.callsign),
                _format_altitude(ac.altitude),
                f"{ac.speed:.0f}" if ac.speed is not None else "",
                _heading_arrow(ac.heading),
                Text(_format_vr(ac.vertical_rate), style=_vr_style_name(ac.vertical_rate)),
                str(ac.messages),
                Text(_format_seen_text(age), style=_age_style_name(age)),
                style=row_style,
            )

        table.caption = (
            f"Tracking: {len(data.aircraft)}  ┆  "
            f"Msgs: {data.total_messages}  ┆  "
            f"{data.messages_per_second:.0f}/s  ┆  "
            f"ICAOs: {data.unique_icaos}"
        )

        self.update(table)


class ADSBWidget(Horizontal, PanelWidget):
    """ADS-B aircraft map + table display."""

    _refresh_timer = None

    def __init__(self) -> None:
        super().__init__()
        self._map = _ADSBMapPanel()
        self._table = _ADSBTablePanel()

    def compose(self):
        yield self._map
        yield self._table

    def on_mount(self) -> None:
        self.border_title = "ADS-B"
        # Widget is only mounted while ADS-B decoder is active; tick the
        # age refresh as long as we're alive.
        self._refresh_timer = self.set_interval(1.0, self._tick_ages)

    def on_unmount(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def _tick_ages(self) -> None:
        """Re-render map and table to update age-based colors and Seen column."""
        self._map.refresh_ages()
        self._table.refresh_ages()

    def update_messages(self, event: DecoderOutputEvent) -> None:
        adsb_data = None
        for msg in event.messages:
            if isinstance(msg.data, ADSBData):
                adsb_data = msg.data

        if adsb_data is None:
            return

        self._map.update_data(adsb_data)
        self._table.update_data(adsb_data)
