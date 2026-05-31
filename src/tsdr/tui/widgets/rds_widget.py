from textual.containers import Horizontal
from textual.widgets import Static

from tsdr.core.events.events import DecoderOutputEvent
from tsdr.radio.decoders.rds import RDSData
from tsdr.tui.markup import escape_forced
from tsdr.tui.widgets.panel import PanelWidget

# Number of group columns (after stats + main)
_GROUP_COLS = 3


class RDSWidget(Horizontal, PanelWidget):
    """Display RDS data in a multi-column layout.

    Col 1: Stats (sync, BER, offset)
    Col 2: Main RDS data (PI, PS, PTY, Radio Text)
    Cols 3+: Group summaries, flowed top-to-bottom then left-to-right
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_rds: RDSData | None = None
        self._group_grid: dict[str, str] = {}
        self._col_stats = Static("Waiting...", id="rds-stats")
        self._col_main = Static("", id="rds-main")
        self._col_grps = [Static("", id=f"rds-grp{i}") for i in range(_GROUP_COLS)]

    def compose(self):
        yield self._col_stats
        yield self._col_main
        yield from self._col_grps

    def on_mount(self) -> None:
        self.border_title = "RDS"

    def update_messages(self, event: DecoderOutputEvent) -> None:
        """Update widget with new RDS data from DecoderOutputEvent."""
        # Find the last message with RDSData payload
        rds_data = None
        for msg in event.messages:
            if isinstance(msg.data, RDSData):
                rds_data = msg.data

        if rds_data is None:
            return

        self._current_rds = rds_data

        for group in rds_data.recent_groups:
            summary = group.summary
            key = summary.split(" ", 1)[0] if summary else ""
            if key:
                self._group_grid[key] = summary

        if not rds_data.sync_locked:
            self._group_grid.clear()

        self._refresh_display()

    def _refresh_display(self) -> None:
        if self._current_rds is None:
            self._col_stats.update("Waiting...")
            self._col_main.update("")
            for col in self._col_grps:
                col.update("")
            return

        rds = self._current_rds

        # Column 1: Stats
        stats: list[str] = []
        sync_icon = "[green]●[/green] SYNC" if rds.sync_locked else "[red]○[/red] NO SYNC"
        conf_pct = rds.sync_confidence * 100
        stats.append(f"{sync_icon} {conf_pct:.0f}%")

        offset = rds.baseband_offset_hz
        offset_color = "green" if abs(offset) < 200 else "yellow" if abs(offset) < 500 else "red"
        stats.append(f"[{offset_color}]{offset:+.0f} Hz[/{offset_color}]")

        stats.append(f"Groups: {rds.groups_received}")
        if rds.groups_received > 0:
            ber_pct = rds.block_error_rate * 100
            ber_color = "green" if ber_pct < 10 else "yellow" if ber_pct < 50 else "red"
            stats.append(f"BER: [{ber_color}]{ber_pct:.0f}%[/{ber_color}]")
        if rds.uncorrectable_blocks > 0:
            stats.append(f"Uncorr: {rds.uncorrectable_blocks}")

        self._col_stats.update("\n".join(stats))

        # Column 2: Main RDS data
        main: list[str] = []
        if not rds.sync_locked:
            main.append("[dim]Searching...[/dim]")
        else:
            if rds.pi_code is not None:
                main.append(f"PI: {rds.pi_code:04X}")
            if rds.ps_name:
                main.append(f"[bold]{escape_forced(rds.ps_name)}[/bold]")
            if rds.pty_name:
                main.append(f"PTY: {escape_forced(rds.pty_name)}")
            if rds.radio_text:
                main.append(escape_forced(rds.radio_text))

        self._col_main.update("\n".join(main))

        # Group columns: flow sorted entries into available rows
        entries = [self._group_grid[k] for k in sorted(self._group_grid)]

        # Determine rows per column from widget height
        # content_size.height accounts for padding/border; fall back to entry count
        h = self.content_size.height
        rows_per_col = max(h, 1) if h > 0 else max(len(entries), 1)

        col_lines: list[list[str]] = [[] for _ in range(_GROUP_COLS)]
        for i, entry in enumerate(entries):
            col_idx = min(i // rows_per_col, _GROUP_COLS - 1)
            col_lines[col_idx].append(f"[dim cyan]{escape_forced(entry)}[/dim cyan]")

        for col, lines in zip(self._col_grps, col_lines, strict=True):
            col.update("\n".join(lines))
