from __future__ import annotations

from tsdr.core.events.events import DecoderOutputEvent
from tsdr.radio.decoders.rds import RDSData
from tsdr.tui.markup import escape_forced
from tsdr.tui.widgets.section_panel import Section, SectionPanel

# Number of group columns (after stats + main) when laid out horizontally
_GROUP_COLS = 3


class RDSWidget(SectionPanel):
    """Display RDS data in a multi-column layout.

    Col 1: Stats (sync, BER, offset)
    Col 2: Main RDS data (PI, PS, PTY, Radio Text)
    Cols 3+: Group summaries, flowed top-to-bottom then left-to-right
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_rds: RDSData | None = None
        self._group_grid: dict[str, str] = {}

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

        self.refresh()

    def build_sections(self) -> list[Section]:
        rds = self._current_rds
        if rds is None:
            return [Section("Waiting...")]
        sections = [
            Section("\n".join(self._stats_lines(rds)), width=16, min_width=14),
            Section("\n".join(self._main_lines(rds)), width=24, min_width=18),
        ]
        sections.extend(self._group_sections())
        return sections

    @staticmethod
    def _stats_lines(rds: RDSData) -> list[str]:
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
        return stats

    @staticmethod
    def _main_lines(rds: RDSData) -> list[str]:
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
        return main

    def _group_sections(self) -> list[Section]:
        entries = [self._group_grid[k] for k in sorted(self._group_grid)]
        if not entries:
            return []
        rendered = [f"[dim cyan]{escape_forced(e)}[/dim cyan]" for e in entries]

        # Tall layout: a single stacked column. Wide: flow into up to _GROUP_COLS
        # columns sized by the available content height.
        if self.dock_edge != "bottom":
            return [Section("\n".join(rendered))]

        h = self.content_size.height
        rows_per_col = max(h, 1) if h > 0 else max(len(rendered), 1)
        cols: list[list[str]] = [[] for _ in range(_GROUP_COLS)]
        for i, line in enumerate(rendered):
            cols[min(i // rows_per_col, _GROUP_COLS - 1)].append(line)
        return [Section("\n".join(c)) for c in cols if c]
