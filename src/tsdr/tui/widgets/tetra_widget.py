"""TETRA domain-specific display widget.

Mirrors `RDSWidget`: reads a `TetraSnapshot` payload off the shared
`DecoderOutputEvent`, renders four columns (Network, Slots, Calls, Quality)
and also consumes `StatsUpdateEvent` for SNR.
"""

from __future__ import annotations

from textual.containers import Horizontal
from textual.widgets import Static

from tsdr.core.events.events import DecoderOutputEvent, StatsUpdateEvent
from tsdr.radio.decoders.tetra.state import (
    CARRIER_ROLE_MULTI,
    CARRIER_ROLE_SINGLE,
    CARRIER_ROLE_TCH,
    SLOT_USAGE_CONTROL,
    SLOT_USAGE_IDLE,
    SLOT_USAGE_SYNC,
    SLOT_USAGE_TRAFFIC,
    TetraSnapshot,
)
from tsdr.tui.widgets.panel import PanelWidget

_BAR_WIDTH = 5
_ALLOC_LOG_MAX_ROWS = 6


class TETRAWidget(Horizontal, PanelWidget):
    """Four-column TETRA state display.

    Col 1: Network / Cell identity + carrier role
    Col 2: Per-timeslot (TS1..TS4) usage + traffic bar
    Col 3: Active calls + rolling allocation log
    Col 4: Signal quality (CRC%, BFI%, freq offset, SNR, bursts)
    """

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: TetraSnapshot | None = None
        self._snr_db: float | None = None
        self._col_network = Static("Waiting...", id="tetra-network")
        self._col_slots = Static("", id="tetra-slots")
        self._col_calls = Static("", id="tetra-calls")
        self._col_quality = Static("", id="tetra-quality")

    def compose(self):
        yield self._col_network
        yield self._col_slots
        yield self._col_calls
        yield self._col_quality

    def on_mount(self) -> None:
        self.border_title = "TETRA"

    # event handlers

    def update_messages(self, event: DecoderOutputEvent) -> None:
        latest: TetraSnapshot | None = None
        for msg in event.messages:
            if isinstance(msg.data, TetraSnapshot):
                latest = msg.data
        if latest is None:
            return
        self._snapshot = latest
        self._refresh()

    def update_stats(self, event: StatsUpdateEvent) -> None:
        if not self.display:
            return
        self._snr_db = event.channel_snr
        self._refresh()

    # render

    def _refresh(self) -> None:
        snap = self._snapshot
        if snap is None:
            self._col_network.update("Waiting...")
            self._col_slots.update("")
            self._col_calls.update("")
            self._col_quality.update("")
            return

        self._col_network.update(self._render_network(snap))
        self._col_slots.update(self._render_slots(snap))
        self._col_calls.update(self._render_calls(snap))
        self._col_quality.update(self._render_quality(snap))

    @staticmethod
    def _render_network(snap: TetraSnapshot) -> str:
        lines: list[str] = []
        net = snap.network
        if net is None:
            lines.append("[dim]unlocked[/dim]")
            return "\n".join(lines)

        lines.append(f"MCC {net.mcc} MNC {net.mnc} CC {net.colour_code}")

        lines.append(_role_badge(snap.carrier_role))

        cell = snap.cell
        if cell is not None:
            lines.append(f"LA {cell.location_area} {cell.dl_freq_hz / 1e6:.4f} MHz")
            if cell.services:
                svcs = ",".join(cell.services[:5])
                if len(cell.services) > 5:
                    svcs += "+"
                lines.append(f"[dim]{svcs}[/dim]")

        return "\n".join(lines)

    @staticmethod
    def _render_slots(snap: TetraSnapshot) -> str:
        lines = ["[bold]SLOTS[/bold]"]
        for slot in snap.slots:
            label = _slot_label(slot.usage_label)
            bar = _bar(slot.traffic_ratio, _BAR_WIDTH)
            suffix = ""
            if slot.active_call_id is not None and slot.usage_label == SLOT_USAGE_TRAFFIC:
                suffix = f" #{slot.active_call_id}"
            lines.append(f"TS{slot.tn} {label} {bar}{suffix}")
        return "\n".join(lines)

    @staticmethod
    def _render_calls(snap: TetraSnapshot) -> str:
        lines: list[str] = []
        if snap.active_calls:
            lines.append("[bold]ACTIVE[/bold]")
            for call in snap.active_calls:
                slot_str = f"TS{call.assigned_slot}" if call.assigned_slot else "TS?"
                freq_str = ""
                if call.assigned_dl_freq_hz is not None:
                    freq_str = f" {call.assigned_dl_freq_hz / 1e6:.4f}"
                local = " [dim](local)[/dim]" if not call.is_offcarrier else ""
                if call.encryption_algo == "clear":
                    enc = "[green]clr[/green]"
                else:
                    enc = f"[yellow]{call.encryption_algo}[/yellow]"
                lines.append(f"#{call.call_id} {enc} -> {slot_str}{freq_str}{local}")
        else:
            lines.append("[dim]no active calls[/dim]")

        if snap.allocation_log:
            lines.append("")
            lines.append("[bold]ALLOCS[/bold]")
            recent = list(snap.allocation_log)[-_ALLOC_LOG_MAX_ROWS:][::-1]
            for entry in recent:
                freq_str = f"{entry.dl_freq_hz / 1e6:.4f}" if entry.dl_freq_hz is not None else "?"
                tag = "[magenta]↗[/magenta]" if entry.is_offcarrier else "[cyan]·[/cyan]"
                call_id_str = f"#{entry.call_id}" if entry.call_id is not None else "?"
                lines.append(f"{tag} {call_id_str} TS{entry.timeslot} {freq_str}")
        return "\n".join(lines)

    def _render_quality(self, snap: TetraSnapshot) -> str:
        q = snap.quality
        lines: list[str] = []
        lines.append(_sync_line(q.sync_state))

        crc_color = "green" if q.crc_pct >= 80 else "yellow" if q.crc_pct >= 50 else "red"
        lines.append(f"CRC [{crc_color}]{q.crc_pct:.0f}%[/{crc_color}]")

        if q.bfi_pct is not None:
            bfi_color = "green" if q.bfi_pct < 10 else "yellow" if q.bfi_pct < 30 else "red"
            lines.append(f"BFI [{bfi_color}]{q.bfi_pct:.0f}%[/{bfi_color}]")

        if q.freq_offset_hz is not None:
            off = q.freq_offset_hz
            offset_color = "green" if abs(off) < 100 else "yellow" if abs(off) < 500 else "red"
            lines.append(f"Δf [{offset_color}]{off:+.0f}Hz[/{offset_color}]")

        if self._snr_db is not None:
            snr = self._snr_db
            snr_color = "green" if snr > 20 else "yellow" if snr > 10 else "red"
            lines.append(f"SNR [{snr_color}]{snr:.1f}dB[/{snr_color}]")

        lines.append(f"[dim]{q.burst_count} bursts[/dim]")
        lines.append(f"[dim]codec {snap.voice_codec}[/dim]")
        return "\n".join(lines)


# render helpers


def _role_badge(role: str) -> str:
    if role == CARRIER_ROLE_SINGLE:
        return "[cyan bold]SINGLE[/cyan bold]"
    if role == CARRIER_ROLE_MULTI:
        return "[magenta bold]MULTI[/magenta bold]"
    if role == CARRIER_ROLE_TCH:
        return "[yellow bold]TCH[/yellow bold]"
    return "[dim]? role[/dim]"


_SLOT_LABEL_RENDER: dict[str, str] = {
    SLOT_USAGE_SYNC: "[cyan]SYNC [/cyan]",
    SLOT_USAGE_CONTROL: "[dim]CTRL [/dim]",
    SLOT_USAGE_TRAFFIC: "[green bold]VOICE[/green bold]",
    SLOT_USAGE_IDLE: "[dim]IDLE [/dim]",
}


def _slot_label(usage: str) -> str:
    return _SLOT_LABEL_RENDER.get(usage, "[dim]?    [/dim]")


def _bar(ratio: float, width: int) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return "[green]" + "█" * filled + "[/green]" + "[dim]" + "░" * (width - filled) + "[/dim]"


def _sync_line(state: str) -> str:
    if state == "locked":
        return "[green]●[/green] LOCKED"
    if state == "unlocking":
        return "[yellow]◐[/yellow] UNLOCKING"
    return "[red]○[/red] UNLOCKED"
