"""DMR domain-specific display widget.

Reads a DMRSnapshot payload off DecoderOutputEvent and renders three
columns: Repeater info, Slot status, Signal quality.
"""

from __future__ import annotations

from textual.containers import Horizontal
from textual.widgets import Static

from tsdr.core.events.events import DecoderOutputEvent, StatsUpdateEvent
from tsdr.radio.decoders.dmr.decoder import DMRSnapshot

_BAR_WIDTH = 5


class DMRWidget(Horizontal):
    """Three-column DMR state display.

    Col 1: Repeater info (color code, modulation)
    Col 2: Per-timeslot status (voice/idle)
    Col 3: Signal quality (lock%, CACH%, SNR, burst count)
    """

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: DMRSnapshot | None = None
        self._snr_db: float | None = None
        self._col_repeater = Static("Waiting...", id="dmr-repeater")
        self._col_slots = Static("", id="dmr-slots")
        self._col_quality = Static("", id="dmr-quality")

    def compose(self):
        yield self._col_repeater
        yield self._col_slots
        yield self._col_quality

    def on_mount(self) -> None:
        self.border_title = "DMR"
        self.display = False

    def clear(self) -> None:
        self._snapshot = None
        self._snr_db = None
        self._refresh()

    # event handlers

    def update_messages(self, event: DecoderOutputEvent) -> None:
        if not self.display:
            return
        latest: DMRSnapshot | None = None
        for msg in event.messages:
            if isinstance(msg.data, DMRSnapshot):
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
            self._col_repeater.update("Waiting...")
            self._col_slots.update("")
            self._col_quality.update("")
            return

        self._col_repeater.update(self._render_repeater(snap))
        self._col_slots.update(self._render_slots(snap))
        self._col_quality.update(self._render_quality(snap))

    @staticmethod
    def _render_repeater(snap: DMRSnapshot) -> str:
        lines: list[str] = []
        if snap.color_code is not None:
            lines.append(f"CC [bold]{snap.color_code}[/bold]")
        else:
            lines.append("[dim]no CC[/dim]")
        lines.append("[dim]4FSK 12.5kHz[/dim]")
        lines.append("[dim]codec AMBE+2[/dim]")
        return "\n".join(lines)

    @staticmethod
    def _render_slots(snap: DMRSnapshot) -> str:
        lines = ["[bold]SLOTS[/bold]"]
        for slot in snap.slots:
            if slot.in_voice_call:
                label = "[green bold]VOICE[/green bold]"
                bar = _bar(1.0, _BAR_WIDTH)
                suffix = f"  #{slot.voice_burst_count}"
            else:
                label = "[dim]IDLE [/dim]"
                bar = _bar(0.0, _BAR_WIDTH)
                suffix = ""
            lines.append(f"TS{slot.timeslot} {label} {bar}{suffix}")
        return "\n".join(lines)

    def _render_quality(self, snap: DMRSnapshot) -> str:
        q = snap.quality
        lines: list[str] = []

        if q.locked:
            lines.append("[green]\u25cf[/green] LOCKED")
        else:
            lines.append("[red]\u25cb[/red] SEARCHING")

        lock_color = "green" if q.lock_pct >= 80 else "yellow" if q.lock_pct >= 50 else "red"
        lines.append(f"Lock [{lock_color}]{q.lock_pct:.0f}%[/{lock_color}]")

        cach_color = "green" if q.cach_pct >= 80 else "yellow" if q.cach_pct >= 50 else "red"
        lines.append(f"CACH [{cach_color}]{q.cach_pct:.0f}%[/{cach_color}]")

        if self._snr_db is not None:
            snr = self._snr_db
            snr_color = "green" if snr > 20 else "yellow" if snr > 10 else "red"
            lines.append(f"SNR [{snr_color}]{snr:.1f}dB[/{snr_color}]")

        lines.append(f"[dim]{q.burst_count} bursts[/dim]")
        return "\n".join(lines)


# render helpers


def _bar(ratio: float, width: int) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return (
        "[green]"
        + "\u2588" * filled
        + "[/green]"
        + "[dim]"
        + "\u2591" * (width - filled)
        + "[/dim]"
    )
