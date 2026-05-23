import time
from argparse import Namespace
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from tsdr.core.clock_sync import ClockSyncMonitor, SyncResult, get_clock_sync_monitor
from tsdr.tui.commands._format import error, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.model.store import get_ui_store

_DEFAULT_NTP_SERVER = "pool.ntp.org"

_STATE_COLOR = {
    "synced": "green",
    "drift": "yellow",
    "unsynced": "red",
    "unknown": "dim",
}


def _format_age(measured_at: float) -> str:
    age_s = int(round(time.monotonic() - measured_at))
    if age_s < 120:
        return f"{age_s} s ago"
    return f"{age_s // 60} m ago"


def _ntp_lines(monitor: ClockSyncMonitor, indent: str) -> list[str]:
    """Return the 1–3 detail lines (state / offset / probed) for the NTP block.

    Header line ("server …" or "ntp …") is rendered by callers — they want
    different prefixes for the standalone vs. nested layouts.
    """
    snap: SyncResult = monitor.get()
    if not snap.is_current(monitor.server):
        return [f"{indent}state    [dim]pending[/]"]
    if snap.offset_s is None:
        lines = [f"{indent}state    [red]no answer[/]"]
        if snap.measured_at is not None:
            lines.append(f"{indent}probed   {_format_age(snap.measured_at)}")
        return lines
    color = _STATE_COLOR[snap.state]
    offset_ms = snap.offset_s * 1000
    lines = [
        f"{indent}state    [{color}]{snap.state}[/]",
        f"{indent}offset   [{color}]{offset_ms:+.1f} ms[/]",
    ]
    if snap.measured_at is not None:
        lines.append(f"{indent}probed   {_format_age(snap.measured_at)}")
    return lines


class TimeCommand(Command):
    @property
    def description(self) -> str:
        return "Configure the tuner clock (timezone, visibility, NTP sync)"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="action")
        sub.add_parser("show", help="Show the clock section")
        sub.add_parser("hide", help="Hide the clock section")
        tz = sub.add_parser("timezone", help="Set or show display timezone")
        tz.add_argument(
            "zone",
            nargs="?",
            help="IANA name (e.g. Europe/Amsterdam, UTC), or 'clear' for system local",
        )
        ntp = sub.add_parser("ntp", help="Configure SNTP probe (off by default)")
        ntp.add_argument(
            "target",
            nargs="?",
            help="Server hostname, 'on' (pool.ntp.org), or 'off' to disable",
        )

    def run(self, args: Namespace) -> str:
        if args.action == "show":
            return self._set_visible(True)
        if args.action == "hide":
            return self._set_visible(False)
        if args.action == "timezone":
            return self._timezone(args.zone)
        if args.action == "ntp":
            return self._ntp(args.target)
        return self._status()

    def _status(self) -> str:
        model = get_ui_store().model
        tz_name = model.timezone or "system local"
        vis = "[green]visible[/]" if model.clock_visible else "[dim]hidden[/]"
        monitor = get_clock_sync_monitor()
        lines = [
            f"clock      {vis}",
            f"timezone   {tz_name}",
        ]
        if monitor.server is None:
            lines.append("ntp        [dim]off[/]")
        else:
            lines.append(f"ntp        {monitor.server}")
            lines.extend(_ntp_lines(monitor, indent="  "))
        return "\n".join(lines)

    def _set_visible(self, visible: bool) -> str:
        get_ui_store().update(clock_visible=visible)
        return success(f"clock {'shown' if visible else 'hidden'}")

    def _timezone(self, zone: str | None) -> str:
        store = get_ui_store()
        if zone is None:
            current = store.model.timezone or "system local"
            return f"timezone={current}"
        if zone == "clear":
            store.update(timezone=None)
            return success("timezone reset to system local")
        try:
            ZoneInfo(zone)
        except ZoneInfoNotFoundError:
            return error(f"unknown timezone '{zone}'")
        store.update(timezone=zone)
        return success(f"timezone set to {zone}")

    def _ntp(self, target: str | None) -> str:
        store = get_ui_store()
        monitor = get_clock_sync_monitor()
        if target is None:
            if monitor.server is None:
                return "ntp probe is [dim]off[/]"
            lines = [f"server   {monitor.server}", *_ntp_lines(monitor, indent="")]
            return "\n".join(lines)
        if target == "off":
            store.update(ntp_server=None)
            monitor.set_server(None)
            return success("ntp probe disabled")
        server = _DEFAULT_NTP_SERVER if target == "on" else target
        store.update(ntp_server=server)
        monitor.set_server(server)
        return success(f"ntp probing {server} (first result in a few seconds)")

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if subcommand == "timezone":
            matches = sorted(z for z in available_timezones() if z.startswith(prefix))
            if "clear".startswith(prefix):
                matches = ["clear", *matches]
            return [Completion(z) for z in matches[:50]]
        if subcommand == "ntp":
            return [
                Completion(t) for t in ("on", "off", _DEFAULT_NTP_SERVER) if t.startswith(prefix)
            ]
        return []
