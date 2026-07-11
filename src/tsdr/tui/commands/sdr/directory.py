from __future__ import annotations

from argparse import Namespace
from typing import TypeVar

from tsdr.core.directory import cache, connect
from tsdr.core.directory.display import (
    bandwidth_hz,
    bw_label,
    default_sort_key,
    probe_label,
    source_label,
    status_text,
    users_label,
)
from tsdr.core.directory.favorites import FavoriteDevice, get_favorites_store
from tsdr.core.directory.model import PublicDevice, Source
from tsdr.core.directory.probe import probe_device
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.units import parse_hz
from tsdr.tui.commands._format import error, header, safe, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.model import Edge
from tsdr.tui.model.store import UIStore, get_ui_store

DIRECTORY_PANEL_ID = "directory"
_DEFAULT_LIMIT = 40

_Device = TypeVar("_Device", PublicDevice, FavoriteDevice)


class DirectoryCommand(Command):
    @property
    def description(self) -> str:
        return "Browse public SDR directories (SpyServer, KiwiSDR)"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="action")

        list_p = sub.add_parser("list", help="List cached directory receivers")
        list_p.add_argument("--source", choices=["spyserver", "kiwisdr"])
        list_p.add_argument("--online", action="store_true", help="Only receivers reporting online")
        list_p.add_argument("--free", action="store_true", help="Only receivers usable right now")
        list_p.add_argument("--band", help="Only receivers tunable to this frequency (e.g. 14.2M)")
        list_p.add_argument("--near", help="Sort by distance to LAT,LON (e.g. 52.1,5.2)")
        list_p.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)

        sub.add_parser("refresh", help="Re-fetch both directories (blocks briefly)")

        fav_p = sub.add_parser("favorite", help="Save a receiver to favorites")
        fav_p.add_argument("query", help="host, name, or id substring")

        unfav_p = sub.add_parser("unfavorite", help="Remove a favorite")
        unfav_p.add_argument("query", help="host, name, or id substring")

        sub.add_parser("favorites", help="List saved favorites")
        sub.add_parser("show", help="Open the directory panel")

        add_p = sub.add_parser("add", help="Add + start a receiver (SpyServer only)")
        add_p.add_argument("query", help="host, name, or id substring")

        rm_p = sub.add_parser("remove", help="Stop + remove a receiver added from the directory")
        rm_p.add_argument("query", help="host, name, or id substring")

        flag_p = sub.add_parser("flag", help="Flag a receiver as dead (greyed out on refresh)")
        flag_p.add_argument("query", help="host, name, or id substring")

        unflag_p = sub.add_parser("unflag", help="Clear a receiver's flag")
        unflag_p.add_argument("query", help="host, name, or id substring")

        note_p = sub.add_parser("note", help="Set a favorite's note (empty text clears)")
        note_p.add_argument("query", help="host, name, or id substring")
        note_p.add_argument("text", nargs="*", help="Note text")

        ping_p = sub.add_parser("ping", help="Probe a receiver's live reachability")
        ping_p.add_argument("query", help="host, name, or id substring")

    def runs_in_background(self, argv: list[str]) -> bool:
        # Only these do network I/O; the rest touch UI-store/favorites/engine state.
        return bool(argv) and argv[0] in {"refresh", "ping"}

    def run(self, args: Namespace) -> str:
        if args.action == "list":
            return self._list(args)
        if args.action == "refresh":
            return self._refresh()
        if args.action == "favorite":
            return self._favorite(args.query)
        if args.action == "unfavorite":
            return self._unfavorite(args.query)
        if args.action == "favorites":
            return self._favorites()
        if args.action == "show":
            return self._show()
        if args.action == "add":
            return self._add(args.query)
        if args.action == "remove":
            return self._remove(args.query)
        if args.action == "flag":
            return self._flag(args.query, True)
        if args.action == "unflag":
            return self._flag(args.query, False)
        if args.action == "note":
            return self._note(args.query, " ".join(args.text))
        if args.action == "ping":
            return self._ping(args.query)
        return self.help_text()

    def _list(self, args: Namespace) -> str:
        devices = cache.cached()
        if not devices:
            return "No directory data yet. Run [bold]directory refresh[/] or open the panel."

        if args.source:
            devices = [d for d in devices if d.source == args.source]
        if args.online:
            devices = [d for d in devices if d.online]
        if args.free:
            devices = [d for d in devices if d.usable]
        if args.band:
            freq = float(parse_hz(args.band))
            devices = [d for d in devices if _covers(d, freq)]

        favorites = get_favorites_store()
        near = _parse_latlon(args.near)
        if near is not None:
            devices.sort(key=lambda d: _distance(d, *near))
        else:
            devices.sort(key=lambda d: (not favorites.is_favorite(d.id), default_sort_key(d)))

        if not devices:
            return "No receivers match those filters"

        total = len(devices)
        shown = devices[: max(1, args.limit)]
        lines = [_row(d, favorites.is_favorite(d.id)) for d in shown]
        head = header(f"Receivers ({total})")
        if total > len(shown):
            lines.append(f"[dim]… and {total - len(shown)} more (use --limit)[/]")
        return "\n".join([head, *lines])

    def _refresh(self) -> str:
        result = cache.get_directory()
        counts: dict[Source, int] = {}
        for d in result.devices:
            counts[d.source] = counts.get(d.source, 0) + 1
        parts = ", ".join(f"{source_label(s)}={n}" for s, n in sorted(counts.items()))
        summary = success(f"Fetched {len(result.devices)} receivers ({parts or 'none'})")
        if not result.errors:
            return summary
        failed = "; ".join(
            f"{source_label(s)} failed: {m}" for s, m in sorted(result.errors.items())
        )
        return f"{summary}\n{error(failed)}"

    def _favorite(self, query: str) -> str:
        device = self._resolve(query, cache.cached())
        get_favorites_store().add(device)
        return success(f"Favorited {_name(device)} [dim]{device.host}[/]")

    def _unfavorite(self, query: str) -> str:
        favorites = get_favorites_store()
        match = self._resolve(query, favorites.all())
        favorites.remove(match.id)
        return success(f"Removed favorite {_name(match)}")

    def _favorites(self) -> str:
        favorites = get_favorites_store().all()
        if not favorites:
            return "No favorites saved"
        lines = [header(f"Favorites ({len(favorites)})")]
        for f in favorites:
            src = source_label(f.source)
            endpoint = f.url or f.host
            lines.append(
                f"[green]★[/] [dim]{src:4}[/] [bold]{safe(f.name)[:28]}[/] [dim]{endpoint}[/]"
            )
        return "\n".join(lines)

    def _show(self) -> str:
        store = get_ui_store()
        edge = _panel_edge(store)
        if edge is None:
            return "Directory panel is not docked"
        store.set_panel_active(edge, DIRECTORY_PANEL_ID)
        return f"shown panel={DIRECTORY_PANEL_ID} edge={edge}"

    def _add(self, query: str) -> str:
        device = self._resolve(query, cache.cached())
        return _result_message(connect.add_directory_device(device))

    def _remove(self, query: str) -> str:
        device = self._resolve(query, cache.cached())
        return _result_message(connect.remove_directory_device(device))

    def _flag(self, query: str, flagged: bool) -> str:
        device = self._resolve(query, cache.cached())
        store = get_favorites_store()
        if flagged:
            store.flag(device.id)
            return success(f"Flagged {_name(device)}")
        store.unflag(device.id)
        return success(f"Unflagged {_name(device)}")

    def _note(self, query: str, text: str) -> str:
        device = self._resolve(query, cache.cached())
        store = get_favorites_store()
        if not store.is_favorite(device.id):
            return error(f"{_name(device)} is not a favorite; favorite it first")
        text = text.strip()
        store.set_note(device.id, text)
        if text:
            return success(f"Noted {_name(device)}: [dim]{safe(text)}[/]")
        return success(f"Cleared note on {_name(device)}")

    def _ping(self, query: str) -> str:
        device = self._resolve(query, cache.cached())
        label, color = probe_label(probe_device(device), probing=False)
        return f"{_name(device)} [dim]{device.host}[/] [{color}]{label}[/]"

    def _resolve(self, query: str, devices: list[_Device]) -> _Device:
        if not devices:
            raise SDRException("No directory data. Run 'directory refresh' first.")
        exact = [d for d in devices if d.id == query]
        if exact:
            return exact[0]
        q = query.casefold()
        matches = [
            d
            for d in devices
            if q in d.id.casefold() or q in d.host.casefold() or q in d.name.casefold()
        ]
        if not matches:
            raise SDRException(f"No receiver matches '{query}'")
        if len(matches) > 1:
            sample = ", ".join(d.host for d in matches[:5])
            raise SDRException(f"Ambiguous '{query}' matches {len(matches)}: {sample}")
        return matches[0]

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if subcommand in ("favorite", "add", "remove", "flag", "unflag", "note", "ping"):
            return _host_completions(cache.cached(), prefix)
        if subcommand == "unfavorite":
            return _host_completions(get_favorites_store().all(), prefix)
        return []


def _host_completions(
    devices: list[PublicDevice] | list[FavoriteDevice], prefix: str
) -> list[Completion]:
    seen: set[str] = set()
    out: list[Completion] = []
    for d in devices:
        if d.host in seen or not d.host.startswith(prefix):
            continue
        seen.add(d.host)
        out.append(Completion(d.host, d.name[:40]))
    return out


def _row(d: PublicDevice, favorited: bool) -> str:
    star = "[green]★[/]" if favorited else " "
    src = source_label(d.source)
    name = safe(d.name)[:24]
    loc = safe(d.location or "")[:18]
    users = users_label(d)
    bw = bw_label(bandwidth_hz(d))
    snr = str(d.snr) if d.snr is not None else "-"
    label, color = status_text(d)
    return (
        f"{star} [dim]{src:4}[/] [bold]{name:<24}[/] [dim]{loc:<18}[/] "
        f"[cyan]{users:>7}[/] [yellow]{bw:>6}[/] [dim]{snr:>3}[/] [{color}]{label}[/]"
    )


def _name(d: PublicDevice | FavoriteDevice) -> str:
    return f"[bold]{safe(d.name)[:32]}[/]"


def _result_message(result: connect.ConnectResult) -> str:
    text = safe(result.message)
    return success(text) if result.ok else error(text)


def _covers(d: PublicDevice, freq: float) -> bool:
    return d.freq_min is not None and d.freq_max is not None and d.freq_min <= freq <= d.freq_max


def _distance(d: PublicDevice, lat: float, lon: float) -> float:
    if d.lat is None or d.lon is None:
        return float("inf")
    return float(((d.lat - lat) ** 2 + (d.lon - lon) ** 2) ** 0.5)


def _parse_latlon(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise SDRException("--near expects LAT,LON (e.g. 52.1,5.2)")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise SDRException("--near expects numeric LAT,LON")


def _panel_edge(store: UIStore) -> Edge | None:
    layout = store.model.layout
    edges: tuple[Edge, ...] = ("left", "right", "bottom")
    for edge in edges:
        if DIRECTORY_PANEL_ID in getattr(layout, edge).panels:
            return edge
    return None
