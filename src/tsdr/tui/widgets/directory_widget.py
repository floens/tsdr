from __future__ import annotations

import logging
import textwrap

from rich.markup import escape
from rich.segment import Segment, Segments
from rich.style import Style
from rich.text import Text
from textual import events, on, work
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from tsdr.core.directory import cache, connect
from tsdr.core.directory.display import (
    bandwidth_hz,
    bw_label,
    default_sort_key,
    is_full,
    location_display,
    probe_label,
    range_label,
    snr_color,
    source_color,
    source_label,
    status_text,
    users_label,
)
from tsdr.core.directory.favorites import FavoritesStore, get_favorites_store
from tsdr.core.directory.model import PublicDevice, Source
from tsdr.core.directory.probe import ProbeResult, probe_device
from tsdr.tui.inline_edit import InlineEditor
from tsdr.tui.widgets.panel import PanelWidget, set_orientation_classes

logger = logging.getLogger(__name__)


class _DirectoryList(OptionList):
    """OptionList with per-row hover buttons and click-to-expand."""

    can_focus = False  # click-only; the app owns the keyboard

    class RowAction(Message):
        def __init__(self, action: str, device_id: str) -> None:
            self.action = action
            self.device_id = device_id
            super().__init__()

    class RowHover(Message):
        def __init__(self, current: int | None) -> None:
            self.current = current
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hover_index: int | None = None
        self._button_ranges: list[tuple[int, int, str]] = []

    def set_button_ranges(self, ranges: list[tuple[int, int, str]]) -> None:
        self._button_ranges = ranges

    def reset_hover(self) -> None:
        self._hover_index = None
        self._button_ranges = []

    async def _on_click(self, event: events.Click) -> None:
        # Never call super(): OptionList's own action_select/highlight would double-fire.
        event.stop()
        content_y = event.y + self.scroll_offset.y
        lines = self._lines
        if not (0 <= content_y < len(lines)):
            return
        option_index, line_offset = lines[content_y]
        option_id = self.get_option_at_index(option_index).id
        if option_id is None:  # disabled placeholder row ("No receivers")
            return
        if line_offset == _BUTTON_LINE and option_index == self._hover_index:
            for start, end, action in self._button_ranges:
                if start <= event.x < end:
                    self.post_message(self.RowAction(action, option_id))
                    return
        self.post_message(self.RowAction("toggle", option_id))

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        super()._on_mouse_move(event)
        self._set_hover(self._mouse_hovering_over)

    def _on_leave(self, event: events.Leave) -> None:
        super()._on_leave(event)
        self._set_hover(None)

    def _set_hover(self, index: int | None) -> None:
        if index != self._hover_index:
            self._hover_index = index
            self.post_message(self.RowHover(index))


class _FilterBar(Static):
    """The line above the list: shows the filter, or the live buffer while the
    inline editor is active. Clicking it starts a filter-edit session (the app
    owns the keyboard, so there is no focusable Input)."""

    class Clicked(Message):
        pass

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(self.Clicked())


class DirectoryWidget(Vertical, PanelWidget):
    """Docked panel browsing the public receiver directories (SpyServer, KiwiSDR).

    Click-based: hovering a row reveals its Add/Remove · Favorite · Flag · Note
    buttons on the spacing line below it, and clicking the row body expands it to
    show details and probes the receiver's live reachability (a `ping …` line).
    The probe and the directory fetch run on worker threads so the network never
    blocks the UI. Filter and note text are edited inline via `InlineEditor`
    rendered in the top bar (no focusable Textual Input — the app owns keys).
    """

    dock_edge = reactive(None)

    def __init__(self) -> None:
        super().__init__()
        self._filter = ""
        self._devices: list[PublicDevice] = []
        self._rows: list[PublicDevice] = []
        self._pos: dict[str, int] = {}
        self._hover_id: str | None = None
        self._expanded: set[str] = set()
        self._probes: dict[str, ProbeResult] = {}
        self._probing: set[str] = set()
        self._errors: dict[Source, str] = {}
        self._loading = False
        self._last_width = -1
        self._filter_bar = _FilterBar(id="directory-filter")
        self._error_bar = Static(id="directory-error")
        self._list = _DirectoryList(id="directory-list")
        self._editor = InlineEditor(self)

    def compose(self):
        yield self._filter_bar
        yield self._error_bar
        yield self._list

    def on_mount(self) -> None:
        cached = cache.cached()
        if cached:
            self._apply_fetch(cache.FetchResult(devices=cached, errors=cache.cached_errors()))
        else:
            self._loading = True
            self._rebuild()
            self._fetch()
        self._refresh_filter_bar()

    def on_unmount(self) -> None:
        self._editor.cancel()

    def watch_dock_edge(self, edge) -> None:
        set_orientation_classes(self, edge)

    def on_resize(self, event: events.Resize) -> None:
        if event.size.width != self._last_width:
            self._last_width = event.size.width
            self.call_after_refresh(self._rebuild)

    @work(thread=True, exclusive=True, group="directory-fetch")
    def _fetch(self) -> None:
        result = cache.get_directory()
        self.app.call_from_thread(self._apply_fetch, result)

    def _apply_fetch(self, result: cache.FetchResult) -> None:
        self._loading = False
        self._errors = result.errors
        favorites = get_favorites_store()
        self._devices = sorted(
            result.devices, key=lambda d: (not favorites.is_favorite(d.id), default_sort_key(d))
        )
        self._refresh_error_bar()
        self._rebuild()

    def _refresh_error_bar(self) -> None:
        if not self._errors:
            self._error_bar.update("")
            return
        text = Text(end="")
        for i, (name, message) in enumerate(sorted(self._errors.items())):
            if i:
                text.append("\n")
            text.append(f"⚠ {source_label(name)}: {message}", style="red")
        self._error_bar.update(text)

    def _visible(self) -> list[PublicDevice]:
        q = self._filter.casefold()
        if not q:
            return self._devices
        return [
            d
            for d in self._devices
            if q in d.name.casefold()
            or q in d.host.casefold()
            or q in (d.location or "").casefold()
            or q in d.source
        ]

    def _width(self) -> int:
        return int(max(8, self._list.content_size.width - 1))

    def _view_row(
        self, device: PublicDevice, favorites: FavoritesStore, added: set[tuple[str, int]]
    ) -> tuple[Text, list[tuple[int, int, str]]]:
        fav = favorites.get(device.id)
        return _render_row(
            device,
            hovered=device.id == self._hover_id,
            expanded=device.id in self._expanded,
            favorited=fav is not None,
            flagged=favorites.is_flagged(device.id),
            added=connect.device_endpoint(device) in added,
            note=fav.note if fav is not None else None,
            probe=self._probes.get(device.id),
            probing=device.id in self._probing,
            width=self._width(),
        )

    def _rebuild(self) -> None:
        self._list.clear_options()
        self._rows = []
        self._pos = {}
        self._hover_id = None
        self._list.reset_hover()
        favorites = get_favorites_store()
        added = connect.added_endpoints()
        for d in self._visible():
            if d.id in self._pos:  # OptionList rejects duplicate ids
                continue
            self._pos[d.id] = len(self._rows)
            self._rows.append(d)
            self._list.add_option(Option(self._view_row(d, favorites, added)[0], id=d.id))
        if not self._rows:
            if self._loading:
                message = "Fetching public receivers…"
            elif self._filter:
                message = "No matches"
            else:
                message = "No receivers"
            self._list.add_option(_message_option(message))

    def _repaint(self, device_id: str) -> None:
        pos = self._pos.get(device_id)
        if pos is None:
            return
        text, ranges = self._view_row(
            self._rows[pos], get_favorites_store(), connect.added_endpoints()
        )
        self._list.replace_option_prompt_at_index(pos, text)
        if device_id == self._hover_id:
            self._list.set_button_ranges(ranges)

    def _device_id_at(self, index: int | None) -> str | None:
        if index is None or not (0 <= index < len(self._rows)):
            return None
        return self._rows[index].id

    @on(_DirectoryList.RowHover)
    def _on_row_hover(self, event: _DirectoryList.RowHover) -> None:
        previous = self._hover_id
        self._hover_id = self._device_id_at(event.current)
        if previous is not None:
            self._repaint(previous)
        if self._hover_id is not None:
            self._repaint(self._hover_id)
        else:
            self._list.set_button_ranges([])

    @on(_DirectoryList.RowAction)
    def _on_row_action(self, event: _DirectoryList.RowAction) -> None:
        pos = self._pos.get(event.device_id)
        if pos is None:
            return
        device = self._rows[pos]
        favorites = get_favorites_store()
        if event.action == "toggle":
            self._toggle_expand(device.id)
            if device.id in self._expanded:
                self._start_probe(device)
        elif event.action == "add":
            self._show_result(connect.add_directory_device(device), device.id)
        elif event.action == "start":
            self._show_result(connect.start_directory_device(device), device.id)
        elif event.action == "remove":
            self._show_result(connect.remove_directory_device(device), device.id)
        elif event.action == "favorite":
            if favorites.is_favorite(device.id):
                favorites.remove(device.id)
            else:
                favorites.add(device)
            self._repaint(device.id)
        elif event.action == "flag":
            favorites.toggle_flag(device.id)
            self._repaint(device.id)
        elif event.action == "note":
            self._begin_note(device.id)

    def _show_result(self, result: connect.ConnectResult, device_id: str) -> None:
        message = escape(result.message)
        self.app.show_status(message if result.ok else f"[red]{message}[/]")
        self._repaint(device_id)

    @on(_FilterBar.Clicked)
    def _on_filter_bar_clicked(self, event: _FilterBar.Clicked) -> None:
        self._begin_filter()

    def _refresh_filter_bar(self) -> None:
        self._filter_bar.update(self._filter_bar_content())

    def _filter_bar_content(self) -> Segments | Text:
        editor = self._editor
        if editor.active and editor.buffer is not None:
            context = editor.context
            if isinstance(context, tuple):  # ("note", device_id)
                prefix = f"note {self._device_name(str(context[1]))}: "
            else:
                prefix = "filter: "  # match the committed "filter: …" display
            segments = [
                Segment(prefix, Style(color="cyan")),
                *editor.buffer.render_segments(Style()),
            ]
            return Segments(segments)
        if self._filter:
            return Text(f"filter: {self._filter}", style="dim", no_wrap=True)
        return Text("filter receivers… (click to type)", style="dim", no_wrap=True)

    def _device_name(self, device_id: str) -> str:
        fav = get_favorites_store().get(device_id)
        if fav is not None:
            return fav.name[:24]
        pos = self._pos.get(device_id)
        return self._rows[pos].name[:24] if pos is not None else device_id

    def _begin_filter(self) -> None:
        self._editor.start(
            self._filter,
            redraw=self._refresh_filter_bar,
            on_change=self._filter_changed,
            on_commit=lambda _value: None,
            context="filter",
        )

    def _filter_changed(self, value: str) -> None:
        self._filter = value.strip()  # keep the typed case; matching lowercases in _visible
        self._rebuild()

    def _begin_note(self, device_id: str) -> None:
        fav = get_favorites_store().get(device_id)
        if fav is None:
            return
        self._editor.start(
            fav.note or "",
            redraw=self._refresh_filter_bar,
            on_commit=lambda value: self._note_committed(device_id, value),
            context=("note", device_id),
        )

    def _note_committed(self, device_id: str, value: str) -> None:
        get_favorites_store().set_note(device_id, value.strip())
        self._repaint(device_id)

    def _start_probe(self, device: PublicDevice) -> None:
        if device.id in self._probing:
            return
        self._probing.add(device.id)
        self._repaint(device.id)
        self._probe(device)

    @work(thread=True)
    def _probe(self, device: PublicDevice) -> None:
        result = probe_device(device)
        self.app.call_from_thread(self._apply_probe, device.id, result)

    def _apply_probe(self, device_id: str, result: ProbeResult) -> None:
        self._probing.discard(device_id)
        self._probes[device_id] = result
        self._repaint(device_id)

    def _toggle_expand(self, device_id: str) -> None:
        if device_id not in self._pos:
            return
        self._expanded.symmetric_difference_update({device_id})
        self._repaint(device_id)
        # replace_option_prompt repaints but doesn't reflow; force layout for the taller row.
        self._list.refresh(layout=True)


def _message_option(text: str) -> Option:
    return Option(Text(text, style="dim italic"), disabled=True)


def _line1(
    d: PublicDevice, *, favorited: bool, flagged: bool, added: bool, note: str | None, width: int
) -> Text:
    """`[marks] SRC  name … loc`.

    Status marks lead the line: `A` currently added to tsdr (by host:port), `F`
    favorited, `D` flagged dead. A favorite's note, when set, stands in for the
    name (the full name is still shown in the expanded detail). The (country-first)
    location is right-aligned, capped at half the available width so a long one
    can't crowd out the name; when it's shorter than half, the name takes the rest.
    """
    line = Text(no_wrap=True, end="")
    if added:
        line.append("A", style="cyan")
    if favorited:
        line.append("F", style="green")
    if flagged:
        line.append("D", style="dim")
    if line.cell_len:
        line.append(" ")
    line.append(source_label(d.source), style=source_color(d.source))
    line.append(" ")
    avail = max(2, width - line.cell_len)
    loc = location_display(d)
    name = Text(note or d.name, style="bold", end="")
    if loc:
        loc_text = Text(loc, style="dim", end="")
        loc_text.truncate(max(1, avail // 2), overflow="ellipsis")
        name.truncate(max(1, avail - loc_text.cell_len - 1), overflow="ellipsis")
        line.append_text(name)
        line.append(" " * max(1, width - line.cell_len - loc_text.cell_len))
        line.append_text(loc_text)
    else:
        name.truncate(avail, overflow="ellipsis")
        line.append_text(name)
    return line


def _line2(d: PublicDevice, width: int) -> Text:
    """`IQ  RANGE  USERS  SNR  ERROR` — IQ bandwidth first; error fills the rest.

    Location moved to line 1, so line 2 is uniform across sources.
    """
    line = Text(no_wrap=True, end="")
    line.append(bw_label(bandwidth_hz(d)), style="yellow")
    line.append(" ")
    line.append(range_label(d), style="dim")
    line.append(" ")
    line.append(users_label(d), style="red" if is_full(d) else "cyan")
    line.append(" ")
    line.append(
        str(d.snr) if d.snr is not None else "-",
        style=snr_color(d.snr) if d.snr is not None else "dim",
    )
    # Skip "full": the USERS column already shows that in red.
    if not d.usable and not d.usable_reason.startswith("full"):
        line.append(" ")
        error = Text(d.usable_reason, style="red", end="")
        error.truncate(max(0, width - line.cell_len), overflow="ellipsis")
        line.append_text(error)
    return line


_BUTTON_LINE = 2  # line offset within an option where hover buttons render
_BUTTON_GAP = "   "


def _row_buttons(*, added: bool, favorited: bool, flagged: bool) -> list[tuple[str, str, str]]:
    """(action, label, color) triples for a row's hover buttons. The primary slot
    swaps Add↔Remove on whether the receiver is already added; an added receiver also
    gets Start (retune to the active frequency + start). Note only appears once
    favorited."""
    buttons: list[tuple[str, str, str]] = []
    if added:
        buttons.append(("remove", "Remove", "red"))
        buttons.append(("start", "Start", "green"))
    else:
        buttons.append(("add", "Add", "cyan"))
    buttons.append(("favorite", "Unfavorite" if favorited else "Favorite", "green"))
    buttons.append(("flag", "Unflag" if flagged else "Flag", "dim"))
    if favorited:
        buttons.append(("note", "Note", "green"))
    return buttons


def _buttons_line(
    buttons: list[tuple[str, str, str]],
) -> tuple[Text, list[tuple[int, int, str]]]:
    """Render the hover-button line and its click-hit column ranges together, so
    the painted labels and the clickable spans measure from the same text and
    can't drift apart."""
    line = Text(no_wrap=True, end="")
    ranges: list[tuple[int, int, str]] = []
    for i, (action, label, color) in enumerate(buttons):
        if i:
            line.append(_BUTTON_GAP)
        start = line.cell_len
        line.append(label, style=f"{color} underline")  # style string, so "dim" works
        ranges.append((start, line.cell_len, action))
    return line, ranges


def _render_row(
    d: PublicDevice,
    *,
    hovered: bool,
    expanded: bool,
    favorited: bool,
    flagged: bool,
    added: bool,
    note: str | None,
    probe: ProbeResult | None,
    probing: bool,
    width: int,
) -> tuple[Text, list[tuple[int, int, str]]]:
    # The button line is always present (blank when idle) so hover never reflows the list.
    button_line = Text(no_wrap=True, end="")
    ranges: list[tuple[int, int, str]] = []
    if hovered or expanded:
        button_line, ranges = _buttons_line(
            _row_buttons(added=added, favorited=favorited, flagged=flagged)
        )
    lines = [
        _line1(d, favorited=favorited, flagged=flagged, added=added, note=note, width=width),
        _line2(d, width),
        button_line,
    ]
    if expanded:
        lines.extend(_detail_lines(d, probe, probing, note, width))
        lines.append(Text(end=""))  # one cell of padding below the details

    text = Text(no_wrap=True, end="")
    for i, line in enumerate(lines):
        if i:
            text.append("\n")
        text.append_text(line)
    if flagged:
        text.stylize("dim")
    return text, ranges


def _detail(text: str, width: int, *, style: str = "dim", indent: bool = True) -> Text:
    """One detail row: two cells of left padding (except the header), no wrap,
    truncated to width."""
    line = Text(no_wrap=True, end="")
    if indent:
        line.append("  ")
    line.append(text, style=style)
    line.truncate(width, overflow="ellipsis")
    return line


def _detail_lines(
    d: PublicDevice, probe: ProbeResult | None, probing: bool, note: str | None, width: int
) -> list[Text]:
    """The click-to-expand block: everything about the receiver, untruncated fields
    (name/location get cut on the summary lines) laid out one topic per line. The
    endpoint header sits flush; every line below it is indented under it. The live
    probe result (once pinged) sits just under the endpoint, then the favorite's
    note if any. The full name is shown here (wrapped), since the summary line
    truncates it."""
    lines = [_detail(f"↳ {d.url or f'{d.host}:{d.port}'}", width, indent=False)]
    if probing or probe is not None:
        label, color = probe_label(probe, probing=probing)
        lines.append(_detail(f"ping {label}", width, style=color))
    if note:
        lines.append(_detail(f"note: {note}", width, style="green"))
    for chunk in textwrap.wrap(d.name, max(1, width - 2)) or [d.name]:
        lines.append(_detail(chunk, width))

    location = d.location or (
        f"{d.lat:.4f}, {d.lon:.4f}" if d.lat is not None and d.lon is not None else ""
    )
    if location:
        lines.append(_detail(location, width))

    if d.freq_min is not None and d.freq_max is not None:
        lines.append(_detail(f"{d.freq_min / 1e6:.3f}–{d.freq_max / 1e6:.3f} MHz", width))

    caps: list[str] = []
    if d.sample_rate:
        caps.append(f"IQ {bw_label(d.sample_rate)}")
    if d.bandwidth:
        caps.append(f"BW {bw_label(d.bandwidth)}")
    if d.channels:
        caps.append(f"{d.channels} ch")
    if d.device_hw:
        caps.append(d.device_hw)
    if d.grid:
        caps.append(d.grid)
    if caps:
        lines.append(_detail(" · ".join(caps), width))

    stats: list[str] = []
    if d.users is not None and d.users_max is not None:
        stats.append(f"{d.users}/{d.users_max} users")
    if d.snr is not None:
        stats.append(f"SNR {d.snr}")
    stats.append(status_text(d)[0])
    lines.append(_detail(" · ".join(stats), width))
    return lines
