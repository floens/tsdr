import asyncio

from textual.app import App, ComposeResult

from tsdr.core.directory import cache, connect
from tsdr.core.directory.favorites import init_favorites_store
from tsdr.core.directory.model import PublicDevice
from tsdr.core.directory.probe import ProbeResult
from tsdr.tui.widgets.directory_widget import (
    _BUTTON_LINE,
    DirectoryWidget,
    _buttons_line,
    _render_row,
    _row_buttons,
)

_DEVICE = PublicDevice(source="spyserver", id="s", name="x", host="h", usable=True)


def _row(
    *,
    hovered: bool,
    favorited: bool = False,
    flagged: bool = False,
    added: bool = False,
    note: str | None = None,
    expanded: bool = False,
    probe: ProbeResult | None = None,
    probing: bool = False,
    device: PublicDevice = _DEVICE,
) -> list[str]:
    text, _ = _render_row(
        device,
        hovered=hovered,
        expanded=expanded,
        favorited=favorited,
        flagged=flagged,
        added=added,
        note=note,
        probe=probe,
        probing=probing,
        width=80,
    )
    return text.plain.split("\n")


def _actions(**kwargs) -> list[str]:
    return [action for action, _label, _color in _row_buttons(**kwargs)]


def _flag_label(flagged: bool) -> str:
    buttons = _row_buttons(added=False, favorited=False, flagged=flagged)
    return next(label for action, label, _color in buttons if action == "flag")


def test_add_remove_swaps_on_added() -> None:
    assert _row_buttons(added=False, favorited=False, flagged=False)[0] == ("add", "Add", "cyan")
    assert _row_buttons(added=True, favorited=False, flagged=False)[0] == (
        "remove",
        "Remove",
        "red",
    )


def test_start_button_only_when_added() -> None:
    assert "start" in _actions(added=True, favorited=False, flagged=False)
    assert "start" not in _actions(added=False, favorited=False, flagged=False)


def test_button_colors_align_with_markers() -> None:
    colors = {a: c for a, _label, c in _row_buttons(added=True, favorited=True, flagged=True)}
    assert colors["favorite"] == "green"
    assert colors["flag"] == "dim"
    assert _row(
        hovered=True, added=True, favorited=True, flagged=True
    )  # renders "dim" without error


def test_note_button_only_when_favorited() -> None:
    assert "note" not in _actions(added=False, favorited=False, flagged=False)
    assert "note" in _actions(added=False, favorited=True, flagged=False)


def test_flag_label_toggles() -> None:
    assert _flag_label(False) == "Flag"
    assert _flag_label(True) == "Unflag"


def test_buttons_line_ranges_match_labels() -> None:
    """The rendered hover-button line must place each label exactly where its
    click-hit range says, or position-based dispatch fires the wrong action."""
    buttons = _row_buttons(added=True, favorited=True, flagged=True)
    line, ranges = _buttons_line(buttons)
    for (start, end, action), (b_action, label, _color) in zip(ranges, buttons, strict=True):
        assert action == b_action
        assert line.plain[start:end] == label


def test_idle_row_has_no_buttons() -> None:
    assert _row(hovered=False)[_BUTTON_LINE] == ""


def test_expanded_row_shows_buttons_without_hover() -> None:
    assert "Add" in _row(hovered=False, expanded=True)[_BUTTON_LINE]


def test_favorite_mark_leads_line() -> None:
    line1 = _row(hovered=False, favorited=True)[0]
    assert line1.startswith("F")
    assert not line1.rstrip().endswith("F")


def test_added_mark_leads_line() -> None:
    assert _row(hovered=False, added=True)[0].startswith("A")
    assert "A" not in _row(hovered=False, added=False)[0]


def test_marks_share_the_leading_slot() -> None:
    line1 = _row(hovered=False, added=True, favorited=True, flagged=True)[0]
    assert line1.startswith("AFD ")


def test_note_replaces_name_on_summary_line() -> None:
    line1 = _row(hovered=False, favorited=True, note="home base")[0]
    assert "home base" in line1
    assert _DEVICE.name not in line1


def test_flagged_row_marked_and_dim() -> None:
    text, _ = _render_row(
        _DEVICE,
        hovered=False,
        expanded=False,
        favorited=False,
        flagged=True,
        added=False,
        note=None,
        probe=None,
        probing=False,
        width=80,
    )
    assert text.plain.split("\n")[0].startswith("D")
    assert any(span.start == 0 and span.style == "dim" for span in text.spans)


def test_note_shown_in_details() -> None:
    lines = _row(hovered=False, expanded=True, favorited=True, note="great at night")
    assert any("note: great at night" in line for line in lines)


def test_details_layout() -> None:
    device = PublicDevice(
        source="spyserver",
        id="s",
        name="A Long Receiver Name",
        host="h",
        port=5555,
        freq_min=0.0,
        freq_max=30_000_000.0,
        usable=True,
    )
    lines = _row(
        hovered=False, expanded=True, device=device, probe=ProbeResult(reachable=True, rtt_ms=12.0)
    )
    details = lines[3:]  # after line1, line2, buttons
    assert details[0] == "↳ h:5555"
    assert details[1] == "  ping 12 ms"
    assert details[2] == "  A Long Receiver Name"
    assert lines[-1] == ""  # one cell of padding below the details


def test_spyserver_location_right_aligned() -> None:
    # SpyServer's short country code hugs the right edge (name fills the left).
    device = PublicDevice(
        source="spyserver", id="s", name="Airspy HF+", host="h", lat=52.37, lon=4.9, usable=True
    )
    line1 = _row(hovered=False, device=device)[0]
    assert line1.startswith("spy Airspy HF+")
    assert line1.endswith("NL")
    assert line1.rstrip("N L") != line1  # padding sits between name and code


def test_kiwisdr_location_right_aligned() -> None:
    # KiwiSDR also right-aligns; a short location leaves the name the remainder.
    device = PublicDevice(source="kiwisdr", id="k", name="Rx", host="h", location="Amsterdam")
    line1 = _row(hovered=False, device=device)[0]
    assert line1.startswith("kiwi Rx")
    assert line1.endswith("Amsterdam")


class _Harness(App):
    active_inline_editor = None  # KeyboardMixin provides this on the real app

    def compose(self) -> ComposeResult:
        yield DirectoryWidget()


class _FakeKey:
    def __init__(self, key: str, character: str | None = None) -> None:
        self.key = key
        self.character = character
        self.is_printable = bool(character) and character.isprintable()

    def prevent_default(self) -> None: ...


def test_filter_persists_across_escape(monkeypatch) -> None:
    """Typing a filter, pressing Escape, then reopening keeps the text: Escape
    leaves edit mode without discarding the applied filter."""
    init_favorites_store()
    monkeypatch.setattr(cache, "cached", lambda: [_DEVICE])
    monkeypatch.setattr(connect, "added_endpoints", lambda: set())
    captured: dict[str, object] = {}

    async def go() -> None:
        async with _Harness().run_test(size=(80, 24)) as pilot:
            w = pilot.app.query_one(DirectoryWidget)
            w._begin_filter()
            for ch in "alpha":
                w._editor.handle_key(_FakeKey(ch, ch))
            assert w._filter == "alpha"
            w._editor.handle_key(_FakeKey("escape"))
            assert not w._editor.active
            captured["after_escape"] = w._filter
            w._begin_filter()
            captured["reopened"] = w._editor.buffer.value

    asyncio.run(go())
    assert captured["after_escape"] == "alpha"
    assert captured["reopened"] == "alpha"
