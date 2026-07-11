"""InlineEditor lifecycle without mounting Textual — a stub host stands in for the
widget/app, and a fake key event carries just the attributes handle_key reads."""

from tsdr.tui.inline_edit import InlineEditor


class _StubApp:
    def __init__(self) -> None:
        self.active_inline_editor: InlineEditor | None = None


class _StubTimer:
    def stop(self) -> None: ...
    def reset(self) -> None: ...


class _StubHost:
    def __init__(self) -> None:
        self.app = _StubApp()

    def set_interval(self, interval: float, callback) -> _StubTimer:
        return _StubTimer()


class _FakeKey:
    def __init__(self, key: str, character: str | None = None) -> None:
        self.key = key
        self.character = character
        self.is_printable = bool(character) and character.isprintable()

    def prevent_default(self) -> None: ...


def test_typing_editing_and_commit() -> None:
    host = _StubHost()
    editor = InlineEditor(host)
    committed: list[str] = []
    editor.start("ab", redraw=lambda: None, on_commit=committed.append)

    assert host.app.active_inline_editor is editor
    editor.handle_key(_FakeKey("c", "c"))
    assert editor.buffer.value == "abc"
    editor.handle_key(_FakeKey("backspace"))
    assert editor.buffer.value == "ab"
    editor.handle_key(_FakeKey("left"))
    editor.handle_key(_FakeKey("x", "x"))
    assert editor.buffer.value == "axb"

    editor.handle_key(_FakeKey("enter"))
    assert committed == ["axb"]
    assert not editor.active
    assert host.app.active_inline_editor is None


def test_on_change_fires_live_and_cancel_reverts() -> None:
    host = _StubHost()
    editor = InlineEditor(host)
    changes: list[str] = []
    cancelled: list[bool] = []
    editor.start(
        "",
        redraw=lambda: None,
        on_commit=lambda _v: None,
        on_change=changes.append,
        on_cancel=lambda: cancelled.append(True),
    )

    editor.handle_key(_FakeKey("h", "h"))
    editor.handle_key(_FakeKey("i", "i"))
    assert changes == ["h", "hi"]

    editor.handle_key(_FakeKey("escape"))
    assert cancelled == [True]
    assert not editor.active


def test_start_takeover_is_silent() -> None:
    host = _StubHost()
    first = InlineEditor(host)
    second = InlineEditor(host)
    cancelled: list[str] = []
    first.start(
        "a",
        redraw=lambda: None,
        on_commit=lambda _v: None,
        on_cancel=lambda: cancelled.append("first"),
    )

    second.start("b", redraw=lambda: None, on_commit=lambda _v: None)
    assert cancelled == []  # a takeover fires neither commit nor cancel
    assert not first.active
    assert host.app.active_inline_editor is second
