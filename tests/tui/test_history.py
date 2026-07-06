from __future__ import annotations

from pathlib import Path

import pytest

from tsdr.core import storage
from tsdr.tui.console import history as history_mod
from tsdr.tui.console.history import HISTORY_FILE, CommandHistory


@pytest.fixture(autouse=True)
def _tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "config_dir", lambda: tmp_path)
    history_mod.reset_history_singleton()


def test_empty_history() -> None:
    h = CommandHistory()
    assert len(h) == 0
    assert h.walk_back(None, "") is None
    assert h.search("anything", None, -1) is None


def test_add_and_persist(tmp_path: Path) -> None:
    h = CommandHistory()
    h.add("echo hello")
    h.add("list")
    assert (tmp_path / HISTORY_FILE).exists()

    reloaded = CommandHistory()
    assert reloaded.entries == ["echo hello", "list"]


def test_consecutive_dedup() -> None:
    h = CommandHistory()
    h.add("echo hello")
    h.add("echo hello")
    h.add("list")
    h.add("echo hello")
    assert h.entries == ["echo hello", "list", "echo hello"]


def test_empty_line_ignored() -> None:
    h = CommandHistory()
    h.add("")
    h.add("   ")
    assert len(h) == 0


def test_walk_back_prefix() -> None:
    h = CommandHistory()
    for line in ["echo a", "list", "echo b", "use 0", "echo c"]:
        h.add(line)

    r = h.walk_back(None, "echo")
    assert r == (4, "echo c")
    r = h.walk_back(4, "echo")
    assert r == (2, "echo b")
    r = h.walk_back(2, "echo")
    assert r == (0, "echo a")
    assert h.walk_back(0, "echo") is None


def test_walk_back_no_prefix() -> None:
    h = CommandHistory()
    for line in ["a", "b", "c"]:
        h.add(line)

    r = h.walk_back(None, "")
    assert r == (2, "c")
    r = h.walk_back(2, "")
    assert r == (1, "b")


def test_walk_back_substring_match() -> None:
    """Query matches anywhere in the entry, not just the start."""
    h = CommandHistory()
    for line in ["echo bar foo baz", "list", "use rtl0"]:
        h.add(line)

    r = h.walk_back(None, "foo")
    assert r == (0, "echo bar foo baz")


def test_walk_back_case_insensitive() -> None:
    h = CommandHistory()
    h.add("echo Foo")

    assert h.walk_back(None, "foo") == (0, "echo Foo")
    assert h.walk_back(None, "FOO") == (0, "echo Foo")


def test_walk_forward_case_insensitive() -> None:
    h = CommandHistory()
    h.add("echo Foo")
    h.add("list")
    h.add("echo FOObar")

    # Jumping from index 0 forward with lowercase query
    assert h.walk_forward(0, "foo") == (2, "echo FOObar")


def test_walk_forward_prefix() -> None:
    h = CommandHistory()
    for line in ["echo a", "list", "echo b", "use 0", "echo c"]:
        h.add(line)

    r = h.walk_forward(0, "echo")
    assert r == (2, "echo b")
    r = h.walk_forward(2, "echo")
    assert r == (4, "echo c")
    assert h.walk_forward(4, "echo") == (None, None)


def test_search_reverse() -> None:
    h = CommandHistory()
    for line in ["echo a", "list", "echo foo", "use 0", "foo bar"]:
        h.add(line)

    # Reverse search from newest
    r = h.search("foo", None, -1)
    assert r == (4, "foo bar")
    r = h.search("foo", 3, -1)
    assert r == (2, "echo foo")
    assert h.search("foo", 1, -1) is None


def test_search_forward() -> None:
    h = CommandHistory()
    for line in ["echo a", "list", "echo foo", "use 0", "foo bar"]:
        h.add(line)

    r = h.search("foo", 0, 1)
    assert r == (2, "echo foo")
    r = h.search("foo", 3, 1)
    assert r == (4, "foo bar")
    assert h.search("foo", 5, 1) is None


def test_search_empty_query() -> None:
    h = CommandHistory()
    h.add("anything")
    assert h.search("", None, -1) is None
