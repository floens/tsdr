from __future__ import annotations

from pathlib import Path

import pytest

import tsdr.tui.console.autosuggest as autosuggest
import tsdr.tui.console.terminal_input as ti_mod
from tsdr.core import storage
from tsdr.tui.console import history as history_mod
from tsdr.tui.console.terminal_input import TerminalInput


def _make_input() -> TerminalInput:
    t = TerminalInput(id="test-input")
    t.post_message = lambda _msg: None  # type: ignore[assignment]
    t.refresh = lambda *a, **k: None  # type: ignore[assignment]
    return t


def _setv(t: TerminalInput, value: str, cursor: int | None = None) -> None:
    t._value = value
    t._cursor_pos = cursor if cursor is not None else len(value)


@pytest.fixture(autouse=True)
def _tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "config_dir", lambda: tmp_path)
    history_mod.reset_history_singleton()


# word movement


def test_word_boundary_left_skips_nonword_then_word() -> None:
    t = _make_input()
    _setv(t, "echo hello world", cursor=16)
    assert t._word_boundary_left(16) == 11  # before "world"
    assert t._word_boundary_left(11) == 5  # before "hello"
    assert t._word_boundary_left(5) == 0


def test_word_boundary_right_skips_nonword_then_word() -> None:
    t = _make_input()
    _setv(t, "echo hello world", cursor=0)
    assert t._word_boundary_right(0) == 4  # end of "echo"
    assert t._word_boundary_right(4) == 10  # end of "hello"
    assert t._word_boundary_right(10) == 16


def test_move_word_left_and_right() -> None:
    t = _make_input()
    _setv(t, "echo hello world", cursor=7)
    t._move_word_left()
    assert t._cursor_pos == 5
    t._move_word_right()
    assert t._cursor_pos == 10


# deletion


def test_delete_word_left_stores_kill() -> None:
    t = _make_input()
    _setv(t, "echo hello", cursor=10)
    t._delete_word_left()
    assert t._value == "echo "
    assert t._cursor_pos == 5
    assert t._last_kill == "hello"


def test_delete_word_right_stores_kill() -> None:
    t = _make_input()
    _setv(t, "echo hello", cursor=5)
    t._delete_word_right()
    assert t._value == "echo "
    assert t._cursor_pos == 5
    assert t._last_kill == "hello"


def test_kill_to_end() -> None:
    t = _make_input()
    _setv(t, "echo hello", cursor=5)
    t._kill_to_end()
    assert t._value == "echo "
    assert t._cursor_pos == 5
    assert t._last_kill == "hello"


def test_kill_to_start() -> None:
    t = _make_input()
    _setv(t, "echo hello", cursor=5)
    t._kill_to_start()
    assert t._value == "hello"
    assert t._cursor_pos == 0
    assert t._last_kill == "echo "


def test_yank_inserts_at_cursor() -> None:
    t = _make_input()
    _setv(t, "echo hello", cursor=5)
    t._kill_to_end()
    # Now _value="echo ", cursor at 5, last_kill="hello"
    _setv(t, "ab", cursor=1)
    t._yank()
    assert t._value == "ahellob"
    assert t._cursor_pos == 6


def test_transpose_chars_mid_line() -> None:
    t = _make_input()
    _setv(t, "abcd", cursor=2)
    t._transpose_chars()
    assert t._value == "acbd"
    assert t._cursor_pos == 3


def test_transpose_chars_at_end() -> None:
    t = _make_input()
    _setv(t, "abcd", cursor=4)
    t._transpose_chars()
    assert t._value == "abdc"
    assert t._cursor_pos == 4


def test_transpose_chars_at_start_noop() -> None:
    t = _make_input()
    _setv(t, "abcd", cursor=0)
    t._transpose_chars()
    assert t._value == "abcd"


def test_transpose_words() -> None:
    t = _make_input()
    _setv(t, "hello world", cursor=11)
    t._transpose_words()
    assert t._value == "world hello"


def test_transpose_words_mid() -> None:
    t = _make_input()
    _setv(t, "one two three", cursor=5)  # inside "two"
    t._transpose_words()
    assert t._value == "two one three"


# history navigation


def test_history_up_down_prefix_search() -> None:
    t = _make_input()
    h = history_mod.get_history()
    for line in ["echo a", "list", "echo b", "focus 0"]:
        h.add(line)

    _setv(t, "echo", cursor=4)
    t.history_up()
    assert t._value == "echo b"
    t.history_up()
    assert t._value == "echo a"
    t.history_up()  # no more matches
    assert t._value == "echo a"

    t.history_down()
    assert t._value == "echo b"
    t.history_down()
    # past most-recent -- restore saved buffer
    assert t._value == "echo"


def test_history_up_empty_buffer_walks_all() -> None:
    t = _make_input()
    h = history_mod.get_history()
    for line in ["a", "b", "c"]:
        h.add(line)

    _setv(t, "", cursor=0)
    t.history_up()
    assert t._value == "c"
    t.history_up()
    assert t._value == "b"
    t.history_up()
    assert t._value == "a"


def test_history_no_entries() -> None:
    t = _make_input()
    _setv(t, "", cursor=0)
    t.history_up()
    assert t._value == ""  # no-op


# reverse-incremental search


def test_enter_search_and_extend() -> None:
    t = _make_input()
    h = history_mod.get_history()
    for line in ["echo a", "list", "echo foo", "focus 0", "foo bar"]:
        h.add(line)

    _setv(t, "", cursor=0)
    t.enter_search()
    assert t.in_search

    # Extend query with 'f'
    t._search.query = "foo"
    t._search_find(direction=-1, from_scratch=True)
    assert t._value == "foo bar"

    # Step to next older match
    t._search_step(direction=-1)
    assert t._value == "echo foo"


def test_search_no_match_sets_failed() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "", cursor=0)
    t.enter_search()
    t._search.query = "zzz"
    t._search_find(direction=-1, from_scratch=True)
    assert t._search.failed is True


def test_exit_search_keep_match() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "orig", cursor=4)
    t.enter_search()
    t._search.query = "hello"
    t._search_find(direction=-1, from_scratch=True)
    t._exit_search(keep_match=True)
    assert not t.in_search
    assert t._value == "echo hello"


def test_exit_search_cancel_restores_original() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "orig", cursor=2)
    t.enter_search()
    t._search.query = "hello"
    t._search_find(direction=-1, from_scratch=True)
    t._exit_search(keep_match=False)
    assert not t.in_search
    assert t._value == "orig"
    assert t._cursor_pos == 2


# Autosuggestions


def test_suggestion_computed_on_set_value() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello world")

    _setv(t, "", cursor=0)
    t._set_value("echo", 4)
    assert t._suggestion == " hello world"


def test_suggestion_stickiness_matching_char() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "", cursor=0)
    t._set_value("ec", 2)
    assert t._suggestion == "ho hello"

    call_count = [0]
    original = autosuggest.compute_suggestion

    def counting(v: str) -> str:
        call_count[0] += 1
        return original(v)

    ti_mod.autosuggest.compute_suggestion = counting  # type: ignore[assignment]
    try:
        t._set_value("ech", 3)
        assert t._suggestion == "o hello"
        assert call_count[0] == 0  # stickiness avoids recompute
    finally:
        ti_mod.autosuggest.compute_suggestion = original  # type: ignore[assignment]


def test_suggestion_non_matching_char_recomputes() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "", cursor=0)
    t._set_value("ec", 2)
    t._set_value("ez", 2)
    assert t._suggestion == ""  # "ez" doesn't match any history entry


def test_suggestion_cleared_during_search_edits() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "", cursor=0)
    t._set_value("ec", 2)
    assert t._suggestion == "ho hello"
    t.enter_search()
    # Any value change while in search mode should clear the suggestion.
    t._set_value("ech", 3)
    assert t._suggestion == ""


def test_accept_suggestion_full() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "", cursor=0)
    t._set_value("echo", 4)
    t._accept_suggestion_full()
    assert t._value == "echo hello"
    assert t._cursor_pos == 10


def test_accept_suggestion_full_noop_mid_line() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello")

    _setv(t, "", cursor=0)
    t._set_value("echo", 4)
    t._cursor_pos = 2  # mid-line
    t._accept_suggestion_full()
    assert t._value == "echo"  # unchanged


def test_accept_suggestion_word() -> None:
    t = _make_input()
    h = history_mod.get_history()
    h.add("echo hello world")

    _setv(t, "", cursor=0)
    t._set_value("echo", 4)
    assert t._suggestion == " hello world"
    t._accept_suggestion_word()
    assert t._value == "echo hello"
    assert t._cursor_pos == 10
    # Remaining suggestion preserved for a second press
    assert t._suggestion == " world"
