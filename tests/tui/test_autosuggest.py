from __future__ import annotations

from pathlib import Path

import pytest

from tsdr.core import storage
from tsdr.tui.console import autosuggest
from tsdr.tui.console import history as history_mod


@pytest.fixture(autouse=True)
def _tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "config_dir", lambda: tmp_path)
    history_mod.reset_history_singleton()


def test_empty_value_returns_empty() -> None:
    history_mod.get_history().add("echo hello")
    assert autosuggest.compute_suggestion("") == ""


def test_no_history_returns_empty() -> None:
    assert autosuggest.compute_suggestion("echo") == ""


def test_prefix_hit_returns_tail() -> None:
    h = history_mod.get_history()
    h.add("echo hello world")
    assert autosuggest.compute_suggestion("echo") == " hello world"
    assert autosuggest.compute_suggestion("echo hello") == " world"


def test_most_recent_match_wins() -> None:
    h = history_mod.get_history()
    h.add("echo alpha")
    h.add("list")
    h.add("echo beta")
    assert autosuggest.compute_suggestion("echo") == " beta"


def test_no_extension_skipped() -> None:
    h = history_mod.get_history()
    h.add("echo")
    # Value is exactly an entry; nothing to suggest.
    assert autosuggest.compute_suggestion("echo") == ""


def test_longer_match_still_valid_when_exact_entry_is_newer() -> None:
    h = history_mod.get_history()
    h.add("echo hello")
    h.add("echo")
    # Most recent is "echo" which doesn't extend "echo"; skip it and pick older extension.
    assert autosuggest.compute_suggestion("echo") == " hello"


def test_no_matching_entry_returns_empty() -> None:
    h = history_mod.get_history()
    h.add("list")
    h.add("use")
    assert autosuggest.compute_suggestion("echo") == ""
