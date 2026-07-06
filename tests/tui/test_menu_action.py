from __future__ import annotations

from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.console.command_input import _menu_action, _should_auto_open


def test_no_items_does_nothing() -> None:
    assert _menu_action([], auto_apply=True) == "none"
    assert _menu_action([], auto_apply=False) == "none"


def test_single_item_tab_applies() -> None:
    items = [MenuItem("rtl0", "", (0,))]
    assert _menu_action(items, auto_apply=True) == "apply"


def test_single_item_auto_open_shows_menu() -> None:
    """Auto-open with a single match must NOT silently complete.

    Regression: after backspacing back to `use `, auto-open would fire and,
    because rtl0 was the only device, auto-apply would revert the edit.
    """
    items = [MenuItem("rtl0", "", (0,))]
    assert _menu_action(items, auto_apply=False) == "show"


def test_multiple_items_shows_menu() -> None:
    items = [MenuItem("wfm", "", ()), MenuItem("nfm", "", ())]
    assert _menu_action(items, auto_apply=True) == "show"
    assert _menu_action(items, auto_apply=False) == "show"


# Auto-open trigger rules


def test_auto_open_fires_after_space() -> None:
    assert _should_auto_open("demod ", dismissed_for="", has_completions=True)


def test_auto_open_fires_mid_token() -> None:
    """Typing `demod A` should narrow the menu, not hide it."""
    assert _should_auto_open("demod A", dismissed_for="", has_completions=True)


def test_auto_open_suppressed_on_empty_line() -> None:
    assert not _should_auto_open("", dismissed_for="", has_completions=True)
    assert not _should_auto_open("   ", dismissed_for="", has_completions=True)


def test_auto_open_suppressed_when_no_completions() -> None:
    assert not _should_auto_open("demod xyz", dismissed_for="", has_completions=False)


def test_auto_open_suppressed_when_dismissed() -> None:
    assert not _should_auto_open("demod ", dismissed_for="demod ", has_completions=True)
    # but a different value re-enables
    assert _should_auto_open("demod A", dismissed_for="demod ", has_completions=True)
