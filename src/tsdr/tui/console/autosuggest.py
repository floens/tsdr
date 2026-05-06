from __future__ import annotations

from tsdr.tui.console.history import get_history


def compute_suggestion(value: str) -> str:
    """Return the chars to render after `value` as a dim ghost suggestion.

    Picks the most recent history entry that strictly extends `value`.
    Returns "" when value is empty, history is empty, or no entry extends it.
    """
    if not value:
        return ""
    entries = get_history().entries
    for entry in reversed(entries):
        if entry.startswith(value) and len(entry) > len(value):
            return entry[len(value) :]
    return ""
