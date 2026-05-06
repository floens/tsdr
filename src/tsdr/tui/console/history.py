from __future__ import annotations

from tsdr.core import storage

HISTORY_FILE = "history"

_singleton: CommandHistory | None = None


def get_history() -> CommandHistory:
    global _singleton
    if _singleton is None:
        _singleton = CommandHistory()
    return _singleton


def reset_history_singleton() -> None:
    """Test-only: drop the cached singleton so the next get_history() reloads."""
    global _singleton
    _singleton = None


class CommandHistory:
    """Newline-delimited history persisted under the user config dir.

    Consecutive duplicates are skipped (bash-style). No line cap (fish-style).
    """

    def __init__(self) -> None:
        self._entries: list[str] = []
        text = storage.read_text(HISTORY_FILE)
        if text:
            self._entries = [line for line in text.splitlines() if line]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[str]:
        return self._entries

    def add(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if self._entries and self._entries[-1] == line:
            return
        self._entries.append(line)
        self._persist()

    def _persist(self) -> None:
        storage.write_text(HISTORY_FILE, "\n".join(self._entries) + "\n")

    def walk_back(self, cursor: int | None, query: str) -> tuple[int, str] | None:
        """Find the next older entry containing query (case-insensitive substring).

        cursor=None starts from the newest entry. Otherwise starts at cursor-1.
        """
        q = query.lower()
        start = len(self._entries) - 1 if cursor is None else cursor - 1
        for i in range(start, -1, -1):
            if q in self._entries[i].lower():
                return i, self._entries[i]
        return None

    def walk_forward(self, cursor: int, query: str) -> tuple[int | None, str | None]:
        """Find the next newer entry containing query (case-insensitive substring).

        Returns (None, None) if we walked past the most-recent match (caller
        should restore the saved in-progress buffer).
        """
        q = query.lower()
        for i in range(cursor + 1, len(self._entries)):
            if q in self._entries[i].lower():
                return i, self._entries[i]
        return None, None

    def search(self, query: str, start: int | None, direction: int) -> tuple[int, str] | None:
        """Case-insensitive substring search used by Ctrl-R / Ctrl-S.

        direction=-1 searches older entries (including `start`); +1 searches newer.
        start=None means "begin at the newest entry" for -1 or "begin at 0" for +1.
        """
        if not query or not self._entries:
            return None
        q = query.lower()
        if direction < 0:
            begin = len(self._entries) - 1 if start is None else start
            indices = range(begin, -1, -1)
        else:
            begin = 0 if start is None else start
            indices = range(begin, len(self._entries))
        for i in indices:
            if q in self._entries[i].lower():
                return i, self._entries[i]
        return None
