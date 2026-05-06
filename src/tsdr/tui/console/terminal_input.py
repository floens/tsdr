from __future__ import annotations

from dataclasses import dataclass, field

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from tsdr.tui.console import autosuggest
from tsdr.tui.console.highlight import highlight_command
from tsdr.tui.console.history import CommandHistory, get_history

DEFAULT_PROMPT: list[tuple[str, Style]] = [("$ ", Style(color="white", dim=True))]
_CURSOR_STYLE = Style(reverse=True)
_SEARCH_PROMPT_STYLE = Style(color="white", dim=True)
_SEARCH_FAIL_STYLE = Style(color="red", dim=True)
_GHOST_STYLE = Style(dim=True, italic=True)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


@dataclass
class _SearchState:
    query: str = ""
    match_index: int | None = None
    match_value: str | None = None
    original_value: str = ""
    original_cursor_pos: int = 0
    original_prompt: list[tuple[str, Style]] = field(default_factory=list)
    failed: bool = False


@dataclass
class _HistoryNav:
    cursor: int
    saved_buffer: str
    query: str


class TerminalInput(Widget):
    """Single-line input that renders prompt + text + blinking cursor via render_line()."""

    can_focus = True

    class Changed(Message):
        def __init__(self, value: str, input: TerminalInput) -> None:
            super().__init__()
            self.value = value
            self.input = input

        @property
        def control(self) -> TerminalInput:
            return self.input

    class Submitted(Message):
        def __init__(self, value: str, input: TerminalInput) -> None:
            super().__init__()
            self.value = value
            self.input = input

        @property
        def control(self) -> TerminalInput:
            return self.input

    def __init__(
        self,
        placeholder: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._value = ""
        self._cursor_pos = 0
        self._cursor_visible = True
        self._blink_timer = None
        self._active = False
        self.placeholder = placeholder
        self._prompt_segments = DEFAULT_PROMPT

        self._last_kill: str = ""
        self._history_nav: _HistoryNav | None = None
        self._search: _SearchState | None = None
        self._highlight_cache: tuple[str, list[tuple[str, Style]]] = ("", [])
        self._suggestion: str = ""
        self._suggestion_source: str = ""

    @property
    def _history(self) -> CommandHistory:
        return get_history()

    @property
    def prompt_len(self) -> int:
        return sum(len(text) for text, _ in self._prompt_segments)

    @property
    def prompt_segments(self) -> list[tuple[str, Style]]:
        return self._prompt_segments

    @prompt_segments.setter
    def prompt_segments(self, segments: list[tuple[str, Style]]) -> None:
        self._prompt_segments = segments
        self.refresh()

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        if value == self._active:
            return
        self._active = value
        if value:
            self._cursor_visible = True
            self._cursor_pos = len(self._value)
            if self._blink_timer is None:
                self._blink_timer = self.set_interval(0.53, self._toggle_cursor)
        else:
            if self._blink_timer is not None:
                self._blink_timer.stop()
                self._blink_timer = None
            self._cursor_visible = True
        self.refresh()

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, new_value: str) -> None:
        if new_value != self._value:
            self._value = new_value
            self._cursor_pos = min(self._cursor_pos, len(new_value))
            self._update_suggestion()
            self.post_message(self.Changed(new_value, self))
            self.refresh()

    @property
    def cursor_position(self) -> int:
        return self._cursor_pos

    @cursor_position.setter
    def cursor_position(self, pos: int) -> None:
        self._cursor_pos = max(0, min(pos, len(self._value)))
        self.refresh()

    @property
    def in_search(self) -> bool:
        return self._search is not None

    def reset_history_cursor(self) -> None:
        self._history_nav = None

    def _on_focus(self, event: events.Focus) -> None:
        self.refresh()

    def _toggle_cursor(self) -> None:
        self._cursor_visible = not self._cursor_visible
        self.refresh()

    def _reset_cursor_blink(self) -> None:
        self._cursor_visible = True
        if self._blink_timer is not None:
            self._blink_timer.reset()

    # Editing primitives

    def _set_value(self, new_value: str, new_cursor: int) -> None:
        changed = new_value != self._value
        self._value = new_value
        self._cursor_pos = max(0, min(new_cursor, len(new_value)))
        if changed:
            self._update_suggestion()
            self.post_message(self.Changed(new_value, self))
        self.refresh()

    def _update_suggestion(self) -> None:
        if not self._value or self._search is not None:
            self._suggestion = ""
            self._suggestion_source = self._value
            return
        if self._value == self._suggestion_source:
            return
        if self._value.startswith(self._suggestion_source) and self._suggestion_source:
            tail = self._value[len(self._suggestion_source) :]
            if self._suggestion.startswith(tail):
                self._suggestion = self._suggestion[len(tail) :]
                self._suggestion_source = self._value
                return
        self._suggestion = autosuggest.compute_suggestion(self._value)
        self._suggestion_source = self._value

    def _insert(self, text: str) -> None:
        if not text:
            return
        new_value = self._value[: self._cursor_pos] + text + self._value[self._cursor_pos :]
        self._set_value(new_value, self._cursor_pos + len(text))

    def _backspace(self) -> None:
        if self._cursor_pos > 0:
            self._set_value(
                self._value[: self._cursor_pos - 1] + self._value[self._cursor_pos :],
                self._cursor_pos - 1,
            )

    def _delete(self) -> None:
        if self._cursor_pos < len(self._value):
            self._set_value(
                self._value[: self._cursor_pos] + self._value[self._cursor_pos + 1 :],
                self._cursor_pos,
            )

    def _word_boundary_left(self, pos: int) -> int:
        i = pos
        while i > 0 and not _is_word_char(self._value[i - 1]):
            i -= 1
        while i > 0 and _is_word_char(self._value[i - 1]):
            i -= 1
        return i

    def _word_boundary_right(self, pos: int) -> int:
        n = len(self._value)
        i = pos
        while i < n and not _is_word_char(self._value[i]):
            i += 1
        while i < n and _is_word_char(self._value[i]):
            i += 1
        return i

    def _move_word_left(self) -> None:
        self._cursor_pos = self._word_boundary_left(self._cursor_pos)
        self.refresh()

    def _move_word_right(self) -> None:
        self._cursor_pos = self._word_boundary_right(self._cursor_pos)
        self.refresh()

    def _delete_word_left(self) -> None:
        target = self._word_boundary_left(self._cursor_pos)
        if target == self._cursor_pos:
            return
        self._last_kill = self._value[target : self._cursor_pos]
        self._set_value(self._value[:target] + self._value[self._cursor_pos :], target)

    def _delete_word_right(self) -> None:
        target = self._word_boundary_right(self._cursor_pos)
        if target == self._cursor_pos:
            return
        self._last_kill = self._value[self._cursor_pos : target]
        self._set_value(self._value[: self._cursor_pos] + self._value[target:], self._cursor_pos)

    def _kill_to_start(self) -> None:
        if self._cursor_pos == 0:
            return
        self._last_kill = self._value[: self._cursor_pos]
        self._set_value(self._value[self._cursor_pos :], 0)

    def _kill_to_end(self) -> None:
        if self._cursor_pos >= len(self._value):
            return
        self._last_kill = self._value[self._cursor_pos :]
        self._set_value(self._value[: self._cursor_pos], self._cursor_pos)

    def _yank(self) -> None:
        if self._last_kill:
            self._insert(self._last_kill)

    def _at_eol_with_suggestion(self) -> bool:
        return (
            self._cursor_pos == len(self._value) and bool(self._suggestion) and self._search is None
        )

    def _accept_suggestion_full(self) -> None:
        if not self._at_eol_with_suggestion():
            return
        new_value = self._value + self._suggestion
        self._suggestion = ""
        self._set_value(new_value, len(new_value))

    def _accept_suggestion_word(self) -> None:
        if not self._at_eol_with_suggestion():
            return
        s = self._suggestion
        i = 0
        n = len(s)
        while i < n and not _is_word_char(s[i]):
            i += 1
        while i < n and _is_word_char(s[i]):
            i += 1
        if i == 0:
            return
        accepted = s[:i]
        rest = s[i:]
        new_value = self._value + accepted
        self._suggestion = rest
        self._suggestion_source = new_value
        self._set_value(new_value, len(new_value))

    def _transpose_chars(self) -> None:
        n = len(self._value)
        if n < 2 or self._cursor_pos == 0:
            return
        # Readline convention: at EOL, transpose the two chars before cursor
        # without advancing; mid-line, transpose across the cursor and advance.
        if self._cursor_pos == n:
            a = self._cursor_pos - 2
            b = self._cursor_pos - 1
            new_cursor = self._cursor_pos
        else:
            a = self._cursor_pos - 1
            b = self._cursor_pos
            new_cursor = self._cursor_pos + 1
        chars = list(self._value)
        chars[a], chars[b] = chars[b], chars[a]
        self._set_value("".join(chars), new_cursor)

    def _transpose_words(self) -> None:
        v = self._value
        if not v:
            return
        # Find the word at/after the cursor, then the word immediately before
        # it, and swap them. Cursor lands after the moved right word.
        end_right = self._word_boundary_right(self._cursor_pos)
        start_right = end_right
        while start_right > 0 and _is_word_char(v[start_right - 1]):
            start_right -= 1
        if start_right == end_right:
            return
        end_left = start_right
        while end_left > 0 and not _is_word_char(v[end_left - 1]):
            end_left -= 1
        start_left = end_left
        while start_left > 0 and _is_word_char(v[start_left - 1]):
            start_left -= 1
        if start_left == end_left:
            return
        left_word = v[start_left:end_left]
        middle = v[end_left:start_right]
        right_word = v[start_right:end_right]
        new_value = v[:start_left] + right_word + middle + left_word + v[end_right:]
        self._set_value(new_value, end_right)

    # History navigation

    def history_up(self) -> None:
        if len(self._history) == 0 or self._search is not None:
            return
        nav = self._history_nav
        cursor = nav.cursor if nav is not None else None
        query = nav.query if nav is not None else self._value
        result = self._history.walk_back(cursor, query)
        if result is None:
            return
        idx, entry = result
        if nav is None:
            self._history_nav = _HistoryNav(cursor=idx, saved_buffer=self._value, query=query)
        else:
            nav.cursor = idx
        self._set_value(entry, len(entry))

    def history_down(self) -> None:
        nav = self._history_nav
        if nav is None or self._search is not None:
            return
        idx, entry = self._history.walk_forward(nav.cursor, nav.query)
        if idx is None:
            saved = nav.saved_buffer
            self._history_nav = None
            self._set_value(saved, len(saved))
            return
        assert entry is not None
        nav.cursor = idx
        self._set_value(entry, len(entry))

    # Reverse-incremental search

    def enter_search(self) -> None:
        if len(self._history) == 0:
            return
        if self._search is not None:
            # Already in search: step to next older match
            self._search_step(direction=-1)
            return
        self._search = _SearchState(
            original_value=self._value,
            original_cursor_pos=self._cursor_pos,
            original_prompt=self._prompt_segments,
        )
        self._update_search_prompt()
        self.refresh()

    def _exit_search(self, *, keep_match: bool) -> None:
        s = self._search
        if s is None:
            return
        self._prompt_segments = s.original_prompt
        self._search = None
        if keep_match and s.match_value is not None:
            self._set_value(s.match_value, len(s.match_value))
        else:
            self._set_value(s.original_value, s.original_cursor_pos)
        self.refresh()

    def _search_find(self, *, direction: int, from_scratch: bool) -> None:
        s = self._search
        if s is None:
            return
        if from_scratch:
            # Always search from the newest (or oldest for forward) entry
            start: int | None = None
        elif s.match_index is not None:
            start = s.match_index + direction
            if start < 0 or start >= len(self._history):
                s.failed = True
                self._update_search_prompt()
                return
        else:
            start = None
        result = self._history.search(s.query, start, direction)
        if result is None:
            s.failed = True
        else:
            idx, value = result
            s.failed = False
            s.match_index = idx
            s.match_value = value
            self._set_value(value, len(value))
        self._update_search_prompt()

    def _search_step(self, *, direction: int) -> None:
        s = self._search
        if s is None:
            return
        if not s.query:
            return
        self._search_find(direction=direction, from_scratch=False)

    def _update_search_prompt(self) -> None:
        s = self._search
        if s is None:
            return
        label = "failed reverse-i-search" if s.failed else "reverse-i-search"
        style = _SEARCH_FAIL_STYLE if s.failed else _SEARCH_PROMPT_STYLE
        self._prompt_segments = [(f"({label})`{s.query}': ", style)]
        self.refresh()

    # Key handling

    def _handle_search_key(self, event: events.Key) -> None:
        s = self._search
        assert s is not None
        key = event.key

        if key == "enter":
            self._exit_search(keep_match=True)
        elif key in ("escape", "ctrl+g"):
            self._exit_search(keep_match=False)
        elif key == "ctrl+r":
            self._search_step(direction=-1)
        elif key == "ctrl+s":
            self._search_step(direction=1)
        elif key == "backspace":
            if s.query:
                s.query = s.query[:-1]
                s.failed = False
                self._search_find(direction=-1, from_scratch=True)
            else:
                self._update_search_prompt()
        elif event.character and event.is_printable:
            s.query += event.character
            self._search_find(direction=-1, from_scratch=True)
        else:
            # Any other key: accept current match and absorb the keypress.
            self._exit_search(keep_match=True)

        event.prevent_default()
        event.stop()

    def _on_paste(self, event: events.Paste) -> None:
        if not self._active:
            return
        # Single-line input: collapse line breaks (CRLF/CR/LF) to spaces.
        text = " ".join(event.text.splitlines())
        if self._search is not None:
            self._search.query += text
            self._search.failed = False
            self._search_find(direction=-1, from_scratch=True)
        else:
            self._insert(text)
        event.prevent_default()
        event.stop()

    def _on_key(self, event: events.Key) -> None:
        if event.key == "grave_accent":
            event.prevent_default()
            return

        if not self._active:
            return

        self._reset_cursor_blink()

        if self._search is not None:
            self._handle_search_key(event)
            return

        key = event.key

        if key == "enter":
            self.post_message(self.Submitted(self._value, self))
            event.prevent_default()
            event.stop()
            return

        if key in ("left", "ctrl+b"):
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
                self.refresh()
        elif key in ("right", "ctrl+f"):
            if self._at_eol_with_suggestion():
                self._accept_suggestion_full()
            elif self._cursor_pos < len(self._value):
                self._cursor_pos += 1
                self.refresh()
        elif key in ("home", "ctrl+a"):
            self._cursor_pos = 0
            self.refresh()
        elif key in ("end", "ctrl+e"):
            self._cursor_pos = len(self._value)
            self.refresh()
        elif key in ("alt+b", "alt+left", "ctrl+left"):
            self._move_word_left()
        elif key in ("alt+f", "alt+right", "ctrl+right"):
            if self._at_eol_with_suggestion():
                self._accept_suggestion_word()
            else:
                self._move_word_right()
        elif key in ("backspace", "ctrl+h"):
            self._backspace()
        elif key == "delete":
            self._delete()
        elif key == "ctrl+d":
            # Emacs convention: forward-delete on non-empty line; ignore on empty
            # (we don't want to accidentally quit the app here).
            if self._value:
                self._delete()
            else:
                return  # let it bubble (app may handle)
        elif key in ("ctrl+w", "alt+backspace"):
            self._delete_word_left()
        elif key == "alt+d":
            self._delete_word_right()
        elif key == "ctrl+u":
            self._kill_to_start()
        elif key == "ctrl+k":
            self._kill_to_end()
        elif key == "ctrl+y":
            self._yank()
        elif key == "ctrl+t":
            self._transpose_chars()
        elif key == "alt+t":
            self._transpose_words()
        elif event.character and event.is_printable:
            self._insert(event.character)
        else:
            return  # let it bubble

        event.prevent_default()
        event.stop()

    # Rendering

    def _spans_for(self, text: str) -> list[tuple[str, Style]]:
        if self._search is not None:
            return [(text, Style())]
        cached_text, cached_spans = self._highlight_cache
        if text == cached_text:
            return cached_spans
        spans = highlight_command(text)
        self._highlight_cache = (text, spans)
        return spans

    def render_line(self, y: int) -> Strip:
        base_style = self.rich_style
        if y != 0:
            return Strip.blank(self.size.width, base_style)

        width = self.size.width
        prompt_len = self.prompt_len

        # Placeholder when unfocused and empty
        if not self._active and not self._value:
            placeholder_segments = [
                Segment(self.placeholder[:width], base_style + Style(dim=True)),
            ]
            return Strip(placeholder_segments).extend_cell_length(width, base_style)

        text = self._value
        cursor_pos = self._cursor_pos

        available = width - prompt_len
        if available <= 0:
            return Strip.blank(width, base_style)

        scroll_offset = 0
        if cursor_pos >= available:
            scroll_offset = cursor_pos - available + 1

        visible_start = scroll_offset
        visible_end = scroll_offset + available
        cursor_in_view = cursor_pos - scroll_offset

        spans = self._spans_for(text)

        segments: list[Segment] = [Segment(t, base_style + s) for t, s in self._prompt_segments]

        show_cursor = self._active and self._cursor_visible

        # Emit the visible portion of text, splitting the cursor char out if needed.
        pos = 0
        for span_text, span_style in spans:
            span_start = pos
            span_end = pos + len(span_text)
            pos = span_end
            start = max(span_start, visible_start)
            end = min(span_end, visible_end)
            if start >= end:
                continue
            local_start = start - span_start
            local_end = end - span_start
            chunk = span_text[local_start:local_end]
            style = base_style + span_style

            if show_cursor and start <= cursor_pos < end:
                rel = cursor_pos - start
                before = chunk[:rel]
                cursor_char = chunk[rel]
                after = chunk[rel + 1 :]
                if before:
                    segments.append(Segment(before, style))
                segments.append(Segment(cursor_char, style + _CURSOR_STYLE))
                if after:
                    segments.append(Segment(after, style))
            else:
                segments.append(Segment(chunk, style))

        ghost = self._suggestion if self._search is None else ""

        if show_cursor and cursor_pos >= len(text) and cursor_in_view < available:
            if ghost:
                segments.append(Segment(ghost[0], base_style + _CURSOR_STYLE))
                if len(ghost) > 1:
                    segments.append(Segment(ghost[1:], base_style + _GHOST_STYLE))
            else:
                segments.append(Segment(" ", base_style + _CURSOR_STYLE))
        elif ghost:
            segments.append(Segment(ghost, base_style + _GHOST_STYLE))

        return Strip(segments).extend_cell_length(width, base_style)
