from rich.segment import Segment
from rich.style import Style

_CURSOR_STYLE = Style(reverse=True)


class InlineEditBuffer:
    """Text buffer with cursor, for inline editing in any widget.

    Handles text manipulation and cursor-aware segment rendering.
    Cursor blink timing is managed by the hosting widget.
    """

    __slots__ = ("value", "cursor_pos", "cursor_visible")

    def __init__(self, value: str, cursor_pos: int | None = None) -> None:
        self.value = value
        self.cursor_pos = cursor_pos if cursor_pos is not None else len(value)
        self.cursor_visible = True

    def insert(self, char: str) -> None:
        self.value = self.value[: self.cursor_pos] + char + self.value[self.cursor_pos :]
        self.cursor_pos += 1

    def backspace(self) -> None:
        if self.cursor_pos > 0:
            self.value = self.value[: self.cursor_pos - 1] + self.value[self.cursor_pos :]
            self.cursor_pos -= 1

    def delete(self) -> None:
        if self.cursor_pos < len(self.value):
            self.value = self.value[: self.cursor_pos] + self.value[self.cursor_pos + 1 :]

    def move_left(self) -> None:
        if self.cursor_pos > 0:
            self.cursor_pos -= 1

    def move_right(self) -> None:
        if self.cursor_pos < len(self.value):
            self.cursor_pos += 1

    def home(self) -> None:
        self.cursor_pos = 0

    def end(self) -> None:
        self.cursor_pos = len(self.value)

    def toggle_cursor(self) -> None:
        self.cursor_visible = not self.cursor_visible

    def reset_cursor(self) -> None:
        self.cursor_visible = True

    def render_segments(self, style: Style) -> list[Segment]:
        """Render text with cursor as Rich Segments.

        When cursor_visible is False, renders plain text without cursor highlight.
        When cursor is at end of text, a trailing space shows the cursor block.
        """
        if not self.cursor_visible:
            return [Segment(self.value, style)] if self.value else []

        before = self.value[: self.cursor_pos]
        if self.cursor_pos < len(self.value):
            cursor_char = self.value[self.cursor_pos]
            after = self.value[self.cursor_pos + 1 :]
        else:
            cursor_char = " "
            after = ""

        segments: list[Segment] = []
        if before:
            segments.append(Segment(before, style))
        segments.append(Segment(cursor_char, _CURSOR_STYLE))
        if after:
            segments.append(Segment(after, style))
        return segments
