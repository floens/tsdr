from __future__ import annotations

import shlex

from rich.style import Style

from tsdr.tui.commands import registry

_DEFAULT = Style()
_COMMAND_OK = Style(color="cyan")
_COMMAND_ERR = Style(color="red")
_PARSE_ERR = Style(color="red")


def highlight_command(text: str) -> list[tuple[str, Style]]:
    """Return styled spans that reassemble to `text` exactly.

    v1: colors the first token cyan if it's a registered command, red otherwise.
    On shlex parse error, returns the whole text in red.
    """
    if not text:
        return []

    try:
        shlex.split(text)
    except ValueError:
        return [(text, _PARSE_ERR)]

    i = 0
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    lead = text[:i]

    j = i
    while j < n and not text[j].isspace():
        j += 1
    cmd = text[i:j]
    rest = text[j:]

    cmd_style = _COMMAND_OK if cmd in registry.COMMANDS else _COMMAND_ERR

    spans: list[tuple[str, Style]] = []
    if lead:
        spans.append((lead, _DEFAULT))
    if cmd:
        spans.append((cmd, cmd_style))
    if rest:
        spans.append((rest, _DEFAULT))
    return spans
