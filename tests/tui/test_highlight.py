from __future__ import annotations

import pytest

from tsdr.tui.commands import registry
from tsdr.tui.console.highlight import highlight_command


@pytest.fixture
def fake_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "COMMANDS", {"echo": object(), "list": object()})


def _assemble(spans: list) -> str:
    return "".join(text for text, _ in spans)


def test_empty_input() -> None:
    assert highlight_command("") == []


def test_known_command(fake_commands: None) -> None:
    spans = highlight_command("echo hello world")
    assert _assemble(spans) == "echo hello world"
    # First span is the command name, cyan
    cmd_text, cmd_style = spans[0]
    assert cmd_text == "echo"
    assert cmd_style.color is not None
    assert cmd_style.color.name == "cyan"


def test_unknown_command(fake_commands: None) -> None:
    spans = highlight_command("unknown arg")
    assert _assemble(spans) == "unknown arg"
    cmd_text, cmd_style = spans[0]
    assert cmd_text == "unknown"
    assert cmd_style.color is not None
    assert cmd_style.color.name == "red"


def test_leading_whitespace(fake_commands: None) -> None:
    spans = highlight_command("  echo hi")
    assert _assemble(spans) == "  echo hi"
    # First span is the leading whitespace
    assert spans[0][0] == "  "


def test_parse_error_is_red() -> None:
    spans = highlight_command('echo "unclosed')
    assert _assemble(spans) == 'echo "unclosed'
    assert len(spans) == 1
    _, style = spans[0]
    assert style.color is not None
    assert style.color.name == "red"


def test_command_only(fake_commands: None) -> None:
    spans = highlight_command("echo")
    assert _assemble(spans) == "echo"
    assert spans[0][0] == "echo"
