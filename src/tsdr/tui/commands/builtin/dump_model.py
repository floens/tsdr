from argparse import Namespace
from dataclasses import fields, is_dataclass
from typing import Any

from rich.markup import escape

from tsdr.tui.commands.base import Command, CommandParser
from tsdr.tui.model.store import get_ui_store


class DumpModelCommand(Command):
    @property
    def description(self) -> str:
        return "Print the current UIModel (debug)"

    def configure(self, parser: CommandParser) -> None:
        pass

    def run(self, _args: Namespace) -> str:
        return _format_dc(get_ui_store().model)


def _format_dc(obj: Any, indent: int = 0) -> str:
    pad = "  " * indent
    lines = [f"{pad}[bold]{type(obj).__name__}[/]:"]
    for f in fields(obj):
        lines.append(_format_field(f.name, getattr(obj, f.name), indent + 1))
    return "\n".join(lines)


def _format_field(name: str, value: Any, indent: int) -> str:
    pad = "  " * indent
    if _is_dc_instance(value):
        return f"{pad}[dim]{name}=[/]\n{_format_dc(value, indent + 1)}"
    if isinstance(value, tuple) and value and all(_is_dc_instance(v) for v in value):
        body = "\n".join(_format_dc(v, indent + 1) for v in value)
        return f"{pad}[dim]{name}=[/]\n{body}"
    return f"{pad}[dim]{name}=[/]{_format_value(value)}"


def _format_value(v: Any) -> str:
    if v is None:
        return "[dim italic]None[/]"
    if isinstance(v, bool):
        return "[green]True[/]" if v else "[dim]False[/]"
    if isinstance(v, (int, float)):
        return f"[yellow]{v}[/]"
    if isinstance(v, str):
        return f"[cyan]{escape(repr(v))}[/]"
    if isinstance(v, tuple):
        return "[dim]()[/]" if not v else f"[dim]({len(v)} items)[/]"
    return str(escape(repr(v)))


def _is_dc_instance(v: Any) -> bool:
    return is_dataclass(v) and not isinstance(v, type)
