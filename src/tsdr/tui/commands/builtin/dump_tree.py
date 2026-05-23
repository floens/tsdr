from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from rich.markup import escape

from tsdr.tui.commands.base import Command, CommandParser
from tsdr.tui.model.store import get_ui_store
from tsdr.tui.view.spec import WidgetSpec
from tsdr.tui.view.tree import derive_tree


class DumpTreeCommand(Command):
    @property
    def description(self) -> str:
        return "Print the current derive_tree(UIModel) as an indented tree (debug)"

    def configure(self, parser: CommandParser) -> None:
        pass

    def run(self, _args: Namespace) -> str:
        return _format_spec(derive_tree(get_ui_store().model))


def _format_spec(spec: WidgetSpec, indent: int = 0) -> str:
    pad = "  " * indent
    head = f"{pad}[bold]{escape(spec.kind)}[/][dim]#[/][cyan]{escape(spec.key)}[/]"
    if spec.props:
        head += f"  [dim]([/]{_format_props(spec.props)}[dim])[/]"
    if not spec.children:
        return head
    body = "\n".join(_format_spec(c, indent + 1) for c in spec.children)
    return f"{head}\n{body}"


def _format_props(props: Mapping[str, Any]) -> str:
    return ", ".join(f"[dim]{escape(k)}=[/]{_format_value(v)}" for k, v in props.items())


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
