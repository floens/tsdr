from argparse import Namespace
from itertools import groupby

from tsdr.tui.commands._format import header, safe
from tsdr.tui.commands.base import Command
from tsdr.tui.keybindings import BINDINGS, format_keys


class KeysCommand(Command):
    @property
    def description(self) -> str:
        return "Show keyboard shortcuts"

    def run(self, args: Namespace) -> str:
        width = max(len(format_keys(b)) for b in BINDINGS) + 2
        blocks = [
            "\n".join(
                [header(category)]
                + [safe(f"  {format_keys(b):<{width}}{b.description}") for b in group]
            )
            for category, group in groupby(BINDINGS, key=lambda b: b.category)
        ]
        return "\n\n".join(blocks)
