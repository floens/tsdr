from argparse import Namespace

from tsdr.tui.commands.base import Command


class HelpCommand(Command):
    @property
    def description(self) -> str:
        return "Display available commands"

    def run(self, args: Namespace) -> str:
        from tsdr.tui.commands.registry import COMMANDS  # noqa: PLC0415

        commands = sorted(COMMANDS.keys())
        if not commands:
            return "No commands available"
        return f"Available commands: {', '.join(commands)}"
