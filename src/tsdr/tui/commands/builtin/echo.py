from argparse import Namespace

from tsdr.tui.commands.base import Command, CommandParser


class EchoCommand(Command):
    @property
    def description(self) -> str:
        return "Echo back the provided text"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("words", nargs="*")

    def run(self, args: Namespace) -> str:
        if not args.words:
            return self.help_text()
        return " ".join(args.words)
