from argparse import Namespace

from tsdr.tui.commands.base import Command


class ExitCommand(Command):
    @property
    def description(self) -> str:
        return "Exit the application"

    def run(self, args: Namespace) -> str:
        from tsdr.tui.app import get_app  # noqa: PLC0415

        get_app().exit()
        return "Exiting..."
