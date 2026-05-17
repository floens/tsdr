from argparse import Namespace

from tsdr.core.preferences import save_device
from tsdr.core.sdr.engine import get_engine
from tsdr.tui.commands._format import device_id, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions


class SDRFocusCommand(Command):
    @property
    def description(self) -> str:
        return "Focus (select) an SDR device"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("device_id")

    def run(self, args: Namespace) -> str:
        engine = get_engine()
        engine.set_focused_device(args.device_id)
        save_device(engine)
        return success(f"Focused {device_id(args.device_id)}")

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        return device_id_completions(prefix)
