from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.tui.commands._format import device_id, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id


class SDRStartCommand(Command):
    @property
    def description(self) -> str:
        return "Start an SDR device"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("--device", dest="device_id")

    def run(self, args: Namespace) -> str:
        did = args.device_id or get_focused_device_id()
        get_engine().start_device(did)
        return success(f"Started {device_id(did)}")

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if flag == "--device":
            return device_id_completions(prefix)
        return []
