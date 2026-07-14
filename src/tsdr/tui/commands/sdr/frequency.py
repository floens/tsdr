from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.core.units import parse_hz
from tsdr.tui.commands._format import freq_mhz, success
from tsdr.tui.commands.base import Command, CommandParser
from tsdr.tui.commands.sdr._utils import get_focused_device_id


class FrequencyCommand(Command):
    @property
    def description(self) -> str:
        return "Set frequency (e.g. f 100.1M, f 430k, f 100_100_000)"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("frequency", help="Frequency with optional SI suffix (k/M/G/Hz)")

    def run(self, args: Namespace) -> str:
        freq_hz = float(parse_hz(args.frequency))
        did = get_focused_device_id()
        get_engine().update_device_config(did, tuned_frequency=freq_hz)
        return success(f"Frequency: {freq_mhz(freq_hz)}")
