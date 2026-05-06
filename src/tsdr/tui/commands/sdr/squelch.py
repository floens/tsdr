from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.tui.commands._format import db, fields, state
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id


class SDRSquelchCommand(Command):
    @property
    def description(self) -> str:
        return "Configure audio squelch (mute when signal is weak)"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("state", nargs="?", choices=["on", "off"])
        parser.add_argument(
            "--threshold",
            type=float,
            help="Squelch threshold in dBFS (typical range -100 to 0)",
        )
        parser.add_argument(
            "--hang",
            type=float,
            help="Hang time in milliseconds (gate stays open this long after signal drops)",
        )
        parser.add_argument("--device", dest="device_id")

    def run(self, args: Namespace) -> str:
        engine = get_engine()
        did = args.device_id or get_focused_device_id()

        context = engine.get_device(did)
        if "audio" not in context.config.pipelines:
            raise SDRException(f"No audio demodulator active on '{did}'. Use 'demod' first.")

        kwargs: dict = {}
        if args.state == "on":
            kwargs["enabled"] = True
        elif args.state == "off":
            kwargs["enabled"] = False
        if args.threshold is not None:
            kwargs["threshold_db"] = args.threshold
        if args.hang is not None:
            kwargs["hang_ms"] = args.hang

        if kwargs:
            engine.update_squelch(did, "audio", **kwargs)

        pc = engine.get_device(did).config.pipelines["audio"]
        st = "on" if pc.squelch_enabled else "off"
        params = fields(
            {
                "threshold": db(pc.squelch_threshold_db),
                "hang": f"[yellow]{pc.squelch_hang_ms:.0f} ms[/]",
            }
        )
        return f"Squelch {state(st)} ({params})"

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
