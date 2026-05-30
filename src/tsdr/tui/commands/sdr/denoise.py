from argparse import Namespace

from tsdr.core.preferences import save_engine_config
from tsdr.core.sdr.engine import get_engine
from tsdr.radio.dsp.rnnoise import rnnoise_available
from tsdr.tui.commands._format import error, state, success
from tsdr.tui.commands.base import Command, CommandParser


class SDRDenoiseCommand(Command):
    @property
    def description(self) -> str:
        return "Toggle RNNoise speech denoising on audio output (global)"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("state", nargs="?", choices=["on", "off"])

    def run(self, args: Namespace) -> str:
        engine = get_engine()
        if args.state == "on":
            want = True
        elif args.state == "off":
            want = False
        else:
            want = not engine.config.denoise

        if want and not rnnoise_available():
            return error("Denoise unavailable — install the 'denoise' extra")

        engine.update_global_config(denoise=want)
        save_engine_config(engine)
        return success(f"Denoise {state('on' if want else 'off')}")
