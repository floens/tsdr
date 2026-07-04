from argparse import Namespace

from tsdr.core.audio_spec import AudioDemodSpec
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import ConfigurationError
from tsdr.core.units import parse_hz
from tsdr.radio.decoders.sstv import MODES_BY_NAME as SSTV_MODES_BY_NAME
from tsdr.radio.registry import DEMODULATORS
from tsdr.tui.commands._format import device_id, fields, rate_sps, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id


class SDRDemodCommand(Command):
    @property
    def description(self) -> str:
        return "Enable/disable audio demodulation or protocol decoding"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("mode", choices=[*sorted(DEMODULATORS), "off"])
        parser.add_argument("--device", dest="device_id")
        parser.add_argument(
            "--offset",
            default=None,
            help="Frequency offset with SI suffix (e.g. -25k, 100k)",
        )
        parser.add_argument(
            "--deviation",
            default=None,
            help="FM deviation override with SI suffix (NFM only; default: bandwidth/2)",
        )
        parser.add_argument(
            "--sstv-mode",
            dest="sstv_mode",
            default=None,
            help="Force a specific SSTV submode (e.g. martin_m1); SSTV only",
        )

    def run(self, args: Namespace) -> str:
        manager = get_engine()
        did = args.device_id or get_focused_device_id()
        mode = args.mode.upper()
        frequency_offset = float(parse_hz(args.offset)) if args.offset is not None else 0.0
        deviation = float(parse_hz(args.deviation)) if args.deviation is not None else None
        sstv_mode: str | None = args.sstv_mode
        if sstv_mode is not None and mode != "SSTV":
            raise ConfigurationError("--sstv-mode is only valid with mode 'sstv'")
        if sstv_mode is not None and sstv_mode.lower() not in SSTV_MODES_BY_NAME:
            known = ", ".join(sorted(SSTV_MODES_BY_NAME))
            raise ConfigurationError(f"unknown SSTV mode '{sstv_mode}'. Known: {known}")

        if mode == "OFF":
            manager.stop_audio_output(did)
            manager.remove_pipeline(did, "audio")
            return success(f"Disabled demodulation for {device_id(did)}")

        if mode not in DEMODULATORS:
            available = ", ".join(sorted(DEMODULATORS))
            raise ConfigurationError(f"Unknown mode '{mode}'. Available: {available}")

        context = manager.get_device(did)

        if frequency_offset != 0.0:
            max_offset = context.config.sample_rate / 2.0
            if abs(frequency_offset) > max_offset:
                raise ConfigurationError(
                    f"Frequency offset {frequency_offset / 1000:.1f} kHz exceeds Nyquist limit "
                    f"(±{max_offset / 1000:.1f} kHz for sample rate "
                    f"{rate_sps(context.config.sample_rate)})"
                )

        spec = AudioDemodSpec(
            mode=mode,
            frequency_offset=frequency_offset,
            fm_deviation_hz=deviation,
            sstv_mode=sstv_mode,
        )
        manager.set_audio_demod(did, spec)

        head = success(f"Enabled [bold green]{mode}[/] demodulation for {device_id(did)}")
        extras: dict[str, str] = {}
        if frequency_offset != 0.0:
            extras["offset"] = f"[cyan]{frequency_offset / 1000:.1f} kHz[/]"
        if deviation is not None:
            extras["deviation"] = f"[yellow]±{deviation / 1000:.1f} kHz[/]"
        if sstv_mode is not None:
            extras["sstv_mode"] = f"[cyan]{sstv_mode}[/]"

        if extras:
            return f"{head} ({fields(extras)})"
        return head

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
        if flag == "--sstv-mode":
            return [
                Completion(value=name)
                for name in sorted(SSTV_MODES_BY_NAME)
                if name.startswith(prefix.lower())
            ]
        return []
