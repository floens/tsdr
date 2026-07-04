from argparse import Namespace

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import ConfigurationError, SDRException
from tsdr.core.units import parse_hz
from tsdr.radio.decoders.fsk.profile import FRAMINGS as _FSK_FRAMINGS
from tsdr.radio.decoders.fsk.profile import PROFILES as FSK_PRESETS
from tsdr.radio.decoders.fsk.tables import ALPHABETS as _FSK_ALPHABET_TABLES
from tsdr.radio.decoders.sstv import MODES_BY_NAME as SSTV_MODES_BY_NAME
from tsdr.radio.registry import DEMODULATORS
from tsdr.tui.commands._format import device_id, fields, rate_sps, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

_FSK_MODES = {"RTTY", "FSK"}  # accept --baud/--shift/--reverse/--preset
_FSK_ALPHABETS = tuple(_FSK_ALPHABET_TABLES)


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
        parser.add_argument(
            "--preset",
            default=None,
            help=f"FSK preset ({', '.join(sorted(FSK_PRESETS))}); rtty/fsk only",
        )
        parser.add_argument("--baud", type=float, default=None, help="FSK baud; rtty/fsk only")
        parser.add_argument(
            "--shift",
            default=None,
            help="FSK shift in Hz (SI suffix) or 'auto' (default); rtty/fsk only",
        )
        parser.add_argument(
            "--reverse", action="store_true", help="Invert mark/space (LSB); rtty/fsk only"
        )
        parser.add_argument(
            "--alphabet", choices=_FSK_ALPHABETS, default=None, help="fsk mode only"
        )
        parser.add_argument("--framing", choices=_FSK_FRAMINGS, default=None, help="fsk mode only")

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

        fsk = self._resolve_fsk(args, mode)

        if mode == "OFF":
            manager.stop_audio_output(did)
            manager.remove_pipeline(did, "audio")
            return success(f"Disabled demodulation for {device_id(did)}")

        if mode not in DEMODULATORS:
            available = ", ".join(sorted(DEMODULATORS))
            raise ConfigurationError(f"Unknown mode '{mode}'. Available: {available}")

        context = manager.get_device(did)

        if context.state != DeviceState.RUNNING:
            raise SDRException(f"Device {did} must be running")

        if frequency_offset != 0.0:
            max_offset = context.config.sample_rate / 2.0
            if abs(frequency_offset) > max_offset:
                raise ConfigurationError(
                    f"Frequency offset {frequency_offset / 1000:.1f} kHz exceeds Nyquist limit "
                    f"(±{max_offset / 1000:.1f} kHz for sample rate "
                    f"{rate_sps(context.config.sample_rate)})"
                )

        spec = DemodSpec(
            mode=mode,
            frequency_offset=frequency_offset,
            fm_deviation_hz=deviation,
            sstv_mode=sstv_mode,
            **fsk,
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
        if mode in _FSK_MODES:
            extras["baud"] = f"[cyan]{fsk['fsk_baud']:g}[/]" if fsk["fsk_baud"] else "[cyan]auto[/]"
            extras["shift"] = (
                f"[cyan]{fsk['fsk_shift_hz']:.0f} Hz[/]" if fsk["fsk_shift_hz"] else "[cyan]auto[/]"
            )
            if fsk["fsk_reverse"]:
                extras["polarity"] = "[cyan]reverse[/]"

        if extras:
            return f"{head} ({fields(extras)})"
        return head

    def _resolve_fsk(self, args: Namespace, mode: str) -> dict[str, object]:
        """Resolve preset + overrides into DemodSpec fsk_* fields (None = auto)."""
        used = [
            name
            for name, val in (
                ("--preset", args.preset),
                ("--baud", args.baud),
                ("--shift", args.shift),
            )
            if val is not None
        ]
        if args.reverse:
            used.append("--reverse")
        if used and mode not in _FSK_MODES:
            raise ConfigurationError(f"{used[0]} is only valid with mode 'rtty' or 'fsk'")
        if (args.alphabet or args.framing) and mode != "FSK":
            raise ConfigurationError("--alphabet/--framing are only valid with mode 'fsk'")

        preset = None
        if args.preset is not None:
            preset = FSK_PRESETS.get(args.preset.lower())
            if preset is None:
                known = ", ".join(sorted(FSK_PRESETS))
                raise ConfigurationError(f"unknown preset '{args.preset}'. Known: {known}")
            if mode == "RTTY" and preset.framing != "start_stop":
                raise ConfigurationError(f"preset '{args.preset}' needs mode 'fsk'")

        if mode not in _FSK_MODES:
            return {}

        if args.shift is None:
            shift = preset.shift_hz if preset else None
        elif args.shift.lower() == "auto":
            shift = None
        else:
            shift = float(parse_hz(args.shift))
        reverse: bool | None
        if args.reverse:
            reverse = True
        elif preset is not None:
            reverse = preset.polarity == "reverse"
        else:
            reverse = None
        return {
            "fsk_baud": args.baud if args.baud is not None else (preset.baud if preset else None),
            "fsk_shift_hz": shift,
            "fsk_reverse": reverse,
            "fsk_alphabet": args.alphabet
            or (preset.alphabet if preset and mode == "FSK" else None),
            "fsk_framing": args.framing or (preset.framing if preset and mode == "FSK" else None),
        }

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
        if flag == "--preset":
            return [
                Completion(value=n) for n in sorted(FSK_PRESETS) if n.startswith(prefix.lower())
            ]
        if flag == "--alphabet":
            return [Completion(value=n) for n in _FSK_ALPHABETS if n.startswith(prefix.lower())]
        if flag == "--framing":
            return [Completion(value=n) for n in _FSK_FRAMINGS if n.startswith(prefix.lower())]
        return []
