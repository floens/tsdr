from argparse import Namespace
from typing import Any

from tsdr.core.preferences import save_device
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.engine import get_engine
from tsdr.core.units import parse_hz
from tsdr.tui.commands._format import db, device_id, fields, freq_mhz, rate_msps, state, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

_DEVICE_FIELDS = frozenset(DeviceConfig.__dataclass_fields__.keys())
_GLOBAL_FIELDS = frozenset(SDRConfig.__dataclass_fields__.keys())


class SDRConfigCommand(Command):
    @property
    def description(self) -> str:
        return "Configure SDR device parameters"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument(
            "frequency",
            nargs="?",
            default=None,
            help="Center frequency with SI suffix (e.g. 100.1M, 430k)",
        )
        parser.add_argument("--device", dest="device_id")
        parser.add_argument(
            "--frequency",
            dest="frequency_flag",
            help="Center frequency with SI suffix (e.g. 100.1M, 430k)",
        )
        parser.add_argument("--sample-rate", help="Sample rate with SI suffix (e.g. 2.4M, 250k)")
        parser.add_argument("--gain", "--rf-gain", type=float, help="RF gain in dB (disables AGC)")
        parser.add_argument("--agc", choices=["on", "off"], help="Client-side AGC")
        parser.add_argument("--hw-agc", choices=["on", "off"], help="Hardware AGC")
        parser.add_argument(
            "--bias-t",
            dest="bias_t",
            choices=["on", "off"],
            help="Antenna bias-T (RTL-SDR / Airspy / HackRF)",
        )
        parser.add_argument("--fft-size", type=int, help="FFT size")
        parser.add_argument(
            "--bandwidth", help="Channel bandwidth with SI suffix (e.g. 200k, 1.5M)"
        )
        parser.add_argument("--fps", type=float, help="Target UI update rate")
        parser.add_argument(
            "--network-buffer",
            dest="network_buffer_seconds",
            type=float,
            help="Jitter buffer pre-fill, seconds (rtltcp/spyserver)",
        )

    def run(self, args: Namespace) -> str:
        manager = get_engine()
        did = args.device_id or get_focused_device_id()

        changes: dict[str, Any] = {}

        if args.frequency is not None:
            changes["center_frequency"] = float(parse_hz(args.frequency))
        if args.frequency_flag is not None:
            changes["center_frequency"] = float(parse_hz(args.frequency_flag))
        if args.sample_rate is not None:
            changes["sample_rate"] = float(parse_hz(args.sample_rate))
        if args.gain is not None:
            changes["rf_gain"] = args.gain
            changes["enable_agc"] = False
        if args.agc is not None:
            if args.agc == "on":
                changes["enable_agc"] = True
                changes["auto_gain"] = False
            else:
                changes["enable_agc"] = False
        if args.hw_agc is not None:
            if args.hw_agc == "on":
                changes["auto_gain"] = True
                changes["enable_agc"] = False
            else:
                changes["auto_gain"] = False
        if args.bias_t is not None:
            device = manager.get_device(did)
            if not device.device.supports_bias_tee:
                return f"bias-T not supported by {device_id(did)}"
            changes["bias_tee"] = args.bias_t == "on"
        if args.fft_size is not None:
            changes["fft_size"] = args.fft_size
        if args.bandwidth is not None:
            changes["channel_bandwidth"] = float(parse_hz(args.bandwidth))
        if args.fps is not None:
            changes["target_fps"] = args.fps
        if args.network_buffer_seconds is not None:
            context = manager.get_device(did)
            if context.device_type not in ("rtltcp", "spyserver"):
                return (
                    f"--network-buffer is not applicable to "
                    f"{device_id(did)} ({context.device_type})"
                )
            changes["network_buffer_seconds"] = args.network_buffer_seconds

        if not changes:
            return self.help_text()

        device_changes = {k: v for k, v in changes.items() if k in _DEVICE_FIELDS}
        global_changes = {k: v for k, v in changes.items() if k in _GLOBAL_FIELDS}

        if device_changes:
            manager.update_device_config(did, **device_changes)
        if global_changes:
            manager.update_global_config(**global_changes)

        save_device(manager)

        summary: dict[str, str] = {}
        for key, value in changes.items():
            if key == "center_frequency":
                summary["frequency"] = freq_mhz(value, precision=2)
            elif key == "sample_rate":
                summary["sample_rate"] = rate_msps(value)
            elif key == "rf_gain":
                summary["gain"] = db(value)
            elif key == "auto_gain":
                summary["hw-agc"] = state("on" if value else "off")
            elif key == "enable_agc":
                summary["agc"] = state("on" if value else "off")
            elif key == "bias_tee":
                summary["bias-t"] = state("on" if value else "off")
            elif key == "channel_bandwidth":
                summary["bandwidth"] = f"[yellow]{value / 1000:.1f} kHz[/]"
            elif key == "network_buffer_seconds":
                summary["network-buffer"] = f"[yellow]{value:.2f} s[/]"
            else:
                summary[key] = str(value)

        return f"{success('Updated ' + device_id(did))}: {fields(summary)}"

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
