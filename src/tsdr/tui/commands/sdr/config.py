from argparse import Namespace
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from tsdr.core.preferences import save_device
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine, get_engine
from tsdr.core.units import parse_hz
from tsdr.devices import NetworkDeviceParams
from tsdr.tui.commands._format import (
    db,
    device_id,
    error,
    fields,
    freq_mhz,
    header,
    rate_msps,
    state,
    success,
)
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

_DEVICE_FIELDS = frozenset(DeviceConfig.__dataclass_fields__.keys())
_GLOBAL_FIELDS = frozenset(SDRConfig.__dataclass_fields__.keys())


def _format_field(name: str, value: Any) -> str:
    """Render a single config field. Generic by default; specialized only
    where the unit is well-known so it's easy to add new fields without
    touching this file."""
    if value is None:
        return "[dim]None[/]"
    if isinstance(value, bool):
        return state("on" if value else "off")
    if isinstance(value, MappingProxyType):
        return f"[dim]{{{len(value)} entries}}[/]"
    if name == "center_frequency" and isinstance(value, (int, float)):
        return freq_mhz(float(value), precision=3)
    if name == "sample_rate" and isinstance(value, (int, float)):
        return rate_msps(float(value))
    if name == "rf_gain" and isinstance(value, (int, float)):
        return db(float(value))
    if name == "channel_bandwidth" and isinstance(value, (int, float)):
        return f"[yellow]{value / 1000:.1f} kHz[/]"
    if name == "network_buffer_seconds" and isinstance(value, (int, float)):
        return f"[yellow]{value:.2f} s[/]"
    return repr(value)


def _dump_dataclass(obj: Any) -> list[str]:
    """Render every dataclass field of `obj` as `name: value`, one per line.

    Iterates fields via `dataclasses.fields()` so new fields appear here
    automatically. Only the formatter (`_format_field`) needs touching to
    customize unit display.
    """
    rows = [(f.name, _format_field(f.name, getattr(obj, f.name))) for f in dataclass_fields(obj)]
    if not rows:
        return ["  [dim](no fields)[/]"]
    width = max(len(name) for name, _ in rows)
    return [f"  [dim]{name:<{width}}[/]  {rendered}" for name, rendered in rows]


class SDRConfigCommand(Command):
    @property
    def description(self) -> str:
        return "Configure SDR device parameters"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument(
            "frequency",
            nargs="?",
            default=None,
            help="Center frequency with SI suffix (e.g. 100.1M, 430k), or 'show' to dump config",
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
        parser.add_argument(
            "--host",
            help="Reconnect to a different host (rtltcp/spyserver). Auto-restarts.",
        )
        parser.add_argument(
            "--port",
            type=int,
            help="Reconnect to a different port (rtltcp/spyserver). Auto-restarts.",
        )

    def run(self, args: Namespace) -> str:
        manager = get_engine()
        did = args.device_id or get_focused_device_id()

        if args.frequency == "show":
            return self._show_config(manager, did)

        # Allow `--host host:port` as a shorthand for `--host host --port port`.
        # Explicit `--port` wins over the embedded one.
        if args.host is not None and ":" in args.host:
            host_part, _, port_part = args.host.rpartition(":")
            if not port_part.isdigit():
                return error(f"invalid port in --host {args.host!r}")
            args.host = host_part
            if args.port is None:
                args.port = int(port_part)

        endpoint_error = self._apply_endpoint(manager, did, args)
        if endpoint_error is not None:
            return endpoint_error

        changes: dict[str, Any] = {}

        if args.frequency is not None:
            changes["center_frequency"] = float(parse_hz(args.frequency))
        if args.frequency_flag is not None:
            changes["center_frequency"] = float(parse_hz(args.frequency_flag))
        if args.sample_rate is not None:
            changes["sample_rate"] = float(parse_hz(args.sample_rate))
        if args.gain is not None or args.agc is not None or args.hw_agc is not None:
            device = manager.get_device(did)
            if not device.device.supports_gain_control:
                return f"gain locked by {device_id(did)} (--gain/--agc/--hw-agc unavailable)"
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

        endpoint_changed = args.host is not None or args.port is not None
        if not changes and not endpoint_changed:
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

        if args.host is not None:
            summary["host"] = f"[yellow]{args.host}[/]"
        if args.port is not None:
            summary["port"] = f"[yellow]{args.port}[/]"

        return f"{success('Updated ' + device_id(did))}: {fields(summary)}"

    def _show_config(self, manager: SDREngine, did: str | None) -> str:
        if did is None:
            return error("No device focused")
        context = manager.get_device(did)

        lines: list[str] = []
        lines.append(
            f"{device_id(did)} [dim]({context.device_type}, {state(context.state.value)})[/]"
        )
        lines.append(header("Params"))
        lines.extend(_dump_dataclass(context.params))
        lines.append(header("Device"))
        lines.extend(_dump_dataclass(context.config))
        lines.append(header("Global"))
        lines.extend(_dump_dataclass(manager.config))
        return "\n".join(lines)

    def _apply_endpoint(self, manager, did: str, args: Namespace) -> str | None:
        """Stop-swap-restart the device with new host/port from args.

        No-op when neither --host nor --port given. Returns an error message
        when the device type doesn't support endpoint reconfig, otherwise None.
        """
        if args.host is None and args.port is None:
            return None

        context = manager.get_device(did)
        if not isinstance(context.params, NetworkDeviceParams):
            return f"--host/--port not applicable to {device_id(did)} ({context.device_type})"

        # `replace` requires a dataclass; mypy can't see that the Protocol is
        # implemented by frozen dataclasses (RTLTCPParams / SpyServerParams).
        new_params = replace(
            context.params,  # type: ignore[type-var]
            host=context.params.host if args.host is None else args.host,
            port=context.params.port if args.port is None else args.port,
        )

        was_running = context.state == DeviceState.RUNNING
        if was_running:
            manager.stop_device(did)
        manager.reconfigure_device_params(did, new_params)
        if was_running:
            manager.start_device(did)
        return None

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
        if flag is None and "show".startswith(prefix):
            return [Completion("show", "Dump current device and global config")]
        return []
