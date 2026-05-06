from argparse import Namespace
from pathlib import Path

from tsdr.core.preferences import save_device
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import ConfigurationError
from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices import (
    DeviceParams,
    IQFileParams,
    MockParams,
    RTLSDRParams,
    RTLTCPParams,
    SoapySDRParams,
)
from tsdr.devices.iq_file import parse_sample_rate_from_filename
from tsdr.tui.commands._format import device_id, success
from tsdr.tui.commands.base import Command, CommandParser


class SDRAddCommand(Command):
    @property
    def description(self) -> str:
        return "Add a new SDR device"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("device_id")
        parser.add_argument(
            "--type",
            required=True,
            dest="device_type",
            choices=["rtltcp", "rtlsdr", "mock", "iq-file", "soapy"],
        )
        parser.add_argument("--host", default="localhost", help="TCP host (for rtltcp)")
        parser.add_argument("--port", type=int, default=1234, help="TCP port (for rtltcp)")
        parser.add_argument("--path", help="File path (for iq-file)")
        parser.add_argument(
            "--format", dest="sample_format", choices=["cu8", "cf32"], help="Sample format"
        )
        parser.add_argument("--driver", default="", help="SoapySDR driver (for soapy)")
        parser.add_argument("--serial", default="", help="Device serial (for soapy, rtlsdr)")
        parser.add_argument("--antenna", default="", help="Antenna port (for soapy)")
        parser.add_argument("--device-args", default="", help="Extra SoapySDR args (for soapy)")
        parser.add_argument(
            "--device-index", type=int, default=0, help="USB device index (for rtlsdr)"
        )
        parser.add_argument(
            "--frequency", type=float, default=100.0, help="Center frequency in MHz"
        )
        parser.add_argument("--sample-rate", type=float, help="Sample rate in Hz")
        parser.add_argument("--buffer-size", type=int, help="Bytes per read")

    def run(self, args: Namespace) -> str:
        manager = get_engine()

        config_overrides: dict[str, object] = {}
        config_overrides["center_frequency"] = args.frequency * 1e6
        if args.sample_rate is not None:
            config_overrides["sample_rate"] = args.sample_rate
        if args.buffer_size is not None:
            config_overrides["buffer_size"] = args.buffer_size

        params: DeviceParams
        if args.device_type == "rtltcp":
            params = RTLTCPParams(host=args.host, port=args.port)
        elif args.device_type == "rtlsdr":
            params = RTLSDRParams(serial=args.serial, device_index=args.device_index)
        elif args.device_type == "mock":
            params = MockParams()
        elif args.device_type == "iq-file":
            if not args.path:
                raise ConfigurationError("--path is required for iq-file device")
            fmt: SampleFormat | None = None
            if args.sample_format:
                format_map = {"cu8": SampleFormat.UINT8_IQ, "cf32": SampleFormat.COMPLEX64}
                fmt = format_map.get(args.sample_format)
            params = IQFileParams(path=args.path, sample_format=fmt)
            if args.sample_rate is None:
                parsed_rate = parse_sample_rate_from_filename(Path(args.path).name)
                if parsed_rate is not None:
                    config_overrides["sample_rate"] = parsed_rate
        elif args.device_type == "soapy":
            device_args = args.device_args
            if args.host != "localhost":
                remote = (
                    f"tcp://{args.host}" if args.port == 1234 else f"tcp://{args.host}:{args.port}"
                )
                device_args = (
                    f"remote={remote},{device_args}" if device_args else f"remote={remote}"
                )
            params = SoapySDRParams(
                driver=args.driver or ("remote" if args.host != "localhost" else ""),
                serial=args.serial,
                antenna=args.antenna,
                device_args=device_args,
            )
        else:
            raise ConfigurationError(f"Unknown device type: {args.device_type}")

        device_config: DeviceConfig | None = None
        if config_overrides:
            device_config = DeviceConfig(**config_overrides)  # type: ignore[arg-type]

        manager.add_device(args.device_id, args.device_type, params, device_config)
        save_device(manager)
        return success(f"Added {device_id(args.device_id)} [dim]({args.device_type})[/]")
