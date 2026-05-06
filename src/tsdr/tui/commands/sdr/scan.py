from argparse import Namespace

from tsdr.core.sdr.exceptions import ConfigurationError
from tsdr.devices.soapy import _HAS_SOAPY, _SoapySDR
from tsdr.tui.commands._format import field, header
from tsdr.tui.commands.base import Command, CommandParser


class ScanCommand(Command):
    @property
    def description(self) -> str:
        return "Scan for available SoapySDR devices"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("--driver", default="", help="Filter by driver (e.g. rtlsdr, remote)")
        parser.add_argument("--remote", default="", help="Remote hostname (SoapyRemote)")

    def run(self, args: Namespace) -> str:
        if not _HAS_SOAPY:
            raise ConfigurationError("SoapySDR not installed. Install via system package manager.")

        kwargs: dict[str, str] = {}
        if args.driver:
            kwargs["driver"] = args.driver
        if args.remote:
            kwargs["driver"] = "remote"
            kwargs["remote"] = args.remote
        results = _SoapySDR.Device.enumerate(kwargs)

        if not results:
            return "No SoapySDR devices found"

        lines = [header(f"Found [cyan]{len(results)}[/] device(s)")]
        for i, dev in enumerate(results):
            lines.append(f"  [dim][{i}][/]")
            for key in ("driver", "label", "serial", "product", "manufacturer", "remote"):
                if key in dev:
                    value = f"[cyan]{dev[key]}[/]" if key == "driver" else str(dev[key])
                    lines.append(f"    {field(key, value)}")
            extra = set(dev.keys()) - {
                "driver",
                "label",
                "serial",
                "product",
                "manufacturer",
                "remote",
            }
            for key in sorted(extra):
                lines.append(f"    {field(key, str(dev[key]))}")

        return "\n".join(lines)
