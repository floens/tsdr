from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.tui.commands._format import device_id, freq_mhz, header, rate_msps, state
from tsdr.tui.commands.base import Command


class SDRListCommand(Command):
    @property
    def description(self) -> str:
        return "List all SDR devices"

    def run(self, args: Namespace) -> str:
        devices = get_engine().list_devices()

        if not devices:
            return "No devices configured"

        lines = [header("Devices")]
        for dev in devices:
            marker = "[bold yellow]*[/]" if dev["focused"] else " "
            mode = dev["mode"]
            mode_md = f"[green]{mode}[/]" if mode and mode != "OFF" else f"[dim]{mode}[/]"
            desc = dev["description"]
            desc_md = f" | [dim]{desc}[/]" if desc else ""
            lines.append(
                f"{marker} {device_id(dev['id'])}: "
                f"[dim]{dev['type']}[/] | "
                f"{state(dev['state'].lower())} | "
                f"{freq_mhz(dev['frequency'], precision=2, width=10)} | "
                f"{rate_msps(dev['sample_rate'])} | "
                f"{mode_md}"
                f"{desc_md}"
            )

        return "\n".join(lines)
