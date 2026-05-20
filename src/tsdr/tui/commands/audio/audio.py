from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import ConfigurationError
from tsdr.tui.commands._format import field, header, safe, success
from tsdr.tui.commands.base import Command, CommandParser


class AudioCommand(Command):
    @property
    def description(self) -> str:
        return "Audio device management"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="subcommand")

        # device
        device_p = sub.add_parser("device")
        device_sub = device_p.add_subparsers(dest="device_action")
        device_sub.add_parser("list")
        set_p = device_sub.add_parser("set")
        set_p.add_argument("device_spec", help="Device name (substring match), id, or 'default'")

        # sources
        sub.add_parser("sources")

    def run(self, args: Namespace) -> str:
        if not args.subcommand:
            return self.help_text()

        if args.subcommand == "device":
            return self._device(args)
        elif args.subcommand == "sources":
            return self._sources()
        else:
            raise ConfigurationError(f"Unknown subcommand '{args.subcommand}'")

    def _device(self, args: Namespace) -> str:
        if not getattr(args, "device_action", None):
            return "Usage: audio device <list|set>"

        engine = get_engine()

        if args.device_action == "list":
            devices = engine.list_audio_devices()
            if not devices:
                return "No audio output devices found"
            lines = [header("Audio Output Devices")]
            for dev in devices:
                params = f"{field('channels', f'[cyan]{dev["channels"]}[/]')}"
                lines.append(f"  [bold]{dev['name']}[/] [dim]({params})[/]")
            return "\n".join(lines)

        elif args.device_action == "set":
            device_spec = args.device_spec
            device_name = None if device_spec.lower() == "default" else device_spec
            engine.set_audio_output_device(device_name)
            shown = "default" if device_name is None else device_name
            return success(f"Set audio output device to [bold cyan]{safe(shown)}[/]")

        raise ConfigurationError(f"Unknown device action '{args.device_action}'")

    def _sources(self) -> str:
        sources = get_engine().get_active_audio_sources()
        if not sources:
            return "No active audio sources"
        lines = [header("Active Audio Sources")]
        for source_id in sources:
            lines.append(f"  [dim cyan]{source_id}[/]")
        return "\n".join(lines)
