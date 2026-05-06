from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.stages.demodulator_stage import DemodulatorStage
from tsdr.radio.decoders.dab import DABDecoder
from tsdr.radio.decoders.dab.fig import _build_ensemble
from tsdr.tui.commands._format import success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id


def _get_dab_decoder(device_id: str) -> DABDecoder:
    """Find the DABDecoder on a device, or raise SDRException."""
    # TODO: this would be better if this was abstracted on the engine, somehow
    # if pipeline changes occur, this state can be lost.
    # needs to be part of configuration, so that configuration -> drives state
    engine = get_engine()
    context = engine.get_device(device_id)
    for pipeline in context.pipelines.values():
        for stage in pipeline.stages:
            if isinstance(stage, DemodulatorStage) and isinstance(stage.demodulator, DABDecoder):
                return stage.demodulator
    raise SDRException(f"No DAB decoder active on device '{device_id}'")


class DABCommand(Command):
    @property
    def description(self) -> str:
        return "DAB station selection and info"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="subcommand")
        select_p = sub.add_parser("select")
        select_p.add_argument("station", nargs="*")
        select_p.add_argument("--device", dest="device_id")
        list_p = sub.add_parser("list")
        list_p.add_argument("--device", dest="device_id")

    def run(self, args: Namespace) -> str:
        if not args.subcommand:
            return self.help_text()

        device_id = getattr(args, "device_id", None) or get_focused_device_id()
        decoder = _get_dab_decoder(device_id)

        if args.subcommand == "list":
            return self._list(decoder)
        if args.subcommand == "select":
            return self._select(decoder, device_id, args.station)
        raise SDRException(f"Unknown subcommand: {args.subcommand}")

    def _list(self, decoder: DABDecoder) -> str:
        ensemble = _build_ensemble(decoder._fig_state)
        if not ensemble.services:
            return "No services discovered yet"

        lines = [f"[bold cyan]{ensemble.label}[/] [dim]({ensemble.ensemble_id:#06x})[/]"]
        selected_id = decoder._selected_service.service_id if decoder._selected_service else None
        for svc in ensemble.services:
            is_selected = svc.service_id == selected_id
            marker = "[bold yellow]>[/]" if is_selected else " "
            kind_md = "[green]audio[/]" if svc.is_audio else "[dim]data[/]"
            label_md = f"[bold]{svc.label}[/]" if is_selected else svc.label
            lines.append(f"  {marker} {label_md} [dim]({svc.service_id:#06x})[/] [{kind_md}]")
        return "\n".join(lines)

    def _select(self, decoder: DABDecoder, device_id: str, station: list[str]) -> str:
        station_str = " ".join(station) if station else None
        if station_str is None:
            result = decoder.select_service()
            if result is None or result.startswith("Error:"):
                raise SDRException(
                    result.removeprefix("Error: ") if result else "No audio services found"
                )
            return success(f"Selected: [bold]{result}[/]")

        # Try hex service ID
        service_id = None
        try:
            service_id = int(station_str, 16)
        except ValueError:
            pass

        if service_id is not None:
            result = decoder.select_service(service_id)
            if result is None or result.startswith("Error:"):
                raise SDRException(
                    result.removeprefix("Error: ") if result else f"Service {station_str} not found"
                )
            return success(f"Selected: [bold]{result}[/]")

        # Match by label substring (case-insensitive)
        ensemble = _build_ensemble(decoder._fig_state)
        query = station_str.lower()
        matches = [s for s in ensemble.services if s.is_audio and query in s.label.lower()]

        if not matches:
            raise SDRException(f"No audio service matching '{station_str}'")
        if len(matches) > 1:
            names = ", ".join(s.label for s in matches)
            raise SDRException(f"Ambiguous match '{station_str}': {names}")

        result = decoder.select_service(matches[0].service_id)
        if result is None or result.startswith("Error:"):
            raise SDRException(result.removeprefix("Error: ") if result else "Selection failed")
        return success(f"Selected: [bold]{result}[/]")

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

        # Called for the "station" positional inside the "select" subparser
        # (base class strips "select" before calling complete)
        return self._complete_stations(prefix)

    def _complete_stations(self, prefix: str) -> list[Completion]:
        try:
            device_id = get_focused_device_id()
            decoder = _get_dab_decoder(device_id)
        except SDRException, RuntimeError:
            return []

        ensemble = _build_ensemble(decoder._fig_state)
        prefix_lower = prefix.lower()
        return [
            Completion(svc.label, f"{svc.service_id:#06x}")
            for svc in ensemble.services
            if svc.is_audio and svc.label.lower().startswith(prefix_lower)
        ]
