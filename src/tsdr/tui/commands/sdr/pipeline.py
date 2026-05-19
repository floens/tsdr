from argparse import Namespace
from dataclasses import replace
from types import MappingProxyType

from tsdr.core.sdr.config import StageType
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.tui.commands._format import freq_mhz, rate_sps, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

_STAGE_TYPE_VALUES = [st.value for st in StageType]


class SDRPipelineCommand(Command):
    @property
    def description(self) -> str:
        return "Show or modify pipelines (add/remove stages)"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("--device", dest="device_id")
        sub = parser.add_subparsers(dest="action")

        add_p = sub.add_parser("add", help="Add a stage to a pipeline")
        add_p.add_argument("stage_type", choices=_STAGE_TYPE_VALUES)
        add_p.add_argument("--name", default="visualization", help="Pipeline name")

        rm_p = sub.add_parser("remove", help="Remove a stage from a pipeline")
        rm_p.add_argument("index", type=int)
        rm_p.add_argument("--name", default="visualization", help="Pipeline name")

    def run(self, args: Namespace) -> str:
        action = getattr(args, "action", None)
        if action == "add":
            return self._add(args)
        if action == "remove":
            return self._remove(args)
        return self._show(args)

    def _show(self, args: Namespace) -> str:
        manager = get_engine()
        device_id = args.device_id or get_focused_device_id()

        if device_id not in manager.devices:
            raise SDRException(f"Device {device_id} not found")

        context = manager.devices[device_id]

        running = bool(context.pipelines)
        title = f"Pipelines for [bold cyan]{device_id}[/]"
        if not running:
            title += " [dim](not running)[/]"

        lines = [
            f"{title}:",
            "",
            f"  [bold white]IQ Samples[/] "
            f"[dim]@[/] {rate_sps(context.config.sample_rate)} "
            f"[dim]·[/] {freq_mhz(context.config.center_frequency)}",
        ]

        if running:
            for pipeline_name, pipeline in context.pipelines.items():
                lines.append("")
                lines.append(f"  [bold blue]\\[{pipeline_name}][/]")
                n = len(pipeline.stages)
                for i, stage in enumerate(pipeline.stages):
                    lines.append(f"    [dim]{i}[/]  {self._format_stage(stage)}")
                    if i < n - 1:
                        lines.append("       [dim]↓[/]")
        else:
            configured = context.config.pipelines
            if not configured:
                lines.append("")
                lines.append("  [dim](no pipelines configured)[/]")
            else:
                for pipeline_name, pc in configured.items():
                    lines.append("")
                    lines.append(f"  [bold blue]\\[{pipeline_name}][/]")
                    n = len(pc.stages)
                    for i, st in enumerate(pc.stages):
                        lines.append(f"    [dim]{i}[/]  {self._format_stage_type(st)}")
                        if i < n - 1:
                            lines.append("       [dim]↓[/]")

        return "\n".join(lines)

    def _format_stage(self, stage) -> str:
        stage_type = type(stage).__name__

        if stage_type == "FFTStage":
            return (
                f"[magenta]FFTStage[/] "
                f"[dim]size=[/]{stage.fft_size} "
                f"[dim]window=[/]{stage.window_type}"
            )
        elif stage_type == "AGCStage":
            return "[yellow]AGCStage[/]"
        elif stage_type == "EventEmitterStage":
            return "[cyan]EventEmitterStage[/]"
        elif stage_type == "FrequencyShiftStage":
            offset_khz = stage.frequency_offset / 1e3
            return f"[yellow]FrequencyShiftStage[/] [dim]offset=[/]{offset_khz:+.1f} kHz"
        elif stage_type == "DemodulatorStage":
            info = stage.demodulator.info()
            return (
                f"[green]DemodulatorStage[/] "
                f"[bold green]{info.label}[/] "
                f"[dim]({info.modulation})[/]"
            )
        elif stage_type == "RecordStage":
            return f"[cyan]RecordStage[/] [dim]→[/] {stage._path.name}"
        else:
            return f"[white]{stage_type}[/]"

    @staticmethod
    def _format_stage_type(st: StageType) -> str:
        color = {
            StageType.AGC: "yellow",
            StageType.FREQUENCY_SHIFT: "yellow",
            StageType.FFT: "magenta",
            StageType.EVENT_EMITTER: "cyan",
            StageType.DEMODULATOR: "green",
            StageType.RECORD: "cyan",
        }.get(st, "white")
        return f"[{color}]{st.value}[/]"

    def _add(self, args: Namespace) -> str:
        manager = get_engine()
        device_id = args.device_id or get_focused_device_id()
        context = manager.get_device(device_id)
        pipeline_name = args.name

        pipeline_config = context.config.pipelines.get(pipeline_name)
        if pipeline_config is None:
            raise SDRException(f"Pipeline '{pipeline_name}' doesn't exist for device {device_id}")

        new_stage = StageType(args.stage_type)
        new_stages = pipeline_config.stages + (new_stage,)
        new_pipeline_config = replace(pipeline_config, stages=new_stages)

        new_pipelines = MappingProxyType(
            dict(context.config.pipelines) | {pipeline_name: new_pipeline_config}
        )
        manager.update_device_config(device_id, pipelines=new_pipelines)
        return success(f"Added [magenta]{args.stage_type}[/] to [bold blue]\\[{pipeline_name}][/]")

    def _remove(self, args: Namespace) -> str:
        manager = get_engine()
        device_id = args.device_id or get_focused_device_id()
        context = manager.get_device(device_id)
        pipeline_name = args.name

        pipeline_config = context.config.pipelines.get(pipeline_name)
        if pipeline_config is None:
            raise SDRException(f"Pipeline '{pipeline_name}' doesn't exist for device {device_id}")

        stages = list(pipeline_config.stages)
        if not (0 <= args.index < len(stages)):
            raise SDRException(f"Stage index {args.index} out of range (0-{len(stages) - 1})")
        removed = stages.pop(args.index)
        new_pipeline_config = replace(pipeline_config, stages=tuple(stages))

        new_pipelines = MappingProxyType(
            dict(context.config.pipelines) | {pipeline_name: new_pipeline_config}
        )
        manager.update_device_config(device_id, pipelines=new_pipelines)
        return success(
            f"Removed [magenta]{removed.value}[/] from [bold blue]\\[{pipeline_name}][/]"
        )

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
