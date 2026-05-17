from __future__ import annotations

import logging
from argparse import Namespace
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from tsdr.core.sdr.config import PipelineConfig, StageType
from tsdr.core.sdr.engine import SDREngine, get_engine
from tsdr.core.sdr.exceptions import ConfigurationError, SDRException
from tsdr.core.units import parse_hz
from tsdr.tui.commands._format import device_id, safe, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path("samples")
PIPELINE_NAME = "recording"
_MIN_SR_FOR_AUTOCOMPLETE = 30_000
_RATIONAL_DENOM_LIMIT = 10_000


class SDRRecordCommand(Command):
    """Record IQ from the focused device to samples/.

    Three call patterns:
      * ``record for SECONDS`` - record for N seconds, then auto-stop.
      * ``record start [--sr H] [--device D] [--output P]`` - begin background
        recording. Runs until ``record stop`` is called.
      * ``record stop [--device D]`` - end a running recording.
    """

    @property
    def description(self) -> str:
        return "Record IQ to samples/ (record for <sec> | record start | record stop)"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="action")

        for_p = sub.add_parser("for")
        for_p.add_argument("seconds", type=float, help="Duration in seconds")
        _add_common(for_p)

        start_p = sub.add_parser("start")
        _add_common(start_p)

        stop_p = sub.add_parser("stop")
        stop_p.add_argument("--device", dest="device_id", default=None)

    def run(self, args: Namespace) -> str:
        if args.action is None:
            return self.help_text()

        engine = get_engine()
        did = args.device_id or get_focused_device_id()

        if args.action == "stop":
            return self._stop(engine, did)
        if args.action == "start":
            return self._start(engine, did, args, duration=None)
        if args.action == "for":
            if args.seconds <= 0:
                raise ConfigurationError("Duration must be positive")
            return self._start(engine, did, args, duration=args.seconds)
        raise SDRException(f"Unknown record action: {args.action}")

    def _start(
        self,
        engine: SDREngine,
        did: str,
        args: Namespace,
        *,
        duration: float | None,
    ) -> str:
        device = engine.get_device(did)
        if PIPELINE_NAME in device.config.pipelines:
            raise SDRException(f"Already recording on {did} - stop it first with 'record stop'.")

        device_rate = device.config.sample_rate
        target_rate = (
            parse_hz(args.sample_rate) if args.sample_rate is not None else int(device_rate)
        )
        if target_rate > device_rate:
            raise ConfigurationError(
                f"Target rate {target_rate} Hz exceeds device rate {int(device_rate)} Hz - "
                f"choose a lower --sr."
            )

        resample, out_rate = _compute_resample(device_rate, target_rate)
        max_samples = int(duration * out_rate) if duration is not None else None

        output_path = (
            Path(args.output)
            if args.output
            else _autogenerate_path(
                center_frequency=device.config.center_frequency,
                out_rate=out_rate,
                duration=int(duration) if duration is not None else None,
                rf_gain=device.config.rf_gain,
            )
        )

        pipeline_config = PipelineConfig(
            stages=(StageType.RECORD,),
            record_path=str(output_path),
            record_resample=resample,
            record_max_samples=max_samples,
        )

        engine.add_pipeline(did, PIPELINE_NAME, pipeline_config)

        path_md = safe(str(output_path))
        rate_md = f"[yellow]{int(out_rate)} Hz[/]"
        if duration is not None:
            return success(f"Recording {duration:g}s on {device_id(did)} @ {rate_md} → {path_md}")
        return success(
            f"Recording on {device_id(did)} @ {rate_md} → {path_md} "
            "[dim](stop with 'record stop')[/]"
        )

    def _stop(self, engine: SDREngine, did: str) -> str:
        device = engine.get_device(did)
        pipeline_config = device.config.pipelines.get(PIPELINE_NAME)
        if pipeline_config is None:
            raise SDRException(f"Not recording on '{did}'")
        path = pipeline_config.record_path or "<unknown>"
        engine.remove_pipeline(did, PIPELINE_NAME)
        return success(f"Stopped recording → {safe(path)}")

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
        if flag == "--sr":
            return _sr_divisor_completions(prefix)
        if flag == "--output":
            return []
        if not tokens:
            return [
                Completion("for", "record for N seconds"),
                Completion("start", "begin a background recording"),
                Completion("stop", "end a running recording"),
            ]
        return []


def _add_common(parser: CommandParser) -> None:
    parser.add_argument(
        "--sr",
        dest="sample_rate",
        default=None,
        help="Target output sample rate with SI suffix (e.g. 250k, 1.2M; default: device rate)",
    )
    parser.add_argument(
        "--device",
        dest="device_id",
        default=None,
        help="Device id (default: focused device)",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Output file path (default: samples/<auto>.cu8.zst)",
    )


def _compute_resample(device_rate: float, target_rate: int) -> tuple[tuple[int, int] | None, float]:
    """Return (resample_tuple, effective_output_rate).

    ``None`` means no resampling is needed. For integer divisors the returned
    tuple is ``(1, N)`` which the stage routes through StreamingDecimFilter.
    """
    device_int = int(device_rate)
    if target_rate == device_int:
        return None, device_rate
    if device_int % target_rate == 0:
        return (1, device_int // target_rate), float(target_rate)
    frac = Fraction(target_rate, device_int).limit_denominator(_RATIONAL_DENOM_LIMIT)
    out_rate = device_rate * frac.numerator / frac.denominator
    return (frac.numerator, frac.denominator), out_rate


def _autogenerate_path(
    center_frequency: float, out_rate: float, duration: int | None, rf_gain: float
) -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    fmhz = f"{center_frequency / 1e6:g}"
    if "." not in fmhz and "e" not in fmhz:
        fmhz += ".0"
    sr_khz = int(round(out_rate / 1000))
    gain = int(round(rf_gain))
    ts = datetime.now().strftime("%Y%m%dT%H%M")
    dur_segment = f"_dur={int(round(duration))}s" if duration is not None else ""
    name = f"freq={fmhz}M_sr={sr_khz}k{dur_segment}_gain={gain}_{ts}.cu8.zst"
    return SAMPLES_DIR / name


def _sr_divisor_completions(prefix: str) -> list[Completion]:
    try:
        engine = get_engine()
    except RuntimeError:
        return []
    if engine.focused_device is None:
        return []
    device = engine.devices.get(engine.focused_device)
    if device is None:
        return []
    device_rate = int(device.config.sample_rate)

    candidates: list[tuple[int, int]] = []
    n = 2
    while device_rate // n >= _MIN_SR_FOR_AUTOCOMPLETE:
        if device_rate % n == 0:
            candidates.append((device_rate // n, n))
        n += 1

    return [
        Completion(str(rate), f"÷{divisor}")
        for rate, divisor in candidates
        if str(rate).startswith(prefix)
    ]
