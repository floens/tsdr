import logging
import signal
import sys
import threading
from argparse import Namespace

import tsdr.tui.commands  # noqa: F401  # trigger command self-registration
from tsdr.core.band_stack import init_band_stack
from tsdr.core.bandplans import init_bandplan_store
from tsdr.core.clock_sync import get_clock_sync_monitor, init_clock_sync_monitor
from tsdr.core.devices import init_device_store
from tsdr.core.events.events import (
    DecoderOutputEvent,
    DeviceCapabilitiesChangedEvent,
    DeviceErrorEvent,
    DeviceStateChangedEvent,
    Event,
    PipelineChangedEvent,
    PipelineErrorEvent,
    RecordingFinishedEvent,
)
from tsdr.core.memories import init_memory_store
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.tuning_state import init_tuning_state
from tsdr.tui.commands.base import Command
from tsdr.tui.commands.registry import COMMANDS, execute
from tsdr.tui.state import init_ui_state

logger = logging.getLogger(__name__)

_shutdown = threading.Event()


class _HeadlessExitCommand(Command):
    @property
    def description(self) -> str:
        return "Exit the application"

    def run(self, args: Namespace) -> str:
        _shutdown.set()
        return "Exiting..."


def _emit(line: str) -> None:
    print(line, flush=True)


def _on_device_state(event: Event) -> None:
    if not isinstance(event, DeviceStateChangedEvent):
        return
    _emit(f"EVENT device_state device={event.device_id} running={event.running}")


def _on_device_error(event: Event) -> None:
    if not isinstance(event, DeviceErrorEvent):
        return
    _emit(f"EVENT device_error device={event.device_id} error={event.error!r}")


def _on_device_capabilities(event: Event) -> None:
    if not isinstance(event, DeviceCapabilitiesChangedEvent):
        return
    _emit(f"EVENT device_capabilities device={event.device_id} capabilities={event.capabilities}")


def _on_pipeline_changed(event: Event) -> None:
    if not isinstance(event, PipelineChangedEvent):
        return
    _emit(
        f"EVENT pipeline device={event.device_id} name={event.pipeline_name} "
        f"active={event.active} mode={event.mode!r}"
    )


def _on_pipeline_error(event: Event) -> None:
    if not isinstance(event, PipelineErrorEvent):
        return
    _emit(
        f"EVENT pipeline_error device={event.device_id} pipeline={event.pipeline_id} "
        f"stage={event.stage_name} error={event.error!r}"
    )


def _on_recording_finished(event: Event) -> None:
    if not isinstance(event, RecordingFinishedEvent):
        return
    _emit(
        f"EVENT recording_finished device={event.device_id} pipeline={event.pipeline_name} "
        f"path={event.path!r} samples={event.samples_written}"
    )


def _on_decoder_output(event: Event) -> None:
    if not isinstance(event, DecoderOutputEvent):
        return
    for msg in event.messages:
        _emit(
            f"EVENT decoder_output device={event.device_id} protocol={event.protocol} "
            f"text={msg.text!r}"
        )


def _subscribe_events(engine: SDREngine) -> None:
    bus = engine.event_bus
    bus.subscribe(DeviceStateChangedEvent, _on_device_state)
    bus.subscribe(DeviceErrorEvent, _on_device_error)
    bus.subscribe(DeviceCapabilitiesChangedEvent, _on_device_capabilities)
    bus.subscribe(PipelineChangedEvent, _on_pipeline_changed)
    bus.subscribe(PipelineErrorEvent, _on_pipeline_error)
    bus.subscribe(RecordingFinishedEvent, _on_recording_finished)
    bus.subscribe(DecoderOutputEvent, _on_decoder_output)


def _run_line(line: str) -> None:
    line = line.strip()
    if not line:
        return
    result = execute(line)
    if result:
        print(result, flush=True)


def _stdin_loop() -> None:
    for raw in sys.stdin:
        _run_line(raw)
        if _shutdown.is_set():
            return
    _shutdown.set()


def run_headless(startup_commands: list[str]) -> int:
    engine = SDREngine()
    init_memory_store()
    init_band_stack()
    init_bandplan_store()
    init_device_store()
    init_tuning_state()
    init_ui_state()
    init_clock_sync_monitor()

    COMMANDS["exit"] = _HeadlessExitCommand()
    COMMANDS["quit"] = _HeadlessExitCommand()

    _subscribe_events(engine)

    def _on_signal(signum, frame) -> None:
        logger.info("signal_received signum=%s action=shutdown", signum)
        _shutdown.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    for line in startup_commands:
        _run_line(line)
        if _shutdown.is_set():
            break

    if not _shutdown.is_set():
        threading.Thread(target=_stdin_loop, name="headless-stdin", daemon=True).start()
        _shutdown.wait()

    try:
        engine.shutdown(timeout=2.0)
    except Exception as e:  # noqa: BLE001 - last-resort cleanup
        logger.error("headless_shutdown_failed error=%r", e, exc_info=True)

    get_clock_sync_monitor().shutdown()

    return 0
