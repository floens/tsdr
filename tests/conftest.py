import sys
import time
from types import SimpleNamespace

import pytest

from tsdr.core.events.events import Event
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, StageType
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import IQFileParams


class _SilentOutputStream:
    """Stand-in for sounddevice.OutputStream that never touches the hardware."""

    def __init__(self, **_kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass

    def write(self, *_args, **_kwargs) -> None:
        pass

    @property
    def active(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _block_audio_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from opening a real audio output stream."""
    fake_sd = SimpleNamespace(
        OutputStream=_SilentOutputStream,
        query_devices=lambda *a, **kw: [],
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)


@pytest.fixture
def run_pipeline():
    """Factory fixture: run an IQ file through a demod pipeline, collect events.

    Starts the device first (creating the default visualization pipeline),
    then adds the audio/demod pipeline. The IQ file device loops at EOF,
    so use `duration` to control how long the pipeline runs.
    """

    def _run(
        iq_path: str,
        mode: str,
        sample_rate: float,
        event_type: type[Event],
        duration: float | None = None,
        timeout: float = 15.0,
        min_events: int = 1,
        buffer_size: int | None = None,
    ) -> list[Event]:
        engine = SDREngine()
        config_kwargs: dict = {"sample_rate": sample_rate}
        if buffer_size is not None:
            config_kwargs["buffer_size"] = buffer_size
        config = DeviceConfig(**config_kwargs)

        params = IQFileParams(path=iq_path)
        engine.add_device("test", "iq-file", params, config)

        collected: list[Event] = []

        def handler(event: Event) -> None:
            collected.append(event)

        engine.event_bus.subscribe(event_type, handler)

        # Start device first - creates default visualization pipeline (FFT, stats)
        engine.start_device("test")

        # Add audio pipeline via config
        audio_config = PipelineConfig(
            stages=(StageType.DEMODULATOR,),
            demod_mode=mode,
        )
        engine.add_pipeline("test", "audio", audio_config)

        if duration is not None:
            # Run for a fixed duration (e.g. to loop the IQ file multiple times)
            time.sleep(duration)
        else:
            # Wait until we have enough events or timeout
            deadline = time.monotonic() + timeout
            while len(collected) < min_events and time.monotonic() < deadline:
                time.sleep(0.1)

        engine.stop_device("test")
        engine.shutdown(timeout=5.0)
        return collected

    return _run
