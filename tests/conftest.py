import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsdr.core import storage
from tsdr.core.devices import init_device_store
from tsdr.core.events.events import Event
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, StageType
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.workers import audio_worker as audio_worker_module
from tsdr.devices import IQFileParams


class _FakeSpeaker:
    """SoundCard speaker stand-in that never opens a real audio device."""

    def __init__(self, id_: str = "fake-default", name: str = "Fake Speaker") -> None:
        self.id = id_
        self.name = name
        self.channels = 2

    def player(self, **_kwargs) -> _FakePlayerCM:
        return _FakePlayerCM()


class _FakePlayer:
    """SoundCard _Player stand-in with the public-ish ``_queue`` attribute."""

    def __init__(self) -> None:
        self._queue: deque = deque()

    def play(self, _data, wait: bool = True) -> None:
        # Mimic SoundCard: append to internal queue. We don't actually play.
        self._queue.append(None)


class _FakePlayerCM:
    def __enter__(self) -> _FakePlayer:
        return _FakePlayer()

    def __exit__(self, *_args) -> None:
        return None


@pytest.fixture(autouse=True)
def _block_audio_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from opening a real audio output stream."""
    fake_sc = SimpleNamespace(
        default_speaker=lambda: _FakeSpeaker(),
        get_speaker=lambda spec: _FakeSpeaker(id_=spec, name=spec),
        all_speakers=lambda: [_FakeSpeaker()],
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)


@pytest.fixture(autouse=True)
def _isolate_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect storage to a tmp dir and init the device store so save_device works."""
    monkeypatch.setattr(storage, "config_dir", lambda: tmp_path)
    init_device_store()


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
        buffer_samples: int | None = None,
    ) -> list[Event]:
        engine = SDREngine()
        config_kwargs: dict = {"sample_rate": sample_rate}
        if buffer_samples is not None:
            config_kwargs["buffer_samples"] = buffer_samples
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
