import queue
from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.core.sdr.config import SDRConfig
from tsdr.core.sdr.datatypes import AudioBatch
from tsdr.core.sdr.pipeline.stages.denoiser_stage import DenoiserStage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.radio.dsp.rnnoise import rnnoise_available


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(audio_queue=queue.Queue())


def _audio(n: int, rate: float = 48000.0) -> AudioBatch:
    return AudioBatch(samples=np.zeros(n, dtype=np.float32), sample_rate=rate)


def test_passthrough_drains_unchanged_when_disabled():
    stage = DenoiserStage(enabled=False)
    batch = _audio(1000)
    ctx = _ctx()
    out = stage.process(SamplesBatch(audio_batches=(batch,)), ctx)
    assert out.audio_batches == ()  # cleared after draining
    assert ctx.audio_queue.get_nowait() is batch  # same object, not denoised


def test_no_audio_queue_is_safe():
    stage = DenoiserStage(enabled=False)
    out = stage.process(
        SamplesBatch(audio_batches=(_audio(1000),)), SimpleNamespace(audio_queue=None)
    )
    assert out.audio_batches == ()


@pytest.mark.skipif(not rnnoise_available(), reason="librnnoise not available on this platform")
def test_denoises_and_drains_when_enabled():
    stage = DenoiserStage(enabled=True)
    ctx = _ctx()
    out = stage.process(SamplesBatch(audio_batches=(_audio(1000),)), ctx)
    assert out.audio_batches == ()
    drained = ctx.audio_queue.get_nowait()
    assert drained.sample_rate == 48000.0
    assert len(drained.samples) == 960  # 1000 -> two 480 frames, 40 carried
    stage.close()


@pytest.mark.skipif(not rnnoise_available(), reason="librnnoise not available on this platform")
def test_on_config_change_toggles_denoiser():
    stage = DenoiserStage(enabled=False)
    assert stage._denoiser is None
    stage.on_config_change(SDRConfig(denoise=True))
    assert stage._denoiser is not None
    stage.on_config_change(SDRConfig(denoise=False))
    assert stage._denoiser is None
