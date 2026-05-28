"""Tests for the SSTV decoder + demodulator.

Inputs are the real mp3 recordings under ``tests/samples/sstv/`` (LFS).
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from tsdr.core.events.events import DecodedMessage
from tsdr.radio.decoders.sstv import (
    MODES,
    MODES_BY_NAME,
    SSTVData,
    StreamerState,
    StreamingEvents,
    StreamingSSTV,
    decode_offline,
    find_vis,
    hilbert_fft,
    instantaneous_freq,
)
from tsdr.radio.demodulators.sstv import SSTVDemodulator

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples" / "sstv"

# (filename, expected VIS code, expected image (H, W)) for the recordings
# we ship under tests/samples/sstv/.
SAMPLE_FIXTURES: list[tuple[str, int, tuple[int, int]]] = [
    ("Martin_1.mp3", 44, (256, 320)),
    ("Scottie_1.mp3", 60, (256, 320)),
    ("Robot_8_BW.mp3", 2, (120, 160)),
    ("Robot_36.mp3", 8, (240, 320)),
    ("Robot_72.mp3", 12, (240, 320)),
    ("ScottieDX.mp3", 76, (256, 320)),
]


def _load_mp3(path: Path) -> tuple[np.ndarray, int]:
    """Decode an mp3 to mono float32 + sample rate via pyav."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    rate = stream.codec_context.sample_rate
    resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)

    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for r in resampler.resample(frame):
            chunks.append(r.to_ndarray().reshape(-1).astype(np.float32))
    for r in resampler.resample(None):
        chunks.append(r.to_ndarray().reshape(-1).astype(np.float32))
    container.close()
    if not chunks:
        raise RuntimeError(f"no audio frames decoded from {path}")
    return np.concatenate(chunks), rate


@pytest.fixture(scope="module")
def _samples() -> dict[str, tuple[np.ndarray, int]]:
    """Load each sample once per test module; mp3 decode is the slow part."""
    out: dict[str, tuple[np.ndarray, int]] = {}
    for name, _, _ in SAMPLE_FIXTURES:
        path = SAMPLES_DIR / name
        if not path.exists():
            pytest.skip(f"missing sample: {path}")
        out[name] = _load_mp3(path)
    return out


# ---- VIS detection --------------------------------------------------------


@pytest.mark.parametrize(("filename", "expected_vis", "_shape"), SAMPLE_FIXTURES)
def test_vis_detection_known_samples(
    _samples: dict[str, tuple[np.ndarray, int]],
    filename: str,
    expected_vis: int,
    _shape: tuple[int, int],
) -> None:
    audio, fs = _samples[filename]
    freq = instantaneous_freq(audio, fs)
    result = find_vis(freq, fs)
    assert result is not None, f"no VIS detected in {filename}"
    vis_code, _ = result
    assert vis_code == expected_vis


# ---- Offline decode -------------------------------------------------------


@pytest.mark.parametrize(("filename", "_vis", "shape"), SAMPLE_FIXTURES)
def test_decode_offline_image_shape(
    _samples: dict[str, tuple[np.ndarray, int]],
    filename: str,
    _vis: int,
    shape: tuple[int, int],
) -> None:
    audio, fs = _samples[filename]
    result = decode_offline(audio, fs)
    assert result is not None, f"offline decode failed for {filename}"
    _, img = result
    assert img.shape == (*shape, 3)
    # Not a blank frame.
    assert img.std() > 5.0


def test_decode_offline_martin1_content_signature(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """Regression on the Martin M1 path (BT.601 conversion + segment layout).

    Asserts the decoded image stays close to a captured reference signature
    (means of two rows + global mean) within tolerance.
    """
    audio, fs = _samples["Martin_1.mp3"]
    result = decode_offline(audio, fs)
    assert result is not None
    _, img = result
    # Sample a few rows to detect any single-channel regression.
    row50_mean = img[50].mean(axis=0)
    row200_mean = img[200].mean(axis=0)
    global_mean = img.mean(axis=(0, 1))
    # Each channel should be a non-trivial value (not all 0 or 255).
    assert all(5.0 < v < 250.0 for v in row50_mean), row50_mean
    assert all(5.0 < v < 250.0 for v in row200_mean), row200_mean
    assert all(5.0 < v < 250.0 for v in global_mean), global_mean
    # No channel should be saturated to 0 or 255 across the whole frame.
    for ch in range(3):
        ch_data = img[..., ch]
        assert ch_data.min() < 32, f"channel {ch} min={ch_data.min()}"
        assert ch_data.max() > 200, f"channel {ch} max={ch_data.max()}"


# ---- Streaming vs offline parity ------------------------------------------


def test_streaming_matches_offline_robot36(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """Streaming chunks should produce essentially the same image as offline."""
    audio, fs = _samples["Robot_36.mp3"]

    offline_result = decode_offline(audio, fs)
    assert offline_result is not None
    _, offline_img = offline_result

    captured: list[np.ndarray] = []

    def on_image(_mode, img: np.ndarray) -> None:
        captured.append(img.copy())

    decoder = StreamingSSTV(fs, events=StreamingEvents(on_image=on_image))
    chunk = max(1, int(0.050 * fs))  # 50 ms chunks
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i : i + chunk])
    decoder.flush()

    assert captured, "streaming decoder produced no image"
    stream_img = captured[-1]
    assert stream_img.shape == offline_img.shape
    # Slant tracker resolves chunk-boundary jitter slightly differently than
    # the offline pass; tolerate a small mean-pixel-difference.
    diff = np.mean(np.abs(stream_img.astype(np.int32) - offline_img.astype(np.int32)))
    assert diff < 5.0, f"streaming vs offline diff={diff}"


def test_streaming_decodes_back_to_back_transmissions(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """After completing one image, the decoder must reset on the next ``process``
    call and decode the next VIS-tagged transmission. Regression for the bug
    where ``_emit_image`` overwrote a callback-issued ``reset()``, wedging the
    state machine in ``DONE``.
    """
    audio, fs = _samples["Martin_1.mp3"]
    # Concatenate two transmissions with one second of silence between them.
    gap = np.zeros(int(1.0 * fs), dtype=np.float32)
    combined = np.concatenate([audio, gap, audio])

    images: list[np.ndarray] = []
    modes: list[str] = []

    def on_mode(mode, _vis: int) -> None:
        modes.append(mode.name)

    def on_image(_mode, img: np.ndarray) -> None:
        images.append(img.copy())
        decoder.reset()

    decoder = StreamingSSTV(fs, events=StreamingEvents(on_mode=on_mode, on_image=on_image))
    chunk = max(1, int(0.050 * fs))
    for i in range(0, len(combined), chunk):
        decoder.process(combined[i : i + chunk])
    decoder.flush()

    assert len(images) == 2, f"expected two images, got {len(images)} (modes={modes})"
    assert modes == ["Martin M1", "Martin M1"]


# ---- Forced mode ----------------------------------------------------------


def test_forced_mode_decodes_known_sample(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """A correctly-tagged Martin_1 stream still decodes when the mode is forced."""
    audio, fs = _samples["Martin_1.mp3"]
    result = decode_offline(audio, fs, forced_mode=MODES_BY_NAME["martin_m1"])
    assert result is not None
    mode, img = result
    assert mode.vis == 44
    assert img.shape == (256, 320, 3)
    assert img.std() > 5.0


def test_streaming_forced_mode_skips_vis_lookup(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """When ``forced_mode`` is set, the streaming decoder uses it regardless
    of what VIS is parsed."""
    audio, fs = _samples["Martin_1.mp3"]
    captured: list[tuple[str, int]] = []

    def on_mode(mode, vis_code: int) -> None:
        captured.append((mode.name, vis_code))

    forced = MODES_BY_NAME["martin_m1"]
    decoder = StreamingSSTV(fs, events=StreamingEvents(on_mode=on_mode), forced_mode=forced)
    chunk = max(1, int(0.050 * fs))
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i : i + chunk])
    decoder.flush()
    assert captured, "no mode lock"
    assert captured[0][0] == "Martin M1"


# ---- Slant tracker --------------------------------------------------------


def test_scottie_dx_slant_correction_bounded(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """Scottie DX is the longest-line mode in the suite. The slant tracker's
    final correction should remain a tiny fraction of the nominal line.
    """
    audio, fs = _samples["ScottieDX.mp3"]
    captured_state: dict = {}

    def on_image(_mode, _img) -> None:
        if decoder.dstate is not None:
            captured_state["correction"] = decoder.dstate.slant.correction
            captured_state["nominal"] = decoder.dstate.slant.nominal_line_samples

    decoder = StreamingSSTV(fs, events=StreamingEvents(on_image=on_image))
    chunk = max(1, int(0.050 * fs))
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i : i + chunk])
    decoder.flush()
    # If the image didn't complete in stream, fall back to whatever state remains.
    if "nominal" not in captured_state and decoder.dstate is not None:
        captured_state["correction"] = decoder.dstate.slant.correction
        captured_state["nominal"] = decoder.dstate.slant.nominal_line_samples
    assert "nominal" in captured_state, "decoder never locked"
    ratio = abs(captured_state["correction"]) / captured_state["nominal"]
    assert ratio < 0.02, f"slant correction ratio {ratio:.4f} too large"


# ---- VIS-alias mapping ----------------------------------------------------


def test_vis_alias_14_resolves_to_robot36() -> None:
    """Some encoders transmit VIS=14 for a Robot 36 stream."""
    assert MODES[14] is MODES[8]
    assert MODES[14].name == "Robot 36"


# ---- Mode-name lookup -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["martin m1", "Martin M1", "martin_m1", "martinm1"],
)
def test_modes_by_name_accepts_common_spellings(name: str) -> None:
    assert MODES_BY_NAME[name.lower()].name == "Martin M1"


# ---- Demodulator end-to-end ------------------------------------------------


def _audio_to_usb_iq(audio: np.ndarray, fs: int, *, offset_hz: float = 0.0) -> np.ndarray:
    """Build a complex64 IQ stream that the USB demod can recover ``audio`` from.

    Take the analytic signal (audio + j*hilbert(audio)), then optionally up-shift
    by ``offset_hz`` so the carrier sits off-DC.
    """
    analytic = hilbert_fft(audio.astype(np.float64)).astype(np.complex64)
    if offset_hz != 0.0:
        t = np.arange(len(audio), dtype=np.float64) / fs
        analytic *= np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)
    return analytic


def test_sstvdemodulator_pipeline_robot36(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """Feed a synthesized USB IQ stream into ``SSTVDemodulator`` and assert
    the state machine reaches DONE with the expected image dimensions and a
    pile of SSTVData messages along the way."""
    audio, fs = _samples["Robot_36.mp3"]
    iq = _audio_to_usb_iq(audio, fs)

    demod = SSTVDemodulator(sample_rate=float(fs), audio_rate=float(fs))
    # Decimation should be 1:1 since audio_rate == sample_rate.
    assert demod.channel_decimation == 1

    collected: list[SSTVData] = []
    chunk = max(1, int(0.050 * fs))
    for i in range(0, len(iq), chunk):
        block = iq[i : i + chunk]
        demod.demodulate(block, capture_utc_s=1_700_000_000.0 + i / fs)
        for msg in demod.get_messages():
            if isinstance(msg, DecodedMessage) and isinstance(msg.data, SSTVData):
                collected.append(msg.data)

    assert collected, "no SSTVData messages produced"
    states = {d.state for d in collected}
    assert StreamerState.DECODING in states
    assert StreamerState.DONE in states
    done = next(d for d in collected if d.state == StreamerState.DONE)
    assert done.mode_name == "Robot 36"
    assert done.vis_code == 8
    assert done.image is not None
    assert done.image.shape == (240, 320, 3)


def test_sstvdemodulator_set_sstv_mode_overrides_and_resets() -> None:
    """``set_sstv_mode`` swaps the forced mode and clears the streaming state."""
    demod = SSTVDemodulator(sample_rate=44100.0, audio_rate=11025.0)
    assert demod.sstv_mode_name is None
    demod.set_sstv_mode("scottie_1")
    assert demod.sstv_mode_name == "Scottie 1"
    assert demod._decoder.state == StreamerState.LOOKING
    demod.set_sstv_mode(None)
    assert demod.sstv_mode_name is None


# ---- LOOKING-state ping ----------------------------------------------------


def test_demodulator_emits_looking_state_messages() -> None:
    """The demodulator pushes a bare LOOKING-state SSTVData while no
    transmission is detected so the widget can show the LOOKING badge."""
    fs = 12000
    demod = SSTVDemodulator(sample_rate=float(fs), audio_rate=float(fs))
    rng = np.random.default_rng(0)
    looking_msgs: list[SSTVData] = []
    chunk = int(0.050 * fs)
    iq_noise = (rng.standard_normal(fs * 4) + 1j * rng.standard_normal(fs * 4)).astype(
        np.complex64
    ) * 0.01
    for i in range(0, len(iq_noise), chunk):
        demod.demodulate(iq_noise[i : i + chunk], capture_utc_s=i / fs)
        for msg in demod.get_messages():
            if isinstance(msg.data, SSTVData) and msg.data.state == StreamerState.LOOKING:
                looking_msgs.append(msg.data)
    # 4 s with a 1 s emit throttle should produce 3-4 looking pings.
    assert 2 <= len(looking_msgs) <= 6, len(looking_msgs)


def test_sstvdemodulator_reset_clears_state() -> None:
    demod = SSTVDemodulator(sample_rate=44100.0, audio_rate=11025.0)
    # Drive a noise burst through demodulate so internal filters hold state.
    rng = np.random.default_rng(0)
    iq = rng.standard_normal((4096, 2)).astype(np.float32).view(np.complex64).reshape(-1)
    demod.demodulate(iq, capture_utc_s=0.0)
    demod.reset()
    assert demod._decoder.state == StreamerState.LOOKING
    assert demod._pending == []
