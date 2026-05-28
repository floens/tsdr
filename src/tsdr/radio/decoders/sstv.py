"""SSTV (Slow-Scan Television) decoder.

Streaming decoder that turns demodulated SSB audio (mono float32) into images.
Supported modes (auto-detected from VIS or force-selected by name):

    Martin M1/M2, Scottie 1/2/DX,
    Robot 8/12/24 BW, Robot 24 color, Robot 36, Robot 72,
    PD 50/90/120/160/180/240/290,
    Wrasse SC2-180, Pasokon P3/P5/P7.

Tones (BW=1500 Hz, white=2300 Hz, sync=1200 Hz, leader=1900 Hz) follow N7CXI's
Dayton 2000 SSTV specification. YCrCb→RGB uses the BT.601 limited-range
coefficients from that paper's Appendix B.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from tsdr.radio.dsp import StreamingFilter

logger = logging.getLogger(__name__)


FREQ_BLACK = 1500.0
FREQ_WHITE = 2300.0
FREQ_SYNC = 1200.0
FREQ_LEADER = 1900.0


# A line is a sequence of timed segments. The decoder walks each segment in
# order, mapping `scan` segments to pixels of the named channel.


@dataclass(frozen=True)
class Segment:
    kind: str  # "sync" | "porch" | "scan" | "sep" | "chroma_sep"
    ms: float
    channel: int = -1  # for "scan": output channel index, see Mode.color


@dataclass(frozen=True)
class Mode:
    name: str
    vis: int
    width: int
    height: int
    color: str  # "rgb" | "mono" | "ycrcb420" | "ycrcb422" | "pd_ycrcb"
    segments: tuple[Segment, ...]
    has_start_sync: bool = False  # Scottie: 9 ms leading sync after VIS
    chroma_width: int = 0  # width of half-rate chroma scan (Robot color)

    @property
    def line_ms(self) -> float:
        return sum(s.ms for s in self.segments)

    @property
    def scan_lines(self) -> int:
        """Number of decoder iterations per image. PD modes emit two output
        rows per scan-line, so the loop iterates ``height // 2`` times.
        """
        return self.height // 2 if self.color == "pd_ycrcb" else self.height


def _martin(name: str, vis: int, scan_ms: float) -> Mode:
    return Mode(
        name=name,
        vis=vis,
        width=320,
        height=256,
        color="rgb",
        segments=(
            Segment("sync", 4.862),
            Segment("porch", 0.572),
            Segment("scan", scan_ms, channel=1),
            Segment("sep", 0.572),
            Segment("scan", scan_ms, channel=2),
            Segment("sep", 0.572),
            Segment("scan", scan_ms, channel=0),
            Segment("sep", 0.572),
        ),
    )


def _scottie(name: str, vis: int, scan_ms: float) -> Mode:
    # Scottie quirk: sync sits between blue and red, not at line start. The
    # first line has a leading 9 ms sync to anchor timing.
    return Mode(
        name=name,
        vis=vis,
        width=320,
        height=256,
        color="rgb",
        segments=(
            Segment("sep", 1.5),
            Segment("scan", scan_ms, channel=1),
            Segment("sep", 1.5),
            Segment("scan", scan_ms, channel=2),
            Segment("sync", 9.0),
            Segment("sep", 1.5),
            Segment("scan", scan_ms, channel=0),
        ),
        has_start_sync=True,
    )


def _robot_bw(name: str, vis: int, width: int, height: int, scan_ms: float, line_ms: float) -> Mode:
    return Mode(
        name=name,
        vis=vis,
        width=width,
        height=height,
        color="mono",
        segments=(
            Segment("sync", 7.0),
            Segment("scan", scan_ms, channel=0),
            Segment("porch", line_ms - 7.0 - scan_ms),
        ),
    )


def _robot36() -> Mode:
    # Y full-rate, one chroma per line, alternating Cr/Cb (4:2:0).
    return Mode(
        name="Robot 36",
        vis=8,
        width=320,
        height=240,
        color="ycrcb420",
        segments=(
            Segment("sync", 9.0),
            Segment("porch", 3.0),
            Segment("scan", 88.0, channel=0),
            Segment("chroma_sep", 4.5),
            Segment("porch", 1.5),
            Segment("scan", 44.0, channel=-2),
        ),
        chroma_width=160,
    )


def _robot24_color() -> Mode:
    # 160x120, 4:2:2 (Cr then Cb every line, no chroma_sep encoding). Total
    # line = 200 ms. Empirically derived from the sample's frequency profile:
    # 6 ms sync, 88 ms Y, two 8 ms gaps (color burst + porch), 44 ms each chroma.
    return Mode(
        name="Robot 24",
        vis=4,
        width=160,
        height=120,
        color="ycrcb422",
        segments=(
            Segment("sync", 6.0),
            Segment("porch", 2.0),
            Segment("scan", 88.0, channel=0),
            Segment("porch", 8.0),
            Segment("scan", 44.0, channel=1),
            Segment("porch", 8.0),
            Segment("scan", 44.0, channel=2),
        ),
        chroma_width=80,
    )


def _wrasse_sc2_180() -> Mode:
    # N7CXI Dayton spec: VIS 55, 320x256 RGB, sync 5.5225 ms, porch 0.500 ms,
    # three back-to-back 235 ms scans in R/G/B order.
    return Mode(
        name="Wrasse SC2-180",
        vis=55,
        width=320,
        height=256,
        color="rgb",
        segments=(
            Segment("sync", 5.5225),
            Segment("porch", 0.500),
            Segment("scan", 235.000, channel=0),
            Segment("scan", 235.000, channel=1),
            Segment("scan", 235.000, channel=2),
        ),
    )


def _pasokon(name: str, vis: int, sync_ms: float, porch_ms: float, scan_ms: float) -> Mode:
    # Pasokon "P" modes: 640x496 RGB. Sync, porch, then each scan separated
    # by a porch (4 porches total around 3 scans).
    return Mode(
        name=name,
        vis=vis,
        width=640,
        height=496,
        color="rgb",
        segments=(
            Segment("sync", sync_ms),
            Segment("porch", porch_ms),
            Segment("scan", scan_ms, channel=0),
            Segment("porch", porch_ms),
            Segment("scan", scan_ms, channel=1),
            Segment("porch", porch_ms),
            Segment("scan", scan_ms, channel=2),
            Segment("porch", porch_ms),
        ),
    )


def _pd(name: str, vis: int, w: int, h: int, scan_ms: float) -> Mode:
    """Martin Bruchanov's PD format: 20 ms sync, 2.08 ms porch, then four
    equal-length scans per line: Y_odd, R-Y (averaged for both rows), B-Y
    (averaged for both rows), Y_even. Each scan-line outputs two image rows.
    """
    return Mode(
        name=name,
        vis=vis,
        width=w,
        height=h,
        color="pd_ycrcb",
        segments=(
            Segment("sync", 20.0),
            Segment("porch", 2.08),
            Segment("scan", scan_ms, channel=0),  # Y odd
            Segment("scan", scan_ms, channel=1),  # R-Y
            Segment("scan", scan_ms, channel=2),  # B-Y
            Segment("scan", scan_ms, channel=3),  # Y even
        ),
    )


def _robot72() -> Mode:
    # Y full-rate, both chromas every line at half-rate (4:2:2).
    return Mode(
        name="Robot 72",
        vis=12,
        width=320,
        height=240,
        color="ycrcb422",
        segments=(
            Segment("sync", 9.0),
            Segment("porch", 3.0),
            Segment("scan", 138.0, channel=0),
            Segment("sep", 4.5),
            Segment("porch", 1.5),
            Segment("scan", 69.0, channel=1),
            Segment("sep", 4.5),
            Segment("porch", 1.5),
            Segment("scan", 69.0, channel=2),
        ),
        chroma_width=160,
    )


_MODE_LIST = [
    _martin("Martin M1", 44, 146.432),
    _martin("Martin M2", 40, 73.216),
    _scottie("Scottie 1", 60, 138.240),
    _scottie("Scottie 2", 56, 88.064),
    _scottie("Scottie DX", 76, 345.600),
    _robot_bw("Robot 8 BW", 2, 160, 120, 60.0, 66.9),
    _robot_bw("Robot 12 BW", 6, 160, 120, 93.0, 100.0),
    _robot_bw("Robot 24 BW", 10, 320, 240, 93.0, 100.0),
    _robot24_color(),
    _robot36(),
    _robot72(),
    _pd("PD 50", 93, 320, 256, 91.520),
    _pd("PD 90", 99, 320, 256, 170.240),
    _pd("PD 120", 95, 640, 496, 121.600),
    _pd("PD 160", 98, 512, 400, 195.584),
    _pd("PD 180", 96, 640, 496, 183.040),
    _pd("PD 240", 97, 640, 496, 244.480),
    _pd("PD 290", 94, 800, 616, 228.800),
    _wrasse_sc2_180(),
    _pasokon("Pasokon P3", 113, 5.208, 1.042, 133.333),
    _pasokon("Pasokon P5", 114, 7.813, 1.563, 200.000),
    _pasokon("Pasokon P7", 115, 10.417, 2.083, 266.666),
]
MODES: dict[int, Mode] = {m.vis: m for m in _MODE_LIST}
# Some encoders (notably the source for sigidwiki's "Robot72SSTV_Sound.mp3")
# transmit VIS=14 for what is otherwise a Robot 36 stream.
MODES[14] = MODES[8]


def _name_aliases(m: Mode) -> tuple[str, str, str]:
    slug = m.name.lower()
    return slug, slug.replace(" ", "_"), slug.replace(" ", "")


MODES_BY_NAME: dict[str, Mode] = {alias: m for m in _MODE_LIST for alias in _name_aliases(m)}


@dataclass(frozen=True)
class SSTVData:
    """Snapshot of the streaming decoder's state for the SSTV widget.

    ``image`` is a fresh copy of the running frame buffer; the decoder
    continues to mutate its own buffer after this snapshot is taken.
    """

    state: StreamerState
    vis_code: int | None
    mode_name: str | None
    line_index: int  # last completed line; -1 in LOOKING
    total_lines: int
    image_width: int
    image_height: int
    image: NDArray[np.uint8] | None
    forced_mode: bool
    error: str | None = None


def _ms_to_samples(ms: float, fs: int) -> int:
    return int(round(ms * fs / 1000.0))


def hilbert_fft(x: np.ndarray) -> np.ndarray:
    """FFT-based Hilbert transform (offline, whole signal)."""
    n = len(x)
    spectrum = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def instantaneous_freq(x: np.ndarray, fs: int) -> np.ndarray:
    """Per-sample instantaneous frequency in Hz, same length as `x`."""
    z = hilbert_fft(x)
    phi = np.angle(z)
    dphi = np.diff(phi)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    freq = dphi * fs / (2 * np.pi)
    return np.concatenate([freq, freq[-1:]]).astype(np.float32)


class StreamingHilbert:
    """FIR Hilbert transformer producing the analytic signal in chunks.

    Antisymmetric Hamming-windowed FIR; I is delayed by ``(n_taps-1)//2``
    samples to align with the Q branch. State persists across calls so
    chunk boundaries are seamless.
    """

    def __init__(self, n_taps: int = 65):
        if n_taps % 2 == 0:
            raise ValueError("n_taps must be odd")
        # Ideal Hilbert FIR sampled and Hamming-windowed. Antisymmetric
        # impulse response, non-zero only at odd offsets from center.
        m = (n_taps - 1) // 2
        h = np.zeros(n_taps, dtype=np.float32)
        for i in range(n_taps):
            k = i - m
            if k != 0 and k % 2 != 0:
                h[i] = 2.0 / (np.pi * k)
        h *= np.hamming(n_taps).astype(np.float32)
        self.delay = m
        self._fir = StreamingFilter(h, [1.0], dtype=np.float32)
        self._i_delay = np.zeros(m, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        q = self._fir.process(x)
        extended = np.concatenate([self._i_delay, x])
        i = extended[: len(x)]
        self._i_delay = extended[-self.delay :].copy() if self.delay else self._i_delay
        # Build complex64 directly: ``i + 1j*q`` promotes through complex128.
        out = np.empty(len(x), dtype=np.complex64)
        out.real = i
        out.imag = q
        return out

    def reset(self) -> None:
        self._fir.reset()
        self._i_delay[:] = 0


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centered boxcar smoothing of length n."""
    if n <= 1:
        return x.astype(np.float32, copy=False)
    kernel = np.ones(n, dtype=np.float32) / n
    pad = n // 2
    padded = np.pad(x.astype(np.float32, copy=False), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def _close_gaps(mask: np.ndarray, gap_n: int) -> np.ndarray:
    """Morphological closing: flip any False-run shorter than gap_n to True."""
    out = mask.copy()
    for rs, re in _runs(~mask):
        if re - rs < gap_n:
            out[rs:re] = True
    return out


def find_vis(freq: np.ndarray, fs: int) -> tuple[int, int] | None:
    """Locate a VIS header. Returns (vis_code, sample_index_after_stop_bit).

    Anchors on the leader tone, not the sync band, so the detector tolerates
    operator mistuning up to ±200 Hz. For each ≥200 ms run of smoothed freq
    in [1700, 2100] Hz (with single-sample demod glitches closed up to 10 ms),
    require the run to be stable (MAD ≤ 200 Hz) and its median to fall inside
    the band — voice/noise traversing the band has MAD ≥ 280 Hz, and tones
    sitting above the band can otherwise be pulled in by gap-closing of
    short in-band excursions. Then look 0–60 ms past the leader's end for a
    15 ms window stably below leader_freq − 400 Hz — that's the start-bit
    body. Sample 9 bit means at 30 ms each from there (start bit + 7 data
    bits + parity); the stop bit is skipped because its sampling window
    abuts the first image pixel and is routinely contaminated. Bit decisions
    compare each bit mean against the canonical sync reference
    (leader_freq − 700 Hz), which tracks mistune via the per-candidate
    leader_freq. Only candidates that decode to a known VIS are kept —
    surfacing an undecodable VIS is useless and known-mode is the strongest
    signal-vs-noise discriminator we have.
    """
    smooth_freq = _smooth(freq, _ms_to_samples(2.0, fs))
    bit_n = _ms_to_samples(30.0, fs)
    min_leader_n = _ms_to_samples(200.0, fs)
    # 10 ms gap-closing only handles single-sample demod glitches; anything
    # longer is genuinely out-of-band content (voice, image data, drift)
    # that should split the run.
    leader_close_n = _ms_to_samples(10.0, fs)
    drop_look_n = _ms_to_samples(60.0, fs)
    drop_stable_n = _ms_to_samples(15.0, fs)
    leader_lo, leader_hi = 1700.0, 2100.0
    max_leader_mad = 200.0

    leader_mask = (smooth_freq >= leader_lo) & (smooth_freq <= leader_hi)
    leader_mask = _close_gaps(leader_mask, leader_close_n)

    candidates: list[tuple[float, int, int]] = []  # (score, vis, stop_end)
    for leader_start, leader_end in _runs(leader_mask):
        if leader_end - leader_start < min_leader_n:
            continue
        leader_seg = smooth_freq[leader_start:leader_end]
        leader_freq = float(np.median(leader_seg))
        # Reject runs whose median falls outside the band: gap-closing can
        # bridge out-of-band tones (e.g. a 2275 Hz carrier with brief dips
        # into 2050) into a synthetic "leader" run.
        if not (leader_lo <= leader_freq <= leader_hi):
            continue
        leader_mad = float(np.median(np.abs(leader_seg - leader_freq)))
        if leader_mad > max_leader_mad:
            continue

        drop_thresh = leader_freq - 400.0
        region_end = min(leader_end + drop_look_n + drop_stable_n, len(smooth_freq))
        below = smooth_freq[leader_end:region_end] < drop_thresh
        start_offset = -1
        for i in range(len(below) - drop_stable_n):
            if below[i : i + drop_stable_n].all():
                start_offset = i
                break
        if start_offset < 0:
            logger.info(
                "sstv_vis_near_miss reason=no_start_bit leader_freq=%.0f t=%.3fs",
                leader_freq,
                leader_end / fs,
            )
            continue
        start_bit_idx = leader_end + start_offset

        # Sample the start bit (i=0), 7 data bits (i=1..7), and parity (i=8).
        # The stop bit (i=9) abuts image data — its sampling window often
        # bleeds into the first pixel, contaminating any sync reference
        # derived from it. Skip it.
        bit_means: list[float] = []
        for i in range(9):
            mid = start_bit_idx + i * bit_n + bit_n // 2
            lo, hi = mid - bit_n // 4, mid + bit_n // 4
            if hi >= len(smooth_freq):
                break
            bit_means.append(float(smooth_freq[lo:hi].mean()))
        if len(bit_means) < 9:
            continue

        # Canonical SSTV sync is 700 Hz below the leader (1900 → 1200 Hz);
        # this tracks mistune since leader_freq is measured per-candidate.
        sync_ref = leader_freq - 700.0
        data_bits = [1 if b < sync_ref else 0 for b in bit_means[1:8]]
        parity_bit = 1 if bit_means[8] < sync_ref else 0
        parity_ok = (sum(data_bits) + parity_bit) % 2 == 0
        bit_conf = float(np.mean([abs(b - sync_ref) for b in bit_means[1:9]]))

        vis = 0
        for i, b in enumerate(data_bits):
            vis |= b << i
        # Require a known mode: image data and voice can synthesize bit
        # patterns that satisfy parity but decode to garbage VIS codes, and
        # any VIS we can't render is useless to surface anyway.
        if vis not in MODES:
            continue
        global_start = start_bit_idx
        stop_end = start_bit_idx + 10 * bit_n

        leader_dev = abs(leader_freq - FREQ_LEADER)
        score = (200.0 if parity_ok else 0.0) + bit_conf * 0.5 - leader_dev * 0.25
        log = logger.info if not parity_ok else logger.debug
        log(
            "sstv_vis_candidate vis=%d parity=%s leader_freq=%.0f bit_conf=%.0f "
            "score=%.0f at_sample=%d t=%.3fs",
            vis,
            parity_ok,
            leader_freq,
            bit_conf,
            score,
            global_start,
            global_start / fs,
        )
        candidates.append((score, vis, stop_end))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    score, vis, stop_end = candidates[0]
    logger.info(
        "sstv_vis_detected vis=%d score=%.0f candidates=%d",
        vis,
        score,
        len(candidates),
    )
    return vis, stop_end


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive)] runs of True in `mask`."""
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends, strict=False))


def freq_to_pixel(f: np.ndarray) -> np.ndarray:
    v = (f - FREQ_BLACK) / (FREQ_WHITE - FREQ_BLACK) * 255.0
    return np.clip(v, 0, 255).astype(np.uint8)


def scan_to_pixels(scan: np.ndarray, width: int) -> np.ndarray:
    """Average instantaneous-frequency samples into ``width`` pixel bins."""
    n = len(scan)
    if n == 0:
        return np.zeros(width, dtype=np.uint8)
    edges = np.linspace(0, n, width + 1).astype(np.int32)
    sums = np.add.reduceat(scan, edges[:-1])
    counts = np.maximum(np.diff(edges), 1).astype(np.float32)
    return freq_to_pixel(sums / counts)


def find_sync(freq: np.ndarray, expected: int, search_ms: float, sync_ms: float, fs: int) -> int:
    """Locate a 1200 Hz sync pulse near ``expected``. Returns the index where
    the sync pulse starts, or ``expected`` if nothing convincing is found.
    """
    half = _ms_to_samples(search_ms, fs)
    sync_n = _ms_to_samples(sync_ms, fs)
    lo = max(0, expected - half)
    hi = min(len(freq), expected + half + sync_n)
    if hi - lo < sync_n + 1:
        return expected
    window = freq[lo:hi]
    window_s = _smooth(window, _ms_to_samples(2.0, fs))
    is_sync = ((window_s > 1050) & (window_s < 1350)).astype(np.float32)
    kernel = np.ones(sync_n, dtype=np.float32)
    conv = np.convolve(is_sync, kernel, mode="valid")
    if conv.max() < sync_n * 0.6:
        return expected
    return lo + int(np.argmax(conv))


@dataclass
class SlantTracker:
    """Integral-feedback tracker for per-line sample-rate drift.

    Each line, the decoder observes ``drift = actual_sync - expected_sync``.
    We absorb a fraction of every observation into ``correction``, which is
    added to the nominal line length for subsequent lines. With a true drift
    of C samples per line, the loop converges geometrically to correction=C.
    """

    nominal_line_samples: int = 0
    correction: int = 0
    alpha: float = 0.3

    def observe(self, drift: int) -> None:
        delta = int(round(self.alpha * drift))
        # Cap correction at ±5% of nominal to defend against runaway
        # detection noise; this is far more than any real clock mismatch.
        cap = max(1, int(self.nominal_line_samples * 0.05))
        self.correction = max(-cap, min(cap, self.correction + delta))

    def line_samples(self) -> int:
        return self.nominal_line_samples + self.correction


@dataclass
class DecodeState:
    """Persistent state during one image's decoding.

    For RGB/mono modes only ``image`` is used. For Robot YCrCb modes the Y and
    chroma planes are tracked separately so each line's RGB row can be
    composed with whatever chroma is available so far.
    """

    mode: Mode
    image: np.ndarray  # H x W x 3 uint8
    y_plane: np.ndarray | None = None  # H x W (Y for Robot color)
    cr_plane: np.ndarray | None = None  # H x chroma_width
    cb_plane: np.ndarray | None = None
    cr_seen: np.ndarray | None = None  # H bool
    cb_seen: np.ndarray | None = None
    last_cr_row: np.ndarray | None = None  # most-recent Cr row, full width
    last_cb_row: np.ndarray | None = None
    # Image rows updated by the most recent ``decode_line`` call (typically
    # the new line plus any rows that were re-rendered by progressive chroma
    # fix-up). Callers drain this to fire on_line events.
    written_rows: list[int] = field(default_factory=list)
    slant: SlantTracker = field(default_factory=SlantTracker)


def make_decode_state(mode: Mode, fs: int) -> DecodeState:
    img = np.zeros((mode.height, mode.width, 3), dtype=np.uint8)
    slant = SlantTracker(nominal_line_samples=_ms_to_samples(mode.line_ms, fs))
    if mode.color in ("ycrcb420", "ycrcb422"):
        cw = mode.chroma_width
        cr = np.full((mode.height, cw), 128, dtype=np.uint8)
        cb = np.full((mode.height, cw), 128, dtype=np.uint8)
        return DecodeState(
            mode=mode,
            image=img,
            y_plane=np.zeros((mode.height, mode.width), dtype=np.uint8),
            cr_plane=cr,
            cb_plane=cb,
            cr_seen=np.zeros(mode.height, dtype=bool),
            cb_seen=np.zeros(mode.height, dtype=bool),
            last_cr_row=np.full(mode.width, 128, dtype=np.uint8),
            last_cb_row=np.full(mode.width, 128, dtype=np.uint8),
            slant=slant,
        )
    if mode.color == "pd_ycrcb":
        # PD: one chroma row per scan-line (shared by the pair of output rows).
        return DecodeState(
            mode=mode,
            image=img,
            y_plane=np.zeros((mode.height, mode.width), dtype=np.uint8),
            cr_plane=np.full((mode.scan_lines, mode.width), 128, dtype=np.uint8),
            cb_plane=np.full((mode.scan_lines, mode.width), 128, dtype=np.uint8),
            cr_seen=np.zeros(mode.scan_lines, dtype=bool),
            cb_seen=np.zeros(mode.scan_lines, dtype=bool),
            slant=slant,
        )
    return DecodeState(mode=mode, image=img, slant=slant)


def _align_to_sync(freq: np.ndarray, fs: int, state: DecodeState, cursor: int) -> int:
    """Find the sync pulse near its expected position, observe drift, and
    return a cursor anchored to the line start. Search window grows with the
    current slant correction so a large per-line drift can't slip past it.
    """
    mode = state.mode
    sync_seg = next((s for s in mode.segments if s.kind == "sync"), None)
    sync_offset_ms = _segment_offset(mode, "sync")
    if sync_seg is None or sync_offset_ms is None:
        return cursor
    expected_sync = cursor + _ms_to_samples(sync_offset_ms, fs)
    slant_mag_ms = abs(state.slant.correction) * 1000.0 / fs
    sync_at = find_sync(freq, expected_sync, 8.0 + slant_mag_ms, sync_seg.ms, fs)
    state.slant.observe(sync_at - expected_sync)
    return sync_at - _ms_to_samples(sync_offset_ms, fs)


def decode_line(freq: np.ndarray, fs: int, state: DecodeState, line_index: int, cursor: int) -> int:
    """Decode line ``line_index`` into ``state.image``. Returns the cursor
    (sample index) where the next line is expected to begin.

    Pre-condition: enough samples in ``freq`` for the whole line plus a small
    sync-search margin.
    """
    if state.mode.color == "pd_ycrcb":
        return _decode_pd_line(freq, fs, state, line_index, cursor)
    mode = state.mode
    cursor = _align_to_sync(freq, fs, state, cursor)

    seg_cursor = cursor
    chroma_kind = 0  # 1=Cr, 2=Cb, only meaningful when chroma_sep is in this mode
    cr_seen_now = False
    cb_seen_now = False

    for seg in mode.segments:
        seg_n = _ms_to_samples(seg.ms, fs)
        end = min(seg_cursor + seg_n, len(freq))
        if seg.kind == "chroma_sep" and end > seg_cursor:
            # Median ignores transition slopes at segment edges; classify by
            # nearest canonical tone (1500=Cr, 2300=Cb).
            mean_f = float(np.median(freq[seg_cursor:end]))
            chroma_kind = 1 if abs(mean_f - 1500.0) < abs(mean_f - 2300.0) else 2
        elif seg.kind == "scan":
            if mode.color == "rgb":
                pixels = scan_to_pixels(freq[seg_cursor:end], mode.width)
                state.image[line_index, :, seg.channel] = pixels
            elif mode.color == "mono":
                pixels = scan_to_pixels(freq[seg_cursor:end], mode.width)
                state.image[line_index, :, :] = pixels[:, None]
            elif mode.color in ("ycrcb420", "ycrcb422"):
                assert state.y_plane is not None
                assert state.cr_plane is not None
                assert state.cb_plane is not None
                if seg.channel == 0:
                    state.y_plane[line_index, :] = scan_to_pixels(freq[seg_cursor:end], mode.width)
                elif seg.channel == 1:
                    state.cr_plane[line_index, :] = scan_to_pixels(
                        freq[seg_cursor:end], mode.chroma_width
                    )
                    cr_seen_now = True
                elif seg.channel == 2:
                    state.cb_plane[line_index, :] = scan_to_pixels(
                        freq[seg_cursor:end], mode.chroma_width
                    )
                    cb_seen_now = True
                elif seg.channel == -2:
                    pixels = scan_to_pixels(freq[seg_cursor:end], mode.chroma_width)
                    if chroma_kind == 1:
                        state.cr_plane[line_index, :] = pixels
                        cr_seen_now = True
                    elif chroma_kind == 2:
                        state.cb_plane[line_index, :] = pixels
                        cb_seen_now = True
        seg_cursor += seg_n

    state.written_rows.clear()
    state.written_rows.append(line_index)
    if mode.color in ("ycrcb420", "ycrcb422"):
        assert state.y_plane is not None
        assert state.cr_plane is not None
        assert state.cb_plane is not None
        assert state.cr_seen is not None
        assert state.cb_seen is not None
        if cr_seen_now:
            state.cr_seen[line_index] = True
            state.last_cr_row = _upsample_x(state.cr_plane[line_index], mode.width)
        if cb_seen_now:
            state.cb_seen[line_index] = True
            state.last_cb_row = _upsample_x(state.cb_plane[line_index], mode.width)
        assert state.last_cr_row is not None and state.last_cb_row is not None
        state.image[line_index, :, :] = _ycrcb_to_rgb(
            state.y_plane[line_index], state.last_cr_row, state.last_cb_row
        )
        # Progressive chroma fix-up: if line N-1 was rendered without the
        # chroma kind we just got, re-render it with the new (closer) chroma.
        prev = line_index - 1
        if prev >= 0:
            improved = False
            if cr_seen_now and not state.cr_seen[prev]:
                improved = True
            if cb_seen_now and not state.cb_seen[prev]:
                improved = True
            if improved:
                state.image[prev, :, :] = _ycrcb_to_rgb(
                    state.y_plane[prev], state.last_cr_row, state.last_cb_row
                )
                state.written_rows.append(prev)

    return cursor + state.slant.line_samples()


def _decode_pd_line(
    freq: np.ndarray, fs: int, state: DecodeState, scan_line: int, cursor: int
) -> int:
    """Decode a PD scan-line. Writes two image rows (2*scan_line and
    2*scan_line+1) and one shared Cr/Cb pair indexed by ``scan_line``.
    """
    mode = state.mode
    assert state.y_plane is not None
    assert state.cr_plane is not None
    assert state.cb_plane is not None
    assert state.cr_seen is not None
    assert state.cb_seen is not None
    cursor = _align_to_sync(freq, fs, state, cursor)
    seg_cursor = cursor
    row_odd = 2 * scan_line
    row_even = row_odd + 1
    for seg in mode.segments:
        seg_n = _ms_to_samples(seg.ms, fs)
        end = min(seg_cursor + seg_n, len(freq))
        if seg.kind == "scan":
            pixels = scan_to_pixels(freq[seg_cursor:end], mode.width)
            if seg.channel == 0:
                state.y_plane[row_odd, :] = pixels
            elif seg.channel == 1:
                state.cr_plane[scan_line, :] = pixels
                state.cr_seen[scan_line] = True
            elif seg.channel == 2:
                state.cb_plane[scan_line, :] = pixels
                state.cb_seen[scan_line] = True
            elif seg.channel == 3 and row_even < mode.height:
                state.y_plane[row_even, :] = pixels
        seg_cursor += seg_n

    cr_row = state.cr_plane[scan_line]
    cb_row = state.cb_plane[scan_line]
    state.image[row_odd, :, :] = _ycrcb_to_rgb(state.y_plane[row_odd], cr_row, cb_row)
    state.written_rows.clear()
    state.written_rows.append(row_odd)
    if row_even < mode.height:
        state.image[row_even, :, :] = _ycrcb_to_rgb(state.y_plane[row_even], cr_row, cb_row)
        state.written_rows.append(row_even)
    return cursor + state.slant.line_samples()


def finalize_image(state: DecodeState) -> np.ndarray:
    """Apply chroma fixups (backward fill for missing rows) and return the
    final RGB image. Idempotent; safe to call multiple times.
    """
    mode = state.mode
    if mode.color in ("ycrcb420", "ycrcb422"):
        assert state.y_plane is not None
        assert state.cr_plane is not None and state.cr_seen is not None
        assert state.cb_plane is not None and state.cb_seen is not None
        cr = _fill_missing_rows(state.cr_plane, state.cr_seen)
        cb = _fill_missing_rows(state.cb_plane, state.cb_seen)
        cr_full = _upsample_x(cr, mode.width)
        cb_full = _upsample_x(cb, mode.width)
        state.image = _ycrcb_to_rgb(state.y_plane, cr_full, cb_full)
    return state.image


def _segment_offset(mode: Mode, kind: str) -> float | None:
    """Cumulative ms before the first segment of ``kind``, else None."""
    t = 0.0
    for s in mode.segments:
        if s.kind == kind:
            return t
        t += s.ms
    return None


def _fill_missing_rows(plane: np.ndarray, seen: np.ndarray) -> np.ndarray:
    if seen.all():
        return plane
    out = plane.copy()
    last = -1
    for i in range(len(seen)):
        if seen[i]:
            last = i
        elif last >= 0:
            out[i, :] = out[last, :]
    nxt = -1
    for i in range(len(seen)):
        if seen[i]:
            nxt = i
            break
    if nxt > 0:
        for i in range(nxt):
            out[i, :] = plane[nxt, :]
    return out


def _upsample_x(plane: np.ndarray, width: int) -> np.ndarray:
    """Nearest-neighbor horizontal upsample. Works for 1D rows or 2D planes."""
    src_w = plane.shape[-1]
    if src_w == width:
        return plane
    idx = np.linspace(0, src_w - 1, width).astype(np.int32)
    return plane[..., idx]


def _ycrcb_to_rgb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """BT.601 limited-range YCbCr → RGB, per N7CXI's Dayton 2000 SSTV spec
    (Appendix B). Y is encoded as 16-235, Cr/Cb as 16-240, all packed into
    0-255 sample range. Coefficients match the ITU BT.601 (NTSC) standard.
    """
    yf = (y.astype(np.float32) - 16.0) * (1.0 / 256.0)
    crf = cr.astype(np.float32) - 128.0
    cbf = cb.astype(np.float32) - 128.0
    r = 298.082 * yf + 1.596 * crf
    g = 298.082 * yf - 0.392 * cbf - 0.813 * crf
    b = 298.082 * yf + 2.017 * cbf
    out: np.ndarray = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
    return out


def decode_offline(
    audio: np.ndarray,
    fs: int,
    *,
    forced_mode: Mode | None = None,
) -> tuple[Mode, np.ndarray] | None:
    """Decode an entire SSTV transmission from a numpy audio buffer."""
    freq = instantaneous_freq(audio, fs)

    vis = find_vis(freq, fs)
    if forced_mode is not None:
        mode = forced_mode
        start_sample = vis[1] if vis else 0
    else:
        if vis is None:
            return None
        vis_code, start_sample = vis
        if vis_code not in MODES:
            logger.error("sstv_unknown_vis vis=%d", vis_code)
            return None
        mode = MODES[vis_code]

    cursor = start_sample
    if mode.has_start_sync:
        cursor += _ms_to_samples(9.0, fs)

    state = make_decode_state(mode, fs)
    for line in range(mode.scan_lines):
        if cursor + state.slant.line_samples() > len(freq):
            break
        cursor = decode_line(freq, fs, state, line, cursor)

    return mode, finalize_image(state)


class StreamerState(Enum):
    LOOKING = "looking"  # scanning for VIS header
    DECODING = "decoding"  # filling the image line by line
    DONE = "done"  # last image complete, awaiting reset


@dataclass
class StreamingEvents:
    """Hooks fired by StreamingSSTV as decoding progresses."""

    on_mode: Callable[[Mode, int], None] | None = None
    on_line: Callable[[Mode, int, np.ndarray], None] | None = None
    on_image: Callable[[Mode, np.ndarray], None] | None = None


class StreamingSSTV:
    """Decode SSTV from a stream of audio samples.

    Push samples via :meth:`process`. The decoder runs a small state machine:
    waits for a VIS header, then decodes lines as enough samples arrive,
    finally emits the completed image and resets.

    When ``forced_mode`` is set, the parsed VIS code is ignored and the given
    mode is used instead. The detected VIS pulse position still anchors timing.
    """

    # Drop old samples when we have more than this much buffer outside the
    # active decode window. Keeps memory bounded for long idle periods.
    _COMPACT_THRESHOLD_S = 5.0

    def __init__(
        self,
        fs: int,
        *,
        events: StreamingEvents | None = None,
        hilbert_taps: int = 65,
        forced_mode: Mode | None = None,
    ):
        self.fs = fs
        self.events = events or StreamingEvents()
        self.forced_mode = forced_mode
        self._hilbert = StreamingHilbert(hilbert_taps)
        self._prev_phase = 0.0
        self._has_prev_phase = False
        # freq buffer of global samples, indexed by absolute sample number.
        # Holds samples from _buf_offset onward.
        self._freq = np.zeros(0, dtype=np.float32)
        self._buf_offset = 0
        # Total samples consumed (always == _buf_offset + len(_freq)).
        self._produced = 0

        self.state = StreamerState.LOOKING
        self.mode: Mode | None = None
        self.vis_code: int | None = None
        self._dstate: DecodeState | None = None
        self._line_index = 0
        self._cursor = 0  # global sample index where current line begins

    def process(self, samples: np.ndarray) -> None:
        """Push a chunk of mono audio samples."""
        if len(samples) == 0:
            return
        z = self._hilbert.process(samples)
        new_freq = self._inst_freq(z)
        self._freq = np.concatenate([self._freq, new_freq])
        self._produced += len(new_freq)
        self._advance()
        self._maybe_compact()

    def flush(self) -> None:
        """Emit the in-progress image even if the audio ended mid-stream."""
        if self.state == StreamerState.DECODING and self._dstate is not None:
            self._emit_image()

    def reset(self) -> None:
        self._hilbert.reset()
        self._has_prev_phase = False
        self._prev_phase = 0.0
        self._freq = np.zeros(0, dtype=np.float32)
        self._buf_offset = 0
        self._produced = 0
        self.state = StreamerState.LOOKING
        self.mode = None
        self.vis_code = None
        self._dstate = None
        self._line_index = 0
        self._cursor = 0

    def set_forced_mode(self, mode: Mode | None) -> None:
        """Override VIS-code interpretation. Takes effect on the next VIS
        detection; call :meth:`reset` afterwards to abandon the current frame.
        """
        self.forced_mode = mode

    @property
    def line_index(self) -> int:
        return self._line_index

    @property
    def dstate(self) -> DecodeState | None:
        return self._dstate

    @property
    def samples_processed(self) -> int:
        """Total audio samples ever fed to the decoder. Use as a monotonic
        audio-time clock when wall-clock isn't suitable (e.g. file replay)."""
        return self._produced

    def _inst_freq(self, z: np.ndarray) -> np.ndarray:
        phi = np.angle(z).astype(np.float32)
        # Seed _prev_phase from the first sample on the very first chunk so
        # every output index k holds the phase diff between samples k-1 and k
        # — no padding, no off-by-one between IF samples and audio samples.
        if not self._has_prev_phase:
            self._prev_phase = float(phi[0])
            self._has_prev_phase = True
        full = np.concatenate([[self._prev_phase], phi])
        self._prev_phase = float(phi[-1])
        dphi = np.diff(full)
        # Unwrap to (-π, π] in place to avoid three full-size intermediates.
        np.add(dphi, np.pi, out=dphi)
        np.mod(dphi, 2 * np.pi, out=dphi)
        np.subtract(dphi, np.pi, out=dphi)
        dphi *= self.fs / (2 * np.pi)
        out: np.ndarray = dphi.astype(np.float32, copy=False)
        return out

    def _local(self, global_index: int) -> int:
        return global_index - self._buf_offset

    def _advance(self) -> None:
        while True:
            if self.state == StreamerState.LOOKING:
                if not self._try_find_vis():
                    return
            elif self.state == StreamerState.DECODING:
                if not self._try_decode_line():
                    return
            else:
                return

    def _try_find_vis(self) -> bool:
        # VIS detection needs at least 200 ms leader + 300 ms bits = 500 ms.
        # Search only within the current buffer.
        min_n = _ms_to_samples(500.0, self.fs)
        if len(self._freq) < min_n:
            return False
        vis = find_vis(self._freq, self.fs)
        if vis is None:
            return False
        vis_code, post_stop_local = vis
        if self.forced_mode is not None:
            mode: Mode = self.forced_mode
        else:
            looked_up = MODES.get(vis_code)
            if looked_up is None:
                logger.error("sstv_unknown_vis_streaming vis=%d", vis_code)
                # Discard the rejected VIS region and keep scanning so the
                # decoder doesn't wedge waiting for a reset that never comes.
                self._freq = self._freq[post_stop_local:]
                self._buf_offset += post_stop_local
                return False
            mode = looked_up
        self.mode = mode
        self.vis_code = vis_code
        self._dstate = make_decode_state(mode, self.fs)
        post_stop_global = post_stop_local + self._buf_offset
        self._cursor = post_stop_global
        if mode.has_start_sync:
            self._cursor += _ms_to_samples(9.0, self.fs)
        self._line_index = 0
        self.state = StreamerState.DECODING
        logger.info(
            "sstv_stream_mode_locked vis=%d mode=%s forced=%s",
            vis_code,
            mode.name,
            self.forced_mode is not None,
        )
        if self.events.on_mode:
            self.events.on_mode(mode, vis_code)
        return True

    def _try_decode_line(self) -> bool:
        if self.mode is None or self._dstate is None:
            return False
        line_samples = self._dstate.slant.line_samples()
        # Need full line + sync search margin to decode this line. Margin
        # scales with current slant correction so we don't dispatch decode_line
        # before the actual sync arrives.
        slant_mag = abs(self._dstate.slant.correction)
        margin = _ms_to_samples(10.0, self.fs) + slant_mag
        need_end_global = self._cursor + line_samples + margin
        if need_end_global - self._buf_offset > len(self._freq):
            return False  # wait for more samples
        local_cursor = self._local(self._cursor)
        next_local = decode_line(self._freq, self.fs, self._dstate, self._line_index, local_cursor)
        self._cursor = next_local + self._buf_offset
        if self.events.on_line:
            for r in self._dstate.written_rows:
                self.events.on_line(self.mode, r, self._dstate.image[r])
        self._line_index += 1
        if self._line_index >= self.mode.scan_lines:
            self._emit_image()
        return True

    def _emit_image(self) -> None:
        if self.mode is None or self._dstate is None:
            return
        img = finalize_image(self._dstate)
        # Transition to DONE before the callback so the callback can mutate
        # state (e.g. reset() to resume scanning for the next transmission)
        # and have it stick.
        self.state = StreamerState.DONE
        if self.events.on_image:
            self.events.on_image(self.mode, img)

    def _maybe_compact(self) -> None:
        if self.state == StreamerState.LOOKING:
            # VIS detection needs ~500 ms of buffer; keep a bit more for safety.
            keep_from_global = self._produced - _ms_to_samples(800.0, self.fs)
        elif self.state == StreamerState.DECODING:
            # In DECODING state we don't reach into samples before _cursor.
            keep_from_global = self._cursor - _ms_to_samples(20.0, self.fs)
        else:
            return
        drop_local = keep_from_global - self._buf_offset
        if drop_local > _ms_to_samples(self._COMPACT_THRESHOLD_S * 1000.0, self.fs):
            self._freq = self._freq[drop_local:]
            self._buf_offset += drop_local
