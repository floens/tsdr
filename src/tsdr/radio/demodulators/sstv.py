"""SSTV demodulator: USB SSB → ``StreamingSSTV`` line decoder.

USB demodulation mirrors ``SSBDemodulator`` minus the squelch, since the SSTV
decoder needs the audio even when the signal is below a typical voice squelch
threshold. The recovered audio is both played back via ``_emit_audio`` and
forked into a streaming SSTV decoder; line-decoded images surface through
``get_messages()`` as ``DecodedMessage(data=SSTVData)``.
"""

from __future__ import annotations

import time

import numpy as np

from tsdr.core.clock_sync import now_utc_seconds
from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.radio.decoders.sstv import (
    MODES_BY_NAME,
    Mode,
    SSTVData,
    StreamerState,
    StreamingEvents,
    StreamingSSTV,
)
from tsdr.radio.demodulators import NYQUIST_MARGIN, Demodulator
from tsdr.radio.dsp import (
    AGC,
    DCBlocker,
    StreamingFilter,
    firwin,
)
from tsdr.radio.dsp._kernels import apply_freq_shift_c64


class SSTVDemodulator(Demodulator):
    """USB SSTV demodulator with an inline streaming line decoder."""

    HAS_AUDIO = True
    LABEL = "SSTV"
    MODULATION = "USB"
    MESSAGE_TYPE = "sstv"
    SIDEBAND = "upper"

    DEFAULT_CHANNEL_BANDWIDTH = 3_000
    MAX_CHANNEL_BANDWIDTH = 48_000 * NYQUIST_MARGIN
    DC_BLOCKER_CUTOFF = 100.0

    # Min interval between per-line snapshots emitted to the UI. Fast modes
    # (Robot 8 BW: 30 lines/s) get coalesced so the message bus doesn't see
    # 30 KB×30 copies/s.
    _LINE_EMIT_MIN_INTERVAL_S = 0.1
    _LOOKING_EMIT_MIN_INTERVAL_S = 1.0

    def __init__(
        self,
        sample_rate: float,
        audio_rate: float = 11_025,
        channel_bandwidth: float | None = None,
        sstv_mode: str | None = None,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.audio_rate = float(audio_rate)
        self.channel_bandwidth = float(channel_bandwidth or self.DEFAULT_CHANNEL_BANDWIDTH)
        self.sstv_mode_name: str | None = None

        self._fwd_phase = 0.0
        self._back_phase = 0.0

        self._pending: list[DecodedMessage] = []
        self._capture_utc_s = 0.0
        self._build_audio_chain()

        if sstv_mode is not None:
            self.set_sstv_mode(sstv_mode)

    def _build_audio_chain(self) -> None:
        self._setup_channel_filter()
        self._dc_blocker = DCBlocker(self.decimated_rate, cutoff_hz=self.DC_BLOCKER_CUTOFF)
        self._agc = AGC(
            self.decimated_rate,
            attack_ms=5.0,
            decay_ms=200.0,
            setpoint=0.5,
        )
        self._decoder = self._build_decoder()
        self._last_line_emit_mono = 0.0
        self._next_looking_emit_sample = 0

    def _setup_channel_filter(self) -> None:
        self._decim = self._install_channel_frontend(
            self.sample_rate, self.audio_rate, self.channel_bandwidth
        )
        self._channel = self._build_channel_filter(self.channel_bandwidth)

    def _build_channel_filter(self, bandwidth: float) -> StreamingFilter:
        return StreamingFilter(
            firwin(128, bandwidth / 2, fs=self.decimated_rate),
            [1.0],
            dtype=np.complex64,
        )

    def _build_decoder(self) -> StreamingSSTV:
        forced = MODES_BY_NAME[self.sstv_mode_name.lower()] if self.sstv_mode_name else None
        return StreamingSSTV(
            int(self.decimated_rate),
            events=StreamingEvents(
                on_mode=self._on_mode,
                on_line=self._on_line,
                on_image=self._on_image,
            ),
            forced_mode=forced,
        )

    @property
    def _fwd_offset_hz(self) -> float:
        # USB selection: shift the band of interest down by bw/2 so the
        # upper sideband centres at DC. Lower sideband ends up at -bw and
        # is removed by the channel filter.
        return self.channel_bandwidth / 2.0

    def set_channel_bandwidth(self, bandwidth: float) -> None:
        self.channel_bandwidth = min(float(bandwidth), self.decimated_rate * NYQUIST_MARGIN)
        self._channel = self._build_channel_filter(self.channel_bandwidth)

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = float(rate)
        self._fwd_phase = 0.0
        self._back_phase = 0.0
        self._pending.clear()
        self._build_audio_chain()

    def set_sstv_mode(self, name: str | None) -> None:
        """Force a specific mode (skips VIS-code interpretation) or clear the
        override. Drops any in-flight image — the new mode takes effect at the
        next detected VIS pulse.
        """
        if name is None:
            self.sstv_mode_name = None
            self._decoder.set_forced_mode(None)
        else:
            mode = MODES_BY_NAME.get(name.lower())
            if mode is None:
                raise ValueError(f"unknown SSTV mode: {name}")
            self.sstv_mode_name = mode.name
            self._decoder.set_forced_mode(mode)
        self._decoder.reset()
        self._pending.clear()
        self._last_line_emit_mono = 0.0
        self._next_looking_emit_sample = 0

    def reset(self) -> None:
        super().reset()
        self._fwd_phase = 0.0
        self._back_phase = 0.0
        self._decim.reset()
        self._channel.reset()
        self._dc_blocker.reset()
        self._agc.reset()
        self._decoder.reset()
        self._pending.clear()
        self._last_line_emit_mono = 0.0
        self._next_looking_emit_sample = 0

    def status(self) -> DemodStatus:
        """Thread-safe: callable from any thread."""
        mode = self._decoder.mode
        return DemodStatus(description=mode.name if mode is not None else None)

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        if len(iq_samples) == 0:
            return

        self._capture_utc_s = capture_utc_s

        iq_lo = self._decim.process(iq_samples)
        if len(iq_lo) == 0:
            return

        offset = self._fwd_offset_hz
        iq_shifted, self._fwd_phase = apply_freq_shift_c64(
            iq_lo, offset, self.decimated_rate, self._fwd_phase
        )
        iq_filt = self._channel.process(iq_shifted)
        iq_back, self._back_phase = apply_freq_shift_c64(
            iq_filt, -offset, self.decimated_rate, self._back_phase
        )

        audio = iq_back.real.astype(np.float32, copy=False)
        audio = self._dc_blocker.process(audio)
        audio = self._agc.process(audio)
        np.clip(audio, -1.0, 1.0, out=audio)

        self._decoder.process(audio)
        self._maybe_emit_looking()
        self._emit_audio(audio, self.decimated_rate)

    def get_messages(self) -> list[DecodedMessage]:
        out, self._pending = self._pending, []
        return out

    def _push(self, text: str, data: SSTVData) -> None:
        ts = self._capture_utc_s if self._capture_utc_s > 0 else now_utc_seconds()
        self._pending.append(DecodedMessage(text=text, timestamp=ts, data=data))

    def _make_data(
        self,
        state: StreamerState,
        mode: Mode | None,
        *,
        line_index: int,
        image: np.ndarray | None,
        error: str | None = None,
    ) -> SSTVData:
        return SSTVData(
            state=state,
            vis_code=self._decoder.vis_code,
            mode_name=mode.name if mode is not None else None,
            line_index=line_index,
            total_lines=mode.scan_lines if mode is not None else 0,
            image_width=mode.width if mode is not None else 0,
            image_height=mode.height if mode is not None else 0,
            image=image,
            forced_mode=self.sstv_mode_name is not None,
            error=error,
        )

    def _maybe_emit_looking(self) -> None:
        """Push a bare LOOKING-state ping while we're scanning, rate-limited
        on the audio-sample clock (which makes the cadence right for both
        live audio and faster-than-realtime file replay).
        """
        if self._decoder.state != StreamerState.LOOKING:
            return
        if self._decoder.samples_processed < self._next_looking_emit_sample:
            return
        self._next_looking_emit_sample = self._decoder.samples_processed + int(
            self._LOOKING_EMIT_MIN_INTERVAL_S * self._decoder.fs
        )
        data = self._make_data(StreamerState.LOOKING, None, line_index=-1, image=None)
        self._push("SSTV listening", data)

    def _on_mode(self, mode: Mode, _vis_code: int) -> None:
        self._last_line_emit_mono = 0.0
        data = self._make_data(StreamerState.DECODING, mode, line_index=-1, image=None)
        self._push(f"SSTV {mode.name} (start)", data)

    def _on_line(self, mode: Mode, idx: int, _row: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_line_emit_mono < self._LINE_EMIT_MIN_INTERVAL_S:
            return
        self._last_line_emit_mono = now
        dstate = self._decoder.dstate
        snapshot = dstate.image.copy() if dstate is not None else None
        data = self._make_data(StreamerState.DECODING, mode, line_index=idx, image=snapshot)
        self._push(f"SSTV {mode.name} L{idx + 1}/{mode.scan_lines}", data)

    def _on_image(self, mode: Mode, img: np.ndarray) -> None:
        data = self._make_data(
            StreamerState.DONE, mode, line_index=mode.scan_lines - 1, image=img.copy()
        )
        self._push(f"SSTV {mode.name} (done)", data)
        self._decoder.reset()
        self._last_line_emit_mono = 0.0
        self._next_looking_emit_sample = 0
