"""ACARS decoder: single tuned VHF channel, AM envelope -> coherent MSK -> text.

Pipeline (see the module DSP notes): device IQ -> decimate to a ~24 kHz
intermediate -> complex channel-select FIR -> AM envelope (|.|) -> DC block ->
decimate to 12 kHz -> `MSKDemod` -> `AcarsFramer`. Envelope detection is
insensitive to carrier offset, so the tune need not be exact; the fixed channel
filter width sets the slack vs adjacent-channel rejection.
"""

from __future__ import annotations

import numpy as np

from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.radio.decoders.acars.constants import (
    BAUD,
    CHANNEL_BANDWIDTH,
    INTERMEDIATE_RATE,
    INTERNAL_RATE,
    MSK_CENTER_HZ,
    MSK_SPACE_HZ,
)
from tsdr.radio.decoders.acars.frame import AcarsFramer
from tsdr.radio.decoders.acars.msk import MSKDemod
from tsdr.radio.demodulators import NYQUIST_MARGIN, Demodulator
from tsdr.radio.dsp import DCBlocker, StreamingDecimFilter, StreamingFilter, firwin


class ACARSDecoder(Demodulator):
    LABEL = "ACARS"
    MODULATION = "MSK"
    MESSAGE_TYPE = "text"
    HAS_TEXT = True
    FIXED_CHANNEL_BANDWIDTH = CHANNEL_BANDWIDTH

    def __init__(self, sample_rate: float) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self._framer = AcarsFramer()
        self._build()

    def _build(self) -> None:
        sr = self.sample_rate
        # Decimate to a ~24 kHz intermediate (leaving offset headroom); a clean
        # multiple of 12 kHz collapses to interm_decim=1.
        interm_decim = max(1, round(sr / INTERMEDIATE_RATE))
        self.intermediate_rate = sr / interm_decim

        self._decim1 = (
            StreamingDecimFilter(
                firwin(64, self.intermediate_rate * 0.45, fs=sr, window=("kaiser", 6.0)),
                decimation=interm_decim,
                dtype=np.complex64,
            )
            if interm_decim > 1
            else None
        )
        cutoff = min(CHANNEL_BANDWIDTH / 2, self.intermediate_rate * NYQUIST_MARGIN / 2)
        self._channel = StreamingFilter(
            firwin(128, cutoff, fs=self.intermediate_rate), [1.0], dtype=np.complex64
        )
        self._dc = DCBlocker(self.intermediate_rate, cutoff_hz=16.0)

        audio_decim = max(1, round(self.intermediate_rate / INTERNAL_RATE))
        self._decim2 = (
            StreamingDecimFilter(
                firwin(64, INTERNAL_RATE * 0.45, fs=self.intermediate_rate),
                decimation=audio_decim,
                dtype=np.float32,
            )
            if audio_decim > 1
            else None
        )
        msk_rate = self.intermediate_rate / audio_decim
        self._msk = MSKDemod(msk_rate, BAUD, center_hz=MSK_CENTER_HZ, space_hz=MSK_SPACE_HZ)

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = float(rate)
        self._build()
        self._framer.reset()

    def status(self) -> DemodStatus:
        return DemodStatus()

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        if len(iq_samples) == 0:
            return
        iq = iq_samples if self._decim1 is None else self._decim1.process(iq_samples)
        iq = self._channel.process(iq)
        env = np.abs(iq)
        env = self._dc.process(env)
        if self._decim2 is not None:
            env = self._decim2.process(env)
        soft = self._msk.process(env)
        self._framer.process(soft, capture_utc_s)

    def get_messages(self) -> list:
        return self._framer.drain()

    def reset(self) -> None:
        super().reset()
        if self._decim1 is not None:
            self._decim1.reset()
        self._channel.reset()
        self._dc.reset()
        if self._decim2 is not None:
            self._decim2.reset()
        self._msk.reset()
        self._framer.reset()
