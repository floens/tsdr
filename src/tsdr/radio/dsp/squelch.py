from __future__ import annotations

import math

import numpy as np


class SquelchGate:
    """Audio-muting gate driven by pre-demod RMS power."""

    _HYSTERESIS_DB = 3.0
    _EMA_TC_MS = 50.0

    def __init__(self, audio_rate: float, ramp_ms: float = 10.0) -> None:
        self._audio_rate = float(audio_rate)
        self._ema_tc_s = self._EMA_TC_MS / 1000.0
        self._ramp_samples = max(1, int(self._audio_rate * ramp_ms / 1000.0))

        self._enabled = False
        self._threshold_db = -50.0
        self._hang_samples = int(self._audio_rate * 0.5)

        self._power_ema_db = -120.0
        self._is_open = False
        self._hang_counter = 0
        self._gain = 1.0

    def configure(
        self,
        enabled: bool,
        threshold_db: float,
        hang_ms: float,
    ) -> None:
        self._enabled = bool(enabled)
        self._threshold_db = float(threshold_db)
        self._hang_samples = max(0, int(self._audio_rate * hang_ms / 1000.0))

    def reset(self) -> None:
        self._power_ema_db = -120.0
        self._is_open = False
        self._hang_counter = 0
        self._gain = 1.0 if not self._enabled else 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def threshold_db(self) -> float:
        return self._threshold_db

    @property
    def hang_ms(self) -> float:
        return self._hang_samples / self._audio_rate * 1000.0

    @property
    def power_ema_db(self) -> float:
        return self._power_ema_db

    def process(self, power_db: float, n_audio_samples: int) -> np.ndarray | None:
        if not self._enabled:
            self._gain = 1.0
            return None

        # Chunk-duration EMA: alpha adapts so the effective time constant is
        # ~EMA_TC_MS regardless of chunk size.
        dt = n_audio_samples / self._audio_rate
        alpha = math.exp(-dt / self._ema_tc_s)
        self._power_ema_db = alpha * self._power_ema_db + (1.0 - alpha) * power_db

        open_threshold = self._threshold_db + self._HYSTERESIS_DB
        close_threshold = self._threshold_db - self._HYSTERESIS_DB

        if self._power_ema_db >= open_threshold:
            self._is_open = True
            self._hang_counter = self._hang_samples
        elif self._power_ema_db <= close_threshold:
            if self._hang_counter > 0:
                self._hang_counter = max(0, self._hang_counter - n_audio_samples)
            else:
                self._is_open = False

        target_gain = 1.0 if self._is_open else 0.0
        envelope = _build_envelope(
            start=self._gain,
            target=target_gain,
            n=n_audio_samples,
            ramp_samples=self._ramp_samples,
        )
        self._gain = float(envelope[-1])
        return envelope


def iq_power_db(iq: np.ndarray) -> float:
    """RMS power of complex IQ in dBFS (0 dB = unity amplitude)."""
    if len(iq) == 0:
        return -120.0
    mean_power = float(np.vdot(iq, iq).real) / len(iq)
    return 10.0 * math.log10(mean_power + 1e-20)


def _build_envelope(start: float, target: float, n: int, ramp_samples: int) -> np.ndarray:
    """Linear ramp toward target at 1/ramp_samples per sample, clamped at target."""
    envelope = np.empty(n, dtype=np.float32)
    if start == target:
        envelope.fill(target)
        return envelope

    step = 1.0 / ramp_samples
    offsets = np.arange(1, n + 1, dtype=np.float32) * np.float32(step)
    if target > start:
        ramp = start + offsets
        np.minimum(ramp, target, out=ramp)
    else:
        ramp = start - offsets
        np.maximum(ramp, target, out=ramp)
    envelope[:] = ramp
    return envelope
