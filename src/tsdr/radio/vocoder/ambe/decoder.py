"""Public AMBE+2 2450 decoder

Wires all six layers into a stateful decoder class. Three public
entry points at three levels of abstraction, all converging at the
parameter-decode stage:

    decode_frame(voice_bits, bfi=False) -> int16[160]
        TSDR-facing. 9 bytes (72 raw bits) of DMR voice frame in,
        160 int16 samples out. `bfi=True` treats the frame as a bad
        frame indicator and returns silence.

    decode_codewords(ambe_fr) -> int16[160]
        Takes a (4, 24) int8 array with C0..C3 codewords.

    decode_ambe_d(ambe_d, errs2=0) -> int16[160]
        Low-level. Takes the 49-bit post-FEC data vector directly --
        same input format as DSD's `.amb` file records.

The decoder carries three `MbeParams` instances (`cur_mp`, `prev_mp`,
`prev_mp_enh`) across calls to maintain the inter-frame state that
MBE needs for smooth harmonic synthesis.
"""

from __future__ import annotations

import math

import numpy as np

from tsdr.radio.vocoder.ambe._rng import Xorshift32
from tsdr.radio.vocoder.ambe.enhance import spectral_amp_enhance
from tsdr.radio.vocoder.ambe.frame import demodulate_c1, ecc_c0, pack_ambe_d
from tsdr.radio.vocoder.ambe.params import MbeParams
from tsdr.radio.vocoder.ambe.params_decode import decode_ambe2450_parms
from tsdr.radio.vocoder.ambe.synthesis import (
    float_to_short,
    synthesize_silencef,
    synthesize_speechf,
)

# AMBE silence frame marker: b0=124 or 125 sets w0 to this exact value.
_SILENCE_W0 = float(np.float32(2.0 * math.pi / 32.0))


class AmbePlus2Decoder:
    """Stateful AMBE+2 2450/1150 decoder for DMR voice.

    One instance per logical voice stream (e.g. one per DMR timeslot).
    Do not share an instance between unrelated streams -- inter-frame
    state would bleed across.

    Example::

        >>> dec = AmbePlus2Decoder()
        >>> pcm = dec.decode_frame(voice_bytes)  # int16[160]
    """

    def __init__(self, uvquality: int = 3, rng_seed: int = 1) -> None:
        self.uvquality = uvquality
        self.cur_mp = MbeParams()
        self.prev_mp = MbeParams()
        self.prev_mp_enh = MbeParams()
        self._rng = Xorshift32(rng_seed)

    def decode_frame(self, voice_bits: bytes, bfi: bool = False) -> np.ndarray:
        """Decode a 9-byte (72-bit) DMR voice frame into ``int16[160]``.

        ``voice_bits`` is the post-BPTC frame payload, 9 bytes MSB-first
        in transmission order. ``bfi=True`` bypasses decode and outputs
        silence -- use it on bursts that failed channel FEC.
        """
        if bfi:
            self._reset_state()
            return float_to_short(synthesize_silencef())
        if len(voice_bits) < 9:
            raise ValueError(f"expected at least 9 bytes, got {len(voice_bits)}")
        return self.decode_codewords(_unpack_voice_bits(voice_bits))

    def decode_codewords(self, ambe_fr: np.ndarray) -> np.ndarray:
        """Decode a ``(4, 24)`` int8 codeword matrix into ``int16[160]``.

        ``ambe_fr`` format: row 0 is C0 (24
        bits), row 1 is C1 (23 bits, bit 23 unused), row 2 is C2 (11
        bits), row 3 is C3 (14 bits), with unused slots zero-filled.
        """
        fr = np.asarray(ambe_fr, dtype=np.int8).copy()
        c0_errs = ecc_c0(fr)
        demodulate_c1(fr)
        ambe_d, c1_errs = pack_ambe_d(fr)
        return self.decode_ambe_d(ambe_d, errs2=c0_errs + c1_errs)

    def decode_ambe_d(self, ambe_d: np.ndarray, errs2: int = 0) -> np.ndarray:
        """Decode an already-FEC'd 49-bit ambe_d vector into ``int16[160]``.

        This is the `.amb` file path. ``errs2`` is the channel error
        count carried alongside each record (DSD sets this during
        recording); values > 3 trigger the "use previous params"
        repeat behaviour.
        """
        return float_to_short(self.decode_ambe_d_float(np.asarray(ambe_d, dtype=np.int8), errs2))

    def decode_ambe_d_float(self, ambe_d: np.ndarray, errs2: int = 0) -> np.ndarray:
        """Full pipeline from ambe_d through float32[160] audio.

        Same as ``decode_ambe_d`` but without the final int16 gain+clip
        step, so the output preserves fractional amplitude. Useful for
        feeding into downstream DSP without two rounds of quantisation.
        """
        bad = decode_ambe2450_parms(ambe_d, self.cur_mp, self.prev_mp)
        if bad != 0:
            self._reset_state()
            return synthesize_silencef()

        if errs2 > 3:
            self.cur_mp.use_last(self.prev_mp)
            self.cur_mp.repeat += 1
        else:
            self.cur_mp.repeat = 0

        if self.cur_mp.repeat > 3:
            self._reset_state()
            return synthesize_silencef()

        self.prev_mp.copy_from(self.cur_mp)
        spectral_amp_enhance(self.cur_mp)
        audio = synthesize_speechf(self.cur_mp, self.prev_mp_enh, self.uvquality, self._rng)
        self.prev_mp_enh.copy_from(self.cur_mp)

        # Silence-coded frames (b0=124/125) produce low-level unvoiced
        # noise from the speaker's ambient mic pickup. Mute them to avoid
        # the periodic frame-rate buzz that vocoders produce during
        # silence. State is already updated so speech resumes cleanly.
        if self.cur_mp.w0 == _SILENCE_W0:
            audio[:] = 0.0

        return audio

    def _reset_state(self) -> None:
        self.cur_mp.reset()
        self.prev_mp.reset()
        self.prev_mp_enh.reset()


def _unpack_voice_bits(voice_bits: bytes) -> np.ndarray:
    """Unpack 9 bytes (72 bits MSB-first) into the ``(4, 24)`` codeword layout.

    DMR TS 102 361-1 transmits the four AMBE+2 codewords concatenated
    in order C0 (24) + C1 (23) + C2 (11) + C3 (14) = 72 bits. The
    in-memory layout reverses the bit order within each codeword so
    that ``fr[i, j]`` holds the j-th bit from the LSB side -- the
    Golay/descramble/pack layers rely on this.
    """
    bits = np.unpackbits(np.frombuffer(voice_bits[:9], dtype=np.uint8))
    fr = np.zeros((4, 24), dtype=np.int8)
    # Each assignment reverses the slice to LSB-first codeword order.
    fr[0, :24] = bits[:24][::-1]
    fr[1, :23] = bits[24:47][::-1]
    fr[2, :11] = bits[47:58][::-1]
    fr[3, :14] = bits[58:72][::-1]
    return fr


def _pack_voice_bits(ambe_fr: np.ndarray) -> bytes:
    """Inverse of ``_unpack_voice_bits``. Used by round-trip tests."""
    bits = np.zeros(72, dtype=np.uint8)
    bits[:24] = ambe_fr[0, :24][::-1]
    bits[24:47] = ambe_fr[1, :23][::-1]
    bits[47:58] = ambe_fr[2, :11][::-1]
    bits[58:72] = ambe_fr[3, :14][::-1]
    return bytes(np.packbits(bits))
