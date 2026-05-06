import numpy as np

from .decoder import (
    build_noise_filter,
    decode_algebraic_code,
    decode_gains,
    decode_lsp,
    decode_pitch_sf0,
    decode_pitch_sfn,
    interpolate_lpc,
    unpack_frame,
)
from .kernels import pred_lt, syn_filt
from .tables import (
    INTER_COEF_1_3,
    INTER_COEF_M1_3,
    L_FRAME,
    L_INTER,
    L_SUBFR,
    LSPOLD_INIT,
    PIT_MAX,
    P,
)

# RMS threshold for silence gating (post 6 dB boost, int16 scale).
# Frames below this are ambient mic noise, not speech.
_SILENCE_RMS_THRESHOLD = 60.0


class TetraAcelpVocoder:
    """TETRA ACELP speech decoder (ETSI EN 300 395-2).

    Decodes 137-bit coded frames into 240-sample PCM frames (30ms @ 8kHz).
    """

    def __init__(self) -> None:
        self.lsp_old = LSPOLD_INIT.copy()
        self.exc = np.zeros(PIT_MAX + L_INTER + L_FRAME, dtype=np.float64)
        self.mem_syn = np.zeros(P, dtype=np.float64)
        self.old_T0 = 60
        self.last_ener_pit = 0.0
        self.last_ener_cod = 0.0
        self.old_parm = np.zeros(23, dtype=np.int32)

    @property
    def _exc_offset(self) -> int:
        """Offset into exc buffer where current frame starts."""
        return PIT_MAX + L_INTER

    def decode_frame(self, coded: bytes, bfi: bool = False) -> np.ndarray:
        """Decode one ACELP frame.

        coded: 18 bytes of packed codec data
        bfi: bad frame indicator

        Returns 240 int16 PCM samples.
        """
        parm = unpack_frame(coded)

        if not bfi:
            lsp_new = decode_lsp(parm[:3], self.lsp_old)
            self.old_parm[:] = parm
        else:
            lsp_new = self.lsp_old.copy()
            parm = self.old_parm.copy()

        # Interpolate LPC for 4 subframes
        a_all = interpolate_lpc(self.lsp_old, lsp_new)
        self.lsp_old[:] = lsp_new

        # Subframe parameter pointer (skip 3 LSP indices)
        sf_parm_idx = 3

        synth = np.empty(L_FRAME, dtype=np.float64)
        exc_off = self._exc_offset
        t0_min = 0

        for sf in range(4):
            a_sf = a_all[sf]
            index = int(parm[sf_parm_idx])
            sf_parm_idx += 1

            # Pitch decoding
            if sf == 0:
                if not bfi:
                    t0, t0_frac, t0_min, t0_max = decode_pitch_sf0(index)
                else:
                    t0 = self.old_T0
                    t0_frac = 0
                    t0_min = max(t0 - 5, 20)
                    t0_max = min(t0_min + 9, 143)
                    if t0_max > 143:
                        t0_max = 143
                        t0_min = t0_max - 9
            else:
                if not bfi:
                    t0, t0_frac = decode_pitch_sfn(index, t0_min)

            # Adaptive codebook
            sf_start = exc_off + sf * L_SUBFR
            pred_lt(self.exc, sf_start, t0, t0_frac, L_SUBFR, INTER_COEF_1_3, INTER_COEF_M1_3)
            adaptive_exc = self.exc[sf_start : sf_start + L_SUBFR].copy()

            # Noise filter + algebraic codebook
            f_padded = build_noise_filter(a_sf, t0)
            code_index = int(parm[sf_parm_idx])
            sign_code = int(parm[sf_parm_idx + 1])
            shift_code = int(parm[sf_parm_idx + 2])
            sf_parm_idx += 3

            code = decode_algebraic_code(code_index, sign_code, shift_code, f_padded)

            # Gain decoding
            gain_index = int(parm[sf_parm_idx])
            sf_parm_idx += 1

            gain_pit, gain_cod, self.last_ener_pit, self.last_ener_cod = decode_gains(
                gain_index,
                bfi,
                a_sf,
                adaptive_exc,
                code,
                self.last_ener_pit,
                self.last_ener_cod,
            )

            # Excitation: gain_pit is Q12, gain_cod is Q0
            for i in range(L_SUBFR):
                val = adaptive_exc[i] * gain_pit / 4096.0 + code[i] * gain_cod
                if val > 32767.0:
                    val = 32767.0
                elif val < -32768.0:
                    val = -32768.0
                self.exc[sf_start + i] = val

            # Synthesis filter
            sf_synth = syn_filt(a_sf, self.exc[sf_start : sf_start + L_SUBFR], self.mem_syn, True)
            synth[sf * L_SUBFR : (sf + 1) * L_SUBFR] = sf_synth

        # Shift excitation buffer
        self.exc[: PIT_MAX + L_INTER] = self.exc[L_FRAME:]
        self.exc[PIT_MAX + L_INTER :] = 0.0

        self.old_T0 = t0

        # Post-processing: multiply by 2 (6dB boost)
        synth *= 2.0

        # Silence gate: mute frames where the codec is reproducing
        # low-level ambient mic noise rather than speech. State is
        # already updated so speech resumes cleanly.
        rms = float(np.sqrt(np.mean(synth * synth)))
        if rms < _SILENCE_RMS_THRESHOLD:
            synth[:] = 0.0

        # Clip and convert to int16
        synth = np.clip(synth, -32768, 32767)
        return synth.astype(np.int16)
