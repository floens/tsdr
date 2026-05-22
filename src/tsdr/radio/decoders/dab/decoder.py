from __future__ import annotations

import logging
import time

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, SignalInfo
from tsdr.core.tracing import span
from tsdr.radio.demodulators import Demodulator

from .aac import _AACDecoder
from .constants import N_FIC_SYMBOLS, T_FRAME, T_S, VITERBI_K
from .data import DABData, DABServiceInfo, DABSlide, DABStats
from .fec import _generate_prbs
from .fic import _decode_fic
from .fig import DABEnsemble, DABService, _build_ensemble, _FIGParserState, _parse_figs
from .msc import (
    _build_eep_depuncture_index,
    _demod_frame_msc,
    _eep_depuncture,
    _extract_subchannel,
    _msc_to_cifs,
    _TimeDeinterleaver,
)
from .ofdm import (
    OFDMState,
    _dqpsk_constellation,
    _dqpsk_to_soft_bits,
    _ofdm_demod_frame,
    detect_null_symbols,
)
from .pad import _CONTENT_SUB_TYPE_JFIF, _CONTENT_SUB_TYPE_PNG, PADDecoder
from .superframe import SuperframeFormat, _DabPlusSuperframe
from .viterbi import _viterbi_decode

logger = logging.getLogger(__name__)


def _mot_file_to_slide(mot: PADDecoder) -> DABSlide | None:
    """Convert PADDecoder's current MOT file to a frozen DABSlide."""
    f = mot.slide
    if f is None:
        return None
    if f.content_sub_type == _CONTENT_SUB_TYPE_JFIF:
        mime = "image/jpeg"
    elif f.content_sub_type == _CONTENT_SUB_TYPE_PNG:
        mime = "image/png"
    else:
        mime = "image/unknown"
    return DABSlide(
        data=f.data,
        content_type=mime,
        content_name=f.content_name,
        category_title=f.category_title,
    )


class DABDecoder(Demodulator):
    """DAB+ Mode I decoder.

    Decodes DAB OFDM frames from 2048 kHz complex IQ samples. A DAB Mode I
    frame is 196608 samples (~96ms) containing a null symbol followed by 76
    OFDM symbols with 1536 active subcarriers each. The decoder extracts
    ensemble metadata from the FIC (Fast Information Channel) and optionally
    decodes one audio subchannel from the MSC (Main Service Channel) to PCM.

    Pipeline overview
    -----------------

    **Frame synchronization** (ofdm.py):
    The input IQ stream is buffered and scanned for null symbols, periods of
    near-zero power that mark frame boundaries. Null detection divides the
    stream into 2656-sample blocks and compares block power against the median;
    runs of >=2 low-power blocks indicate a null symbol. This gives coarse
    frame boundaries quantized to ~664 samples.

    **OFDM demodulation** (ofdm.py):
    Within each frame, fine timing is established in two stages: cyclic prefix
    (guard interval) correlation narrows the PRS start to ±4 samples, then
    cross-correlation of the candidate FFT with the known PRS reference
    (ETSI EN 300 401 section 14.3.2) pins it to single-sample accuracy.
    All 76 symbols are extracted (guard stripped) and batch-FFT'd to produce
    a (76, 2048) complex array of frequency-domain symbols.

    **Frequency tracking** (ofdm.py, OFDMState):
    A PLL tracks frequency drift across frames. On bootstrap (first frame or
    after reset), the full fractional offset is estimated from cyclic prefix
    correlation and seeded into the accumulator. On subsequent frames, the
    accumulated correction is applied to the raw IQ before timing/FFT, then
    the residual offset on the corrected data is measured and integrated with
    IIR gain 0.1. When the accumulated offset exceeds +-500 Hz (half the 1 kHz
    carrier spacing), it snaps to the nearest integer carrier offset to prevent
    drift into adjacent carriers.

    **DQPSK demodulation** (ofdm.py):
    Differential decoding: each symbol's carriers are multiplied by the
    conjugate of the previous symbol's carriers, producing one complex
    differential product per carrier per symbol. The products are frequency
    de-interleaved (ETSI section 14.6 permutation) and rotated by π/2 to
    align the DQPSK constellation ({±45°, ±135°}) onto the I/Q axes for
    soft-bit extraction. Soft bits are computed as (-imag, +real) of the
    normalized products, yielding 3072 soft bits per symbol (1536 carriers
    × 2 bits/carrier).

    **FIC decoding** (fic.py):
    Symbols 1-3 carry the FIC: 3 × 3072 = 9216 soft bits, split into 4
    blocks of 2304. Each block is depunctured (PI_16 for the main part,
    PI_15 for the tail), expanding 2304 -> 3096 soft bits at the input to
    the rate 1/4, K=7 convolutional decoder. The Viterbi decoder (viterbi.py,
    numba JIT-compiled) produces 768 hard bits per block. These are XOR'd
    with a PRBS sequence (LFSR x^9+x^5+1) for energy dispersal, then split
    into 3 FIBs of 256 bits (32 bytes) each, CRC-16 checked. This yields
    up to 12 FIBs per frame.

    **FIG parsing** (fig.py):
    Valid FIBs are parsed for Fast Information Groups: FIG 0/0 (ensemble ID),
    FIG 0/1 (subchannel configuration: start address, size, protection level,
    EEP option), FIG 0/2 (service-to-subchannel mapping), FIG 0/13 (user
    application types - identifies MOT slideshow), FIG 1/0 (ensemble label),
    and FIG 1/1 (service labels). Once enough FIGs are collected, a
    DABEnsemble is assembled with service list and subchannel parameters.

    **Service selection and MSC decoding** (decoder.py, msc.py):
    After ensemble info is available, the first audio service with subchannel
    parameters is auto-selected (or a specific service can be selected via
    select_service()). Symbols 4-75 (72 data symbols) are DQPSK-demodulated
    to 221184 soft bits, split into 4 CIFs (Common Interleaved Frames) of
    55296 bits each. Each CIF represents 864 capacity units (CUs) of 64 bits.

    **Time de-interleaving** (msc.py, _TimeDeinterleaver):
    A 16-frame convolutional de-interleaver buffers CIFs and reassembles bits
    scattered across 16 consecutive CIFs by the transmitter's interleaver.
    Read-before-write order with delay map [0,8,4,12,2,10,6,14,1,9,5,13,
    3,11,7,15] (= 15 - ETSI delay). One shared instance processes all CIFs
    sequentially - using per-position instances breaks the interleaver state.
    Output begins after 16 CIFs of fill.

    **EEP depuncturing and Viterbi** (msc.py, viterbi.py):
    The selected subchannel's CUs are sliced from each de-interleaved CIF.
    Punctured positions (determined by the EEP-A or EEP-B protection level
    and subchannel size) are filled with zero-confidence soft bits. The same
    rate 1/4, K=7 Viterbi decoder used for FIC decodes the depunctured stream.
    Tail bits are discarded and the result is PRBS-descrambled.

    **Superframe and Reed-Solomon** (superframe.py):
    5 consecutive decoded logical frames (110 bytes each) form a DAB+
    superframe of 550 bytes, arranged as a 5×110 matrix. RS(120,110) error
    correction is applied column-wise across the matrix (not row-wise - the
    parity bytes span the 5 frames). The corrected data contains a header
    with AAC AU boundaries and sample rate info.

    **AAC decoding** (aac.py):
    Access Units are extracted from the superframe, stripped of their 2-byte
    CRC suffix, wrapped in ADTS headers (signaling the core sample rate, e.g.
    24 kHz for HE-AAC v2, not the output rate), and fed to pyav's AAC decoder.
    HE-AAC v2 with SBR doubles the sample rate (24->48 kHz) and PS provides
    stereo from a mono core. Because ffmpeg's AAC decoder uses 1024-sample
    transforms instead of DAB+'s 960, the effective output rate is
    48000 × 1024/960 = 51200 Hz; the audio worker resamples to 48 kHz.

    **PAD decoding** (pad.py):
    Programme Associated Data is extracted from each AAC AU via MPEG-4 DSE
    (Data Stream Element) detection. The PAD carries DLS (Dynamic Label
    Segment) for "now playing" text and MOT (Multimedia Object Transfer)
    for slideshow images. DLS segments are reassembled from X-PAD data
    subfields with toggle/first/last markers. MOT objects are assembled
    from header and body segments identified by transport ID.

    Supported features:
        - DAB+ (AAC audio) only, not classic DAB (MP2)
        - Mode I only (T_U=2048, 76 symbols, 1536 carriers)
        - HE-AAC v2 (SBR + PS), mono and stereo
        - EEP-A and EEP-B protection levels
        - DLS (Dynamic Label) text
        - MOT slideshow (JPEG/PNG)

    Follows the Decoder protocol: decode(data) -> list[DecodedMessage], reset().
    Audio output: call get_audio() to retrieve decoded PCM batches.
    """

    has_audio = True

    def __init__(self, sample_rate: float = 2_048_000):
        super().__init__()
        self._sample_rate = sample_rate
        self._buffer = np.array([], dtype=np.complex64)
        self._fig_state = _FIGParserState()
        self._last_ensemble: DABEnsemble | None = None

        # Statistics
        self._frames_processed = 0
        self._fibs_decoded = 0
        self._fibs_crc_ok = 0
        self._null_symbols_found = 0

        # MSC audio decoding state
        self._selected_service: DABService | None = None
        self._time_deinterleaver = _TimeDeinterleaver()
        self._depuncture_index: np.ndarray | None = None
        self._depuncture_len: int = 0
        self._superframe = _DabPlusSuperframe()
        self._aac_decoder: _AACDecoder | None = None
        self._audio_batches: list[AudioBatch] = []
        self._pending_messages: list[DecodedMessage] = []
        self._prbs_cache: np.ndarray | None = None  # cached PRBS for subchannel size
        self._ofdm_state = OFDMState()
        self._audio_sample_rate: float | None = None
        self._audio_channels: int | None = None
        self._sf_format: SuperframeFormat | None = None
        self._pad_decoder = PADDecoder()
        self._constellation_points: np.ndarray | None = None

    @property
    def audio_prebuffer_seconds(self) -> float:
        # 16-frame time de-interleaver + superframe assembly; need runway for frame drops.
        return 1.0

    def select_service(self, service_id: int | None = None) -> str | None:
        """Select an audio service to decode. None = first audio service.

        Returns service label on success, or error string prefixed with "Error:".
        """
        ensemble = _build_ensemble(self._fig_state)
        if not ensemble.services:
            return "Error: No ensemble data yet"

        if service_id is None:
            # Select first audio service with resolved label and subchannel info.
            # Labels arrive via FIG 1/1; services without labels may be data-only
            # or not yet fully described by the FIC.
            for svc in ensemble.services:
                if (
                    svc.is_audio
                    and svc.start_address is not None
                    and not svc.label.startswith("Service 0x")
                ):
                    return self._activate_service(svc)
            return "Error: No audio services found"

        for svc in ensemble.services:
            if svc.service_id == service_id and svc.is_audio:
                return self._activate_service(svc)

        return f"Error: Service {service_id:#06x} not found"

    def _activate_service(self, svc: DABService) -> str:
        """Activate audio decoding for a service."""
        self._selected_service = svc

        # Build depuncture index
        if svc.subchannel_size is not None and svc.protection_level is not None:
            self._depuncture_index, self._depuncture_len = _build_eep_depuncture_index(
                svc.subchannel_size, svc.protection_level, svc.eep_option
            )

        # Generate PRBS - size determined after first Viterbi run
        self._prbs_cache = None

        # Reset superframe accumulator
        self._superframe.reset()

        # Initialize AAC decoder
        self._aac_decoder = _AACDecoder()

        # Reset time deinterleavers
        self._time_deinterleaver = _TimeDeinterleaver()

        # Reset PAD state (DLS label, slide, MOT app type) for new service
        self._pad_decoder.reset()

        logger.info(
            "dab_service_selected label=%r subchannel=%d start=%d size=%d protection=%d option=%s",
            svc.label,
            svc.subchannel_id,
            svc.start_address,
            svc.subchannel_size,
            svc.protection_level,
            svc.eep_option,
        )
        return svc.label

    def _build_dab_data(self, ensemble: DABEnsemble) -> DABData:
        """Build a DABData snapshot for UI consumption."""
        services = tuple(
            DABServiceInfo(
                service_id=s.service_id,
                label=s.label,
                is_audio=s.is_audio,
                subchannel_id=s.subchannel_id,
                protection_level=s.protection_level,
                subchannel_size=s.subchannel_size,
            )
            for s in ensemble.services
        )
        return DABData(
            ensemble_id=ensemble.ensemble_id,
            ensemble_label=ensemble.label,
            services=services,
            selected_service_id=(
                self._selected_service.service_id if self._selected_service else None
            ),
            fib_crc_rate=(
                self._fibs_crc_ok / self._fibs_decoded if self._fibs_decoded > 0 else 0.0
            ),
            frames_processed=self._frames_processed,
            freq_offset_hz=self._ofdm_state.accumulated_hz,
            audio_sample_rate=self._audio_sample_rate,
            audio_channels=self._audio_channels,
            core_sample_rate=self._sf_format.core_sample_rate if self._sf_format else None,
            sbr=self._sf_format.sbr if self._sf_format else None,
            ps=self._sf_format.ps if self._sf_format else None,
            dynamic_label=self._pad_decoder.dynamic_label,
            slide=_mot_file_to_slide(self._pad_decoder),
        )

    def get_audio(self) -> list[AudioBatch]:
        """Return accumulated audio batches and clear buffer."""
        batches = self._audio_batches
        self._audio_batches = []
        return batches

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._pending_messages
        self._pending_messages = []
        return messages

    def get_constellation(self) -> tuple[np.ndarray, str] | None:
        points = self._constellation_points
        self._constellation_points = None
        if points is None:
            return None
        return points, "DQPSK"

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        self._buffer = np.concatenate([self._buffer, iq_samples.astype(np.complex64)])

        while len(self._buffer) >= 2 * T_FRAME:
            null_positions = detect_null_symbols(self._buffer[: 3 * T_FRAME])

            if len(null_positions) < 1:
                self._buffer = self._buffer[T_FRAME:]
                continue

            frame_start = null_positions[0]
            frame_end = frame_start + T_FRAME

            if frame_end + T_S > len(self._buffer):
                break

            self._null_symbols_found += 1
            # Extra T_S samples: null detection has block-quantized positions
            # (~64 samples/frame drift), so fine timing may place the last symbol
            # well beyond T_FRAME. T_S gives a full symbol width of headroom.
            frame_iq = self._buffer[frame_start : frame_end + T_S]

            frame_msgs = self._process_frame(frame_iq)
            self._pending_messages.extend(frame_msgs)

            self._buffer = self._buffer[frame_end:]

        max_buffer = 4 * T_FRAME
        if len(self._buffer) > max_buffer:
            self._buffer = self._buffer[-max_buffer:]

    def _process_frame(self, frame_iq: np.ndarray) -> list[DecodedMessage]:
        """Process a single DAB frame."""
        self._frames_processed += 1

        with span("ofdm"):
            fft_syms = _ofdm_demod_frame(frame_iq, self._ofdm_state)
        if fft_syms is None:
            return []

        # Constellation: 1 symbol's worth of carriers per frame
        # Take every 3rd carrier (512 of 1536) to keep event size reasonable
        self._constellation_points = _dqpsk_constellation(fft_syms, start_sym=0, n_syms=1)[::3]

        with span("fic"):
            fic_soft = _dqpsk_to_soft_bits(fft_syms, 0, N_FIC_SYMBOLS)
            fibs = _decode_fic(fic_soft)

        messages = []
        for fib_bytes, crc_ok in fibs:
            self._fibs_decoded += 1
            if crc_ok:
                self._fibs_crc_ok += 1
                _parse_figs(fib_bytes, self._fig_state)

        # Build ensemble and generate messages if we have new info
        ensemble = _build_ensemble(self._fig_state)
        if ensemble.label and ensemble != self._last_ensemble:
            self._last_ensemble = ensemble
            svc_list = ", ".join(s.label for s in ensemble.services)
            text = f"[DAB] {ensemble.label}"
            if svc_list:
                text += f" | Services: {svc_list}"
            messages.append(DecodedMessage(text=text, timestamp=time.time()))

        # Always emit DABData snapshot so the widget sees current state
        # (selected service, DLS, slide, stats change independently of ensemble)
        if self._last_ensemble is not None:
            dab_data = self._build_dab_data(self._last_ensemble)
            messages.append(DecodedMessage(text="", timestamp=time.time(), data=dab_data))

        # Set MOT app type from FIG 0/13 user application info
        # App type 0x002 = MOT Slideshow -> X-PAD CI type 12
        if self._pad_decoder.mot_app_type < 0 and self._fig_state.user_app_types:
            for app_types in self._fig_state.user_app_types.values():
                if 0x002 in app_types:
                    self._pad_decoder.mot_app_type = 12
                    break

        # MSC decoding (if a service is selected)
        if self._selected_service is not None:
            self._process_msc(fft_syms)

        return messages

    def _process_msc(self, fft_syms: np.ndarray) -> None:
        """Decode MSC for the selected service."""
        svc = self._selected_service
        if svc is None or svc.start_address is None or svc.subchannel_size is None:
            return

        with span("msc_demod"):
            msc_soft = _demod_frame_msc(fft_syms)
            cifs = _msc_to_cifs(msc_soft)

        # Collect all PCM from all CIFs in this frame into one batch
        frame_pcm_parts: list[np.ndarray] = []

        for cif_soft in cifs:
            deint = self._time_deinterleaver.push(cif_soft)
            if deint is None:
                continue

            subch_soft = _extract_subchannel(deint, svc.start_address, svc.subchannel_size)

            if self._depuncture_index is None:
                continue

            with span("viterbi"):
                depunctured = _eep_depuncture(
                    subch_soft, self._depuncture_index, self._depuncture_len
                )
                decoded = _viterbi_decode(depunctured)

            n_data_bits = len(decoded) - (VITERBI_K - 1)
            decoded = decoded[:n_data_bits]

            if self._prbs_cache is None or len(self._prbs_cache) < n_data_bits:
                self._prbs_cache = _generate_prbs(n_data_bits)
            decoded = decoded ^ self._prbs_cache[:n_data_bits]

            logical_frame = bytes(np.packbits(decoded))

            with span("superframe"):
                sf_result = self._superframe.push(logical_frame)
            if sf_result is None:
                continue

            aus, pad_list, sf_fmt = sf_result
            self._audio_channels = sf_fmt.channels
            self._sf_format = sf_fmt

            # Feed PAD data to decoder
            for pad_entry in pad_list:
                if pad_entry is not None:
                    xpad_data, fpad_data = pad_entry
                    self._pad_decoder.process(xpad_data, fpad_data)

            if self._aac_decoder is None:
                continue

            with span("aac"):
                for au in aus:
                    pcm = self._aac_decoder.decode(au, sf_fmt.core_sample_rate, sf_fmt.channels)
                    if pcm is not None and len(pcm) > 0:
                        frame_pcm_parts.append(pcm)

        if self._aac_decoder is not None:
            self._audio_sample_rate = self._aac_decoder.output_rate
        if frame_pcm_parts and self._aac_decoder is not None:
            combined = np.concatenate(frame_pcm_parts)
            self._audio_batches.append(
                AudioBatch(
                    samples=combined,
                    sample_rate=self._aac_decoder.output_rate,
                    prebuffer_seconds=self.audio_prebuffer_seconds,
                )
            )

    def info(self) -> SignalInfo:
        """Thread-safe: callable from any thread. Reads scalar counters only."""
        quality = None
        quality_label = None
        if self._fibs_decoded > 0:
            quality = self._fibs_crc_ok / self._fibs_decoded
            quality_label = f"FIB CRC {quality * 100:.0f}%"
        description = None
        if self._selected_service is not None and self._selected_service.label:
            description = self._selected_service.label.strip()

        return SignalInfo(
            label="DAB+ Mode I",
            channel_bandwidth=1_536_000,
            modulation="OFDM-DQPSK",
            sample_rate=2_048_000,
            has_audio=True,
            has_text=True,
            message_type="dab",
            quality_label=quality_label,
            quality=quality,
            description=description or None,
        )

    def reset(self) -> None:
        """Clear all decoder state."""
        self._buffer = np.array([], dtype=np.complex64)
        self._fig_state = _FIGParserState()
        self._last_ensemble = None
        self._frames_processed = 0
        self._fibs_decoded = 0
        self._fibs_crc_ok = 0
        self._null_symbols_found = 0
        self._selected_service = None
        self._time_deinterleaver = _TimeDeinterleaver()
        self._depuncture_index = None
        self._depuncture_len = 0
        self._superframe.reset()
        self._aac_decoder = None
        self._audio_batches.clear()
        self._pending_messages.clear()
        self._prbs_cache = None
        self._ofdm_state = OFDMState()
        self._audio_sample_rate = None
        self._audio_channels = None
        self._sf_format = None
        self._pad_decoder.reset()

    @property
    def stats(self) -> DABStats:
        return DABStats(
            frames_processed=self._frames_processed,
            fibs_decoded=self._fibs_decoded,
            fibs_crc_ok=self._fibs_crc_ok,
            null_symbols_found=self._null_symbols_found,
        )
