import logging

import av
import numpy as np

logger = logging.getLogger(__name__)


def _make_adts_header(frame_len: int, sample_rate: int, channels: int, profile: int = 1) -> bytes:
    """Build 7-byte ADTS header for an AAC frame.

    Args:
        frame_len: AAC AU length in bytes (without ADTS header)
        sample_rate: Sample rate (e.g., 48000, 24000)
        channels: Number of channels (1=mono, 2=stereo)
        profile: 0=AAC-LC, 1=HE-AAC (SBR), 2=HE-AACv2 (SBR+PS)
    """
    # Sample rate index (MPEG-4 table)
    sr_table = {
        96000: 0,
        88200: 1,
        64000: 2,
        48000: 3,
        44100: 4,
        32000: 5,
        24000: 6,
        22050: 7,
        16000: 8,
        12000: 9,
        11025: 10,
        8000: 11,
    }
    sri = sr_table.get(sample_rate, 4)  # default 44100

    total_len = frame_len + 7  # ADTS header is 7 bytes (no CRC)

    # For DAB+ HE-AAC: profile=1 in ADTS means AAC-LC (ADTS profile = object_type - 1)
    # The SBR is signaled implicitly by the low sample rate
    adts_profile = 1  # AAC-LC (ADTS uses object_type-1, AAC-LC=2, so profile=1)

    header = bytearray(7)
    header[0] = 0xFF
    header[1] = 0xF1  # syncword + MPEG-4 + Layer 0 + no CRC
    header[2] = ((adts_profile << 6) | (sri << 2) | (channels >> 2)) & 0xFF
    header[3] = (((channels & 0x3) << 6) | (total_len >> 11)) & 0xFF
    header[4] = (total_len >> 3) & 0xFF
    header[5] = (((total_len & 0x7) << 5) | 0x1F) & 0xFF
    header[6] = 0xFC  # buffer fullness 0x7FF + 0 frames

    return bytes(header)


class _AACDecoder:
    """Wraps pyav for HE-AAC v2 -> PCM decoding."""

    def __init__(self):
        self._codec = None
        self.output_rate: float = 48000.0
        self._init_codec()

    def _init_codec(self):
        self._codec = av.CodecContext.create("aac", "r")

    def decode(self, au_data: bytes, core_sr: int = 24000, channels: int = 2) -> np.ndarray | None:
        """Decode one AAC AU to float32 PCM.

        Args:
            au_data: Raw AAC AU (without ADTS header, without CRC).
            core_sr: Core AAC sample rate (before SBR upsampling).
            channels: Number of channels.

        Returns float32 array or None on error.
        """
        if not au_data or len(au_data) < 2:
            return None

        adts = _make_adts_header(len(au_data), core_sr, channels)
        frame_data = adts + au_data

        try:
            packet = av.Packet(frame_data)
            assert self._codec is not None
            frames = self._codec.decode(packet)
            if not frames:
                return None

            pcm_parts = []
            for frame in frames:
                # DAB+ uses 960-sample AAC transforms, but ffmpeg's AAC decoder
                # only supports 1024. ADTS cannot signal frameLengthFlag, and
                # ASC with frameLengthFlag=1 breaks SBR in ffmpeg (unfixed
                # since 2012: https://trac.ffmpeg.org/ticket/1407).
                # Workaround: each AU produces 2048 samples (1024×2 with SBR)
                # instead of 1920 (960×2). Report the effective sample rate so
                # the audio worker resamples to correct duration:
                # 48000 × 1024/960 = 51200 -> resampled to 48000 = ratio 15/16.
                self.output_rate = float(frame.sample_rate) * 1024 / 960
                arr = frame.to_ndarray()
                # av returns shape (channels, samples) in planar format
                if arr.ndim == 2:
                    if frame.format.is_planar:
                        arr = arr[0]  # mono downmix: take first channel
                    else:
                        arr = arr[:, 0] if arr.shape[1] > 1 else arr.ravel()
                if arr.dtype == np.int16:
                    arr = arr.astype(np.float32) / 32768.0
                elif arr.dtype == np.int32:
                    arr = arr.astype(np.float32) / 2147483648.0
                elif arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                pcm_parts.append(arr)

            return np.concatenate(pcm_parts) if pcm_parts else None
        except av.error.FFmpegError:
            return None

    def reset(self):
        self._init_codec()
