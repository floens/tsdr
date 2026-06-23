"""APRS decoder: AX.25 over AFSK1200 (Bell 202) over narrowband FM.

Layering: physical (shared `radio.dsp` FMChannelizer + AFSK1200Demod + MuellerMuller)
→ data-link (`hdlc`, `ax25`, `fec`) → application (`payload`, `mic_e`). The
streaming `APRSDecoder` orchestrator lives in `decoder`.
"""

from tsdr.radio.decoders.aprs.decoder import APRSDecoder

__all__ = ["APRSDecoder"]
