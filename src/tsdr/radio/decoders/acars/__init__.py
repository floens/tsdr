"""ACARS (Aircraft Communications Addressing and Reporting System) decoder.

VHF ACARS: 2400 baud MSK on an AM voice channel, carrying ARINC-618 text blocks.
Layering: physical (`MSKDemod` on the AM envelope) -> link
(`frame` sync + odd parity + CRC-16, `fec` syndrome repair) -> the streaming
`ACARSDecoder` orchestrator.
"""

from tsdr.radio.decoders.acars.decoder import ACARSDecoder

__all__ = ["ACARSDecoder"]
