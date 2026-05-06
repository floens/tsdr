from collections.abc import Callable

from tsdr.radio.demodulators import Demodulator

# mode name -> factory callable (receives sample_rate kwarg, returns Demodulator)
DEMODULATORS: dict[str, Callable[..., Demodulator]] = {}


def register(mode: str, factory: Callable[..., Demodulator]) -> None:
    DEMODULATORS[mode.upper()] = factory


# Built-in audio demodulators
from tsdr.radio.demodulators.am import AMDemodulator  # noqa: E402
from tsdr.radio.demodulators.cw import CWDemodulator  # noqa: E402
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator  # noqa: E402
from tsdr.radio.demodulators.ssb import SSBDemodulator  # noqa: E402
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator  # noqa: E402

register("WFM", lambda sample_rate, **kw: WidebandFMDemodulator(sample_rate=sample_rate, **kw))
register("NFM", lambda sample_rate, **kw: NarrowbandFMDemodulator(sample_rate=sample_rate, **kw))
register("AM", lambda sample_rate, **kw: AMDemodulator(sample_rate=sample_rate, **kw))
register("USB", lambda sample_rate, **kw: SSBDemodulator(mode="USB", sample_rate=sample_rate, **kw))
register("LSB", lambda sample_rate, **kw: SSBDemodulator(mode="LSB", sample_rate=sample_rate, **kw))
register("CW", lambda sample_rate, **kw: CWDemodulator(sample_rate=sample_rate, **kw))

# Protocol decoders
from tsdr.radio.decoders.adsb import ADSBDecoder  # noqa: E402
from tsdr.radio.decoders.dab import DABDecoder  # noqa: E402
from tsdr.radio.decoders.dmr import DMRDecoder  # noqa: E402
from tsdr.radio.decoders.flex import FLEXDecoder  # noqa: E402

# Protocol decoders have fixed, spec-defined bandwidths; they ignore the
# device's channel_bandwidth and any other audio-demod tuning kwargs.
register("ADSB", lambda sample_rate, **_: ADSBDecoder(sample_rate=sample_rate))
register("DAB", lambda sample_rate, **_: DABDecoder(sample_rate=sample_rate))
register("DMR", lambda sample_rate, **_: DMRDecoder(sample_rate=sample_rate))
register("FLEX", lambda sample_rate, **_: FLEXDecoder(sample_rate=sample_rate))

from tsdr.radio.decoders.tetra import TETRADecoder  # noqa: E402

register("TETRA", lambda sample_rate, **_: TETRADecoder(sample_rate=sample_rate))
