from collections.abc import Callable
from functools import partial

from tsdr.core.sdr.datatypes import DemodProfile
from tsdr.radio.demodulators import Demodulator

# mode name -> factory callable (receives sample_rate kwarg, returns Demodulator)
DEMODULATORS: dict[str, Callable[..., Demodulator]] = {}
# Parallel to DEMODULATORS, for class-attribute lookups (e.g. HAS_AUDIO) without instantiating.
DEMODULATOR_CLASSES: dict[str, type[Demodulator]] = {}


def register(mode: str, cls: type[Demodulator], factory: Callable[..., Demodulator]) -> None:
    key = mode.upper()
    DEMODULATORS[key] = factory
    DEMODULATOR_CLASSES[key] = cls


def make_demodulator(
    mode: str,
    sample_rate: float,
    channel_bandwidth: float | None = None,
    fm_deviation_hz: float | None = None,
    sstv_mode: str | None = None,
) -> Demodulator:
    mode = mode.upper()
    if mode not in DEMODULATORS:
        raise ValueError(f"Unknown demodulator mode: {mode}")
    kw: dict = {}
    if channel_bandwidth is not None:
        kw["channel_bandwidth"] = channel_bandwidth
    if mode == "NFM" and fm_deviation_hz is not None:
        kw["deviation"] = fm_deviation_hz
    if mode == "SSTV" and sstv_mode is not None:
        kw["sstv_mode"] = sstv_mode
    return DEMODULATORS[mode](sample_rate=sample_rate, **kw)


def demod_profile(mode: str, channel_bandwidth: float | None = None) -> DemodProfile:
    """Structural profile for `mode` without building the demodulator (desired state)."""
    mode = mode.upper()
    cls = DEMODULATOR_CLASSES.get(mode)
    if cls is None:
        raise ValueError(f"Unknown demodulator mode: {mode}")
    return cls.profile(mode=mode, channel_bandwidth=channel_bandwidth)


# Built-in audio demodulators
from tsdr.radio.demodulators.am import AMDemodulator  # noqa: E402
from tsdr.radio.demodulators.cw import CWDemodulator  # noqa: E402
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator  # noqa: E402
from tsdr.radio.demodulators.ssb import SSBDemodulator  # noqa: E402
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator  # noqa: E402

register(
    "WFM",
    WidebandFMDemodulator,
    lambda sample_rate, **kw: WidebandFMDemodulator(sample_rate=sample_rate, **kw),
)
register(
    "NFM",
    NarrowbandFMDemodulator,
    lambda sample_rate, **kw: NarrowbandFMDemodulator(sample_rate=sample_rate, **kw),
)
register(
    "AM", AMDemodulator, lambda sample_rate, **kw: AMDemodulator(sample_rate=sample_rate, **kw)
)
register(
    "USB",
    SSBDemodulator,
    lambda sample_rate, **kw: SSBDemodulator(mode="USB", sample_rate=sample_rate, **kw),
)
register(
    "LSB",
    SSBDemodulator,
    lambda sample_rate, **kw: SSBDemodulator(mode="LSB", sample_rate=sample_rate, **kw),
)
register(
    "CW", CWDemodulator, lambda sample_rate, **kw: CWDemodulator(sample_rate=sample_rate, **kw)
)

from tsdr.radio.demodulators.sstv import SSTVDemodulator  # noqa: E402

register(
    "SSTV",
    SSTVDemodulator,
    lambda sample_rate, **kw: SSTVDemodulator(sample_rate=sample_rate, **kw),
)

# Protocol decoders
from tsdr.radio.decoders.adsb import ADSBDecoder  # noqa: E402
from tsdr.radio.decoders.dab import DABDecoder  # noqa: E402
from tsdr.radio.decoders.dmr import DMRDecoder  # noqa: E402
from tsdr.radio.decoders.flex import FLEXDecoder  # noqa: E402

# Protocol decoders have fixed, spec-defined bandwidths; they ignore the
# device's channel_bandwidth and any other audio-demod tuning kwargs.
register("ADSB", ADSBDecoder, lambda sample_rate, **_: ADSBDecoder(sample_rate=sample_rate))
register("DAB", DABDecoder, lambda sample_rate, **_: DABDecoder(sample_rate=sample_rate))
register("DMR", DMRDecoder, lambda sample_rate, **_: DMRDecoder(sample_rate=sample_rate))
register("FLEX", FLEXDecoder, lambda sample_rate, **_: FLEXDecoder(sample_rate=sample_rate))

from tsdr.radio.decoders.tetra import TETRADecoder  # noqa: E402

register("TETRA", TETRADecoder, lambda sample_rate, **_: TETRADecoder(sample_rate=sample_rate))

# WSJT-X family demodulators (FT8, FT4). Slot-based 4/8-FSK with LDPC FEC.
from tsdr.radio.demodulators.wsjt import SUPPORTED_MODES as _WSJT_MODES  # noqa: E402
from tsdr.radio.demodulators.wsjt import WSJTDemodulator  # noqa: E402

for _mode in _WSJT_MODES:
    register(_mode, WSJTDemodulator, partial(WSJTDemodulator, mode=_mode))
