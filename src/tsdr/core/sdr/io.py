import gzip
from pathlib import Path

import numpy as np
import zstandard as zstd

from tsdr.core.sdr.samples_batch import SampleFormat
from tsdr.devices import EXTENSION_FORMAT_MAP
from tsdr.radio.dsp._kernels import _uint8_iq_to_complex64


def load_iq(path: str | Path) -> np.ndarray:
    """Load IQ samples from a file, returning complex64 numpy array.

    Supports: .cu8, .iq, .cf32, .raw (optionally .zst or .gz compressed).
    """
    path = Path(path)

    # Detect compression and strip to get the format extension
    name = path.name
    if name.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f_in:
            raw = dctx.stream_reader(f_in).read()
        format_ext = Path(name.removesuffix(".zst")).suffix.lower()
    elif name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            raw = f.read()
        format_ext = Path(name.removesuffix(".gz")).suffix.lower()
    else:
        raw = path.read_bytes()
        format_ext = path.suffix.lower()

    fmt = EXTENSION_FORMAT_MAP.get(format_ext)
    if fmt is None:
        raise ValueError(
            f"Unknown IQ format for extension '{format_ext}'. "
            f"Supported: {', '.join(EXTENSION_FORMAT_MAP)}"
        )

    if fmt == SampleFormat.COMPLEX64:
        return np.frombuffer(raw, dtype=np.complex64)

    # UINT8_IQ: single-pass numba kernel -> complex64 (no intermediate arrays)
    uint8_data = np.frombuffer(raw, dtype=np.uint8)
    result: np.ndarray = _uint8_iq_to_complex64(np.ascontiguousarray(uint8_data))
    return result
