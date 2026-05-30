"""ctypes binding for the RNNoise neural denoiser.

RNNoise is an optional dependency: we depend on the ``pyrnnoise`` wheel solely
to deliver a prebuilt ``librnnoise`` for the current platform, then load that
bundled binary directly with ctypes. We never ``import pyrnnoise`` — its package
``__init__`` pulls heavy deps (audiolab/matplotlib) that the ``denoise`` extra
strips via a uv metadata override (see pyproject.toml). The binary is located
through the distribution's file list instead.

If pyrnnoise (and thus the binary) is absent, ``rnnoise_available()`` returns
False and callers stay in passthrough — mirrors the fail-soft loader in
``tsdr.devices.soapy``.

RNNoise processes mono frames of 480 samples at 48 kHz and expects float samples
in int16 range (±32768); scaling is the caller's job (see ``AudioDenoiser``).
"""

import ctypes
import importlib.metadata
import logging
import platform

import numpy as np

logger = logging.getLogger(__name__)

FRAME_SIZE = 480


def _locate_lib() -> str | None:
    """Return the absolute path to the librnnoise binary bundled by pyrnnoise."""
    if platform.system() == "Windows":
        wanted = "rnnoise.dll"
    elif platform.system() == "Darwin":
        wanted = "librnnoise.dylib"
    else:
        wanted = "librnnoise.so"

    try:
        files = importlib.metadata.files("pyrnnoise")
    except importlib.metadata.PackageNotFoundError:
        return None
    if files is None:
        return None

    for path in files:
        if path.name == wanted:
            return str(path.locate())
    return None


def _load() -> ctypes.CDLL | None:
    lib_path = _locate_lib()
    if lib_path is None:
        return None
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as exc:
        logger.warning("rnnoise_load_failed path=%s error=%r", lib_path, exc)
        return None

    lib.rnnoise_create.restype = ctypes.c_void_p
    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    lib.rnnoise_get_frame_size.restype = ctypes.c_int
    lib.rnnoise_process_frame.restype = ctypes.c_float
    lib.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    logger.debug("rnnoise_loaded path=%s frame_size=%d", lib_path, lib.rnnoise_get_frame_size())
    return lib


_LIB: ctypes.CDLL | None = _load()


def rnnoise_available() -> bool:
    """True when the RNNoise binary is installed and loadable."""
    return _LIB is not None


class RNNoiseState:
    """One RNNoise denoising context. Per-instance state — not shared across threads."""

    def __init__(self) -> None:
        if _LIB is None:
            raise RuntimeError("rnnoise binary not available")
        self._lib: ctypes.CDLL = _LIB
        self._state: int | None = self._lib.rnnoise_create(None)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Denoise one 480-sample mono frame (float32, int16 range). Returns a new array."""
        if self._state is None:
            raise RuntimeError("rnnoise state already closed")
        x = np.ascontiguousarray(frame, dtype=np.float32)
        out = np.empty(FRAME_SIZE, dtype=np.float32)
        self._lib.rnnoise_process_frame(
            self._state,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def close(self) -> None:
        if self._state is not None:
            self._lib.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        self.close()
