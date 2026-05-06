from dataclasses import dataclass


@dataclass(frozen=True)
class DABServiceInfo:
    """A single DAB service as displayed in the UI."""

    service_id: int
    label: str
    is_audio: bool
    subchannel_id: int | None = None
    protection_level: int | None = None
    subchannel_size: int | None = None


@dataclass(frozen=True)
class DABSlide:
    """A MOT slideshow image."""

    data: bytes
    content_type: str  # "image/jpeg" or "image/png"
    content_name: str = ""
    category_title: str = ""


@dataclass(frozen=True)
class DABStats:
    """Raw decoder counters."""

    frames_processed: int
    fibs_decoded: int
    fibs_crc_ok: int
    null_symbols_found: int


@dataclass(frozen=True)
class DABData:
    """Snapshot of DAB ensemble state for UI consumption."""

    ensemble_id: int
    ensemble_label: str
    services: tuple[DABServiceInfo, ...]
    selected_service_id: int | None
    fib_crc_rate: float
    frames_processed: int = 0
    freq_offset_hz: float = 0.0
    audio_sample_rate: float | None = None
    audio_channels: int | None = None
    core_sample_rate: int | None = None
    sbr: bool | None = None
    ps: bool | None = None
    dynamic_label: str | None = None
    slide: DABSlide | None = None
