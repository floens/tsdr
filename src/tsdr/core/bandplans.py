from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tsdr.core import storage

logger = logging.getLogger(__name__)


BAND_TYPE_COLORS: dict[str, str] = {
    "amateur": "#61afef",
    "broadcast": "#e06c75",
    "aviation": "#98c379",
    "satellite": "#c678dd",
    "military": "#e5c07b",
    "marine": "#56b6c2",
}

_DEFAULT_COLOR = "#888888"


def band_type_color(band_type: str) -> str:
    return BAND_TYPE_COLORS.get(band_type, _DEFAULT_COLOR)


def contrast_fg(bg_hex: str) -> str:
    r = int(bg_hex[1:3], 16)
    g = int(bg_hex[3:5], 16)
    b = int(bg_hex[5:7], 16)
    yiq = (r * 299 + g * 587 + b * 114) / 1000
    return "#000000" if yiq >= 128 else "#ffffff"


class Band(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> Band:
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be < end ({self.end})")
        return self


class Bandplan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    country_name: str
    country_code: str
    author_name: str
    author_url: str
    bands: tuple[Band, ...]
    filename: str


BANDPLAN_SUBDIR = "bandplans"


def find_band_at(bandplan: Bandplan, frequency: int) -> Band | None:
    """Return the narrowest band containing `frequency`, or None.

    Mirrors the renderer's "narrower wins" rule so the auto-name matches
    the most specific band shown on screen.
    """
    matching = [b for b in bandplan.bands if b.start <= frequency < b.end]
    if not matching:
        return None
    return min(matching, key=lambda b: b.end - b.start)


class BandplanStore:
    def __init__(self) -> None:
        self._plans: dict[str, Bandplan] = {}
        self._active: Bandplan | None = None

    def load_all(self) -> None:
        self._plans.clear()
        for path in storage.list_files(BANDPLAN_SUBDIR, ".json"):
            plan = self._load_file(path.name)
            if plan is not None:
                self._plans[path.name] = plan

    def _load_file(self, filename: str) -> Bandplan | None:
        data = storage.load_json(f"{BANDPLAN_SUBDIR}/{filename}")
        if data is None:
            return None
        if not isinstance(data, dict):
            logger.warning(
                "bandplan_invalid_top_level filename=%s got=%s",
                filename,
                type(data).__name__,
            )
            return None

        raw_bands = data.get("bands", [])
        if not isinstance(raw_bands, list):
            logger.warning(
                "bandplan_bands_not_array filename=%s got=%s",
                filename,
                type(raw_bands).__name__,
            )
            return None

        valid_bands: list[Band] = []
        for i, entry in enumerate(raw_bands):
            if not isinstance(entry, dict):
                logger.warning("bandplan_band_not_object filename=%s index=%d", filename, i)
                continue
            try:
                valid_bands.append(Band.model_validate(entry))
            except ValidationError as e:
                logger.warning(
                    "bandplan_band_invalid filename=%s index=%d errors=%r",
                    filename,
                    i,
                    e.errors(),
                )

        if raw_bands and not valid_bands:
            logger.warning(
                "bandplan_all_bands_invalid filename=%s count=%d", filename, len(raw_bands)
            )
            return None

        if not raw_bands:
            logger.info("bandplan_empty filename=%s", filename)

        stem = filename.rsplit(".", 1)[0]
        outer = {k: v for k, v in data.items() if k != "bands"}
        outer["bands"] = tuple(valid_bands)
        outer["filename"] = stem

        try:
            plan: Bandplan = Bandplan.model_validate(outer)
        except ValidationError as e:
            logger.warning("bandplan_top_level_invalid filename=%s errors=%r", filename, e.errors())
            return None
        return plan

    def filenames(self) -> list[str]:
        return sorted(self._plans.keys())

    def plans(self) -> list[Bandplan]:
        return sorted(self._plans.values(), key=lambda p: p.filename)

    def get(self, filename: str) -> Bandplan | None:
        return self._plans.get(filename)

    @property
    def active(self) -> Bandplan | None:
        return self._active

    def set_active(self, filename: str) -> Bandplan | None:
        plan = self._plans.get(filename)
        if plan is None:
            logger.warning("bandplan_activate_failed filename=%s reason=not_loaded", filename)
            return None
        self._active = plan
        return plan

    def clear(self) -> None:
        self._active = None


_store: BandplanStore | None = None


def get_bandplan_store() -> BandplanStore:
    if _store is None:
        raise RuntimeError("Bandplan store not initialized")
    return _store


def init_bandplan_store() -> BandplanStore:
    global _store
    _store = BandplanStore()
    _store.load_all()
    return _store
