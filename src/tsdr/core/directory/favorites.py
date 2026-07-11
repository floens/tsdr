from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from tsdr.core import storage
from tsdr.core.directory.model import PublicDevice, Source

logger = logging.getLogger(__name__)

PUBLIC_DEVICES_FILE = "public_devices.toml"


class FavoriteDevice(BaseModel):
    """A favorited public receiver, persisted separately from real SDR devices.

    Stores a self-contained snapshot (host/port/url + last-known caps) so the
    favorite still displays when the directory is unreachable.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    source: Source
    name: str
    host: str
    port: int | None = None
    url: str | None = None
    location: str | None = None
    freq_min: float | None = None
    freq_max: float | None = None
    sample_rate: float | None = None
    note: str | None = None

    @classmethod
    def from_device(cls, device: PublicDevice) -> FavoriteDevice:
        return cls(
            id=device.id,
            source=device.source,
            name=device.name,
            host=device.host,
            port=device.port,
            url=device.url,
            location=device.location,
            freq_min=device.freq_min,
            freq_max=device.freq_max,
            sample_rate=device.sample_rate,
        )


class FavoritesStore:
    """Per-directory-device user state: favorites (with an optional note) and
    flagged ("dead", greyed-out on refresh) ids. Both persist to the same TOML."""

    def __init__(self) -> None:
        self._favorites: dict[str, FavoriteDevice] = {}
        self._flagged: set[str] = set()

    def add(self, device: PublicDevice) -> FavoriteDevice:
        fav = FavoriteDevice.from_device(device)
        self._favorites[fav.id] = fav
        self.save()
        return fav

    def remove(self, device_id: str) -> bool:
        if device_id not in self._favorites:
            return False
        del self._favorites[device_id]
        self.save()
        return True

    def is_favorite(self, device_id: str) -> bool:
        return device_id in self._favorites

    def get(self, device_id: str) -> FavoriteDevice | None:
        return self._favorites.get(device_id)

    def set_note(self, device_id: str, note: str | None) -> bool:
        """Attach a note to an existing favorite (empty text clears it). Returns
        False when the device isn't favorited."""
        fav = self._favorites.get(device_id)
        if fav is None:
            return False
        self._favorites[device_id] = fav.model_copy(update={"note": note or None})
        self.save()
        return True

    def is_flagged(self, device_id: str) -> bool:
        return device_id in self._flagged

    def flag(self, device_id: str) -> None:
        self._flagged.add(device_id)
        self.save()

    def unflag(self, device_id: str) -> None:
        self._flagged.discard(device_id)
        self.save()

    def toggle_flag(self, device_id: str) -> bool:
        """Flip the flag and return the new state."""
        if device_id in self._flagged:
            self._flagged.discard(device_id)
            flagged = False
        else:
            self._flagged.add(device_id)
            flagged = True
        self.save()
        return flagged

    def all(self) -> list[FavoriteDevice]:
        return sorted(self._favorites.values(), key=lambda f: (f.source, f.name.lower()))

    def load(self) -> None:
        data = storage.load_toml(PUBLIC_DEVICES_FILE)
        for entry in data.get("device", []):
            try:
                fav = FavoriteDevice.model_validate(entry)
                self._favorites[fav.id] = fav
            except ValidationError as e:
                logger.warning("favorite_device_invalid error=%r", e)
        self._flagged = {fid for fid in data.get("flagged", []) if isinstance(fid, str)}

    def save(self) -> None:
        data: dict[str, Any] = {
            "device": [f.model_dump(exclude_none=True) for f in self.all()],
            "flagged": sorted(self._flagged),
        }
        storage.save_toml(PUBLIC_DEVICES_FILE, data)


_store: FavoritesStore | None = None


def get_favorites_store() -> FavoritesStore:
    if _store is None:
        raise RuntimeError("Favorites store not initialized")
    return _store


def init_favorites_store() -> FavoritesStore:
    global _store
    _store = FavoritesStore()
    _store.load()
    return _store
