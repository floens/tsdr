from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tsdr.core.directory.model import PublicDevice, Source
from tsdr.core.directory.sources import ALL_SOURCES, DirectoryError, PublicDeviceSource
from tsdr.core.http import HttpError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """Merged directory devices plus per-source failure messages. A source that
    fails contributes an entry in `errors`, not devices, so one source being down
    doesn't hide the other's receivers."""

    devices: list[PublicDevice]
    errors: dict[Source, str]


_devices: list[PublicDevice] = []
_errors: dict[Source, str] = {}


def _fetch_source(source: PublicDeviceSource) -> tuple[list[PublicDevice], str | None]:
    try:
        return source.fetch(), None
    except (HttpError, DirectoryError) as e:
        logger.warning("directory_fetch_failed source=%s error=%r", source.name, e)
        return [], str(e)


def fetch_all() -> FetchResult:
    """Fetch every source concurrently and merge, dropping duplicate ids and
    collecting per-source failures. Public lists (KiwiSDR especially) can list the
    same receiver twice; ids double as widget option keys, so they must be unique."""
    with ThreadPoolExecutor(max_workers=len(ALL_SOURCES)) as pool:
        outcomes = list(pool.map(_fetch_source, ALL_SOURCES))
    devices: list[PublicDevice] = []
    errors: dict[Source, str] = {}
    seen: set[str] = set()
    for source, (found, error) in zip(ALL_SOURCES, outcomes, strict=True):
        if error is not None:
            errors[source.name] = error
            continue
        logger.info("directory_fetched source=%s count=%d", source.name, len(found))
        for d in found:
            if d.id in seen:
                continue
            seen.add(d.id)
            devices.append(d)
    return FetchResult(devices=devices, errors=errors)


def get_directory() -> FetchResult:
    """Fetch both directories fresh and update the module cache."""
    global _devices, _errors
    result = fetch_all()
    _devices = result.devices
    _errors = result.errors
    return result


def cached() -> list[PublicDevice]:
    """The current cached devices without triggering a fetch (may be empty)."""
    return _devices


def cached_errors() -> dict[Source, str]:
    """Per-source failure messages from the last fetch (empty if all succeeded)."""
    return _errors
