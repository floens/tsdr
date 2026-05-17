"""Band-stack registers — per-band tune memory, up to 3 registers per band."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, ValidationError

from tsdr.core import storage

logger = logging.getLogger(__name__)

BAND_STACK_FILE = "band_stack.toml"
REGISTERS_PER_BAND = 3


class BandRegister(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot: int  # 0..REGISTERS_PER_BAND-1
    frequency: int
    mode: str
    bandwidth: int


class Band(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: int  # 1..9 — numeric keyboard binding
    name: str
    start: int
    end: int


class BandStack(BaseModel):
    model_config = ConfigDict(frozen=True)

    band: Band
    registers: tuple[BandRegister, ...] = ()
    current_idx: int = 0


# Ham-centric defaults. SWL/MW users override by editing band_stack.toml.
BAND_DEFAULTS: tuple[Band, ...] = (
    Band(key=1, name="160m", start=1_800_000, end=2_000_000),
    Band(key=2, name="80m", start=3_500_000, end=4_000_000),
    Band(key=3, name="40m", start=7_000_000, end=7_300_000),
    Band(key=4, name="30/20m", start=10_100_000, end=14_350_000),
    Band(key=5, name="17/15m", start=18_068_000, end=21_450_000),
    Band(key=6, name="12/10m", start=24_890_000, end=29_700_000),
    Band(key=7, name="6m", start=50_000_000, end=54_000_000),
    Band(key=8, name="2m", start=144_000_000, end=148_000_000),
    Band(key=9, name="70cm", start=430_000_000, end=450_000_000),
)


class BandStackStore:
    def __init__(self) -> None:
        self._stacks: dict[int, BandStack] = {}

    def _seed_defaults(self) -> None:
        for band in BAND_DEFAULTS:
            self._stacks[band.key] = BandStack(band=band)

    def load(self) -> None:
        data = storage.load_toml(BAND_STACK_FILE)
        for entry in data.get("bandstack", []):
            try:
                stack = BandStack.model_validate(entry)
                self._stacks[stack.band.key] = stack
            except ValidationError as e:
                logger.warning("Skipping invalid band stack entry: %s", e)
        # Backfill any defaults the user hasn't customized.
        for band in BAND_DEFAULTS:
            self._stacks.setdefault(band.key, BandStack(band=band))

    def save(self) -> None:
        data = {"bandstack": [s.model_dump() for s in self.all()]}
        storage.save_toml(BAND_STACK_FILE, data)

    def all(self) -> list[BandStack]:
        return sorted(self._stacks.values(), key=lambda s: s.band.key)

    def get_by_key(self, key: int) -> BandStack | None:
        return self._stacks.get(key)

    def get_register(self, key: int, idx: int) -> BandRegister | None:
        stack = self._stacks.get(key)
        if stack is None:
            return None
        for reg in stack.registers:
            if reg.slot == idx:
                return reg
        return None

    def update_register(self, key: int, idx: int, reg: BandRegister) -> None:
        """In-memory register update. Caller is responsible for save()."""
        stack = self._stacks.get(key)
        if stack is None:
            return
        if reg.slot != idx:
            reg = reg.model_copy(update={"slot": idx})
        kept = tuple(r for r in stack.registers if r.slot != idx)
        new_regs = tuple(sorted((*kept, reg), key=lambda r: r.slot))
        self._stacks[key] = stack.model_copy(update={"registers": new_regs})

    def set_register(self, key: int, idx: int, reg: BandRegister) -> None:
        self.update_register(key, idx, reg)
        self.save()

    def set_current_idx(self, key: int, idx: int) -> None:
        stack = self._stacks.get(key)
        if stack is None:
            return
        if not 0 <= idx < REGISTERS_PER_BAND:
            return
        self._stacks[key] = stack.model_copy(update={"current_idx": idx})
        self.save()


_store: BandStackStore | None = None

# Suspends the core writeback subscriber so a band recall can rewrite freq/mode/bw
# without clobbering the previous band's register or tripping leave-band detection.
_writeback_suspended = False


def get_band_stack() -> BandStackStore:
    if _store is None:
        raise RuntimeError("Band stack store not initialized")
    return _store


def init_band_stack() -> BandStackStore:
    global _store
    _store = BandStackStore()
    _store.load()
    return _store


@contextlib.contextmanager
def suspended_writeback() -> Iterator[None]:
    global _writeback_suspended
    prev = _writeback_suspended
    _writeback_suspended = True
    try:
        yield
    finally:
        _writeback_suspended = prev


def is_writeback_suspended() -> bool:
    return _writeback_suspended
