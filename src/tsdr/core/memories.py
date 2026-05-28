from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, ConfigDict, ValidationError

from tsdr.core import storage
from tsdr.core.audio_spec import AudioDemodSpec
from tsdr.core.preferences import save_device
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.tuning import save_previous_tune_state
from tsdr.core.tuning_state import get_tuning_state
from tsdr.radio.registry import DEMODULATORS

logger = logging.getLogger(__name__)

MODE_COLORS: dict[str, str] = {
    "RAW": "#abb2bf",
    "WFM": "#e06c75",
    "NFM": "#61afef",
    "AM": "#e5c07b",
    "USB": "#98c379",
    "LSB": "#c678dd",
    "CW": "#ffaf5f",
    "TETRA": "#56b6c2",
    "DMR": "#d19a66",
    "DAB": "#be5046",
    "FLEX": "#e06c75",
    "ADSB": "#61afef",
}

_DEFAULT_COLOR = "#888888"


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    frequency: int
    name: str
    audio_spec: AudioDemodSpec
    bandwidth: int
    tags: tuple[str, ...] = ()
    color: str | None = None


def memory_color(memory: Memory) -> str:
    if memory.color is not None:
        return memory.color
    return MODE_COLORS.get(memory.audio_spec.mode, _DEFAULT_COLOR)


MEMORIES_FILE = "memories.toml"


class MemoryStore:
    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}

    def add(
        self,
        frequency: int,
        name: str,
        spec: AudioDemodSpec,
        bandwidth: int,
        tags: tuple[str, ...] = (),
        color: str | None = None,
    ) -> Memory:
        memory_id = uuid.uuid4().hex[:8]
        memory = Memory(
            id=memory_id,
            frequency=frequency,
            name=name,
            audio_spec=spec,
            bandwidth=bandwidth,
            tags=tags,
            color=color,
        )
        self._memories[memory_id] = memory
        self.save()
        return memory

    def remove(self, memory_id: str) -> bool:
        if memory_id not in self._memories:
            return False
        del self._memories[memory_id]
        self.save()
        return True

    def rename(self, memory_id: str, new_name: str) -> Memory | None:
        old = self._memories.get(memory_id)
        if old is None:
            return None
        updated: Memory = old.model_copy(update={"name": new_name})
        self._memories[memory_id] = updated
        self.save()
        return updated

    def get(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def all(self) -> list[Memory]:
        return sorted(self._memories.values(), key=lambda m: m.frequency)

    def find_by_tag(self, tag: str) -> list[Memory]:
        tag_lower = tag.lower()
        return sorted(
            (m for m in self._memories.values() if tag_lower in (t.lower() for t in m.tags)),
            key=lambda m: m.frequency,
        )

    def find_by_name(self, query: str) -> list[Memory]:
        query_lower = query.lower()
        return sorted(
            (m for m in self._memories.values() if query_lower in m.name.lower()),
            key=lambda m: m.frequency,
        )

    def find_by_prefix(self, prefix: str) -> list[Memory]:
        return [m for m in self._memories.values() if m.id.startswith(prefix)]

    def find_nearest(self, frequency: int, max_distance: int) -> Memory | None:
        best: Memory | None = None
        best_dist = max_distance + 1
        for m in self._memories.values():
            dist = abs(m.frequency - frequency)
            if dist <= max_distance and dist < best_dist:
                best = m
                best_dist = dist
        return best

    def tags(self) -> list[str]:
        seen: set[str] = set()
        for m in self._memories.values():
            seen.update(m.tags)
        return sorted(seen)

    def load(self) -> None:
        data = storage.load_toml(MEMORIES_FILE)
        for entry in data.get("memory", []):
            try:
                memory = Memory.model_validate(entry)
                self._memories[memory.id] = memory
            except ValidationError as e:
                logger.warning("memory_entry_invalid error=%r", e)

    def save(self) -> None:
        data = {"memory": [m.model_dump(exclude_none=True) for m in self.all()]}
        storage.save_toml(MEMORIES_FILE, data)


_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    if _store is None:
        raise RuntimeError("Memory store not initialized")
    return _store


def init_memory_store() -> MemoryStore:
    global _store
    _store = MemoryStore()
    _store.load()
    return _store


def recall_memory(memory: Memory, device_id: str) -> None:
    """Tune to memory frequency and set demod mode."""
    engine = get_engine()
    context = engine.get_device(device_id)

    if context.state != DeviceState.RUNNING:
        raise SDRException(f"Device {device_id} must be running")

    ts = get_tuning_state()
    ts.step = None  # context change → reset step ladder to auto
    save_previous_tune_state(context)

    if memory.audio_spec.mode in DEMODULATORS and memory.audio_spec.mode != context.active_mode:
        engine.set_audio_demod(device_id, memory.audio_spec)

    engine.update_device_config(
        device_id,
        center_frequency=float(memory.frequency),
        channel_bandwidth=int(memory.bandwidth),
    )
    save_device(engine)
