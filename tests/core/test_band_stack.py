from __future__ import annotations

from tsdr.core.band_stack import (
    BAND_DEFAULTS,
    REGISTERS_PER_BAND,
    BandRegister,
    BandStackStore,
)
from tsdr.core.demod_spec import DemodSpec


def _spec(mode: str) -> DemodSpec:
    return DemodSpec(mode=mode)


def test_seed_defaults() -> None:
    store = BandStackStore()
    store._seed_defaults()
    stacks = store.all()
    assert len(stacks) == len(BAND_DEFAULTS)
    assert [s.band.key for s in stacks] == sorted(b.key for b in BAND_DEFAULTS)
    # All registers start empty.
    for stack in stacks:
        assert stack.registers == ()
        assert stack.current_idx == 0


def test_set_register_persists() -> None:
    store = BandStackStore()
    store._seed_defaults()
    reg = BandRegister(slot=0, frequency=14_200_000, audio_spec=_spec("USB"), bandwidth=3_000)
    store.set_register(5, 0, reg)
    got = store.get_register(5, 0)
    assert got is not None
    assert got.frequency == 14_200_000
    assert got.audio_spec.mode == "USB"


def test_save_load_roundtrip() -> None:
    store = BandStackStore()
    store._seed_defaults()
    reg = BandRegister(slot=1, frequency=145_500_000, audio_spec=_spec("NFM"), bandwidth=12_500)
    store.set_register(8, 1, reg)
    store.set_current_idx(8, 1)

    fresh = BandStackStore()
    fresh.load()
    got = fresh.get_register(8, 1)
    assert got is not None
    assert got.frequency == 145_500_000
    assert fresh.get_by_key(8).current_idx == 1


def test_set_register_replaces_slot() -> None:
    store = BandStackStore()
    store._seed_defaults()
    r1 = BandRegister(slot=0, frequency=14_100_000, audio_spec=_spec("USB"), bandwidth=3_000)
    r2 = BandRegister(slot=0, frequency=14_200_000, audio_spec=_spec("USB"), bandwidth=2_400)
    store.set_register(5, 0, r1)
    store.set_register(5, 0, r2)
    stack = store.get_by_key(5)
    assert len(stack.registers) == 1
    assert stack.registers[0].frequency == 14_200_000


def test_set_current_idx_bounds() -> None:
    store = BandStackStore()
    store._seed_defaults()
    store.set_current_idx(5, REGISTERS_PER_BAND)  # out of range — no-op
    assert store.get_by_key(5).current_idx == 0
    store.set_current_idx(5, 2)
    assert store.get_by_key(5).current_idx == 2
