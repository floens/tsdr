from __future__ import annotations

import dataclasses

import pytest

from tsdr.tui.model import ConsoleUIState, DeviceUIState, UIModel
from tsdr.tui.model.store import Mutation, UIStore


def test_initial_model() -> None:
    m = UIModel(db_min=2.0)
    store = UIStore(m)
    assert store.model is m


def test_update_returns_new_model() -> None:
    store = UIStore(UIModel())
    store.update(db_min=2.0)
    assert store.model.db_min == 2.0


def test_update_short_circuits_when_unchanged() -> None:
    store = UIStore(UIModel(db_min=2.0))
    events: list[tuple[UIModel, UIModel]] = []
    store.subscribe(lambda old, new: events.append((old, new)))
    store.update(db_min=2.0)
    assert events == []


def test_update_notifies_subscribers() -> None:
    store = UIStore(UIModel())
    events: list[tuple[UIModel, UIModel]] = []
    store.subscribe(lambda old, new: events.append((old, new)))
    store.update(db_min=4.0)
    assert len(events) == 1
    old, new = events[0]
    assert old.db_min == -100.0
    assert new.db_min == 4.0


def test_multiple_subscribers_fire_in_registration_order() -> None:
    store = UIStore(UIModel())
    order: list[int] = []
    store.subscribe(lambda *_: order.append(1))
    store.subscribe(lambda *_: order.append(2))
    store.subscribe(lambda *_: order.append(3))
    store.update(db_min=2.0)
    assert order == [1, 2, 3]


def test_unsubscribe() -> None:
    store = UIStore(UIModel())
    events: list[object] = []
    unsub = store.subscribe(lambda *_: events.append(None))
    store.update(db_min=2.0)
    unsub()
    store.update(db_min=3.0)
    assert len(events) == 1


def test_unsubscribe_idempotent() -> None:
    store = UIStore(UIModel())
    unsub = store.subscribe(lambda *_: None)
    unsub()
    unsub()  # second call must not raise


def test_update_console_replaces_subtree() -> None:
    store = UIStore(UIModel())
    store.update_console(autocomplete_visible=True, selected_index=2)
    assert store.model.console.autocomplete_visible is True
    assert store.model.console.selected_index == 2
    assert store.model.console.autocomplete_items == ()  # unchanged


def test_set_devices() -> None:
    store = UIStore(UIModel())
    devices = (DeviceUIState(device_id="rtl0"), DeviceUIState(device_id="hackrf"))
    store.set_devices(devices)
    assert store.model.devices == devices


def test_update_device_modifies_single() -> None:
    store = UIStore(
        UIModel(
            devices=(
                DeviceUIState(device_id="a"),
                DeviceUIState(device_id="b"),
            )
        )
    )
    store.update_device("b", active_decoder_kind="rds")
    assert store.model.devices[0].active_decoder_kind is None
    assert store.model.devices[1].active_decoder_kind == "rds"


def test_update_device_noop_for_unknown_id() -> None:
    store = UIStore(UIModel(devices=(DeviceUIState(device_id="a"),)))
    events: list[object] = []
    store.subscribe(lambda *_: events.append(None))
    store.update_device("ghost", active_decoder_kind="rds")
    assert events == []


def test_upsert_device_adds() -> None:
    store = UIStore(UIModel())
    store.upsert_device("rtl0", has_audio_pipeline=True)
    assert store.model.devices == (DeviceUIState(device_id="rtl0", has_audio_pipeline=True),)


def test_upsert_device_updates() -> None:
    store = UIStore(UIModel(devices=(DeviceUIState(device_id="rtl0"),)))
    store.upsert_device("rtl0", active_decoder_kind="dab")
    assert store.model.devices == (DeviceUIState(device_id="rtl0", active_decoder_kind="dab"),)


def test_remove_device() -> None:
    store = UIStore(
        UIModel(
            devices=(DeviceUIState(device_id="a"), DeviceUIState(device_id="b")),
            focused_device_id="b",
        )
    )
    store.remove_device("a")
    assert store.model.devices == (DeviceUIState(device_id="b"),)
    assert store.model.focused_device_id == "b"


def test_remove_focused_device_clears_focus() -> None:
    store = UIStore(UIModel(devices=(DeviceUIState(device_id="a"),), focused_device_id="a"))
    store.remove_device("a")
    assert store.model.devices == ()
    assert store.model.focused_device_id is None


def test_remove_device_noop_for_unknown_id() -> None:
    store = UIStore(UIModel(devices=(DeviceUIState(device_id="a"),)))
    events: list[object] = []
    store.subscribe(lambda *_: events.append(None))
    store.remove_device("ghost")
    assert events == []


def test_mutation_log_records_changes() -> None:
    store = UIStore(UIModel())
    store.update(db_min=2.0)
    store.update_console(autocomplete_visible=True)
    log = store.recent_mutations()
    assert [m.op for m in log] == ["update", "update_console"]
    assert log[0].args == {"db_min": 2.0}
    assert log[1].args == {"autocomplete_visible": True}


def test_mutation_log_capped() -> None:
    store = UIStore(UIModel())
    for i in range(250):
        store.update(db_min=float(i + 2))  # 2..251 (avoid initial 1.0)
    log = store.recent_mutations()
    assert len(log) == 200
    # Most-recent at the end of the deque
    assert log[-1].args == {"db_min": 251.0}


def test_mutation_log_skips_unchanged() -> None:
    store = UIStore(UIModel(db_min=1.0))
    store.update(db_min=1.0)
    assert store.recent_mutations() == ()


def test_mutation_is_frozen() -> None:
    m = Mutation(op="x", args={}, ts=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.op = "y"  # type: ignore[misc]


def test_subscriber_exception_does_not_stop_others(caplog) -> None:
    """A flaky subscriber must not block others — EventRouter, EngineSync, and
    PrefsSync all subscribe, and we don't want one bug taking down the others."""
    store = UIStore(UIModel())
    fired: list[int] = []

    def bad(*_):
        raise RuntimeError("boom")

    def good(*_):
        fired.append(1)

    store.subscribe(bad)
    store.subscribe(good)
    store.update(db_min=2.0)
    assert fired == [1]
    assert any("ui_store_subscriber_error" in r.message for r in caplog.records)


def test_console_subtree_equality_short_circuit() -> None:
    store = UIStore(UIModel(console=ConsoleUIState(autocomplete_visible=True)))
    events: list[object] = []
    store.subscribe(lambda *_: events.append(None))
    store.update_console(autocomplete_visible=True)
    assert events == []
