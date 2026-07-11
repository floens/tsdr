from tsdr.core import storage
from tsdr.core.directory.display import default_sort_key
from tsdr.core.directory.favorites import PUBLIC_DEVICES_FILE, FavoriteDevice, FavoritesStore
from tsdr.core.directory.model import PublicDevice


def test_favorite_fields_are_a_subset_of_public_device() -> None:
    # FavoriteDevice snapshots a slim subset of PublicDevice; a PublicDevice field
    # rename must not silently desync FavoriteDevice.from_device.
    snapshot_fields = set(FavoriteDevice.model_fields) - {"note"}
    assert snapshot_fields <= set(PublicDevice.model_fields)


def _device(device_id: str = "spyserver:9.9.9.9:5555", **overrides) -> PublicDevice:
    base = {
        "source": "spyserver",
        "id": device_id,
        "name": "Test Receiver",
        "host": "9.9.9.9",
        "port": 5555,
        "url": "sdr://9.9.9.9:5555",
        "location": "Testville",
        "freq_min": 0.0,
        "freq_max": 1700000000.0,
        "sample_rate": 768000.0,
        "usable": True,
    }
    base.update(overrides)
    return PublicDevice(**base)


def test_add_and_query() -> None:
    store = FavoritesStore()
    device = _device()
    fav = store.add(device)

    assert fav.id == device.id
    assert store.is_favorite(device.id)
    assert [f.id for f in store.all()] == [device.id]
    assert store.all()[0].host == "9.9.9.9"


def test_from_device_snapshots_caps() -> None:
    fav = FavoritesStore().add(_device())
    assert fav.source == "spyserver"
    assert fav.name == "Test Receiver"
    assert (fav.freq_min, fav.freq_max) == (0.0, 1700000000.0)
    assert fav.sample_rate == 768000.0
    # A directory-only field like `usable` is not persisted.
    assert not hasattr(fav, "usable")


def test_round_trip_persists_across_stores() -> None:
    store = FavoritesStore()
    store.add(_device("spyserver:1.1.1.1:5555", host="1.1.1.1"))
    store.add(_device("kiwisdr:http://k.example:8073", source="kiwisdr"))

    reloaded = FavoritesStore()
    reloaded.load()
    ids = {f.id for f in reloaded.all()}
    assert ids == {"spyserver:1.1.1.1:5555", "kiwisdr:http://k.example:8073"}


def test_remove_persists() -> None:
    store = FavoritesStore()
    store.add(_device())
    assert store.remove("spyserver:9.9.9.9:5555")
    assert not store.remove("spyserver:9.9.9.9:5555")  # already gone

    reloaded = FavoritesStore()
    reloaded.load()
    assert reloaded.all() == []


def test_saved_toml_omits_none() -> None:
    # location=None is an optional field left unset.
    FavoritesStore().add(_device(location=None))
    data = storage.load_toml(PUBLIC_DEVICES_FILE)
    entry = data["device"][0]
    assert "port" in entry
    assert "location" not in entry  # exclude_none drops it rather than writing null
    assert "note" not in entry  # unset note dropped too
    assert all(v is not None for v in entry.values())


def test_flag_toggle_persists() -> None:
    store = FavoritesStore()
    assert not store.is_flagged("spyserver:9.9.9.9:5555")
    assert store.toggle_flag("spyserver:9.9.9.9:5555") is True
    assert store.is_flagged("spyserver:9.9.9.9:5555")

    reloaded = FavoritesStore()
    reloaded.load()
    assert reloaded.is_flagged("spyserver:9.9.9.9:5555")

    assert store.toggle_flag("spyserver:9.9.9.9:5555") is False
    again = FavoritesStore()
    again.load()
    assert not again.is_flagged("spyserver:9.9.9.9:5555")


def test_flag_unflag_methods() -> None:
    store = FavoritesStore()
    store.flag("a")
    assert store.is_flagged("a")
    store.unflag("a")
    assert not store.is_flagged("a")


def test_set_note_round_trips() -> None:
    store = FavoritesStore()
    device = _device()
    store.add(device)
    assert store.set_note(device.id, "night dx")
    assert store.get(device.id).note == "night dx"

    reloaded = FavoritesStore()
    reloaded.load()
    assert reloaded.get(device.id).note == "night dx"


def test_set_note_empty_clears() -> None:
    store = FavoritesStore()
    device = _device()
    store.add(device)
    store.set_note(device.id, "x")
    store.set_note(device.id, "")
    assert store.get(device.id).note is None


def test_set_note_requires_favorite() -> None:
    assert FavoritesStore().set_note("missing", "hi") is False


def test_favorites_sort_to_top() -> None:
    store = FavoritesStore()
    aaa = _device("spyserver:1.1.1.1:5555", name="Aaa", host="1.1.1.1")
    zzz = _device("spyserver:9.9.9.9:5555", name="Zzz", host="9.9.9.9")
    store.add(zzz)  # favorite the one that sorts last alphabetically
    ordered = sorted([aaa, zzz], key=lambda d: (not store.is_favorite(d.id), default_sort_key(d)))
    assert [d.id for d in ordered] == [zzz.id, aaa.id]
