from pathlib import Path

import pytest

from tsdr.core.directory import cache, sources
from tsdr.core.directory.model import PublicDevice
from tsdr.core.directory.sources import DirectoryError, KiwiSDRDirectory, SpyServerDirectory
from tsdr.core.http import HttpError

SAMPLES = Path(__file__).parent / "samples" / "directory"


class _DummyClient:
    def __enter__(self) -> _DummyClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _by_id(devices: list[PublicDevice], device_id: str) -> PublicDevice:
    match = next((d for d in devices if d.id == device_id), None)
    assert match is not None, f"{device_id} not in {[d.id for d in devices]}"
    return match


# --- SpyServer -------------------------------------------------------------


def _spyserver_devices() -> list[PublicDevice]:
    return SpyServerDirectory().parse((SAMPLES / "airspy_status.json").read_bytes())


def test_spyserver_skips_malformed_entry() -> None:
    devices = _spyserver_devices()
    # 2 real + 4 valid crafted; the entry without streamingHost is dropped.
    assert len(devices) == 6
    assert all(d.source == "spyserver" for d in devices)


def test_spyserver_field_extraction() -> None:
    d = _by_id(_spyserver_devices(), "spyserver:9.9.9.9:5555")
    assert (d.host, d.port) == ("9.9.9.9", 5555)
    assert d.url == "sdr://9.9.9.9:5555"
    assert d.sample_rate == 768000.0
    assert d.bandwidth == 384000.0
    assert (d.freq_min, d.freq_max) == (0.0, 1700000000.0)
    assert (d.lat, d.lon) == (52.0, 5.0)
    assert d.name == "Test Usable"


def test_spyserver_verdicts() -> None:
    devices = _spyserver_devices()
    usable = _by_id(devices, "spyserver:9.9.9.9:5555")
    assert usable.usable and usable.usable_reason == ""

    full = _by_id(devices, "spyserver:1.1.1.1:5556")
    assert not full.usable and full.usable_reason == "full 1/1"

    offline = _by_id(devices, "spyserver:2.2.2.2:5557")
    assert not offline.usable and offline.usable_reason == "offline"

    view_only = _by_id(devices, "spyserver:3.3.3.3:5558")
    assert view_only.usable and view_only.usable_reason == "view-only"


def test_spyserver_tolerates_unknown_fields() -> None:
    # The view-only entry carries a made-up "someFutureFieldTheServerAdded" key.
    d = _by_id(_spyserver_devices(), "spyserver:3.3.3.3:5558")
    assert d.usable_reason == "view-only"


def test_spyserver_bad_json_raises() -> None:
    with pytest.raises(DirectoryError):
        SpyServerDirectory().parse(b"not json at all")
    with pytest.raises(DirectoryError):
        SpyServerDirectory().parse(b'{"unexpected": true}')


def test_spyserver_all_entries_invalid_raises() -> None:
    # A non-empty servers list that yields no valid device is a parse failure, not
    # a silent empty result.
    with pytest.raises(DirectoryError):
        SpyServerDirectory().parse(b'{"servers": [{"no": "host"}]}')


# --- KiwiSDR ---------------------------------------------------------------


def _kiwi_devices() -> list[PublicDevice]:
    return KiwiSDRDirectory().parse((SAMPLES / "kiwi_public.html").read_bytes())


def test_kiwisdr_skips_entry_without_href() -> None:
    devices = _kiwi_devices()
    # 2 real + 4 valid crafted; the block with no <a href> is dropped.
    assert len(devices) == 6
    assert all(d.source == "kiwisdr" for d in devices)


def test_kiwisdr_field_extraction() -> None:
    d = _by_id(_kiwi_devices(), "kiwisdr:http://kiwi-usable.example.com:8073")
    assert (d.host, d.port) == ("kiwi-usable.example.com", 8073)
    assert (d.users, d.users_max) == (1, 4)
    assert d.snr == 30
    assert d.channels == 4
    assert d.device_hw == "KiwiSDR 2 v1.900"  # badge glyphs stripped
    assert d.grid == "JO22aa"
    assert d.location == "Testville, Testland"
    assert (d.freq_min, d.freq_max) == (0.0, 30000000.0)
    assert (d.lat, d.lon) == (52.10, 5.20)


def test_kiwisdr_verdicts() -> None:
    devices = _kiwi_devices()
    usable = _by_id(devices, "kiwisdr:http://kiwi-usable.example.com:8073")
    assert usable.usable and usable.usable_reason == ""

    full = _by_id(devices, "kiwisdr:http://kiwi-full.example.com:8073")
    assert not full.usable and full.usable_reason == "full 4/4"

    offline = _by_id(devices, "kiwisdr:http://kiwi-offline.example.com:8073")
    assert not offline.usable and offline.usable_reason == "offline"

    noant = _by_id(devices, "kiwisdr:http://kiwi-noant.example.com:8073")
    assert not noant.usable and noant.usable_reason == "no antenna"


def test_kiwisdr_tolerates_unknown_fields() -> None:
    # The usable entry carries a made-up "some_future_field" comment.
    d = _by_id(_kiwi_devices(), "kiwisdr:http://kiwi-usable.example.com:8073")
    assert d.usable


def test_kiwisdr_empty_html_returns_empty() -> None:
    assert KiwiSDRDirectory().parse(b"<html><body>nothing here</body></html>") == []


_CAPTCHA_STUB = (
    b"<!DOCTYPE HTML><html><head><title>rx.kiwisdr.com</title></head><body>"
    b"<script>ajax.setRequestHeader('x-kiwi-auth','deadbeefdeadbeef1234');</script>"
    b"</body></html>"
)


def _patch_responses(monkeypatch: pytest.MonkeyPatch, responses: list[bytes]) -> list:
    """Make KiwiSDRDirectory._get run offline against a queued response list."""
    it = iter(responses)
    seen: list = []

    def fake_get_capped(client, url, *, headers=None, max_bytes=8_000_000):
        seen.append(headers)
        return next(it)

    monkeypatch.setattr(sources, "make_client", lambda *a, **k: _DummyClient())
    monkeypatch.setattr(sources, "get_capped", fake_get_capped)
    return seen


def test_kiwisdr_direct_when_listing_present(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = (SAMPLES / "kiwi_public.html").read_bytes()
    seen = _patch_responses(monkeypatch, [listing])
    devices = KiwiSDRDirectory().fetch()
    assert len(devices) == 6
    assert len(seen) == 1  # listing already present → no handshake


def test_kiwisdr_unlock_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = (SAMPLES / "kiwi_public.html").read_bytes()
    # step1 captcha stub, step2 unlock (empty), step3 listing
    seen = _patch_responses(monkeypatch, [_CAPTCHA_STUB, b"", listing])
    devices = KiwiSDRDirectory().fetch()
    assert len(devices) == 6
    assert seen[1] == {"x-kiwi-auth": "deadbeefdeadbeef1234"}  # token replayed


def test_kiwisdr_locked_without_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_responses(monkeypatch, [b"<html>locked, no token</html>"])
    with pytest.raises(HttpError):
        KiwiSDRDirectory().fetch()


def test_fetch_all_dedupes_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Public lists can repeat a receiver; ids double as widget option keys, so
    # the merge must keep them unique (first occurrence wins).
    dup_a = PublicDevice(source="kiwisdr", id="kiwisdr:http://h:8073", name="A", host="h")
    dup_b = PublicDevice(source="kiwisdr", id="kiwisdr:http://h:8073", name="A copy", host="h")
    other = PublicDevice(source="spyserver", id="spyserver:x:5555", name="B", host="x")

    class _FakeSource:
        name = "fake"

        def fetch(self) -> list[PublicDevice]:
            return [dup_a, dup_b, other]

    monkeypatch.setattr(cache, "ALL_SOURCES", (_FakeSource(),))
    merged = cache.fetch_all()
    assert [d.id for d in merged.devices] == ["kiwisdr:http://h:8073", "spyserver:x:5555"]
    assert merged.devices[0].name == "A"  # first occurrence kept
    assert merged.errors == {}


def test_fetch_all_collects_per_source_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # One source down must not hide the other's receivers; its message is recorded.
    dev = PublicDevice(source="spyserver", id="spyserver:x:5555", name="B", host="x")

    class _OkSource:
        name = "spyserver"

        def fetch(self) -> list[PublicDevice]:
            return [dev]

    class _FailSource:
        name = "kiwisdr"

        def fetch(self) -> list[PublicDevice]:
            raise HttpError("boom")

    monkeypatch.setattr(cache, "ALL_SOURCES", (_OkSource(), _FailSource()))
    result = cache.fetch_all()
    assert [d.id for d in result.devices] == ["spyserver:x:5555"]
    assert result.errors == {"kiwisdr": "boom"}
