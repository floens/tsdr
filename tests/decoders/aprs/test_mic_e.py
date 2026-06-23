"""Mic-E decode tests: a real direwolf vector as ground truth, plus a synthetic
encoder for round-trip coverage of S/W hemispheres and the longitude offsets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tsdr.radio.decoders.aprs import ax25, mic_e, payload


def test_direwolf_vector() -> None:
    # WB2OSZ>TRSW1R: N 42 37.12, W 071 20.83, 0 kt, course 151, symbol "/[".
    d = mic_e.decode("TRSW1R", "`c0ol!O[/>=")
    assert d is not None
    assert d["latitude"] == pytest.approx(42.6187, abs=1e-3)
    assert d["longitude"] == pytest.approx(-71.3472, abs=1e-3)
    assert d["speed"] == pytest.approx(0.0)
    assert d["course"] == 151
    assert d["symbol"] == "/["


def _encode(lat: float, lon: float, speed: int, course: int, symbol: str = "/>") -> tuple[str, str]:
    south, west = lat < 0, lon < 0
    lat, lon = abs(lat), abs(lon)
    dlat = int(lat)
    mlat = round((lat - dlat) * 6000)  # minutes * 100
    digs = [dlat // 10, dlat % 10, mlat // 1000, (mlat // 100) % 10, (mlat // 10) % 10, mlat % 10]

    def ch(d: int, high: bool) -> str:
        return chr((ord("P") if high else ord("0")) + d)

    dlon = int(lon)
    offset = dlon >= 100
    dest = (
        ch(digs[0], False)
        + ch(digs[1], False)
        + ch(digs[2], False)
        + ch(digs[3], not south)  # N=high
        + ch(digs[4], offset)
        + ch(digs[5], west)
    )

    if offset and dlon <= 109:
        b0 = (dlon - 100) + 108
    elif offset:
        b0 = (dlon - 110) + 38
    else:
        b0 = (dlon - 10) + 38
    mlon = (lon - dlon) * 60
    mi = int(round(mlon - 0.5))
    b1 = (mi - 10) + 38 if mi >= 10 else mi + 88
    b2 = round((mlon - mi) * 100) + 28
    sc0 = speed // 10 + 28
    sc1 = (speed % 10) * 10 + course // 100 + 28
    sc2 = course % 100 + 28
    info = "`" + "".join(chr(b) for b in (b0, b1, b2, sc0, sc1, sc2)) + symbol[1] + symbol[0]
    return dest, info


@pytest.mark.parametrize(
    "lat,lon,speed,course",
    [
        (48.1167, 11.55, 42, 90),  # N / E
        (-33.95, 151.2083, 20, 270),  # S / E (Sydney)
        (42.6187, -71.3472, 55, 151),  # N / W
        (-23.5, -46.6, 5, 360),  # S / W (Sao Paulo), course 360 -> 0
    ],
)
def test_roundtrip(lat: float, lon: float, speed: int, course: int) -> None:
    dest, info = _encode(lat, lon, speed, course)
    d = mic_e.decode(dest, info)
    assert d is not None
    assert d["latitude"] == pytest.approx(lat, abs=2e-3)
    assert d["longitude"] == pytest.approx(lon, abs=2e-3)
    assert d["speed"] == pytest.approx(speed)
    assert d["course"] == (0 if course == 360 else course)


def test_payload_end_to_end(synth: SimpleNamespace) -> None:
    dest, info = _encode(48.1167, 11.55, 42, 90)
    frame = ax25.parse_frame(synth.build_ax25("DL1ABC-9", dest, (), info))
    assert frame is not None
    p = payload.parse(frame)
    assert p.info_type == "mic_e"
    assert p.latitude == pytest.approx(48.1167, abs=2e-3)
    assert p.course == 90
