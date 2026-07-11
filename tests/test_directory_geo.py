from tsdr.core.directory.display import location_display
from tsdr.core.directory.geo import country_code
from tsdr.core.directory.model import PublicDevice


def test_country_code_known_points() -> None:
    assert country_code(52.52, 13.40) == "DE"  # Berlin
    assert country_code(52.37, 4.90) == "NL"  # Amsterdam
    assert country_code(56.89, 8.52) == "DK"  # Denmark (airspy sample coords)
    assert country_code(40.71, -74.00) == "US"  # New York
    assert country_code(21.31, -157.86) == "US"  # Honolulu (US Hawaii split box)
    assert country_code(1.35, 103.82) == "SG"  # Singapore (dataset omits it)
    assert country_code(22.32, 114.17) == "HK"  # Hong Kong (dataset omits it)


def test_country_code_ocean_is_none() -> None:
    assert country_code(0.0, 0.0) is None


def test_location_display_spyserver_uses_country_code() -> None:
    spy = PublicDevice(source="spyserver", id="s", name="x", host="h", lat=52.52, lon=13.40)
    assert location_display(spy) == "DE"


def test_location_display_kiwisdr_uses_text() -> None:
    kiwi = PublicDevice(
        source="kiwisdr", id="k", name="x", host="h", location="Victoria, Australia"
    )
    assert location_display(kiwi) == "Australia, Victoria"


def test_location_display_falls_back_to_coords_when_unresolved() -> None:
    spy = PublicDevice(source="spyserver", id="s", name="x", host="h", lat=0.0, lon=0.0)
    assert location_display(spy) == "0.0, 0.0"
