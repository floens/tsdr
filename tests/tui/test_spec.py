import dataclasses

import pytest

from tsdr.tui.view.spec import WidgetSpec, pretty


def test_defaults() -> None:
    spec = WidgetSpec(kind="x", key="x")
    assert spec.props == {}
    assert spec.children == ()


def test_frozen() -> None:
    spec = WidgetSpec(kind="x", key="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.kind = "y"  # type: ignore[misc]


def test_equality_includes_children() -> None:
    a = WidgetSpec("a", "a", {"v": 1}, (WidgetSpec("c", "c"),))
    b = WidgetSpec("a", "a", {"v": 1}, (WidgetSpec("c", "c"),))
    c = WidgetSpec("a", "a", {"v": 1}, ())
    assert a == b
    assert a != c


def test_pretty_single_node() -> None:
    s = WidgetSpec("spectrum", "spectrum", {"zoom": 2.0})
    assert pretty(s) == "spectrum#spectrum (zoom=2.0)"


def test_pretty_nested() -> None:
    s = WidgetSpec(
        "root",
        "root",
        {},
        (
            WidgetSpec("tuner", "tuner"),
            WidgetSpec("viz", "viz", {}, (WidgetSpec("spectrum", "spectrum"),)),
        ),
    )
    expected = "root#root\n  tuner#tuner\n  viz#viz\n    spectrum#spectrum"
    assert pretty(s) == expected
