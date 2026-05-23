"""WidgetSpec — a frozen description of a node in the desired widget tree.

The reactive UI pipeline produces a WidgetSpec tree from `derive_tree(UIModel)`;
the Reconciler diffs it against the live widget tree, mounting/removing widgets
and applying reactive-attr props by key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WidgetSpec:
    kind: str
    """Selects the constructor in view.factory.FACTORY (e.g. 'spectrum', 'tuner')."""

    key: str
    """Stable identity across renders. The reconciler matches children by key,
    so reorders preserve widget instances and their state."""

    props: Mapping[str, Any] = field(default_factory=dict)
    """Values to setattr on the widget. Each prop must correspond to a reactive
    attribute on the widget class (Textual handles equality/refresh)."""

    children: tuple[WidgetSpec, ...] = ()


def pretty(spec: WidgetSpec, indent: int = 0) -> str:
    """Indented multi-line representation, useful for the `dump-tree` command."""
    pad = "  " * indent
    props_repr = ""
    if spec.props:
        items = ", ".join(f"{k}={v!r}" for k, v in spec.props.items())
        props_repr = f" ({items})"
    head = f"{pad}{spec.kind}#{spec.key}{props_repr}"
    if not spec.children:
        return head
    child_lines = "\n".join(pretty(c, indent + 1) for c in spec.children)
    return f"{head}\n{child_lines}"
