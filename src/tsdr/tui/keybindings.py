"""Declarative keyboard shortcut table.

Single source of truth for app-level shortcuts: ``tui/keyboard.py`` dispatches
from ``BINDINGS`` and the ``keys`` command prints it. ``PLACEHOLDER`` sits here
too, hand-written, so the console caption is edited next to the keys it names.
This module imports nothing from ``tsdr``, so widgets, commands, and tests can
pull it in without cycles.

Dispatch lives in ``KeyboardMixin._run_action``, an exhaustive ``match`` over
``Action``. Mypy flags any member without an arm via ``assert_never``, and
``build_lookup`` rejects any member missing from ``BINDINGS``.

Parameterized key families (digit panel hotkeys, ctrl/alt+digit) are dispatched
by residual matchers in ``keyboard.py``. They appear here as documentation-only
rows: ``action`` is ``None`` and ``keys`` holds a display label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from itertools import groupby

from textual.keys import format_key


class Context(Enum):
    GLOBAL = "global"
    CONSOLE = "console"


class Action(StrEnum):
    """Stable action ids; the future user-override layer remaps action -> keys."""

    TUNE_DOWN = "tune.down"
    TUNE_UP = "tune.up"
    TUNE_DOWN_COARSE = "tune.down-coarse"
    TUNE_UP_COARSE = "tune.up-coarse"
    TUNE_DOWN_FINE = "tune.down-fine"
    TUNE_UP_FINE = "tune.up-fine"
    TUNE_TARGET_PREV = "tune.target-prev"
    TUNE_TARGET_NEXT = "tune.target-next"
    TUNE_STEP_NEXT = "tune.step-next"
    TUNE_STEP_PREV = "tune.step-prev"
    TUNE_MODE_TOGGLE = "tune.mode-toggle"
    BANDWIDTH_UP = "bandwidth.up"
    BANDWIDTH_DOWN = "bandwidth.down"
    BANDWIDTH_UP_FINE = "bandwidth.up-fine"
    BANDWIDTH_DOWN_FINE = "bandwidth.down-fine"
    GAIN_DOWN = "gain.down"
    GAIN_UP = "gain.up"
    GAIN_AGC = "gain.agc"
    SQUELCH_DOWN = "squelch.down"
    SQUELCH_UP = "squelch.up"
    SQUELCH_OFF = "squelch.off"
    VOLUME_UP = "volume.up"
    VOLUME_DOWN = "volume.down"
    AUDIO_DENOISE = "audio.denoise"
    DEVICE_TOGGLE = "device.toggle"
    DEMOD_CHORD = "demod.chord"
    DISPLAY_IMAGE = "display.image"
    DISPLAY_SPAN_IN = "display.span-in"
    DISPLAY_SPAN_OUT = "display.span-out"
    DISPLAY_FLOOR_UP = "display.floor-up"
    DISPLAY_FLOOR_DOWN = "display.floor-down"
    DISPLAY_CEILING_UP = "display.ceiling-up"
    DISPLAY_CEILING_DOWN = "display.ceiling-down"
    DISPLAY_CENTER_ON_DIAL = "display.center-on-dial"
    MEMORY_ADD = "memory.add"
    MEMORY_EDIT = "memory.edit"
    MEMORY_REMOVE = "memory.remove"
    CONSOLE_FOCUS = "console.focus"
    CONSOLE_LEAVE = "console.leave"
    CONSOLE_TAB = "console.tab"
    CONSOLE_TAB_BACK = "console.tab-back"
    CONSOLE_ESCAPE = "console.escape"
    CONSOLE_HISTORY_PREV = "console.history-prev"
    CONSOLE_HISTORY_NEXT = "console.history-next"
    CONSOLE_SEARCH = "console.search"
    CONSOLE_CLEAR = "console.clear"


@dataclass(frozen=True)
class Binding:
    action: Action | None  # None = documented only; `keys` then holds a display label
    keys: tuple[str, ...]
    context: Context
    category: str
    description: str


TUNING = "Tuning"
RECEPTION = "Reception"
DISPLAY = "Display"
MEMORIES = "Memories & panels"
CONSOLE = "Console"
CONSOLE_FOCUSED = "Console (input focused)"

# Second keystroke of the `d` chord; drives keyboard.py dispatch and the description below.
DEMOD_CHORD_MODES = {
    "w": "WFM",
    "n": "NFM",
    "a": "AM",
    "u": "USB",
    "l": "LSB",
    "c": "CW",
    "o": "OFF",
}


def demod_chord_prompt() -> str:
    """Rich markup prompting for the chord's second key."""
    keys = " ".join(f"[b]{k}[/]{m[1:].lower()}" for k, m in DEMOD_CHORD_MODES.items())
    return f"Demod: {keys}"


def _chord_description() -> str:
    modes = ", ".join(f"{k} {m}" for k, m in DEMOD_CHORD_MODES.items())
    return f"Switch demodulator: then {modes}"


def _section(
    context: Context,
    category: str,
    *rows: tuple[Action | None, tuple[str, ...], str],
) -> tuple[Binding, ...]:
    """Build one contiguous run of bindings sharing a context and category."""
    return tuple(Binding(a, k, context, category, d) for a, k, d in rows)


BINDINGS: tuple[Binding, ...] = (
    *_section(
        Context.GLOBAL,
        TUNING,
        (Action.TUNE_DOWN, ("left",), "Tune down one step"),
        (Action.TUNE_UP, ("right",), "Tune up one step"),
        (Action.TUNE_DOWN_COARSE, ("shift+left",), "Tune down one coarse step"),
        (Action.TUNE_UP_COARSE, ("shift+right",), "Tune up one coarse step"),
        (Action.TUNE_DOWN_FINE, ("alt+left", "ctrl+left"), "Tune down one fine step"),
        (Action.TUNE_UP_FINE, ("alt+right", "ctrl+right"), "Tune up one fine step"),
        (
            Action.TUNE_TARGET_PREV,
            ("left_square_bracket",),
            "Jump to the previous tuning target (memory or band)",
        ),
        (
            Action.TUNE_TARGET_NEXT,
            ("right_square_bracket",),
            "Jump to the next tuning target (memory or band)",
        ),
        (Action.TUNE_STEP_NEXT, ("s",), "Cycle the tuning step size forward"),
        (Action.TUNE_STEP_PREV, ("S",), "Cycle the tuning step size backward"),
        (Action.TUNE_MODE_TOGGLE, ("c",), "Toggle center and free tuning"),
        (None, ("Ctrl+1-9",), "Recall a band-stack slot"),
        (None, ("Ctrl+0",), "Swap the A/B band stack"),
    ),
    *_section(
        Context.GLOBAL,
        RECEPTION,
        (Action.BANDWIDTH_UP, ("up",), "Increase channel bandwidth"),
        (Action.BANDWIDTH_DOWN, ("down",), "Decrease channel bandwidth"),
        (Action.BANDWIDTH_UP_FINE, ("alt+up", "ctrl+up"), "Increase channel bandwidth finely"),
        (
            Action.BANDWIDTH_DOWN_FINE,
            ("alt+down", "ctrl+down"),
            "Decrease channel bandwidth finely",
        ),
        (Action.GAIN_DOWN, ("g",), "Decrease RF gain"),
        (Action.GAIN_UP, ("G",), "Increase RF gain"),
        (Action.GAIN_AGC, ("ctrl+g",), "Toggle AGC"),
        (Action.SQUELCH_DOWN, ("u",), "Lower the squelch threshold"),
        (Action.SQUELCH_UP, ("U",), "Raise the squelch threshold"),
        (Action.SQUELCH_OFF, ("ctrl+u",), "Disable squelch"),
        (Action.VOLUME_UP, ("shift+up",), "Increase volume"),
        (Action.VOLUME_DOWN, ("shift+down",), "Decrease volume"),
        (Action.AUDIO_DENOISE, ("n",), "Toggle RNNoise denoising"),
        (Action.DEVICE_TOGGLE, ("space",), "Start / stop the focused device"),
        (Action.DEMOD_CHORD, ("d",), _chord_description()),
    ),
    *_section(
        Context.GLOBAL,
        DISPLAY,
        (Action.DISPLAY_IMAGE, ("i",), "Toggle image mode"),
        (Action.DISPLAY_SPAN_IN, ("k",), "Narrow the spectrum span"),
        (Action.DISPLAY_SPAN_OUT, ("j",), "Widen the spectrum span"),
        (Action.DISPLAY_FLOOR_UP, ("h",), "Raise the noise floor (spectrum dB minimum)"),
        (Action.DISPLAY_FLOOR_DOWN, ("l",), "Lower the noise floor (spectrum dB minimum)"),
        (Action.DISPLAY_CEILING_UP, ("H",), "Raise the ceiling (spectrum dB maximum)"),
        (Action.DISPLAY_CEILING_DOWN, ("L",), "Lower the ceiling (spectrum dB maximum)"),
        (Action.DISPLAY_CENTER_ON_DIAL, ("C",), "Re-center a panned view on the dial"),
    ),
    *_section(
        Context.GLOBAL,
        MEMORIES,
        (Action.MEMORY_ADD, ("m",), "Save a memory at the current frequency"),
        (Action.MEMORY_EDIT, ("M",), "Edit the nearest memory"),
        (Action.MEMORY_REMOVE, ("ctrl+m",), "Delete the nearest memory (press y to confirm)"),
        (None, ("1-9",), "Toggle the dockable panel mapped to that key"),
        (None, ("Alt+1-9",), "Move the mapped panel to the next edge (left, bottom, right)"),
    ),
    *_section(
        Context.GLOBAL,
        CONSOLE,
        (Action.CONSOLE_FOCUS, ("grave_accent",), "Focus the command console"),
    ),
    *_section(
        Context.CONSOLE,
        CONSOLE_FOCUSED,
        (Action.CONSOLE_LEAVE, ("grave_accent",), "Return focus to the spectrum"),
        (Action.CONSOLE_TAB, ("tab",), "Open or cycle the autocomplete menu"),
        (Action.CONSOLE_TAB_BACK, ("shift+tab",), "Cycle the autocomplete menu backward"),
        (
            Action.CONSOLE_ESCAPE,
            ("escape",),
            "Dismiss autocomplete, or return focus to the spectrum",
        ),
        (Action.CONSOLE_HISTORY_PREV, ("up", "ctrl+p"), "Previous history entry"),
        (Action.CONSOLE_HISTORY_NEXT, ("down", "ctrl+n"), "Next history entry"),
        (Action.CONSOLE_SEARCH, ("ctrl+r",), "Search history"),
        (Action.CONSOLE_CLEAR, ("ctrl+l",), "Clear the console"),
    ),
)

# Hand-curated typography, not derivable from BINDINGS. Re-check when those keys change.
PLACEHOLDER = (
    " ` console  space run  ←→ tune  ↑↓ bw  d demod  gG gain"
    "  ⇧↕ vol  1-9 panel  mM^m mem  kj span  hl/HL db  i image"
)


def build_lookup() -> dict[tuple[Context, str], Action]:
    """Expand key synonyms into a dispatch map; reject duplicates and gaps."""
    actions = [b.action for b in BINDINGS if b.action is not None]
    if len(set(actions)) != len(actions):
        raise ValueError("duplicate action in bindings")
    missing = set(Action) - set(actions)
    if missing:
        raise ValueError(f"actions missing from bindings: {sorted(m.value for m in missing)}")
    runs = [category for category, _ in groupby(b.category for b in BINDINGS)]
    if len(runs) != len(set(runs)):
        raise ValueError("bindings of one category must be contiguous")
    lookup: dict[tuple[Context, str], Action] = {}
    for binding in BINDINGS:
        if binding.action is None:
            continue
        for key in binding.keys:
            entry = (binding.context, key)
            if entry in lookup:
                raise ValueError(
                    f"duplicate binding for {key!r} in {binding.context.value}: "
                    f"{lookup[entry]} and {binding.action}"
                )
            lookup[entry] = binding.action
    return lookup


def _display_key(key: str) -> str:
    *mods, base = key.split("+")
    # s and S are distinct bindings, so bare letters keep their case.
    base = base.upper() if mods and len(base) == 1 else format_key(base)
    return "+".join([*(m.capitalize() for m in mods), base])


def format_keys(binding: Binding) -> str:
    """Human-readable key label for help output."""
    if binding.action is None:
        return " / ".join(binding.keys)
    return " / ".join(_display_key(k) for k in binding.keys)
