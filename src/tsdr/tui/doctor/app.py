"""Interactive terminal capability diagnostic (the default `tsdr doctor`).

A small self-contained Textual app. Constructed with ``ansi_color=True`` so the
background-passthrough panel faithfully reproduces how the real TSDR app renders
(see ``tui/app.tcss``). Programmatic results are computed *before* the app starts
(the kitty/sync/keyboard probes need the raw tty) and passed in; this screen adds
the visual, eyeball-only checks plus an audible test-tone toggle.

Kitty images live outside the scrolling regions: Textual relocates cached cell
strips on scroll without notifying a (still-mounted) widget that merely moved, so
an out-of-band kitty image cannot track intra-view scrolling - the real app keeps
its visualizations in fixed docks for the same reason. Tab-switch visibility and
content-on-resize are owned by the widgets (``KittyImageWidget.on_hide``/``on_show``
and ``_CheckerImage.on_resize``), not polled by the app.
"""

import logging
import time

import numpy as np
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Resize
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Footer, Label, RichLog, Static, TabbedContent, TabPane

from tsdr.tui.doctor.anim import (
    StripScroller,
    glyph_intensity,
    glyph_row_text,
    spectrum_frame,
    synthetic_row,
)
from tsdr.tui.doctor.checks import (
    CheckResult,
    Status,
    check_pixel_size,
    check_terminal_size,
    check_window_size,
    installed_packages,
    os_details,
)
from tsdr.tui.doctor.export import write_report
from tsdr.tui.doctor.logbuffer import log_lines
from tsdr.tui.doctor.tone import TonePlayer
from tsdr.tui.tty import (
    TTYWindowSpec,
    capabilities,
    resolve_window_spec,
    supports_kitty_images,
    window_spec_from_resize,
)
from tsdr.tui.widgets.kitty_host import KittyHostMixin
from tsdr.tui.widgets.kitty_image import KITTY_TRANSPORT_DESC, KittyImageWidget

_KITTY_TRANSPORT = f"image transport: {KITTY_TRANSPORT_DESC}"

logger = logging.getLogger(__name__)

_STATUS_COLOR = {
    Status.OK: "green",
    Status.WARN: "yellow",
    Status.FAIL: "red",
    Status.UNKNOWN: "dim",
}
_MARKER = {
    Status.OK: " OK ",
    Status.WARN: "WARN",
    Status.FAIL: "FAIL",
    Status.UNKNOWN: "????",
}

# The kitty/size block is named explicitly; the general section is everything
# else in the render group (truecolor, unicode_locale, and any terminal quirks).
_RENDER_KITTY = ("kitty_graphics", "kitty_transports", "window_size", "pixel_size")


def _results_markup(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        color = _STATUS_COLOR[r.status]
        lines.append(f"[{color}]\\[{_MARKER[r.status]}][/] {r.name:<20} {r.summary}")
    return "\n".join(lines)


def _ramp_markup(width: int, color_fn) -> str:
    cells = []
    for i in range(width):
        r, g, b = color_fn(i / (width - 1))
        cells.append(f"[on rgb({r},{g},{b})] [/]")
    return "".join(cells)


def _checker_frame(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    xs = np.arange(w) // 8
    ys = np.arange(h) // 8
    checker = (xs[None, :] ^ ys[:, None]) & 1
    img[:, :, 0] = checker * 255
    img[:, :, 1] = (1 - checker) * 255
    img[:, :, 2] = checker * 128
    return img


def _os_markup() -> str:
    return "\n".join(f"{k:<14} {v}" for k, v in os_details().items())


def _packages_markup() -> str:
    return "\n".join(f"{name:<30} {ver}" for name, ver in installed_packages().items())


def _grayscale(t: float) -> tuple[int, int, int]:
    v = round(t * 255)
    return v, v, v


def _hue(t: float) -> tuple[int, int, int]:
    h = t * 6
    x = round(255 * (1 - abs(h % 2 - 1)))
    if h < 1:
        return 255, x, 0
    if h < 2:
        return x, 255, 0
    if h < 3:
        return 0, 255, x
    if h < 4:
        return 0, x, 255
    if h < 5:
        return x, 0, 255
    return 255, 0, x


class _CheckerImage(KittyImageWidget):
    """Static checkerboard that owns its content: it redraws to fill itself
    whenever it is (re)sized. Visibility on tab switches is handled by the base
    ``KittyImageWidget``."""

    def on_resize(self, event: Resize) -> None:
        super().on_resize(event)
        if self.size.area == 0:  # hidden tab: skip (avoids a zero-region warning)
            return
        w, h = self.available_pixel_size
        if w > 0 and h > 0:
            self.update_image("doctor", _checker_frame(w, h))


class _GlyphWaterfall(Widget):
    """Text-only waterfall (no kitty images) to gauge 60 Hz glyph throughput.

    Each row is rendered to a ``Strip`` exactly once on arrival and cached;
    ``render_line`` just returns the cached strips. So the per-frame cost is one
    new row regardless of how long it has been running - unchanged rows are never
    re-rendered (a plain ``Static`` re-renders the whole visible block each
    frame, which is what makes a naive glyph waterfall degrade as it fills)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._strips: list[Strip] = []

    def push_row(self, row: Text) -> None:
        h = self.size.height
        w = self.size.width
        if h <= 0 or w <= 0:
            return
        options = self.app.console.options.update_width(w)
        lines = self.app.console.render_lines(row, options, pad=True)
        self._strips.insert(0, Strip(lines[0] if lines else [], w))
        del self._strips[h:]
        self.refresh()

    def render_line(self, y: int) -> Strip:
        if y < len(self._strips):
            return self._strips[y].adjust_cell_length(self.size.width)
        return Strip.blank(self.size.width)


class DoctorApp(KittyHostMixin, App[None]):
    CSS = """
    Screen { layout: vertical; }
    TabbedContent { height: 1fr; }
    TabPane { overflow: hidden; }
    VerticalScroll { padding: 0 1; }
    .section-title { color: $accent; text-style: bold; margin-top: 1; }
    .caption { color: $text-muted; padding: 0 1; }
    #bg-panels { height: 5; }
    .bg-default { width: 1fr; border: round $foreground; content-align: center middle; }
    .bg-default { background: ansi_default; }
    .bg-explicit { width: 1fr; border: round $foreground; content-align: center middle;
                   background: rgb(40,40,60); }
    #doctor-image { height: 10; width: 1fr; }
    #spectrum-image { height: 8; width: 1fr; }
    #wf-image { height: 1fr; width: 1fr; }
    #glyph-wf { height: 1fr; width: 1fr; }
    #doctor-log { height: 1fr; width: 1fr; background: $surface; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("a", "toggle_tone", "Toggle test tone"),
        ("e", "export", "Export JSON"),
        ("c", "copy_logs", "Copy logs"),
    ]

    def __init__(self, results: list[CheckResult]) -> None:
        super().__init__(ansi_color=True)
        self._results = results
        self._tone = TonePlayer()
        self._scroller: StripScroller | None = None
        self._anim_t0 = 0.0
        self._last_image_tick = 0.0
        self._fps_image = 0.0
        self._last_text_tick = 0.0
        self._fps_text = 0.0
        # Freshest in-band cell-size source: the latest Resize event that carried
        # pixels. Stays None on terminals (e.g. Windows) whose resize has no pixels.
        self._resize_spec: TTYWindowSpec | None = None
        self._log_cursor = 0
        # Mount the kitty demo widgets only when images can render: the probe must
        # have confirmed kitty graphics + the shared-memory transport (t=s). Same
        # predicate the kitty_transports check uses, so gate and verdict can't drift.
        self._kitty_ok = supports_kitty_images(capabilities())

    def _markup_for(self, *groups: str) -> str:
        return _results_markup([r for r in self._results if r.group in groups])

    def _markup_for_names(self, *names: str) -> str:
        return _results_markup([r for r in self._results if r.name in names])

    def _markup_render_general(self) -> str:
        # Render-group results that aren't the kitty/size block — includes
        # terminal quirks (e.g. apple_terminal_limited), which carry no fixed name.
        return _results_markup(
            [r for r in self._results if r.group == "render" and r.name not in _RENDER_KITTY]
        )

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Rendering", id="tab-render"), VerticalScroll():
                yield Static("General", classes="section-title")
                yield Static(self._markup_render_general(), id="res-render")

                yield Static("Protocol", classes="section-title")
                yield Static(self._markup_for("protocol", "session"), id="res-protocol")

                yield Static("Kitty, window & cell size", classes="section-title")
                yield Static(self._markup_for_names(*_RENDER_KITTY), id="res-kitty")

                yield Static("Background passthrough", classes="section-title")
                yield Static(
                    "Left should match your terminal background; right a distinct color.",
                    classes="caption",
                )
                with Horizontal(id="bg-panels"):
                    yield Static("ansi_default", classes="bg-default")
                    yield Static("explicit bg", classes="bg-explicit")

                yield Static("Text styles", classes="section-title")
                yield Static(
                    "[b]bold[/b]  normal  [dim]dim[/dim]  [i]italic[/i]  "
                    "[u]underline[/u]  [s]strike[/s]  [reverse] reverse [/reverse]"
                )

                yield Static("Color fidelity (smooth, un-banded?)", classes="section-title")
                yield Static(_ramp_markup(64, _grayscale))
                yield Static(_ramp_markup(64, _hue))

                yield Static("Unicode glyphs", classes="section-title")
                yield Static("box   ┌─┬─┐ │ ├─┼─┤ │ └─┴─┘")
                yield Static("block ▁▂▃▄▅▆▇█  ▏▎▍▌▋▊▉█")
                yield Static("braille ⠁⠂⠄⡀⢀⠿⣿   emoji 📻🎶📡")
                yield Static(
                    "cjk   [reverse]日本語[/reverse]  (3 wide chars = 6 cells under ruler)"
                )
                yield Static("ruler 123456")

            # The kitty checkerboard needs its own tab: an out-of-band kitty image
            # can't sit in a scroll, and a dedicated tab keeps hide/show-on-switch clean.
            if self._kitty_ok:
                with TabPane("Image", id="tab-image"):
                    yield Static(
                        f"Kitty image (sharp checkerboard?) - {_KITTY_TRANSPORT}",
                        classes="section-title",
                    )
                    yield _CheckerImage(id="doctor-image")

            with TabPane("System", id="tab-system"), VerticalScroll():
                yield Static(self._markup_for("runtime", "deps", "system"))
                yield Static("Audio test tone", classes="section-title")
                yield Label("♪ test tone: idle  (press 'a' to play)", id="tone-status")
                yield Static("Operating system", classes="section-title")
                yield Static(_os_markup())
                yield Static("Installed packages", classes="section-title")
                yield Static(_packages_markup())

            if self._kitty_ok:
                with TabPane("Live (image)", id="tab-live-image"):
                    yield Static(
                        f"Kitty graphics at 60 Hz - watch for smooth motion. {_KITTY_TRANSPORT}",
                        classes="section-title",
                    )
                    yield Label("warming up…", id="fps-image")
                    yield Static(
                        "Spectrum (kitty full-frame; moving ball = motion check)",
                        classes="section-title",
                    )
                    yield KittyImageWidget(id="spectrum-image")
                    yield Static(
                        "Waterfall (kitty strips; colored left edge marks each strip)",
                        classes="section-title",
                    )
                    yield KittyImageWidget(id="wf-image")

            with TabPane("Live (text)", id="tab-live-text"):
                yield Static(
                    "Glyphs only (no kitty) at 60 Hz - text-render throughput.",
                    classes="section-title",
                )
                yield Label("warming up…", id="fps-text")
                yield Static("Waterfall (colored block glyphs)", classes="section-title")
                yield _GlyphWaterfall(id="glyph-wf")

            with TabPane("Logs", id="tab-logs"):
                yield Static(
                    "Everything the doctor does, including the startup terminal probe.",
                    classes="caption",
                )
                yield RichLog(
                    id="doctor-log", max_lines=10000, wrap=False, highlight=False, markup=False
                )
        yield Footer()

    def on_mount(self) -> None:
        if self._kitty_ok:
            self._scroller = StripScroller(
                self.query_one("#wf-image", KittyImageWidget), mark_strips=True
            )
        self._anim_t0 = time.monotonic()
        self._last_image_tick = self._anim_t0
        self._last_text_tick = self._anim_t0
        self.set_interval(1 / 60, self._animate_live)
        # Drain the in-memory log buffer into the Logs tab (pre-app probe output is
        # already there; later records arrive live).
        self.set_interval(0.5, self._drain_logs)
        self._drain_logs()
        logger.info("doctor_app_mounted")

    def _drain_logs(self) -> None:
        try:
            log_widget = self.query_one("#doctor-log", RichLog)
        except NoMatches:
            return
        lines = log_lines()
        for line in lines[self._log_cursor :]:
            log_widget.write(line)
        self._log_cursor = len(lines)

    def on_resize(self, event: Resize) -> None:
        # Geometry checks are probed once before the app starts and go stale when
        # the terminal resizes; refresh them here so they track the live size.
        #
        # In-band resize pixels (DEC mode 2048) are the freshest cell-size source
        # where the terminal reports them; remember the latest. Textual's Windows
        # driver emits cell dimensions only (no pixels), so there this stays None
        # and the checks fall back to the live ioctl or the CSI startup snapshot.
        pixels = (event.pixel_size.width, event.pixel_size.height) if event.pixel_size else None
        resize_spec = window_spec_from_resize(event.size.width, event.size.height, pixels)
        if resize_spec is not None:
            self._resize_spec = resize_spec
        spec = resolve_window_spec(self._resize_spec)
        for fresh in (check_window_size(spec), check_pixel_size(spec), check_terminal_size()):
            self._results = [fresh if r.name == fresh.name else r for r in self._results]
        self._update_results_widget("res-kitty", self._markup_for_names(*_RENDER_KITTY))
        self._update_results_widget("res-protocol", self._markup_for("protocol", "session"))

    def _update_results_widget(self, widget_id: str, markup: str) -> None:
        try:
            self.query_one(f"#{widget_id}", Static).update(markup)
        except NoMatches:
            pass

    @staticmethod
    def _fps_step(last_tick: float, fps: float) -> tuple[float, float]:
        now = time.monotonic()
        dt = now - last_tick
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
        return now, fps

    def _animate_live(self) -> None:
        """Drive whichever Live tab is visible. Each content widget reports zero
        size when its tab is hidden, so only the active tab does work."""
        self._animate_image_frame()
        self._animate_text_frame()

    def _animate_image_frame(self) -> None:
        if self._scroller is None:
            return
        try:
            spectrum = self.query_one("#spectrum-image", KittyImageWidget)
            waterfall = self.query_one("#wf-image", KittyImageWidget)
            fps_label = self.query_one("#fps-image", Label)
        except NoMatches:
            return
        # When the tab is hidden the widgets have zero cell size; bail before
        # reading available_pixel_size, which warns on a zero region.
        if spectrum.size.area == 0 or waterfall.size.area == 0:
            self._last_image_tick = time.monotonic()  # tab hidden: don't accrue dt
            return
        sw, sh = spectrum.available_pixel_size
        ww, wh = waterfall.available_pixel_size
        if min(sw, sh, ww, wh) <= 0:
            self._last_image_tick = time.monotonic()
            return
        t = time.monotonic() - self._anim_t0
        spectrum.update_image("spectrum", spectrum_frame(sw, sh, t))
        self._scroller.push(synthetic_row(ww, t))
        self._scroller.emit(wh)
        self._last_image_tick, self._fps_image = self._fps_step(
            self._last_image_tick, self._fps_image
        )
        fps_label.update(f"target 60 Hz · measured {self._fps_image:.0f} Hz")

    def _animate_text_frame(self) -> None:
        try:
            glyph = self.query_one("#glyph-wf", _GlyphWaterfall)
            fps_label = self.query_one("#fps-text", Label)
        except NoMatches:
            return
        if glyph.size.width <= 0 or glyph.size.height <= 0:
            self._last_text_tick = time.monotonic()  # tab hidden: don't accrue dt
            return
        t = time.monotonic() - self._anim_t0
        glyph.push_row(glyph_row_text(glyph_intensity(glyph.size.width, t)))
        self._last_text_tick, self._fps_text = self._fps_step(self._last_text_tick, self._fps_text)
        fps_label.update(f"target 60 Hz · measured {self._fps_text:.0f} Hz")

    def action_export(self) -> None:
        path = write_report(self._results)
        logger.info("doctor_export path=%s", path)
        self.notify(f"Diagnostics written to {path}", title="Exported")

    def action_copy_logs(self) -> None:
        lines = log_lines()
        self.copy_to_clipboard("\n".join(lines))
        logger.info("doctor_logs_copied lines=%d", len(lines))
        self.notify(f"Copied {len(lines)} log lines to clipboard", title="Logs")

    def action_toggle_tone(self) -> None:
        label = self.query_one("#tone-status", Label)
        playing = self._tone.toggle(on_finish=self._on_tone_finished)
        logger.info("doctor_tone_toggle playing=%s", playing)
        if playing:
            label.update("♪ test tone: playing…  (press 'a' to stop)")
        else:
            label.update("♪ test tone: idle  (press 'a' to play)")

    def _on_tone_finished(self) -> None:
        self.call_from_thread(self._reset_tone_label)

    def _reset_tone_label(self) -> None:
        try:
            self.query_one("#tone-status", Label).update("♪ test tone: idle  (press 'a' to play)")
        except NoMatches:
            pass

    def on_unmount(self) -> None:
        # KittyImageWidget.on_unmount deletes its own images; just stop the tone.
        self._tone.stop()
