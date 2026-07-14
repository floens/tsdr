from datetime import UTC
from typing import ClassVar, Literal, cast
from zoneinfo import ZoneInfo

from rich.align import Align, AlignMethod
from rich.console import Console, ConsoleOptions
from rich.console import RenderResult as RichRenderResult
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style
from textual.app import ComposeResult, RenderResult
from textual.containers import Horizontal, Vertical
from textual.events import Click, Leave, MouseMove, MouseScrollDown, MouseScrollUp
from textual.message import Message
from textual.reactive import reactive
from textual.renderables.digits import DIGITS, DIGITS3X3, DIGITS3X3_BOLD
from textual.widgets import Digits as DigitsWidget
from textual.widgets import Static

from tsdr.core.band_stack import get_band_stack
from tsdr.core.clock_sync import get_clock_sync_monitor, now
from tsdr.core.events.events import DemodStatusEvent, DeviceStateChangedEvent, StatsUpdateEvent
from tsdr.core.sdr.datatypes import DemodProfile, DemodStatus
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.tracing import span
from tsdr.core.tuning import resolve_auto_step
from tsdr.core.tuning_state import get_tuning_state
from tsdr.core.units import format_hz
from tsdr.tui.commands._format import format_rate
from tsdr.tui.model.store import get_ui_store
from tsdr.tui.widgets.snr_widget import SNRWidget

Half = Literal["top", "bottom"]

_DOT_REPLACEMENTS = str.maketrans({".": "\u2022"})


def _format_signal_info(
    profile: DemodProfile, status: DemodStatus | None, device_sample_rate: float | None
) -> str:
    """Format the demod profile + status for the 3-line tuner widget.

    Layout:
        <label>            (structural, from the profile)
        <modulation>       (structural, from the profile)
        <description, or sample-rate mismatch warning if the decoder requires
        a rate different from the device's: in that case it's producing
        garbage and the description is moot>

    Decoder-specific role badges (e.g. TETRA MCCH/TCH) live inside
    `status.description`; the tuner just renders it as-is.
    """
    lines = [f"[bold]{profile.label}[/bold]", f"[dim]{profile.modulation}[/dim]"]
    if (
        profile.sample_rate is not None
        and device_sample_rate is not None
        and abs(profile.sample_rate - device_sample_rate) > 1.0
    ):
        value, unit = format_rate(profile.sample_rate, precision=3)
        lines.append(f"[red bold]Needs {value} {unit}[/red bold]")
    elif status is not None and status.description:
        lines.append(f"[dim]{status.description}[/dim]")
    return "\n".join(lines)


class _HoverDigitsRenderable:
    """3x3 digit renderable that highlights the top or bottom row of one character."""

    def __init__(
        self,
        text: str,
        style: Style,
        hover: tuple[int, Half] | None,
    ) -> None:
        self._text = text
        self._style = style
        self._hover = hover

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RichRenderResult:
        style = console.get_style(self._style)
        digits = DIGITS3X3_BOLD if style.bold else DIGITS3X3
        highlight = style + Style(reverse=True)

        translated = self._text.translate(_DOT_REPLACEMENTS)
        pieces: list[tuple[str, str, str]] = []
        for ch in translated:
            try:
                position = DIGITS.index(ch) * 3
            except ValueError:
                pieces.append((" ", " ", ch))
            else:
                pieces.append(
                    (
                        digits[position].ljust(3),
                        digits[position + 1].ljust(3),
                        digits[position + 2].ljust(3),
                    )
                )

        hover_idx, hover_half = self._hover if self._hover else (None, None)
        new_line = Segment.line()
        for row_idx in range(3):
            for char_idx, piece in enumerate(pieces):
                is_hover = char_idx == hover_idx and (
                    (hover_half == "top" and row_idx == 0)
                    or (hover_half == "bottom" and row_idx == 2)
                )
                yield Segment(piece[row_idx], highlight if is_hover else style)
            yield new_line

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        width = sum(3 if ch in DIGITS else 1 for ch in self._text)
        return Measurement(width, width)


class HoverableDigits(DigitsWidget):
    """Digits widget with hover highlighting and click/scroll per-digit adjust."""

    ALLOW_SELECT = False

    # Latched the first time we see a mouse event with a sub-cell pointer position.
    # Pixel-precision terminals (iTerm2, WezTerm, Kitty, Alacritty via SGR_PIXELS)
    # let us split the middle row by fractional y; on cell-only terminals it stays
    # False and the middle row remains a dead zone.
    _pixel_mouse: ClassVar[bool] = False

    _hover: reactive[tuple[int, Half] | None] = reactive(None, layout=False)

    class DigitAdjusted(Message):
        """Posted when the user clicks or scrolls on a specific digit."""

        def __init__(self, char_idx: int, direction: int) -> None:
            self.char_idx = char_idx
            self.direction = direction
            super().__init__()

    def render(self) -> RenderResult:
        rich_style = self.rich_style
        if self.text_selection:
            rich_style += self.selection_style
        renderable = _HoverDigitsRenderable(self._value, rich_style, self._hover)
        text_align = self.styles.text_align
        align = "left" if text_align not in {"left", "center", "right"} else text_align
        return Align(renderable, cast(AlignMethod, align), rich_style)

    def _locate_char(self, x: int) -> int | None:
        """Map content-offset x to a digit character index, or None if not on a digit."""
        value = self._value
        display = value.translate(_DOT_REPLACEMENTS)
        col = 0
        for i, ch in enumerate(display):
            w = 3 if ch in DIGITS else 1
            if col + w > x:
                return i if value[i].isdigit() else None
            col += w
        return None

    def _detect_pixel_mouse(
        self, event: MouseMove | Click | MouseScrollUp | MouseScrollDown
    ) -> None:
        if not HoverableDigits._pixel_mouse and (
            event.pointer_y != float(int(event.pointer_y))
            or event.pointer_x != float(int(event.pointer_x))
        ):
            HoverableDigits._pixel_mouse = True

    def _locate(self, event: MouseMove | Click) -> tuple[int, Half] | None:
        """Map a mouse event to (char_idx, half), or None if outside a clickable area."""
        offset = event.get_content_offset(self)
        if offset is None:
            return None
        if offset.y == 0:
            half: Half = "top"
        elif offset.y == 2:
            half = "bottom"
        elif offset.y == 1 and HoverableDigits._pixel_mouse:
            frac_y = event.pointer_y - self.gutter.top
            half = "top" if frac_y < 1.5 else "bottom"
        else:
            return None
        char_idx = self._locate_char(offset.x)
        if char_idx is None:
            return None
        return char_idx, half

    def on_mouse_move(self, event: MouseMove) -> None:
        self._detect_pixel_mouse(event)
        self._hover = self._locate(event)

    def on_leave(self, event: Leave) -> None:
        self._hover = None

    def on_click(self, event: Click) -> None:
        self._detect_pixel_mouse(event)
        # HACK: on pixel-mouse terminals, motion events report fractional
        # pointer_y but click events often arrive cell-aligned, so recomputing
        # from the click at y=1 would disagree with the highlight the user
        # just saw. Fall back to the latched hover state in that case.
        located: tuple[int, Half] | None
        if HoverableDigits._pixel_mouse and self._hover is not None:
            located = self._hover
        else:
            located = self._locate(event)
        if located is None:
            return
        char_idx, half = located
        self.post_message(self.DigitAdjusted(char_idx, 1 if half == "top" else -1))
        event.stop()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._emit_scroll(event, 1)

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._emit_scroll(event, -1)

    def _emit_scroll(self, event: MouseScrollUp | MouseScrollDown, direction: int) -> None:
        self._detect_pixel_mouse(event)
        offset = event.get_content_offset(self)
        if offset is None:
            return
        char_idx = self._locate_char(offset.x)
        if char_idx is None:
            return
        self.post_message(self.DigitAdjusted(char_idx, direction))
        event.stop()


_SYNC_DOT = {
    "synced": "[green]●[/]",
    "drift": "[yellow]●[/]",
    "unsynced": "[red]●[/]",
}


class ClockWidget(Static):
    """Live 3-line clock with configurable display timezone.

    Hardware-transceiver aesthetic:
      * blinking colon (HH<:>MM<:>SS toggles every 500 ms — universal
        radio "alive" indicator)
      * accent color on the changing field (SS bold yellow)
      * SNTP-sync LED next to the time (green/yellow/red), opt-in:
        hidden entirely when probing is disabled

    Line 1  HH:MM:SS [●] TZ
    Line 2  HH:MM:SS UTC
    Line 3  YYYY-MM-DD

    Tick at 4 Hz: enough for a crisp colon blink and second-boundary
    refresh (≤250 ms latency on SS digit changes — well within any
    digital-mode slot tolerance).
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._last_markup: str = ""

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(0.25, self._tick)

    def _tick(self) -> None:
        model = get_ui_store().model
        self.display = model.clock_visible
        if not model.clock_visible:
            return
        tz_name = model.timezone
        local = now(ZoneInfo(tz_name)) if tz_name else now()
        utc = now(UTC)
        tz_short = (local.strftime("%Z") or "LOC")[:4]
        sep = "[b]:[/b]" if local.microsecond < 500_000 else " "

        monitor = get_clock_sync_monitor()
        server = monitor.server
        if server is None:
            dot_section = ""
        else:
            state = monitor.get().state_for(server)
            dot_section = f" {_SYNC_DOT[state]}" if state in _SYNC_DOT else ""

        markup = (
            f"[b]{local:%H}[/b]{sep}[b]{local:%M}[/b]{sep}"
            f"[b yellow]{local:%S}[/b yellow]"
            f"{dot_section} [dim]{tz_short}[/]\n"
            f"[dim]{utc:%H:%M:%S} UTC[/]\n"
            f"[dim]{local:%Y-%m-%d}[/]"
        )
        if markup != self._last_markup:
            self._last_markup = markup
            self.update(markup, layout=False)


class TunerWidget(Vertical):
    """Displays the current tuned frequency with sample rate, gain, and SNR."""

    DEFAULT_CLASSES = "tuner"

    _volume: float = 0.5
    _denoise: bool = False
    _sr_line: str = ""
    _gain_line: str = ""
    _sample_rate: float | None = None
    _demod_profile: DemodProfile | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="tuner-row"):
            with Horizontal(id="tuner-left"):
                yield Static("", id="tuner-mode")
                yield Static("", id="tuner-device")
                yield Static("", id="tuner-state")
            yield HoverableDigits("---.---.--- Hz", id="tuner-frequency")
            with Horizontal(id="tuner-right"):
                yield SNRWidget(id="tuner-meter")
                yield ClockWidget(id="tuner-clock")

    def on_mount(self) -> None:
        self._read_config()
        self._sync_running_class()

    def _sync_running_class(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        running = device is not None and device.state == DeviceState.RUNNING
        locked = (
            running and device is not None and not device.device.capabilities.frequency_controllable
        )
        self.set_class(not running, "stopped")
        self.set_class(locked, "locked")

    def update_running_state(self, event: DeviceStateChangedEvent) -> None:
        engine = get_engine()
        focused = engine.focused_device
        if focused is not None and event.device_id != focused:
            return
        self._sync_running_class()

    def _read_config(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return
        config = device.config

        freq_hz = int(config.tuned_frequency)
        formatted = f"{freq_hz:,} Hz".replace(",", ".").rjust(12, " ")
        self.query_one("#tuner-frequency", HoverableDigits).update(formatted)

        self._sample_rate = config.sample_rate
        sr_value, sr_unit = format_rate(config.sample_rate)
        self._sr_line = f"[bold]{sr_value}[/bold] [dim]{sr_unit}[/dim]"
        caps = device.device.capabilities
        unit = caps.gain_unit
        gain_str = f"{int(config.rf_gain)}" if unit == "index" else f"{config.rf_gain:.1f}"
        if not caps.gain_supported:
            self._gain_line = f"[dim]{gain_str} {unit} locked[/dim]"
        elif config.enable_agc:
            self._gain_line = f"[dim]{gain_str} {unit} AGC[/dim]"
        else:
            self._gain_line = f"[yellow]{gain_str}[/yellow] [dim]{unit}[/dim]"
        if config.bias_tee:
            self._gain_line += " [black on yellow]BT[/black on yellow]"
        self._volume = engine.config.audio_volume
        self._denoise = engine.config.denoise
        self.query_one("#tuner-device", Static).update(
            f"{self._sr_line}\n{self._gain_line}\n{self._vol_line()}"
        )

        profile = device.demod_profile
        self._demod_profile = profile
        status = device.demod_status
        mode_text = _format_signal_info(profile, status, config.sample_rate) if profile else ""
        self.query_one("#tuner-mode", Static).update(mode_text)
        meter = self.query_one("#tuner-meter", SNRWidget)
        if profile is not None and status is not None:
            meter.update_quality(status.quality_label, status.quality)
            meter.update_squelch(status.squelch_threshold_db, status.squelch_open)
        else:
            meter.update_quality(None, None)
            meter.update_squelch(None, None)

        self._render_state_column(device.active_mode, float(config.tuned_frequency))

    def _render_state_column(self, mode: str, freq_hz: float) -> None:
        ts = get_tuning_state()
        if ts.step is None:
            value = resolve_auto_step(mode, freq_hz)
            label = "auto"
        else:
            value = ts.step
            label = "step"
        step_value = format_hz(value, decimals=1, long_suffix=True)
        step_line = f"[bold]{step_value}[/bold] [dim]{label}[/dim]"

        band_line = "[dim]—[/dim]"
        reg_line = ""
        if ts.current_band_key is not None:
            stack = get_band_stack().get_by_key(ts.current_band_key)
            if stack is not None:
                band_line = f"[bold]{stack.band.name}[/bold] [dim]band[/dim]"
                reg_line = f"[bold]{stack.current_idx + 1}/3[/bold] [dim]reg[/dim]"

        self.query_one("#tuner-state", Static).update(f"{step_line}\n{band_line}\n{reg_line}")

    def _vol_line(self) -> str:
        vol = f"VOL {int(self._volume * 100)}%"
        return f"[green]{vol}[/green]" if self._denoise else vol

    def update_config(self) -> None:
        self._read_config()

    def update_stats(self, event: StatsUpdateEvent) -> None:
        self.query_one("#tuner-meter", SNRWidget).update_snr(event.channel_snr)

        clip = event.iq_clip_pct or 0.0
        if clip > 0.1:
            clip_prefix = f"[red bold]CLIP {clip:.0f}%[/red bold] "
        else:
            clip_prefix = ""
        self.query_one("#tuner-device", Static).update(
            f"{self._sr_line}\n{clip_prefix}{self._gain_line}\n{self._vol_line()}"
        )

    def update_status(self, event: DemodStatusEvent) -> None:
        profile = self._demod_profile
        status = event.demod_status
        mode_text = _format_signal_info(profile, status, self._sample_rate) if profile else ""
        self.query_one("#tuner-mode", Static).update(mode_text)
        meter = self.query_one("#tuner-meter", SNRWidget)
        meter.update_quality(status.quality_label, status.quality)
        meter.update_squelch(status.squelch_threshold_db, status.squelch_open)

    def on_hoverable_digits_digit_adjusted(self, event: HoverableDigits.DigitAdjusted) -> None:
        with span("ui.tuner_scroll"):
            value = self.query_one("#tuner-frequency", HoverableDigits).value
            digits_right = sum(1 for c in value[event.char_idx + 1 :] if c.isdigit())
            place_value = 10**digits_right

            engine = get_engine()
            device = engine.get_focused_device()
            if device is None:
                return
            new_freq = device.config.tuned_frequency + event.direction * place_value
            freq_range = device.device.capabilities.frequency_range
            if freq_range is not None:
                lo, hi = freq_range
                new_freq = max(lo, min(new_freq, hi))
            new_freq = max(new_freq, 1.0)  # bands can start at 0 Hz; 0 is invalid
            if new_freq == device.config.tuned_frequency:
                return
            engine.update_device_config(device.device_id, tuned_frequency=new_freq)
