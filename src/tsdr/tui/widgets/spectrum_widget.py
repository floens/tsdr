import logging
import time
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.reactive import reactive
from textual.strip import Strip
from textual.widget import Widget

from tsdr.core.bandplans import Bandplan, band_type_color, contrast_fg
from tsdr.core.events.events import (
    BandplanChangedEvent,
    FFTUpdateEvent,
    MemoriesChangedEvent,
)
from tsdr.core.memories import Memory, get_memory_store, memory_color, recall_memory
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.spectrum_view import full_view_range, resolve_view, view_range
from tsdr.core.tracing import traced
from tsdr.core.tuning import resolve_auto_step, save_previous_tune_state
from tsdr.core.tuning_state import get_tuning_state
from tsdr.core.units import axis_si_prefix
from tsdr.tui.inline_edit import InlineEditor
from tsdr.tui.widgets.dsp_utils import (
    iir_trace_filter,
    normalize_spectrum,
    project_spectrum,
    render_spectrum_to_buf,
    span_rbw,
    status_strip,
    status_text,
    transient_view_shift,
)
from tsdr.tui.widgets.image_mode_mixin import ImageModeMixin

logger = logging.getLogger(__name__)


# Braille lookup: each cell holds 2 cols × 4 rows of dots, indexed by the
# 8-bit dot pattern. Bit layout:
#   col 0 col 1
#   0x01  0x08   (dot row 0, top)
#   0x02  0x10   (dot row 1)
#   0x04  0x20   (dot row 2)
#   0x40  0x80   (dot row 3, bottom)
_BRAILLE_BASE = 0x2800
_BRAILLE_CHARS = tuple(chr(_BRAILLE_BASE | i) for i in range(256))
_BRAILLE_LEFT_BITS = np.array([0x01, 0x02, 0x04, 0x40], dtype=np.uint8)
_BRAILLE_RIGHT_BITS = np.array([0x08, 0x10, 0x20, 0x80], dtype=np.uint8)

_STYLE_NONE = Style()
_STYLE_DIM = Style(dim=True)
_STYLE_GREEN = Style(color="green")
_STYLE_BLUE = Style(color="blue")

# Image mode colors (RGBA)
_TRACE_COLOR = np.array([0x40, 0xA0, 0xFF, 0xFF], dtype=np.uint8)  # bright blue
_FILL_COLOR = np.array([0x20, 0x50, 0x80, 0x60], dtype=np.uint8)  # semi-transparent blue
_BW_BOX_COLOR = np.array([40, 40, 40, 255], dtype=np.uint8)  # dark grey bandwidth box
_CENTER_LINE_COLOR = np.array([255, 0, 0, 200], dtype=np.uint8)  # red center line


@dataclass(frozen=True, slots=True)
class SpectrumFrame:
    """Pre-computed frame ready for rendering."""

    header: str
    cells: tuple[tuple[int, ...], ...]  # rows of cell indices (0-255)
    width: int
    height: int
    freq_axis_labels: tuple[tuple[int, str], ...]  # (col, label) pairs
    bandwidth_range: tuple[int, int] | None  # (col_low, col_high) or None
    memory_labels: tuple[
        tuple[tuple[int, str, str, str], ...], ...
    ] = ()  # rows of (col, name, color_hex, memory_id)
    bandplan_segments: tuple[
        tuple[int, int, str, str], ...
    ] = ()  # (col_start, col_end_exclusive, label, color_hex)


class SpectrumWidget(ImageModeMixin, Widget):
    """Display power spectrum with blue bars, bandwidth indicator, and frequency axis.

    Uses braille characters for 2×4 resolution per terminal cell.
    Supports Kitty image mode for line plot rendering.

    Reactive props:
      db_min, db_max: float — invalidates frame buffer on change.
      image_mode: bool — switches to kitty image rendering.
    """

    db_min = reactive(-100.0)
    db_max = reactive(-30.0)
    image_mode = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.current_event: FFTUpdateEvent | None = None
        self._channel_bandwidth: float | None = None
        self._tuned_frequency: float = 0.0
        self._capture_center: float = 0.0
        self._provides_spectrum: bool = False
        self._sideband: str | None = None
        self._memories: tuple[Memory, ...] = ()
        self._bandplan: Bandplan | None = None
        self._strips: list[Strip] = []
        self._image_key = "spectrum"
        self._editor = InlineEditor(self)
        self._trace_avg: np.ndarray | None = None
        self._trace_key: tuple[str, float, float] | None = None
        self._trace_event: FFTUpdateEvent | None = None
        self._trace_ts: float = 0.0

    def on_mount(self) -> None:
        self._mount_kitty()
        self._read_config()
        # Render overlays on a blank spectrum so bandplan / memories / freq axis
        # are visible before the first FFT arrives.
        if not self.image_mode:
            self._rebuild_strips()
        self.refresh()

    def on_unmount(self) -> None:
        self._editor.cancel()

    def watch_db_min(self, _db_min: float) -> None:
        self.invalidate_frame_buffer()

    def watch_db_max(self, _db_max: float) -> None:
        self.invalidate_frame_buffer()

    def watch_image_mode(self, enabled: bool) -> None:
        if enabled:
            self._on_image_mode_enabled()
        else:
            self._on_image_mode_disabled()

    def update_spectrum(self, event: FFTUpdateEvent) -> None:
        """Update spectrum display from event."""
        self.current_event = event
        if self.image_mode:
            self._render_spectrum_image(event)
        else:
            self._rebuild_strips()
            self.refresh()

    def _read_config(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return
        profile = device.demod_profile
        self._sideband = profile.sideband if profile else None
        self._tuned_frequency = device.config.tuned_frequency
        self._capture_center = device.config.center_frequency
        self._provides_spectrum = device.device.capabilities.provides_spectrum
        if device.config.channel_bandwidth is not None:
            self._channel_bandwidth = device.config.channel_bandwidth
        else:
            self._channel_bandwidth = profile.channel_bandwidth if profile else None

    def update_config(self) -> None:
        """Config changed - bars shift within visible range, overlays follow new config."""
        self._read_config()
        self._refresh_display()

    def update_memories(self, event: MemoriesChangedEvent) -> None:
        """Update memory labels from event snapshot."""
        self._memories = tuple(event.memories)  # type: ignore[arg-type]
        if self._editor.active and not any(m.id == self._editor.context for m in self._memories):
            self._editor.cancel()
            return
        self._refresh_display()

    def update_bandplan(self, event: BandplanChangedEvent) -> None:
        """Update active bandplan from event."""
        self._bandplan = event.bandplan  # type: ignore[assignment]
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Re-render after a non-FFT state change (config, memories, bandplan)."""
        if self.image_mode:
            if self.current_event is not None:
                self._render_spectrum_image(self.current_event)
            else:
                self.refresh()
        else:
            self._rebuild_strips()
            self.refresh()

    def on_resize(self) -> None:
        if self.image_mode:
            self._kitty.remove_image(self._image_key)
            if self.current_event is not None:
                self._render_spectrum_image(self.current_event)
        elif self.current_event is not None:
            self._rebuild_strips()

    def invalidate_frame_buffer(self) -> None:
        """Invalidate after a dB-window change."""
        self._strips = []
        if self.image_mode and self.current_event is not None:
            self._render_spectrum_image(self.current_event)
        self.refresh()

    # Inline memory editing

    def start_edit(self, memory: Memory) -> None:
        self._editor.start(
            memory.name,
            redraw=self._redraw_edit,
            on_commit=lambda value: self._commit_rename(memory.id, value),
            context=memory.id,
        )

    def _redraw_edit(self) -> None:
        self._rebuild_strips()
        self.refresh()

    def _commit_rename(self, memory_id: str, value: str) -> None:
        store = get_memory_store()
        store.rename(memory_id, value)
        get_engine().event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))

    def render_line(self, y: int) -> Strip:
        if self.image_mode:
            if y == 0:
                return self._status_strip()
            live = self._display_range()
            if live is not None:
                freq_min, freq_max = live
                width = self.size.width
                if y in (1, 2) and self._memories:
                    label_rows = self._compute_memory_labels(width, freq_min, freq_max)
                    row_idx = y - 1
                    if row_idx < len(label_rows) and label_rows[row_idx]:
                        return self._render_memory_label_strip(width, label_rows[row_idx])
                if y == self.size.height - 2:
                    segments = self._compute_bandplan_segments(width, freq_min, freq_max)
                    return self._render_bandplan_strip(width, segments)
                if y == self.size.height - 1:
                    labels = self._compute_freq_labels(width, freq_min, freq_max)
                    return self._render_freq_axis_strip(width, labels)
            # Image-area rows: transparent cells so the Kitty image (drawn at
            # z=-1073741825, under text) shows through. The image buffer's own
            # RGB already fills unpainted pixels with $app-bg.
            return Strip.blank(self.size.width)
        if y < len(self._strips):
            return self._strips[y]
        return Strip.blank(self.size.width, self.rich_style)

    def _filter_trace(self, normalized: np.ndarray, freq_min: float, freq_max: float) -> np.ndarray:
        """Calm device-provided spectrum traces with the Rocky IIR.

        Device-provided frames are single un-averaged FFT snapshots;
        local IQ devices already smooth in the pipeline (FFT window +
        spectrum_averaging EMA), so the filter stays off for them. State
        resets on device/view/shape change and advances once per event, not
        per re-render.
        """
        device = get_engine().get_focused_device()
        if device is None or not device.device.capabilities.provides_spectrum:
            self._trace_avg = None
            self._trace_event = None
            return normalized
        key = (device.device_id, freq_min, freq_max)
        if key != self._trace_key or (
            self._trace_avg is not None and self._trace_avg.shape != normalized.shape
        ):
            self._trace_avg = None
            self._trace_key = key
        if self._trace_avg is None or self.current_event is not self._trace_event:
            now = time.monotonic()
            dt = min(max(now - self._trace_ts, 1.0 / 60.0), 0.5)
            self._trace_avg = iir_trace_filter(self._trace_avg, normalized, dt)
            self._trace_event = self.current_event
            self._trace_ts = now
        return self._trace_avg

    def _status_strip(self) -> Strip:
        span, rbw = span_rbw(self._display_range(), self.current_event)
        return status_strip(self.size.width, span, rbw, self.db_min, self.db_max, self.rich_style)

    # Image mode rendering

    @traced("spectrum_image")
    def _render_spectrum_image(self, event: FFTUpdateEvent) -> None:
        # Same as _rebuild_strips: cursor and view range must come from one
        # config snapshot.
        self._read_config()
        full_w, full_h = self._kitty.full_pixel_size
        occ = self._kitty.occlusion_insets
        w = full_w - occ.left - occ.right
        h = full_h - occ.top - occ.bottom
        if w <= 0 or h <= 0:
            return

        cell_h = self._kitty.cell_height_px
        top_rows = 1
        bottom_rows = 2  # bandplan strip + frequency axis
        plot_h = h - (top_rows + bottom_rows) * cell_h
        if plot_h <= 0:
            return

        live = self._display_range()
        if live is None:
            return
        freq_min, freq_max = live

        e_fmin, e_fmax = self._actual_freq_range(event)
        p_fmin, p_fmax = self._projection_window(live, event)
        line = project_spectrum(event.spectrum, e_fmin, e_fmax, p_fmin, p_fmax, w)
        normalized = normalize_spectrum(line, self.db_min, self.db_max)
        normalized = self._filter_trace(normalized, freq_min, freq_max)

        buf = np.zeros((plot_h, w, 4), dtype=np.uint8)

        y_vals = (plot_h - 1) - (normalized * (plot_h - 1)).astype(np.intp)
        bw_range = self._compute_bandwidth_range(w, freq_min, freq_max)
        bw_low, bw_high = bw_range if bw_range else (-1, -1)
        # Dial cursor; deliberately unclamped, an out-of-view x draws no line.
        span_hz = freq_max - freq_min
        center_x = int((self._tuned_frequency - freq_min) / span_hz * w) if span_hz > 0 else w // 2

        render_spectrum_to_buf(
            buf,
            y_vals,
            _FILL_COLOR,
            _TRACE_COLOR,
            bw_low,
            bw_high,
            _BW_BOX_COLOR,
            center_x,
            _CENTER_LINE_COLOR,
        )

        span = freq_max - freq_min
        if self._memories and span > 0:
            # Pre-compute dashed row indices (3px on, 3px off)
            dash_rows = np.concatenate(
                [np.arange(r, min(r + 3, plot_h)) for r in range(0, plot_h, 6)]
            )
            for m in self._memories:
                if m.frequency < freq_min or m.frequency > freq_max:
                    continue
                mx = int((m.frequency - freq_min) / span * w)
                if 0 <= mx < w:
                    color_hex = memory_color(m)
                    r = int(color_hex[1:3], 16)
                    g = int(color_hex[3:5], 16)
                    b = int(color_hex[5:7], 16)
                    buf[dash_rows, mx] = [r, g, b, 200]

        self._kitty.update_image(self._image_key, buf, x=occ.left, y=occ.top + top_rows * cell_h)
        self.refresh()

    def _on_image_mode_enabled(self) -> None:
        self._strips = []
        self.refresh()
        if self.current_event:
            self._render_spectrum_image(self.current_event)

    def _on_image_mode_disabled(self) -> None:
        self._kitty.remove_image(self._image_key)

    # Text mode rendering

    @traced("spectrum_strips")
    def _rebuild_strips(self) -> None:
        """Pre-compute all Strip objects from current data."""
        # Refresh the dial/bandwidth fields here, not only on ConfigChanged:
        # an FFT-triggered render can land between a config swap and its event,
        # and a stale dial against the fresh view flickers the cursor off-center.
        self._read_config()
        base = self.rich_style
        width = self.size.width
        height = self.size.height

        if width < 10 or height < 6:
            self._strips = [Strip([Segment("Too small", base)])]
            return

        live = self._display_range()
        if live is None:
            self._strips = [Strip([Segment("Spectrum: No device", base)])]
            return

        frame = self._prepare_frame(self.current_event, width, height, live)
        self._strips = self._render_frame(frame)

    @traced("prepare_frame")
    def _prepare_frame(
        self,
        event: FFTUpdateEvent | None,
        width: int,
        height: int,
        live_range: tuple[float, float],
    ) -> SpectrumFrame:
        """Transform spectrum data into frame buffer.

        Overlays always use `live_range` (config-led). Bars shift from the event's
        freq range into `live_range`; gaps are blank.
        """
        bars_height = max(1, height - 4)
        freq_min, freq_max = live_range
        target = width * 2

        if event is None:
            normalized = np.zeros(target, dtype=np.float32)
        else:
            e_fmin, e_fmax = self._actual_freq_range(event)
            p_fmin, p_fmax = self._projection_window(live_range, event)
            spectrum = project_spectrum(event.spectrum, e_fmin, e_fmax, p_fmin, p_fmax, target)
            normalized = normalize_spectrum(spectrum, self.db_min, self.db_max)
            normalized = self._filter_trace(normalized, freq_min, freq_max)

        # Build braille cell grid as a line trace: each dot column lights one
        # row at the normalized value, and consecutive columns are joined by a
        # vertical segment so the line stays connected across steep steps
        # (matches the image-mode trace rendering in `render_spectrum_to_buf`).
        total_dots = bars_height * 4
        y = np.clip(
            np.round((1.0 - normalized) * (total_dots - 1)).astype(np.int32),
            0,
            total_dots - 1,
        )
        y_prev = np.empty_like(y)
        y_prev[0] = y[0]
        y_prev[1:] = y[:-1]
        y_lo = np.minimum(y_prev, y)
        y_hi = np.maximum(y_prev, y)

        rows_idx = np.arange(total_dots)
        lit = (rows_idx[np.newaxis, :] >= y_lo[:, np.newaxis]) & (
            rows_idx[np.newaxis, :] <= y_hi[:, np.newaxis]
        )
        lit_cells = lit.reshape(-1, bars_height, 4)

        # Max per-cell sum is 0x40|0x04|0x02|0x01 = 0x47 (71); uint8 is sufficient
        # and avoids the default uint64 promotion on the reduction.
        left_bits = (
            lit_cells[0::2].astype(np.uint8) * _BRAILLE_LEFT_BITS[np.newaxis, np.newaxis, :]
        ).sum(axis=2, dtype=np.uint8)
        right_bits = (
            lit_cells[1::2].astype(np.uint8) * _BRAILLE_RIGHT_BITS[np.newaxis, np.newaxis, :]
        ).sum(axis=2, dtype=np.uint8)
        cells = (left_bits | right_bits).T

        rows = tuple(map(tuple, cells.tolist()))

        freq_axis_labels = self._compute_freq_labels(width, freq_min, freq_max)
        bandwidth_range = self._compute_bandwidth_range(width, freq_min, freq_max)
        span_hz, rbw_hz = span_rbw(live_range, self.current_event)
        header = status_text(span_hz, rbw_hz, self.db_min, self.db_max)
        memory_labels = self._compute_memory_labels(width, freq_min, freq_max)
        bandplan_segments = self._compute_bandplan_segments(width, freq_min, freq_max)

        return SpectrumFrame(
            header=header,
            cells=tuple(rows),
            width=width,
            height=bars_height,
            freq_axis_labels=freq_axis_labels,
            bandwidth_range=bandwidth_range,
            memory_labels=memory_labels,
            bandplan_segments=bandplan_segments,
        )

    @traced("render_frame")
    def _render_frame(self, frame: SpectrumFrame) -> list[Strip]:
        """Render frame buffer to Strip objects."""
        strips: list[Strip] = []
        width = frame.width
        base = self.rich_style

        # Header strip
        strips.append(Strip([Segment(frame.header.ljust(width)[:width], base)], width))

        for label_row in frame.memory_labels:
            if label_row:
                strips.append(self._render_memory_label_strip(width, label_row))

        # Bandwidth indicator strip
        strips.append(self._render_bandwidth_strip(width, frame.bandwidth_range))

        for row_cells in frame.cells:
            strips.append(self._render_bar_row(row_cells, width))

        # Bandplan row (always present; blank when inactive)
        strips.append(self._render_bandplan_strip(width, frame.bandplan_segments))

        # Frequency axis strip
        strips.append(self._render_freq_axis_strip(width, frame.freq_axis_labels))

        return strips

    def _render_bar_row(self, cells: tuple[int, ...], width: int) -> Strip:
        """Render a single bar row with run-length encoding."""
        segments: list[Segment] = []
        base = self.rich_style
        run_char: str | None = None
        run_style: Style | None = None
        run_len = 0

        for cell_idx in cells:
            char = _BRAILLE_CHARS[cell_idx]
            style = _STYLE_BLUE if cell_idx != 0 else base

            if char == run_char and style is run_style:
                run_len += 1
            else:
                if run_len > 0 and run_char is not None:
                    segments.append(Segment(run_char * run_len, run_style))
                run_char = char
                run_style = style
                run_len = 1

        # Flush last run
        if run_len > 0 and run_char is not None:
            segments.append(Segment(run_char * run_len, run_style))

        # Pad if needed
        num_cols = len(cells)
        if num_cols < width:
            segments.append(Segment(" " * (width - num_cols), base))

        return Strip(segments, width)

    def _render_bandwidth_strip(self, width: int, bandwidth_range: tuple[int, int] | None) -> Strip:
        """Render bandwidth indicator strip."""
        base = self.rich_style
        dim = base + _STYLE_DIM
        if bandwidth_range is None:
            return Strip([Segment("─" * width, dim)], width)

        col_low, col_high = bandwidth_range
        segments: list[Segment] = []
        if col_low > 0:
            segments.append(Segment("─" * col_low, dim))
        if col_high > col_low:
            segments.append(Segment("█" * (col_high - col_low), base + _STYLE_GREEN))
        if col_high < width:
            segments.append(Segment("─" * (width - col_high), dim))

        return Strip(segments, width)

    def _render_bandplan_strip(
        self, width: int, segments: tuple[tuple[int, int, str, str], ...]
    ) -> Strip:
        """Render bandplan row: each band is a solid color block with contrasting
        label text on top (same bg across the whole band, including letters).
        """
        base = self.rich_style
        if not segments:
            return Strip.blank(width, base)

        chars: list[str] = [" "] * width
        styles: list[Style] = [base] * width

        for col_start, col_end, name, color in segments:
            seg_style = base + Style(color=contrast_fg(color), bgcolor=color)
            for c in range(col_start, col_end):
                chars[c] = " "
                styles[c] = seg_style
            seg_w = col_end - col_start
            if name and seg_w >= len(name) + 2:
                label_start = col_start + (seg_w - len(name)) // 2
                for i, ch in enumerate(name):
                    chars[label_start + i] = ch
            elif name and seg_w >= 5:
                truncated = name[: seg_w - 3] + "…"
                label_start = col_start + (seg_w - len(truncated)) // 2
                for i, ch in enumerate(truncated):
                    chars[label_start + i] = ch

        segs: list[Segment] = []
        i = 0
        while i < width:
            j = i + 1
            while j < width and chars[j] == chars[i] and styles[j] is styles[i]:
                j += 1
            segs.append(Segment(chars[i] * (j - i), styles[i]))
            i = j
        return Strip(segs, width)

    def _render_freq_axis_strip(self, width: int, labels: tuple[tuple[int, str], ...]) -> Strip:
        """Render frequency axis strip."""
        base = self.rich_style
        dim = base + _STYLE_DIM
        segments: list[Segment] = []
        pos = 0
        for start, label in labels:
            if start > pos:
                segments.append(Segment("─" * (start - pos), dim))
            segments.append(Segment(label, base))
            pos = start + len(label)
        if pos < width:
            segments.append(Segment("─" * (width - pos), dim))

        return Strip(segments, width)

    def _compute_memory_labels(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[tuple[tuple[int, str, str, str], ...], ...]:
        """Compute visible memory label positions across up to 2 rows."""
        if not self._memories:
            return ()
        span = freq_max - freq_min
        if span <= 0:
            return ()

        editing_id = self._editor.context if self._editor.active else None
        edit_buffer = self._editor.buffer
        rows: list[list[tuple[int, str, str, str]]] = [[], []]
        occupied_ends = [-1, -1]
        for m in sorted(self._memories, key=lambda m: m.frequency):
            if m.frequency < freq_min or m.frequency > freq_max:
                continue
            col = int((m.frequency - freq_min) / span * width)
            col = max(0, min(width - 1, col))
            is_editing = m.id == editing_id
            name = edit_buffer.value if is_editing and edit_buffer is not None else m.name
            label = f"▼{name}"
            if col + len(label) > width:
                label = label[: width - col]
            if not label:
                continue
            color = memory_color(m)
            # +1 for cursor space when editing
            end = col + len(label) + (1 if is_editing else 0)
            # Try row 0, then row 1, skip if both occupied
            for r in range(2):
                if col > occupied_ends[r]:
                    rows[r].append((col, label, color, m.id))
                    occupied_ends[r] = end
                    break

        result = tuple(tuple(row) for row in rows if row)
        return result

    def _render_memory_label_strip(
        self, width: int, labels: tuple[tuple[int, str, str, str], ...]
    ) -> Strip:
        """Render memory labels as a colored strip."""
        base = self.rich_style
        segments: list[Segment] = []
        pos = 0
        edit_buffer = self._editor.buffer
        for col, label, color, memory_id in labels:
            if col > pos:
                segments.append(Segment(" " * (col - pos), base))
            if edit_buffer is not None and memory_id == self._editor.context:
                # Render ▼ prefix + editable name with cursor
                mem_style = base + Style(color=color)
                segments.append(Segment("▼", mem_style))
                edit_segments = edit_buffer.render_segments(mem_style)
                segments.extend(edit_segments)
                pos = col + 1 + sum(len(s.text) for s in edit_segments)
            else:
                segments.append(Segment(label, base + Style(color=color)))
                pos = col + len(label)
        if pos < width:
            segments.append(Segment(" " * (width - pos), base))
        return Strip(segments, width)

    # Shared helpers

    def on_click(self, event: Click) -> None:
        live = self._display_range()
        if live is None:
            return
        freq_min, freq_max = live
        if self.image_mode:
            occ = self._kitty.occlusion_insets
            w = self._kitty.full_pixel_size[0] - occ.left - occ.right
            x = event.x * self._kitty.cell_width_px - occ.left
        else:
            w = self.size.width
            x = event.x
        if w <= 0 or x < 0 or x >= w:
            return
        freq = freq_min + (x / w) * (freq_max - freq_min)

        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return

        # Check if click is near a memory label
        if self._memories:
            span = freq_max - freq_min
            # TODO: hardcoded pixels
            tolerance = span / w * 15 if w > 0 else 0
            for m in self._memories:
                if abs(m.frequency - freq) <= tolerance:
                    try:
                        recall_memory(m, device.device_id)
                    except SDRException as e:
                        self.app._show_error(str(e))
                    event.stop()
                    return

        # Snap to the user's current tuning step (auto-resolved if unset).
        ts = get_tuning_state()
        step = (
            ts.step
            if ts.step is not None
            else resolve_auto_step(device.active_mode, device.config.tuned_frequency)
        )
        freq = round(freq / step) * step
        save_previous_tune_state(device)
        try:
            engine.update_device_config(device.device_id, tuned_frequency=freq)
        except SDRException as e:
            self.app._show_error(str(e))
        event.stop()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        if event.shift:
            self._scroll_zoom(1)
        else:
            self._scroll_tune(-1)
        event.stop()

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        if event.shift:
            self._scroll_zoom(-1)
        else:
            self._scroll_tune(1)
        event.stop()

    def _scroll_zoom(self, direction: int) -> None:
        self.app._adjust_spectrum_span(direction)

    def _scroll_tune(self, direction: int) -> None:
        """Plain scroll tunes the dial; in free mode with a span set it pans
        the view instead (in center mode the view follows the dial anyway).
        Pixel-relative step (2 px per notch, min 1 Hz)."""
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return
        cfg = device.config
        caps = device.device.capabilities
        center, span = resolve_view(cfg, caps)
        w = self.size.width
        step = max(span / w * 2, 1.0) if w > 0 else 1.0
        if cfg.tuning_mode == "free" and cfg.spectrum_span is not None:
            lo, hi = full_view_range(cfg, caps)
            new_center = min(max(center + direction * step, lo + span / 2), hi - span / 2)
            try:
                engine.update_device_config(device.device_id, spectrum_center=new_center)
            except SDRException as e:
                self.app._show_error(str(e))
        else:
            new_freq = cfg.tuned_frequency + direction * step
            new_freq = round(new_freq / step) * step
            try:
                engine.update_device_config(device.device_id, tuned_frequency=new_freq)
            except SDRException as e:
                self.app._show_error(str(e))

    def _actual_freq_range(self, event: FFTUpdateEvent) -> tuple[float, float]:
        """Return (freq_min, freq_max) covered by the FFT event."""
        half = event.sample_rate / 2
        return event.center_frequency - half, event.center_frequency + half

    def _projection_window(
        self, view: tuple[float, float], event: FFTUpdateEvent
    ) -> tuple[float, float]:
        """View window for projecting bars; overlays keep the unshifted view."""
        if self._provides_spectrum:
            return view
        return transient_view_shift(view, event.center_frequency, self._capture_center)

    def _display_range(self) -> tuple[float, float] | None:
        """The requested view range, straight from device config.

        Overlays, the axis, and the dial cursor render this directly, so
        zoom/pan/retune move the window instantly. Bars are projected through
        `_projection_window`, which anchors the crop to the stale capture
        during a retune transient so content holds still until data catches
        up.
        """
        device = get_engine().get_focused_device()
        if device is None:
            return None
        return view_range(device.config, device.device.capabilities)

    def _compute_freq_labels(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[tuple[int, str], ...]:
        """Compute frequency axis label positions."""
        span = freq_max - freq_min
        ref = max(abs(freq_min), abs(freq_max))

        nice_intervals = [1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6, 5e6, 10e6]

        # Pick the smallest nice interval where labels don't overlap.
        # Estimate label width from the widest tick (highest magnitude), add min gap.
        min_gap = 3
        interval = nice_intervals[-1]
        for ni in nice_intervals:
            divisor, suffix, decimals = axis_si_prefix(ni, ref)
            cols_per_label = len(f"{ref / divisor:.{decimals}f}{suffix}") + min_gap
            tick_spacing = ni / span * width
            if tick_spacing >= cols_per_label:
                interval = ni
                break

        divisor, suffix, decimals = axis_si_prefix(interval, ref)
        labels: list[tuple[int, str]] = []
        first_tick = ((freq_min // interval) + 1) * interval
        freq = first_tick
        while freq < freq_max:
            col = int((freq - freq_min) / span * width)
            if 0 <= col < width:
                label = f"{freq / divisor:.{decimals}f}{suffix}"
                start = col - len(label) // 2
                end = start + len(label)
                if start >= 0 and end <= width:
                    labels.append((start, label))
            freq += interval

        return tuple(labels)

    def _compute_bandplan_segments(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[tuple[int, int, str, str], ...]:
        """Return (col_start, col_end_exclusive, label, color_hex) for each visible band.

        Narrower bands sort later so they overdraw wider overlapping bands
        when painted in order.
        """
        bp = self._bandplan
        if bp is None:
            return ()
        span = freq_max - freq_min
        if span <= 0:
            return ()

        raw: list[tuple[int, int, str, str, int]] = []
        for band in bp.bands:
            if band.end <= freq_min or band.start >= freq_max:
                continue
            f_lo = max(band.start, freq_min)
            f_hi = min(band.end, freq_max)
            col_start = max(0, floor((f_lo - freq_min) / span * width))
            col_end = min(width, ceil((f_hi - freq_min) / span * width))
            if col_end <= col_start:
                continue
            color = band_type_color(band.type)
            # Un-clipped Hz width is the stacking key: a wide band clipped to
            # a narrow visible slice must still sort as wide.
            raw.append((col_start, col_end, band.name, color, band.end - band.start))

        raw.sort(key=lambda t: -t[4])
        return tuple((cs, ce, name, color) for cs, ce, name, color, _ in raw)

    def _compute_bandwidth_range(
        self, width: int, freq_min: float, freq_max: float
    ) -> tuple[int, int] | None:
        """Compute bandwidth indicator column range."""
        channel_bw = self._channel_bandwidth

        if channel_bw is None:
            return None

        center = self._tuned_frequency
        span = freq_max - freq_min
        if self._sideband == "upper":
            f_low, f_high = center, center + channel_bw
        elif self._sideband == "lower":
            f_low, f_high = center - channel_bw, center
        else:
            f_low, f_high = center - channel_bw / 2, center + channel_bw / 2
        col_low = int((f_low - freq_min) / span * width)
        col_high = int((f_high - freq_min) / span * width)
        # Clamp both ends: the view can pan away from the dial entirely
        # (free mode), putting either column far outside [0, width].
        col_low = max(0, min(col_low, width))
        col_high = max(0, min(col_high, width))
        if col_high <= col_low:
            return None
        return (col_low, col_high)
