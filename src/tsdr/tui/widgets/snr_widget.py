from collections import deque

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from textual.widget import Widget

# Braille sparkline: each cell has 4 vertical dot positions (bottom to top).
# We use the left column only (dots 1,2,3,7 at bit offsets 0,1,2,6).
# Combined with 2 rows, this gives 8 vertical levels.
_BRAILLE_BASE = 0x2800  # ⠀ (empty braille)
# Dot bits for left column, bottom to top: dot7=0x40, dot3=0x04, dot2=0x02, dot1=0x01
_BRAILLE_DOTS = [0x40, 0x04, 0x02, 0x01]
_DOTS_PER_ROW = len(_BRAILLE_DOTS)  # 4 levels per row


def _braille_char(level: int) -> str:
    """Return braille character with `level` dots filled from bottom (0-4)."""
    bits = 0
    for i in range(level):
        bits |= _BRAILLE_DOTS[i]
    return chr(_BRAILLE_BASE | bits)


_SNR_MIN = 0.0
_SNR_MAX = 50.0

_STYLE_DIM = Style(dim=True)
_TRACK_STYLE = Style(dim=True)


def _snr_color(snr: float) -> str:
    if snr > 20:
        return "green"
    elif snr > 10:
        return "yellow"
    return "red"


def _snr_style(snr: float, bold: bool = False) -> Style:
    return Style(color=_snr_color(snr), bold=bold)


def _quality_style(quality: float | None) -> Style:
    if quality is None:
        return _STYLE_DIM
    if quality >= 0.8:
        return Style(color="green", bold=True)
    elif quality >= 0.5:
        return Style(color="yellow", bold=True)
    return Style(color="red", bold=True)


class SNRWidget(Widget):
    """Displays SNR as a slider bar with 2-row braille sparkline history.

    Row 0: ████████░░░░░░░  32.1 dB   (slider + color-coded number)
    Row 1:          ⠠⠰⠸⠸⠰⠠               (upper dots)
    Row 2: ⠁⠃⠇⠧⠧⠇⠧⠧⠧⠧⠧⠧⠧⠇⠃⠁     (lower dots)
    """

    DEFAULT_CSS = """
    SNRWidget {
        width: 1fr;
        max-width: 36;
        height: 3;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: deque[float] = deque(maxlen=200)
        self._current_snr: float | None = None
        self._quality_label: str | None = None
        self._quality: float | None = None
        self._squelch_threshold_db: float | None = None
        self._squelch_open: bool | None = None

    def update_snr(self, snr: float | None) -> None:
        if snr is not None:
            self._history.append(snr)
        self._current_snr = snr
        self.refresh()

    def update_quality(self, label: str | None, quality: float | None) -> None:
        self._quality_label = label
        self._quality = quality
        self.refresh()

    def update_squelch(self, threshold_db: float | None, is_open: bool | None) -> None:
        self._squelch_threshold_db = threshold_db
        self._squelch_open = is_open
        self.refresh()

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if width <= 0:
            return Strip.blank(width, self.rich_style)

        if y == 0:
            return self._render_slider(width)
        elif y == 1:
            return self._render_sparkline_row(width, row=0)
        elif y == 2:
            return self._render_sparkline_row(width, row=1)
        return Strip.blank(width, self.rich_style)

    def _render_slider(self, width: int) -> Strip:
        base = self.rich_style
        quality_suffix = ""
        quality_suffix_len = 0
        if self._quality_label is not None and self._quality is not None:
            quality_suffix = f"  {self._quality_label}"
            quality_suffix_len = len(quality_suffix)

        squelch_suffix, squelch_style = self._squelch_suffix(base)
        squelch_suffix_len = len(squelch_suffix)

        suffix_len = quality_suffix_len + squelch_suffix_len
        available = min(15, width - suffix_len)

        if self._current_snr is None:
            label = "--- dB"
            pad = available - len(label)
            dim = base + _STYLE_DIM
            segments = [Segment(" " * max(0, pad), dim), Segment(label, dim)]
        else:
            snr = self._current_snr
            label = f" {snr:.1f} dB"
            bar_width = max(0, available - len(label))

            ratio = max(0.0, min(1.0, (snr - _SNR_MIN) / (_SNR_MAX - _SNR_MIN)))
            filled_half = ratio * bar_width * 2
            filled_full = int(filled_half) // 2
            has_half = int(filled_half) % 2
            empty = bar_width - filled_full - has_half

            fill_style = base + Style(color=_snr_color(snr))
            segments = [
                Segment("█" * filled_full, fill_style),
            ]
            if has_half:
                segments.append(Segment("▌", fill_style))
            segments.append(Segment("·" * empty, base + _TRACK_STYLE))
            segments.append(Segment(label, base + _snr_style(snr, bold=snr > 20)))

        pad = width - available - suffix_len
        if pad > 0:
            segments.append(Segment(" " * pad, base))
        if squelch_suffix:
            segments.append(Segment(squelch_suffix, squelch_style))
        if quality_suffix:
            segments.append(Segment(quality_suffix, base + _quality_style(self._quality)))

        return Strip(segments, width)

    def _squelch_suffix(self, base: Style) -> tuple[str, Style]:
        if self._squelch_threshold_db is None:
            return "", base
        threshold = int(round(self._squelch_threshold_db))
        if self._squelch_open is None:
            return f"  SQ {threshold:+d}", base + _STYLE_DIM
        glyph = "●" if self._squelch_open else "○"
        color = "green" if self._squelch_open else "red"
        return f"  SQ {threshold:+d} {glyph}", base + Style(color=color, bold=True)

    def _render_sparkline_row(self, width: int, row: int) -> Strip:
        """Render one row of the 2-high braille sparkline.

        row=0 is upper (top half of range), row=1 is lower (bottom half).
        """
        base = self.rich_style
        if not self._history:
            return Strip.blank(width, base)

        samples = list(self._history)[-width:]
        segments = []

        pad = width - len(samples)
        if pad > 0:
            segments.append(Segment(" " * pad, base))

        for val in samples:
            ratio = max(0.0, min(1.0, (val - _SNR_MIN) / (_SNR_MAX - _SNR_MIN)))
            total_dots = int(ratio * (2 * _DOTS_PER_ROW))

            if row == 0:
                # Upper row: dots above _DOTS_PER_ROW
                level = max(0, min(_DOTS_PER_ROW, total_dots - _DOTS_PER_ROW))
            else:
                # Lower row: dots up to _DOTS_PER_ROW
                level = min(_DOTS_PER_ROW, total_dots)

            segments.append(Segment(_braille_char(level), base + _snr_style(val)))

        return Strip(segments, width)
