"""Manual test for KittyImageWidget. Run in a Kitty or Ghostty terminal.

Single-image mode (default):
  Keys: h/l = adjust term xpixel, j/k = adjust term ypixel, q = quit

Multi-image mode (--multi):
  Keys: a/d = move image #2 left/right, w/s = scroll image #3 crop up/down,
        r = remove/re-add image #2, q = quit
"""

import logging
import sys

import numpy as np
from textual.app import App, ComposeResult

from tsdr.tui.widgets.kitty_image import KittyImageWidget

logger = logging.getLogger(__name__)


class KittyTestApp(App):
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "adjust(-1, 0)", "xpixel -1"),
        ("l", "adjust(1, 0)", "xpixel +1"),
        ("k", "adjust(0, -1)", "ypixel -1"),
        ("j", "adjust(0, 1)", "ypixel +1"),
    ]
    CSS = """
    #image {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending_oob_escapes: list[str] = []

    def queue_oob_escape(self, cmd: str) -> None:
        logger.debug(
            "kitty_test_queue_oob_escape bytes=%d pending=%d",
            len(cmd),
            len(self._pending_oob_escapes) + 1,
        )
        self._pending_oob_escapes.append(cmd)

    def post_display_hook(self) -> None:
        logger.debug(
            "kitty_test_post_display_hook pending=%d driver=%s",
            len(self._pending_oob_escapes),
            self._driver is not None,
        )
        if self._pending_oob_escapes and self._driver is not None:
            payload = "".join(self._pending_oob_escapes)
            logger.debug(
                "kitty_test_flush_oob commands=%d bytes=%d",
                len(self._pending_oob_escapes),
                len(payload),
            )
            self._driver.write(payload)
            self._pending_oob_escapes.clear()

    def compose(self) -> ComposeResult:
        yield KittyImageWidget(id="image")

    def on_resize(self) -> None:
        logger.debug("kitty_test_on_resize")
        self.call_after_refresh(self._send_image)

    def action_adjust(self, dx: int, dy: int) -> None:
        widget = self.query_one("#image", KittyImageWidget)
        widget._cell_width_px += dx
        widget._cell_height_px += dy
        logger.debug(
            "kitty_test_adjusted_cell w=%d h=%d",
            widget._cell_width_px,
            widget._cell_height_px,
        )
        self._send_image()

    def _send_image(self) -> None:
        logger.debug("kitty_test_send_image")

        widget = self.query_one("#image", KittyImageWidget)
        w, h = widget.available_pixel_size
        if w <= 0 or h <= 0:
            return

        region = widget.content_region

        # Checkerboard: 4px squares
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        xs = np.arange(w) // 4
        ys = np.arange(h) // 4
        checker = (xs[None, :] ^ ys[:, None]) & 1
        img[:, :, 0] = checker * 255
        img[:, :, 1] = (1 - checker) * 255

        # Render debug text at top-left
        lines = [
            f"cell: {widget._cell_width_px} x {widget._cell_height_px} px",
            f"widget: {region.width}x{region.height} cells -> {w}x{h} px",
        ]
        _draw_text(img, lines)

        widget.update_image("main", img)


class KittyMultiImageApp(App):
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("a", "move_left", "Move left"),
        ("d", "move_right", "Move right"),
        ("w", "scroll_up", "Scroll up"),
        ("s", "scroll_down", "Scroll down"),
        ("r", "toggle_remove", "Remove/re-add"),
    ]
    CSS = """
    #image {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending_oob_escapes: list[str] = []
        self._img2_x = 100
        self._img2_y = 50
        self._img3_crop_y = 0
        self._img2_removed = False

    def queue_oob_escape(self, cmd: str) -> None:
        logger.debug(
            "kitty_multi_queue_oob_escape bytes=%d pending=%d",
            len(cmd),
            len(self._pending_oob_escapes) + 1,
        )
        self._pending_oob_escapes.append(cmd)

    def post_display_hook(self) -> None:
        logger.debug(
            "kitty_multi_post_display_hook pending=%d driver=%s",
            len(self._pending_oob_escapes),
            self._driver is not None,
        )
        if self._pending_oob_escapes and self._driver is not None:
            payload = "".join(self._pending_oob_escapes)
            logger.debug(
                "kitty_multi_flush_oob commands=%d bytes=%d",
                len(self._pending_oob_escapes),
                len(payload),
            )
            self._driver.write(payload)
            self._pending_oob_escapes.clear()

    def compose(self) -> ComposeResult:
        yield KittyImageWidget(id="image")

    def on_resize(self) -> None:
        logger.debug("kitty_multi_image_on_resize")
        self.call_after_refresh(self._send_all)

    def _send_all(self) -> None:
        logger.debug("kitty_test_send_all")
        widget = self.query_one("#image", KittyImageWidget)
        w, h = widget.available_pixel_size
        if w <= 0 or h <= 0:
            logger.debug("kitty_test_zero_size")
            return

        # Image 1: red/green checkerboard at top-left (200x200)
        size1 = min(800, w, h)
        img1 = np.zeros((size1, size1, 4), dtype=np.uint8)
        img1[:, :, 3] = 255
        xs = np.arange(size1) // 8
        ys = np.arange(size1) // 8
        checker = (xs[None, :] ^ ys[:, None]) & 1
        img1[:, :, 0] = checker * 255
        img1[:, :, 1] = (1 - checker) * 255

        # Debug overlay on image 1
        lines = [
            f"img2: x={self._img2_x} y={self._img2_y}",
            f"img3 crop: y={self._img3_crop_y}",
            f"img2 removed: {self._img2_removed}",
        ]
        _draw_text(img1, lines, scale=8)
        widget.update_image("checker", img1)

        # Image 2: blue gradient rectangle, offset
        if not self._img2_removed:
            grad_w, grad_h = 150, 100
            img2 = np.zeros((grad_h, grad_w, 4), dtype=np.uint8)
            img2[:, :, 3] = 255
            img2[:, :, 2] = np.linspace(50, 255, grad_w, dtype=np.uint8)[None, :]
            img2[:, :, 1] = np.linspace(0, 100, grad_h, dtype=np.uint8)[:, None]
            widget.update_image("gradient", img2, x=self._img2_x, y=self._img2_y)

        # Image 3: large image with cropping (400px tall, show 150px window)
        img3_h, img3_w = 400, 200
        img3 = np.zeros((img3_h, img3_w, 4), dtype=np.uint8)
        img3[:, :, 3] = 255
        # Horizontal rainbow bands
        for i in range(img3_h):
            hue = (i / img3_h) * 3
            if hue < 1:
                img3[i, :, 0] = int((1 - hue) * 255)
                img3[i, :, 1] = int(hue * 255)
            elif hue < 2:
                img3[i, :, 1] = int((2 - hue) * 255)
                img3[i, :, 2] = int((hue - 1) * 255)
            else:
                img3[i, :, 2] = int((3 - hue) * 255)
                img3[i, :, 0] = int((hue - 2) * 255)

        widget.update_image(
            "rainbow",
            img3,
            x=max(0, w - img3_w - 20),
            y=10,
            crop_y=self._img3_crop_y,
            crop_h=150,
        )

    def action_move_left(self) -> None:
        self._img2_x = max(0, self._img2_x - 20)
        widget = self.query_one("#image", KittyImageWidget)
        widget.place_image("gradient", x=self._img2_x)
        self._update_overlay(widget)

    def action_move_right(self) -> None:
        self._img2_x += 20
        widget = self.query_one("#image", KittyImageWidget)
        widget.place_image("gradient", x=self._img2_x)
        self._update_overlay(widget)

    def action_scroll_up(self) -> None:
        self._img3_crop_y = max(0, self._img3_crop_y - 20)
        widget = self.query_one("#image", KittyImageWidget)
        widget.place_image("rainbow", crop_y=self._img3_crop_y)
        self._update_overlay(widget)

    def action_scroll_down(self) -> None:
        self._img3_crop_y = min(250, self._img3_crop_y + 20)
        widget = self.query_one("#image", KittyImageWidget)
        widget.place_image("rainbow", crop_y=self._img3_crop_y)
        self._update_overlay(widget)

    def action_toggle_remove(self) -> None:
        widget = self.query_one("#image", KittyImageWidget)
        if self._img2_removed:
            self._img2_removed = False
            # Re-add with a blue gradient
            grad_w, grad_h = 150, 100
            img2 = np.zeros((grad_h, grad_w, 4), dtype=np.uint8)
            img2[:, :, 3] = 255
            img2[:, :, 2] = np.linspace(50, 255, grad_w, dtype=np.uint8)[None, :]
            img2[:, :, 1] = np.linspace(0, 100, grad_h, dtype=np.uint8)[:, None]
            widget.update_image("gradient", img2, x=self._img2_x, y=self._img2_y)
        else:
            self._img2_removed = True
            widget.remove_image("gradient")
        self._update_overlay(widget)

    def _update_overlay(self, widget: KittyImageWidget) -> None:
        """Re-render just the checker overlay with current state."""
        w, h = widget.available_pixel_size
        if w <= 0 or h <= 0:
            return
        size1 = min(800, w, h)
        img1 = np.zeros((size1, size1, 4), dtype=np.uint8)
        img1[:, :, 3] = 255
        xs = np.arange(size1) // 8
        ys = np.arange(size1) // 8
        checker = (xs[None, :] ^ ys[:, None]) & 1
        img1[:, :, 0] = checker * 255
        img1[:, :, 1] = (1 - checker) * 255
        lines = [
            f"img2: x={self._img2_x} y={self._img2_y}",
            f"img3 crop: y={self._img3_crop_y}",
            f"img2 removed: {self._img2_removed}",
        ]
        _draw_text(img1, lines, scale=8)
        widget.update_image("checker", img1)


def _draw_text(img: np.ndarray, lines: list[str], scale: int = 4) -> None:
    """Render text lines as 5x7 bitmap font scaled up."""
    line_h = 7 * scale + scale  # glyph height + spacing
    for line_idx, text in enumerate(lines):
        y0 = line_idx * line_h + scale
        if y0 + 7 * scale > img.shape[0]:
            break
        x = scale
        for ch in text:
            glyph = _FONT.get(ch, _FONT.get("?", []))
            for row_idx, row in enumerate(glyph):
                for col_idx in range(5):
                    if row & (1 << (4 - col_idx)):
                        py = y0 + row_idx * scale
                        px = x + col_idx * scale
                        img[py : py + scale, px : px + scale] = [255, 255, 255, 255]
            x += 6 * scale
            if x >= img.shape[1] - 5 * scale:
                break


# Minimal 5x7 bitmap font for debug overlay
_FONT: dict[str, list[int]] = {
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
    "3": [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
    " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    ".": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
    ":": [0b00000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00100, 0b00000],
    "+": [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
    "-": [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
    "=": [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
    ">": [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
    "x": [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
    "w": [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
    "h": [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
    "c": [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10000, 0b01110],
    "e": [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
    "l": [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "t": [0b00100, 0b00100, 0b01110, 0b00100, 0b00100, 0b00100, 0b00011],
    "r": [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
    "m": [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
    "p": [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
    "i": [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
    "d": [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
    "g": [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110],
    "s": [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
    "o": [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
    "f": [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
    "n": [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
    "a": [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
    "k": [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
    "y": [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01000, 0b10000],
    "v": [0b00000, 0b00000, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
    "u": [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
    "F": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "(": [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    ")": [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    "?": [0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100],
}


if __name__ == "__main__":
    if "--multi" in sys.argv:
        KittyMultiImageApp().run()
    else:
        KittyTestApp().run()
