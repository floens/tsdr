from __future__ import annotations

from textual.strip import Strip

from tsdr.tui.widgets.kitty_image import KittyImageWidget


class ImageModeMixin:
    """Mixin for widgets that support Kitty image mode rendering."""

    _image_mode: bool = False
    _kitty: KittyImageWidget

    @property
    def image_mode(self) -> bool:
        return self._image_mode

    def _mount_kitty(self) -> None:
        """Call from widget's on_mount. Mounts KittyImageWidget child."""
        self._kitty = KittyImageWidget()
        self.mount(self._kitty)  # type: ignore[attr-defined]

    def toggle_image_mode(self, enabled: bool) -> None:
        """Switch between text and image rendering."""
        self._image_mode = enabled
        if enabled:
            self._on_image_mode_enabled()
        else:
            self._on_image_mode_disabled()

    def _on_image_mode_enabled(self) -> None:
        """Override in widget to render initial image when mode turns on."""

    def _on_image_mode_disabled(self) -> None:
        """Override in widget to clean up images when mode turns off."""

    def _image_mode_render_line(self, y: int, width: int) -> Strip:
        """Return blank strip for use in render_line() during image mode."""
        return Strip.blank(width, self.rich_style)  # type: ignore[attr-defined]
