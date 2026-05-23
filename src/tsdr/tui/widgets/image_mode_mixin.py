from __future__ import annotations

from textual.strip import Strip

from tsdr.tui.widgets.kitty_image import KittyImageWidget


class ImageModeMixin:
    """Helpers for widgets that support Kitty image mode rendering.

    Widgets that use this mixin declare ``image_mode = reactive(False)`` as a
    reactive attribute and add a ``watch_image_mode`` that calls
    ``_on_image_mode_enabled`` / ``_on_image_mode_disabled``. The mixin owns
    kitty child mounting and provides default hook stubs.
    """

    _kitty: KittyImageWidget

    def _mount_kitty(self) -> None:
        """Call from widget's on_mount. Mounts KittyImageWidget child."""
        self._kitty = KittyImageWidget()
        self.mount(self._kitty)  # type: ignore[attr-defined]

    def _on_image_mode_enabled(self) -> None:
        """Override in widget to render initial image when mode turns on."""

    def _on_image_mode_disabled(self) -> None:
        """Override in widget to clean up images when mode turns off."""

    def _image_mode_render_line(self, y: int, width: int) -> Strip:
        """Return blank strip for use in render_line() during image mode."""
        return Strip.blank(width, self.rich_style)  # type: ignore[attr-defined]
