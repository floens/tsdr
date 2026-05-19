import logging
import sys
import traceback

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical

import tsdr.tui.commands  # noqa: F401  # trigger command self-registration
import tsdr.tui.tty  # noqa: F401  # install APC-aware XTermParser
from tsdr.core.band_stack import init_band_stack
from tsdr.core.bandplans import get_bandplan_store, init_bandplan_store
from tsdr.core.devices import init_device_store
from tsdr.core.events.events import BandplanChangedEvent, MemoriesChangedEvent
from tsdr.core.memories import get_memory_store, init_memory_store
from tsdr.core.preferences import (
    load_preferences,
    restore_bandplan,
    restore_devices,
    restore_engine_config,
    restore_tuning_state,
    restore_ui_state,
    save_device,
)
from tsdr.core.sdr.engine import SDREngine, get_engine
from tsdr.core.tracing import log_stats, span
from tsdr.core.tuning import flush_band_stack_writeback, subscribe_band_stack_writeback
from tsdr.core.tuning_state import init_tuning_state
from tsdr.tui.commands.registry import MenuItem
from tsdr.tui.console import CommandInputMixin
from tsdr.tui.event_handlers import EventHandlerMixin
from tsdr.tui.keyboard import KeyboardMixin
from tsdr.tui.state import UIState
from tsdr.tui.textual_adapter import TextualEventAdapter
from tsdr.tui.tuning_mixin import TuningMixin
from tsdr.tui.widgets import (
    ADSBWidget,
    ConsoleWidget,
    ConstellationWidget,
    DABWidget,
    DecoderOutputWidget,
    DMRWidget,
    ImageModeMixin,
    PerformanceWidget,
    RDSWidget,
    SpectrumWidget,
    StatsWidget,
    StatusBar,
    TETRAWidget,
    TunerWidget,
    WaterfallWidget,
)

logger = logging.getLogger(__name__)

_active_app: TSDRApp | None = None


def get_app() -> TSDRApp:
    if _active_app is None:
        raise RuntimeError("App not initialized")
    return _active_app


class TSDRApp(App[None], KeyboardMixin, TuningMixin, CommandInputMixin, EventHandlerMixin):
    """A Textual console application"""

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    CSS_PATH = "app.tcss"
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, startup_commands: list[str] | None = None) -> None:
        with span("TSDRApp.__init__"):
            super().__init__(ansi_color=True)

            global _active_app
            _active_app = self

            self._pending_oob_escapes: list[str] = []

            self.ui_state = UIState()
            self._latest_fft_by_device = {}

            # Restore saved preferences
            self._saved_prefs = load_preferences()
            restore_ui_state(self.ui_state, self._saved_prefs)

            sdr_engine = SDREngine()
            restore_engine_config(self._saved_prefs)
            init_memory_store()
            init_band_stack()
            init_bandplan_store()
            init_device_store()
            restore_bandplan(self._saved_prefs)
            tuning_state = init_tuning_state()
            restore_tuning_state(tuning_state, self._saved_prefs)
            subscribe_band_stack_writeback(sdr_engine)

            self.event_adapter = TextualEventAdapter(self, sdr_engine.event_bus)

            self.event_adapter.start()
            logger.info("textual_event_adapter_initialized")

            # Publish initial memories so spectrum widget shows labels on startup
            store = get_memory_store()
            if memories := store.all():
                sdr_engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(memories)))

            # Publish initial bandplan (may be None) so the widget is in a known state.
            bp_store = get_bandplan_store()
            sdr_engine.event_bus.publish(BandplanChangedEvent(bandplan=bp_store.active))

            # Store startup commands for execution after mount
            self.startup_commands = startup_commands or []
            self._startup_index = 0

            self.call_after_refresh(self._restore_saved_devices)

    def compose(self) -> ComposeResult:
        with span("compose"):
            yield TunerWidget()

            with Container(id="main-container"):
                with Container(id="content"), Vertical(id="viz-container"):
                    yield SpectrumWidget(self.ui_state)
                    yield WaterfallWidget(self.ui_state)
                    yield RDSWidget()
                    yield DABWidget()
                    yield ADSBWidget()
                    yield TETRAWidget()
                    yield DMRWidget()
                    yield DecoderOutputWidget()

                with Container(id="sidebar"):
                    yield StatsWidget()
                    yield ConstellationWidget()
                    yield PerformanceWidget()

            yield StatusBar()
            yield ConsoleWidget()

    def on_mount(self) -> None:
        with span("on_mount"):
            self.query_one("#command-input").focus()

            # Restore sidebar state from preferences
            panel = self.ui_state.active_panel
            sidebar = self.query_one("#sidebar")
            if panel:
                sidebar.display = True
                self.query_one(StatsWidget).display = panel == "stats"
                self.query_one(PerformanceWidget).display = panel == "performance"
                self.query_one(ConstellationWidget).display = panel != "performance"
            else:
                sidebar.display = False
                self.query_one(PerformanceWidget).display = False

            if self.ui_state.image_mode:
                self._notify_image_mode_changed()

            if self.startup_commands:
                self.call_after_refresh(self._run_startup_commands)

        log_stats(phase="mounted")

        # TODO: IO-bug
        # On startup (when a lot of CPU/stdout is happening), some widgets are not fully rendered,
        # unless we force a refresh of all widgets. This is the known IO-bug we haven't figured out yet.
        # Here, set some timers that force refresh the screen just after startup.
        self.set_timer(0.5, self._force_refresh_all)
        self.set_timer(1.0, self._force_refresh_all)
        self.set_timer(2.0, self._force_refresh_all)
        self.set_timer(4.0, self._force_refresh_all)

    def _force_refresh_all(self):
        self.screen.refresh()

    def _restore_saved_devices(self) -> None:
        restore_devices()

    def _toggle_panel(self, panel: str) -> None:
        """Toggle a side panel, ensuring mutual exclusivity."""
        sidebar = self.query_one("#sidebar")
        stats = self.query_one(StatsWidget)
        perf = self.query_one(PerformanceWidget)
        constellation = self.query_one(ConstellationWidget)

        if self.ui_state.active_panel == panel:
            # Same panel -> close sidebar
            self.ui_state.active_panel = None
            sidebar.display = False
        else:
            # Different panel or none -> show requested panel
            self.ui_state.active_panel = panel
            sidebar.display = True
            stats.display = panel == "stats"
            perf.display = panel == "performance"
            constellation.display = panel != "performance"

        self._update_constellation_config()

    def action_focus_next(self) -> None:
        # Disable focus next widget
        pass

    def action_focus_previous(self) -> None:
        # Disable focus previous widget
        pass

    def _print_error_renderables(self) -> None:
        for renderable in self._exit_renderables:
            self.error_console.print(renderable)
        self._exit_renderables.clear()

    def _handle_exception(self, error: Exception) -> None:
        # Bypass Textual's default which uses Rich Traceback rendering
        # Rich tracebacks are hundreds of lines with locals, syntax highlighting,
        # and panels, making them harder to read than plain Python tracebacks.
        self._return_code = 1
        if self._exception is None:  # type: ignore[has-type]
            self._exception = error
            self._exception_event.set()
        self._fatal_error()

    def _fatal_error(self) -> None:
        self.bell()
        tb = "".join(traceback.format_exception(*sys.exc_info()))
        self._exit_renderables.append(tb)
        self._close_messages_no_wait()

    def on_unmount(self) -> None:
        """Called when app is unmounting - cleanup resources."""
        logger.info("app_unmounting")

        logger.debug("app_stopping_event_adapter")
        try:
            self.event_adapter.stop()
        except Exception as e:
            logger.error("event_adapter_stop_failed error=%r", e, exc_info=True)

        logger.debug("app_saving_device_state")
        try:
            save_device(get_engine())
        except Exception as e:
            logger.error("device_save_failed error=%r", e, exc_info=True)

        try:
            flush_band_stack_writeback()
        except Exception as e:
            logger.error("band_stack_flush_failed error=%r", e, exc_info=True)

        logger.debug("app_shutting_down_engine")
        try:
            get_engine().shutdown(timeout=2.0)
        except Exception as e:
            logger.error("app_engine_shutdown_failed error=%r", e, exc_info=True)

        logger.info("app_cleanup_complete")

    def show_status(self, message: str) -> None:
        """Show a message in the status bar."""
        self.query_one(StatusBar).show_output(message)

    def _show_error(self, message: str) -> None:
        self.query_one(StatusBar).show_error(message)

    def _show_autocomplete(self, commands: list[MenuItem], selected_index: int) -> None:
        self.query_one(ConsoleWidget).show_autocomplete(commands, selected_index)

    def queue_oob_escape(self, cmd: str) -> None:
        self._pending_oob_escapes.append(cmd)

    def _end_update(self) -> None:
        """Inject OOB kitty escapes"""
        if self._pending_oob_escapes and self._driver is not None:
            combined = "".join(self._pending_oob_escapes)
            self._driver.write(combined)
            self._pending_oob_escapes.clear()
        super()._end_update()

    def _notify_image_mode_changed(self) -> None:
        enabled = self.ui_state.image_mode
        for widget in self.query("*"):
            if isinstance(widget, ImageModeMixin):
                widget.toggle_image_mode(enabled)
        # TODO: why is DABWidget not a ImageModeMixin
        for dab_widget in self.query("DABWidget"):
            dab_widget.set_image_mode(enabled)

        self.query_one(ConstellationWidget).set_image_mode(enabled)
        self._update_constellation_config()

        # TODO: fix hardcoded
        fps = 60 if enabled else 20
        engine = get_engine()
        engine.update_global_config(update_rate_fps=fps)

    def _update_constellation_config(self) -> None:
        """Set calculate_constellation on focused device based on panel/image state."""
        enabled = self.ui_state.image_mode and self.ui_state.active_panel == "stats"
        engine = get_engine()
        device = engine.get_focused_device()
        if device is not None:
            engine.update_device_config(device.device_id, calculate_constellation=enabled)
