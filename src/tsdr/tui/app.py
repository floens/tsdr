import logging
import sys
import traceback

from textual.app import App, ComposeResult

import tsdr.tui.commands  # noqa: F401  # trigger command self-registration
import tsdr.tui.tty  # noqa: F401  # install APC-aware XTermParser
from tsdr.core.band_stack import init_band_stack
from tsdr.core.bandplans import get_bandplan_store, init_bandplan_store
from tsdr.core.clock_sync import get_clock_sync_monitor, init_clock_sync_monitor
from tsdr.core.devices import init_device_store
from tsdr.core.events.events import BandplanChangedEvent, MemoriesChangedEvent
from tsdr.core.memories import get_memory_store, init_memory_store
from tsdr.core.preferences import (
    load_preferences,
    restore_bandplan,
    restore_devices,
    restore_engine_config,
    restore_tuning_state,
    save_device,
)
from tsdr.core.sdr.engine import SDREngine, get_engine
from tsdr.core.tracing import log_stats, span
from tsdr.core.tuning import flush_band_stack_writeback, subscribe_band_stack_writeback
from tsdr.core.tuning_state import init_tuning_state
from tsdr.tui.console import CommandInputMixin
from tsdr.tui.events.engine_sync import EngineSync
from tsdr.tui.events.prefs_sync import PrefsSync
from tsdr.tui.events.router import EventRouter
from tsdr.tui.keyboard import KeyboardMixin
from tsdr.tui.model import UIModel
from tsdr.tui.model.store import UIStore, init_ui_store
from tsdr.tui.textual_adapter import TextualEventAdapter
from tsdr.tui.tuning_mixin import TuningMixin
from tsdr.tui.view.factory import FACTORY
from tsdr.tui.view.reconciler import Reconciler
from tsdr.tui.view.tree import derive_tree

logger = logging.getLogger(__name__)

_active_app: TSDRApp | None = None


def get_app() -> TSDRApp:
    if _active_app is None:
        raise RuntimeError("App not initialized")
    return _active_app


class TSDRApp(App[None], KeyboardMixin, TuningMixin, CommandInputMixin, EventRouter):
    """Reactive Textual SDR app — UIModel drives the widget tree via the Reconciler."""

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
            self._latest_fft_by_device = {}

            self._saved_prefs = load_preferences()

            clock_sync = init_clock_sync_monitor()
            initial_model = UIModel.initial(self._saved_prefs)
            if initial_model.ntp_server:
                clock_sync.set_server(initial_model.ntp_server)

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

            self._store: UIStore = init_ui_store(initial_model)
            self._reconciler: Reconciler = Reconciler(self, FACTORY)
            self._engine: SDREngine = sdr_engine

            self._store.subscribe(lambda _old, new: self._reconciler.schedule(derive_tree(new)))
            self._engine_sync = EngineSync(self._store, sdr_engine)
            self._prefs_sync = PrefsSync(self._store, self)

            self.event_adapter = TextualEventAdapter(self, sdr_engine.event_bus)
            self.event_adapter.start()
            logger.info("textual_event_adapter_initialized")

            # Initial publish so the spectrum has overlays from first paint;
            # widgets also seed in on_mount.
            mem_store = get_memory_store()
            if memories := mem_store.all():
                sdr_engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(memories)))
            bp_store = get_bandplan_store()
            sdr_engine.event_bus.publish(BandplanChangedEvent(bandplan=bp_store.active))

            self.startup_commands = startup_commands or []
            self._startup_index = 0

    def compose(self) -> ComposeResult:
        # Empty — the reconciler mounts the tree in on_mount.
        with span("compose"):
            return
            yield  # pragma: no cover — marks this as a generator

    async def on_mount(self) -> None:
        with span("on_mount"):
            await self._reconciler.run_initial(derive_tree(self._store.model))
            self.query_one("#command-input").focus()
            self.call_after_refresh(self._restore_and_seed)
            if self.startup_commands:
                self.call_after_refresh(self._run_startup_commands)

        log_stats(phase="mounted")

        # Workaround for a terminal-IO drop bug on startup: characters in
        # the initial output burst can be lost. Force-refresh several times
        # to repaint.
        self.set_timer(0.5, self._force_refresh_all)
        self.set_timer(1.0, self._force_refresh_all)
        self.set_timer(2.0, self._force_refresh_all)
        self.set_timer(4.0, self._force_refresh_all)

    def _force_refresh_all(self) -> None:
        self.screen.refresh()

    def _restore_and_seed(self) -> None:
        restore_devices()
        self.seed_from_engine()

    def action_focus_next(self) -> None:
        pass

    def action_focus_previous(self) -> None:
        pass

    def _print_error_renderables(self) -> None:
        for renderable in self._exit_renderables:
            self.error_console.print(renderable)
        self._exit_renderables.clear()

    def _handle_exception(self, error: Exception) -> None:
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
        logger.info("app_unmounting")

        logger.debug("app_stopping_event_adapter")
        try:
            self.event_adapter.stop()
        except Exception as e:  # noqa: BLE001
            logger.error("event_adapter_stop_failed error=%r", e, exc_info=True)

        logger.debug("app_saving_device_state")
        try:
            save_device(get_engine())
        except Exception as e:  # noqa: BLE001
            logger.error("device_save_failed error=%r", e, exc_info=True)

        try:
            flush_band_stack_writeback()
        except Exception as e:  # noqa: BLE001
            logger.error("band_stack_flush_failed error=%r", e, exc_info=True)

        try:
            self._prefs_sync.close()
            self._engine_sync.close()
        except Exception as e:  # noqa: BLE001
            logger.error("sync_close_failed error=%r", e, exc_info=True)

        logger.debug("app_shutting_down_engine")
        try:
            get_engine().shutdown(timeout=2.0)
        except Exception as e:  # noqa: BLE001
            logger.error("app_engine_shutdown_failed error=%r", e, exc_info=True)

        get_clock_sync_monitor().shutdown()

        logger.info("app_cleanup_complete")

    def show_status(self, message: str) -> None:
        status = self._reconciler.get("status-bar")
        if status is not None:
            status.show_output(message)  # type: ignore[attr-defined]

    def _show_error(self, message: str) -> None:
        status = self._reconciler.get("status-bar")
        if status is not None:
            status.show_error(message)  # type: ignore[attr-defined]

    def queue_oob_escape(self, cmd: str) -> None:
        self._pending_oob_escapes.append(cmd)

    def _end_update(self) -> None:
        """Inject OOB kitty escapes."""
        if self._pending_oob_escapes and self._driver is not None:
            combined = "".join(self._pending_oob_escapes)
            self._driver.write(combined)
            self._pending_oob_escapes.clear()
        super()._end_update()
