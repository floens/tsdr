"""Central SDR pipeline manager.

Coordinates multiple SDR devices, their processing pipelines, and audio output.
This is the main entry point for all SDR operations from the UI layer.
"""

import logging
from dataclasses import replace
from queue import Empty, Queue
from types import MappingProxyType
from typing import Any, Unpack

from tsdr.core.audio_spec import AudioDemodSpec
from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import (
    AGCGainChangeEvent,
    ConfigChangedEvent,
    DeviceAddedEvent,
    DeviceCapabilitiesChangedEvent,
    DeviceRemovedEvent,
    DeviceStateChangedEvent,
    Event,
    FocusChangedEvent,
    PipelineChangedEvent,
    RecordingFinishedEvent,
)
from tsdr.core.sdr.config import (
    DeviceConfig,
    DeviceConfigChanges,
    GlobalConfigChanges,
    PipelineConfig,
    SDRConfig,
    StageType,
)
from tsdr.core.sdr.device_context import DeviceState, SDRDeviceContext
from tsdr.core.sdr.exceptions import ConfigurationError, SDRException
from tsdr.core.sdr.workers.audio_worker import AudioOutputWorker, list_audio_devices
from tsdr.core.tracing import span, traced
from tsdr.core.units import find_nearest
from tsdr.core.workers import WorkerRunner
from tsdr.devices import DeviceParams, create_device
from tsdr.radio.registry import DEMODULATOR_CLASSES

logger = logging.getLogger(__name__)

_active_engine: SDREngine | None = None


def get_engine() -> SDREngine:
    if _active_engine is None:
        raise RuntimeError("SDR engine not initialized")
    return _active_engine


class SDREngine:
    """Central coordinator for SDR devices and processing pipelines.

    Manages:
    - Multiple SDR device instances
    - Per-device processing pipelines
    - Audio output routing
    - Default pipeline creation
    - Dynamic pipeline reconfiguration

    Thread Safety:
        - All public methods are thread-safe (called from Textual main thread)
        - Device contexts manage their own thread synchronization
    """

    def __init__(self):
        global _active_engine
        with span("SDREngine init"):
            _active_engine = self
            self.config = SDRConfig()
            self.devices: dict[str, SDRDeviceContext] = {}
            self.focused_device: str | None = None

            self.event_bus = EventBus()
            self.worker_runner = WorkerRunner(self.event_bus)

            self._audio_workers: dict[str, AudioOutputWorker] = {}
            self._audio_output_device: str | None = None

            # TODO: Why is this here?
            self.event_bus.subscribe(AGCGainChangeEvent, self._on_agc_gain_change)
            self.event_bus.subscribe(RecordingFinishedEvent, self._on_recording_finished)
            self.event_bus.subscribe(
                DeviceCapabilitiesChangedEvent, self._on_device_capabilities_changed
            )

    def add_device(
        self,
        device_id: str,
        device_type: str,
        params: DeviceParams,
        device_config: DeviceConfig | None = None,
    ) -> None:
        """Add a new SDR device.

        Args:
            device_id: Unique device identifier (e.g., "rtl0")
            device_type: Device type ("rtltcp", "mock")
            params: Device-specific parameters (RTLTCPParams, MockParams, etc.)
            device_config: Optional per-device configuration (uses defaults if None)
        """
        if device_id in self.devices:
            raise SDRException(f"Device {device_id} already exists")

        try:
            device = create_device(params)
        except Exception as e:
            raise ConfigurationError(f"Failed to create device {device_id}: {e}") from e

        device_config = device_config if device_config is not None else DeviceConfig()

        sample_queue: Queue = Queue(maxsize=device_config.queue_size)
        control_queue: Queue = Queue(maxsize=device_config.queue_size // 2)
        pipeline_control_queue: Queue = Queue(maxsize=10)
        audio_queue: Queue = Queue(maxsize=device_config.queue_size)

        context = SDRDeviceContext(
            device_id=device_id,
            device_type=device_type,
            params=params,
            device=device,
            device_config=device_config,
            sample_queue=sample_queue,
            control_queue=control_queue,
            pipeline_control_queue=pipeline_control_queue,
            audio_queue=audio_queue,
            get_sdr_config=lambda: self.config,
        )

        self.devices[device_id] = context

        focus_assigned = self.focused_device is None
        if focus_assigned:
            self.focused_device = device_id

        self.event_bus.publish(DeviceAddedEvent(device_id=device_id, source_id="engine"))
        if focus_assigned:
            self.event_bus.publish(
                FocusChangedEvent(focused_device_id=device_id, source_id="engine")
            )

    def remove_device(self, device_id: str) -> None:
        """Remove a device (must be stopped first).

        Args:
            device_id: Device to remove

        Raises:
            SDRException: If device doesn't exist or is still running
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]

        if context.state != DeviceState.STOPPED:
            raise SDRException(f"Device {device_id} must be stopped before removal")

        # Clean up
        self.stop_audio_output(device_id)
        del self.devices[device_id]

        focus_changed = self.focused_device == device_id
        if focus_changed:
            self.focused_device = next(iter(self.devices.keys()), None)

        self.event_bus.publish(DeviceRemovedEvent(device_id=device_id, source_id="engine"))
        if focus_changed:
            self.event_bus.publish(
                FocusChangedEvent(focused_device_id=self.focused_device, source_id="engine")
            )

    def reconfigure_device_params(self, device_id: str, params: DeviceParams) -> None:
        """Swap the underlying SDRDevice with one built from new params.

        Keeps device_config, pipelines, audio routing, and focus intact.
        Requires the device to be STOPPED — caller is responsible for stopping
        and restarting around the call if needed.

        The new params must be the same concrete type as the existing ones
        (e.g. RTLTCPParams → RTLTCPParams); cross-type reconfiguration is not
        supported.
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]
        if context.state != DeviceState.STOPPED:
            raise SDRException(f"Device {device_id} must be stopped to reconfigure params")
        if type(params) is not type(context.params):
            raise SDRException(
                f"Cannot reconfigure {device_id}: param type mismatch "
                f"({type(context.params).__name__} → {type(params).__name__})"
            )

        try:
            new_device = create_device(params)
        except Exception as e:  # noqa: BLE001 - device constructors raise opaque driver errors
            raise ConfigurationError(f"Failed to recreate device {device_id}: {e}") from e

        context.params = params
        context.device = new_device

        self.event_bus.publish(ConfigChangedEvent(device_id=device_id, source_id="engine"))

    def start_device(self, device_id: str) -> None:
        """Start a device.

        Pipelines are materialized from DeviceConfig.pipelines by the
        pipeline worker during its setup phase.

        Args:
            device_id: Device to start

        Raises:
            SDRException: If device doesn't exist or is already running
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]

        if context.state != DeviceState.STOPPED:
            raise SDRException(f"Device {device_id} is not stopped")

        # Start device workers (pipeline worker materializes pipelines in setup)
        context.start(worker_runner=self.worker_runner)

        if context.demod_profile and context.demod_profile.has_audio:
            self.start_audio_output(device_id)

        self.event_bus.publish(
            DeviceStateChangedEvent(device_id=device_id, running=True, source_id="engine")
        )

    def stop_device(self, device_id: str) -> None:
        """Stop a device.

        Args:
            device_id: Device to stop

        Raises:
            SDRException: If device doesn't exist
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]

        # Stop audio output
        self.stop_audio_output(device_id)

        # Stop device workers
        context.stop(worker_runner=self.worker_runner, timeout=5.0)

        self.event_bus.publish(
            DeviceStateChangedEvent(device_id=device_id, running=False, source_id="engine")
        )

    @traced("engine.update_device_config")
    def update_device_config(self, device_id: str, **changes: Unpack[DeviceConfigChanges]) -> None:
        """Update per-device configuration.

        DeviceContext handles worker notification (both I/O and pipeline workers).
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]
        old_pipelines = context.config.pipelines

        caps = context.device.capabilities

        if "center_frequency" in changes:
            freq_range = caps.frequency_range
            new_freq = changes["center_frequency"]
            if freq_range is not None:
                lo, hi = freq_range
                if not (lo <= new_freq <= hi):
                    raise SDRException(
                        f"Frequency {new_freq / 1e6:.3f} MHz out of range "
                        f"(device supports {lo / 1e6:.3f}–{hi / 1e6:.3f} MHz)"
                    )

        if "sample_rate" in changes and caps.sample_rates is not None:
            new_rate = changes["sample_rate"]
            if not any(abs(new_rate - r) < 1.0 for r in caps.sample_rates):
                valid = ", ".join(f"{r / 1e6:.3f}M" for r in sorted(caps.sample_rates))
                raise SDRException(
                    f"Sample rate {new_rate / 1e6:.3f} MHz not supported (device supports: {valid})"
                )

        logger.info("device_config_update device=%s changes=%r", device_id, changes)
        context.update_config(**changes)

        self.event_bus.publish(ConfigChangedEvent(device_id=device_id, source_id=device_id))

        # Pipelines mutated → publish PipelineChangedEvent per affected pipeline
        # so the UI can reseed active_decoder_kind / has_audio_pipeline. Without
        # this, callers that touch pipelines via update_device_config directly
        # (e.g. `pipeline add/remove`, replace_pipeline_stage) leave the model
        # stale. add_pipeline/remove_pipeline/set_audio_demod also publish, so
        # those paths see a (harmless) duplicate event.
        if "pipelines" in changes:
            new_pipelines = context.config.pipelines
            for name in set(old_pipelines) | set(new_pipelines):
                if old_pipelines.get(name) != new_pipelines.get(name):
                    self._publish_pipeline_changed(device_id, name, active=name in new_pipelines)

    def update_global_config(self, **changes: Unpack[GlobalConfigChanges]) -> None:
        """Update engine-global configuration (processing/display parameters).

        Notifies all running pipeline workers and audio workers.
        """
        logger.info("global_config_update changes=%r", changes)
        self.config = self.config.with_changes(**changes)
        self.config.validate()

        # Notify audio workers (synchronous - only sets volume, acceptable for a single float)
        # TODO: this is not to design
        for worker in self._audio_workers.values():
            worker.on_config_change(self.config)

        # Notify all devices
        for device_id, context in self.devices.items():
            self.event_bus.publish(ConfigChangedEvent(device_id=device_id, source_id="engine"))
            context.notify_global_config_change(self.config)

    def list_devices(self) -> list[dict[str, Any]]:
        """Get list of all devices and their status.

        Returns:
            List of device info dictionaries

        Example:
            >>> devices = manager.list_devices()
            >>> for dev in devices:
            ...     print(f"{dev['id']}: {dev['state']} @ {dev['frequency']/1e6:.2f} MHz")
        """
        # TODO: who uses this? needs to be typed if used.
        devices = []

        for device_id, context in self.devices.items():
            devices.append(
                {
                    "id": device_id,
                    "type": context.device_type,
                    "state": context.state.name,
                    "frequency": context.config.center_frequency,
                    "sample_rate": context.config.sample_rate,
                    "mode": context.active_mode,
                    "focused": device_id == self.focused_device,
                    "description": context.params.describe(),
                }
            )

        return devices

    def set_focused_device(self, device_id: str) -> None:
        """Set the focused (active) device.

        Args:
            device_id: Device to focus

        Raises:
            SDRException: If device doesn't exist
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        if self.focused_device == device_id:
            return

        self.focused_device = device_id
        self.event_bus.publish(FocusChangedEvent(focused_device_id=device_id, source_id="engine"))

    def get_device(self, device_id: str) -> SDRDeviceContext:
        """Get a device context by ID.

        Raises:
            SDRException: If device doesn't exist
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")
        return self.devices[device_id]

    def get_focused_device(self) -> SDRDeviceContext | None:
        """Get the currently focused device.

        Returns:
            Device context or None
        """
        if self.focused_device is None:
            return None
        return self.devices.get(self.focused_device)

    def add_pipeline(
        self, device_id: str, pipeline_name: str, pipeline_config: PipelineConfig
    ) -> None:
        """Add a pipeline to a device by updating its config.

        The pipeline worker materializes the actual stage instances.

        Args:
            device_id: Device identifier
            pipeline_name: Pipeline name (e.g., "audio")
            pipeline_config: Pipeline configuration
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]
        new_pipelines = MappingProxyType(
            dict(context.config.pipelines) | {pipeline_name: pipeline_config}
        )
        self.update_device_config(device_id, pipelines=new_pipelines)

        logger.info(
            "pipeline_added device=%s name=%s mode=%s",
            device_id,
            pipeline_name,
            pipeline_config.audio_spec.mode if pipeline_config.audio_spec else None,
        )

        self._publish_pipeline_changed(device_id, pipeline_name, active=True)

    def remove_pipeline(self, device_id: str, pipeline_name: str) -> None:
        """Remove a pipeline from a device by updating its config.

        Args:
            device_id: Device identifier
            pipeline_name: Pipeline name to remove
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]
        if pipeline_name not in context.config.pipelines:
            return

        new_pipelines = {k: v for k, v in context.config.pipelines.items() if k != pipeline_name}
        self.update_device_config(device_id, pipelines=MappingProxyType(new_pipelines))

        logger.info("pipeline_removed device=%s name=%s", device_id, pipeline_name)

        self._publish_pipeline_changed(device_id, pipeline_name, active=False)

    def _publish_pipeline_changed(self, device_id: str, name: str, *, active: bool) -> None:
        self.event_bus.publish(
            PipelineChangedEvent(
                source_id=f"engine_{device_id}",
                device_id=device_id,
                pipeline_name=name,
                active=active,
            )
        )

    def _drain_audio_queue(self, device_id: str) -> None:
        context = self.devices[device_id]
        dropped = 0
        try:
            while True:
                context.audio_queue.get_nowait()
                dropped += 1
        except Empty:
            pass
        if dropped:
            logger.info("audio_queue_drained device=%s batches=%d", device_id, dropped)

    def start_audio_output(self, device_id: str) -> None:
        """Start audio output for a device. No-op if device is not running."""
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")
        if device_id in self._audio_workers:
            return

        context = self.devices[device_id]
        if context.state != DeviceState.RUNNING:
            return

        self._drain_audio_queue(device_id)

        worker = AudioOutputWorker(
            source_id=device_id,
            audio_queue=context.audio_queue,
            output_device=self._audio_output_device,
        )
        worker.on_config_change(self.config)
        self.worker_runner.start_worker(worker_id=f"audio_{device_id}", worker=worker, daemon=False)
        self._audio_workers[device_id] = worker

    def stop_audio_output(self, device_id: str, timeout: float = 5.0) -> None:
        """Stop audio output for a device. No-op if not running."""
        if device_id not in self._audio_workers:
            return
        self.worker_runner.stop_worker(f"audio_{device_id}", timeout=timeout)
        self._audio_workers.pop(device_id, None)

    def set_audio_output_device(self, device_name: str | None) -> None:
        """Change the audio output device, restarting any running audio workers."""
        old = self._audio_output_device
        self._audio_output_device = device_name
        logger.info("audio_output_changed old=%r new=%r", old, device_name)
        for device_id in list(self._audio_workers.keys()):
            self.stop_audio_output(device_id)
            self.start_audio_output(device_id)

    def list_audio_devices(self) -> list[dict[str, Any]]:
        """List available audio output devices."""
        return list_audio_devices()

    def get_active_audio_sources(self) -> list[str]:
        """Device IDs with running audio output."""
        return list(self._audio_workers.keys())

    def update_squelch(
        self,
        device_id: str,
        pipeline_name: str = "audio",
        *,
        enabled: bool | None = None,
        threshold_db: float | None = None,
        hang_ms: float | None = None,
    ) -> None:
        """Update squelch parameters on a pipeline's demodulator.

        Only the provided fields are changed; the rest are preserved.
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")
        context = self.devices[device_id]
        current = context.config.pipelines.get(pipeline_name)
        if current is None:
            raise SDRException(f"Pipeline '{pipeline_name}' not found on device {device_id}")

        updates: dict[str, Any] = {}
        if enabled is not None:
            updates["squelch_enabled"] = enabled
        if threshold_db is not None:
            updates["squelch_threshold_db"] = threshold_db
        if hang_ms is not None:
            updates["squelch_hang_ms"] = hang_ms
        if not updates:
            return

        new_pipeline = replace(current, **updates)
        new_pipelines = MappingProxyType(
            dict(context.config.pipelines) | {pipeline_name: new_pipeline}
        )
        self.update_device_config(device_id, pipelines=new_pipelines)

    def set_audio_demod(self, device_id: str, spec: AudioDemodSpec) -> None:
        """Rebuild the 'audio' pipeline from the given demod spec.

        Fast path: when the audio worker is already running and the stage
        tuple is unchanged, swap the demodulator in place — the
        ``AudioOutputWorker``'s sounddevice stream stays open and ``PortAudio``
        is not torn down. Falls back to stop/remove/add/start when the pipeline
        shape changes (FREQUENCY_SHIFT add/remove) or the new mode has no
        audio output (audio→decoder).
        """
        if device_id not in self.devices:
            raise SDRException(f"Device {device_id} not found")

        context = self.devices[device_id]
        old_pipeline = context.config.pipelines.get("audio")
        old_mode = (
            old_pipeline.audio_spec.mode
            if old_pipeline is not None and old_pipeline.audio_spec is not None
            else None
        )
        logger.info("demod_change device=%s old=%s new=%s", device_id, old_mode, spec.mode)

        new_stages: list[StageType] = []
        if spec.frequency_offset != 0.0:
            new_stages.append(StageType.FREQUENCY_SHIFT)
        new_stages.append(StageType.DEMODULATOR)
        new_stages.append(StageType.DENOISER)
        new_stages.append(StageType.EVENT_EMITTER)
        new_stages_tuple = tuple(new_stages)

        new_cls = DEMODULATOR_CLASSES.get(spec.mode.upper())
        new_has_audio = new_cls is not None and new_cls.HAS_AUDIO
        new_bw = (
            new_cls.bandwidth_override_on_mode_switch(context.config.channel_bandwidth)
            if new_cls is not None
            else None
        )

        can_swap_in_place = (
            device_id in self._audio_workers
            and old_pipeline is not None
            and old_pipeline.stages == new_stages_tuple
            and new_has_audio
        )

        if can_swap_in_place:
            self._drain_audio_queue(device_id)
            self._audio_workers[device_id].request_flush()

            new_pc = PipelineConfig(stages=new_stages_tuple, audio_spec=spec)
            new_pipelines = MappingProxyType(dict(context.config.pipelines) | {"audio": new_pc})
            changes: dict[str, Any] = {"pipelines": new_pipelines}
            if new_bw is not None:
                changes["channel_bandwidth"] = new_bw
            self.update_device_config(device_id, **changes)
            self._publish_pipeline_changed(device_id, "audio", active=True)
            return

        self.stop_audio_output(device_id)
        self.remove_pipeline(device_id, "audio")
        if new_bw is not None:
            self.update_device_config(device_id, channel_bandwidth=new_bw)
        self.add_pipeline(
            device_id,
            "audio",
            PipelineConfig(stages=new_stages_tuple, audio_spec=spec),
        )

        if context.demod_profile and context.demod_profile.has_audio:
            self.start_audio_output(device_id)

    def _on_agc_gain_change(self, event: Event) -> None:
        """Handle AGC gain change request."""
        assert isinstance(event, AGCGainChangeEvent)
        if event.device_id in self.devices:
            self.update_device_config(event.device_id, rf_gain=event.rf_gain)

    def _on_recording_finished(self, event: Event) -> None:
        assert isinstance(event, RecordingFinishedEvent)
        if event.device_id in self.devices:
            self.remove_pipeline(event.device_id, event.pipeline_name)

    def _on_device_capabilities_changed(self, event: Event) -> None:
        assert isinstance(event, DeviceCapabilitiesChangedEvent)
        context = self.devices.get(event.device_id)
        if context is None:
            return
        caps = event.capabilities
        config = context.config
        changes: dict[str, Any] = {}

        if caps.gain_supported:
            lo, hi = caps.gain_range
            clamped = max(lo, min(config.rf_gain, hi))
            if clamped != config.rf_gain:
                changes["rf_gain"] = clamped
        elif config.enable_agc:
            changes["enable_agc"] = False

        freq_range = caps.frequency_range
        if freq_range is not None:
            lo, hi = freq_range
            if not (lo <= config.center_frequency <= hi):
                changes["center_frequency"] = max(lo, min(config.center_frequency, hi))

        if caps.sample_rates is not None and not any(
            abs(config.sample_rate - r) < 1.0 for r in caps.sample_rates
        ):
            changes["sample_rate"] = find_nearest(caps.sample_rates, config.sample_rate)

        if not caps.bias_tee_supported and config.bias_tee:
            changes["bias_tee"] = False

        if changes:
            logger.info("device_config_clamped device=%s changes=%r", event.device_id, changes)
            self.update_device_config(event.device_id, **changes)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown all devices and workers.

        Stops all running devices and cleans up all worker threads.
        Should be called when application is exiting.

        Args:
            timeout: Maximum time to wait per worker for graceful shutdown
        """
        logger.info("engine_shutdown_starting")

        # Stop all devices (this stops I/O and pipeline workers)
        device_ids = list(self.devices.keys())
        for device_id in device_ids:
            context = self.devices[device_id]
            if context.state == DeviceState.RUNNING:
                logger.debug("engine_stopping_device device=%s", device_id)
                try:
                    self.stop_device(device_id)
                except Exception as e:
                    logger.error(
                        "engine_stop_device_failed device=%s error=%r",
                        device_id,
                        e,
                        exc_info=True,
                    )

        # Stop any remaining audio workers (devices may not have run stop_device)
        for source_id in list(self._audio_workers.keys()):
            try:
                self.stop_audio_output(source_id, timeout=timeout)
            except (KeyError, RuntimeError) as e:
                logger.error(
                    "engine_stop_audio_failed source=%s error=%r",
                    source_id,
                    e,
                    exc_info=True,
                )

        # Final cleanup: stop any remaining workers
        logger.debug("engine_stopping_remaining_workers")
        try:
            self.worker_runner.stop_all_workers(timeout=timeout)
        except Exception as e:
            logger.error("engine_stop_workers_failed error=%r", e, exc_info=True)

        logger.info("engine_shutdown_complete")
