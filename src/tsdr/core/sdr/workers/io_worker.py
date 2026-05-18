import logging
import queue
import time
from typing import TYPE_CHECKING

from tsdr.core.events.events import (
    DeviceErrorEvent,
    JitterBufferUpdateEvent,
    SamplesDroppedEvent,
)
from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.core.tracing import span
from tsdr.core.workers import WorkerContext
from tsdr.devices.base import HasJitterBuffer

if TYPE_CHECKING:
    from tsdr.core.sdr.config import DeviceConfig
    from tsdr.core.sdr.device_context import SDRDeviceContext

logger = logging.getLogger(__name__)

# Minimum interval between physical device reconfigurations. Rapid UI input
# (e.g. tuner scroll) can produce 100+ config updates per second, clogging up
# devices.
CONFIG_APPLY_MIN_INTERVAL = 0.050

# Coalesce JitterBufferUpdateEvent so the UI doesn't redraw on every tick.
# Source-side: publish only when rebuffering flips, rebuffer_count moves,
# or fill_fraction moves by at least this much.
JITTER_FILL_DELTA = 0.05


class IOWorker:
    """I/O worker for reading from SDR hardware.

    Reads samples from the device, wraps them in SamplesBatch, and enqueues
    them on sample_queue. Applies hardware reconfiguration from control_queue.
    """

    def __init__(self, device_context: SDRDeviceContext) -> None:
        self.device_context = device_context
        self.sample_format: SampleFormat | None = None
        self._pending_config: DeviceConfig | None = None
        self._last_apply_ts: float = 0.0

        # Last-published jitter state (rebuffering, rebuffer_count,
        # fill_fraction) for source-side coalescing; None before first emit.
        self._last_jitter_state: tuple[bool, int, float] | None = None

    def setup(self, context: WorkerContext) -> None:
        device = self.device_context.device
        config = self.device_context.config

        logger.info(f"I/O worker starting for device {self.device_context.device_id}")

        try:
            logger.debug(f"Opening device {self.device_context.device_id}")
            device.open()

            logger.debug(f"Setting frequency to {config.center_frequency / 1e6:.3f} MHz")
            device.set_frequency(config.center_frequency)

            logger.debug(f"Setting sample rate to {config.sample_rate / 1e6:.3f} MHz")
            device.set_sample_rate(config.sample_rate)

            if config.auto_gain:
                logger.debug("Enabling automatic gain control")
                device.set_auto_gain(True)
            else:
                logger.debug(f"Setting RF gain to {config.rf_gain} dB")
                device.set_gain(config.rf_gain)

            if device.supports_bias_tee and config.bias_tee:
                logger.debug("Enabling bias-T")
                device.set_bias_tee(True)

            # Apply network jitter buffer pre-fill (no-op on non-network devices).
            device.set_network_buffer_seconds(config.network_buffer_seconds)

            self.sample_format = device.get_sample_format()
            logger.debug(
                f"Device {self.device_context.device_id} sample format: {self.sample_format.value}"
            )

            logger.info(f"Device {self.device_context.device_id} hardware initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize device {self.device_context.device_id}: {e}")
            context.emit_event(
                DeviceErrorEvent(
                    source_id=context.worker_id,
                    device_id=self.device_context.device_id,
                    error=str(e),
                )
            )
            raise

    def run(self, context: WorkerContext) -> None:
        """Main I/O loop.

        Reads samples from hardware and enqueues them. Handles configuration
        updates from control_queue.

        Args:
            context: Worker execution context
        """
        device = self.device_context.device
        sample_queue = self.device_context.sample_queue
        control_queue = self.device_context.control_queue
        config = self.device_context.config

        assert self.sample_format is not None, "sample_format must be set in setup()"

        while context.should_continue():
            with span("io_worker"):
                try:
                    while True:
                        self._pending_config = control_queue.get(block=False)
                except queue.Empty:
                    pass

                if (
                    self._pending_config is not None
                    and time.perf_counter() - self._last_apply_ts >= CONFIG_APPLY_MIN_INTERVAL
                ):
                    new_config = self._pending_config
                    self._pending_config = None
                    try:
                        if new_config.center_frequency != config.center_frequency:
                            device.set_frequency(new_config.center_frequency)

                        if new_config.sample_rate != config.sample_rate:
                            device.set_sample_rate(new_config.sample_rate)

                        # AGC toggle must come before rf_gain
                        if new_config.auto_gain != config.auto_gain:
                            if new_config.auto_gain:
                                device.set_auto_gain(True)
                            else:
                                device.set_gain(new_config.rf_gain)

                        if not new_config.auto_gain and new_config.rf_gain != config.rf_gain:
                            device.set_gain(new_config.rf_gain)

                        if new_config.bias_tee != config.bias_tee and device.supports_bias_tee:
                            device.set_bias_tee(new_config.bias_tee)

                        if new_config.network_buffer_seconds != config.network_buffer_seconds:
                            device.set_network_buffer_seconds(new_config.network_buffer_seconds)

                        config = new_config

                    except Exception as e:  # noqa: BLE001 - hardware config can fail in various ways
                        error_msg = f"Configuration update failed: {e}"
                        logger.warning(f"Device {self.device_context.device_id}: {error_msg}")
                        context.emit_event(
                            DeviceErrorEvent(
                                source_id=context.worker_id,
                                device_id=self.device_context.device_id,
                                error=error_msg,
                            )
                        )
                    finally:
                        self._last_apply_ts = time.perf_counter()

                with span("device_read"):
                    try:
                        read_bytes = (
                            config.effective_buffer_samples * self.sample_format.bytes_per_sample
                        )
                        raw_bytes = device.read_samples(read_bytes)
                    except DeviceError as e:
                        if not context.should_continue():
                            # Read error after request_stop is the expected
                            # shutdown path (device was interrupted to unblock
                            # us); don't surface it as an error.
                            continue
                        logger.error(f"Device {self.device_context.device_id} read error: {e}")
                        context.emit_event(
                            DeviceErrorEvent(
                                source_id=context.worker_id,
                                device_id=self.device_context.device_id,
                                error=str(e),
                            )
                        )
                        time.sleep(0.1)
                        continue
                    except Exception as e:  # noqa: BLE001 - catch-all for unexpected hardware errors
                        if not context.should_continue():
                            continue
                        error_msg = f"Unexpected read error: {e}"
                        logger.error(f"Device {self.device_context.device_id}: {error_msg}")
                        context.emit_event(
                            DeviceErrorEvent(
                                source_id=context.worker_id,
                                device_id=self.device_context.device_id,
                                error=error_msg,
                            )
                        )
                        time.sleep(0.1)
                        continue
                timestamp = time.perf_counter()

                batch = SamplesBatch(
                    raw_samples=raw_bytes,
                    sample_format=self.sample_format,
                    center_frequency=config.center_frequency,
                    sample_rate=device.actual_sample_rate,
                    rf_gain=config.rf_gain,
                    timestamp=timestamp,
                )

                with span("queue_put"):
                    try:
                        sample_queue.put(batch, timeout=0.1)
                    except queue.Full:
                        try:
                            _ = sample_queue.get(block=False)
                            sample_queue.put(batch, block=False)
                        except (queue.Empty, queue.Full):
                            pass

                        self.device_context.dropped_samples += batch.sample_count
                        context.emit_event(
                            SamplesDroppedEvent(
                                source_id=context.worker_id,
                                device_id=self.device_context.device_id,
                                count=1,
                            )
                        )

                self.device_context.total_samples_read += batch.sample_count

                self._maybe_publish_jitter_state(context)

    def _maybe_publish_jitter_state(self, context: WorkerContext) -> None:
        """Publish a JitterBufferUpdateEvent if the buffer state has shifted.

        Coalesces 20 Hz device-loop ticks into UI events that fire only on
        meaningful change: rebuffering edge, rebuffer count, or ≥5% fill
        delta. Devices without a `jitter` attribute (USB/file/mock) emit
        nothing.
        """
        device = self.device_context.device
        if not isinstance(device, HasJitterBuffer):
            return
        jitter = device.jitter

        rebuffering = bool(jitter.rebuffering)
        rebuffer_count = int(jitter.rebuffer_count)
        fill_fraction = float(jitter.fill_fraction)

        last = self._last_jitter_state
        if last is not None and (
            last[0] == rebuffering
            and last[1] == rebuffer_count
            and abs(fill_fraction - last[2]) < JITTER_FILL_DELTA
        ):
            return
        self._last_jitter_state = (rebuffering, rebuffer_count, fill_fraction)

        context.emit_event(
            JitterBufferUpdateEvent(
                source_id=context.worker_id,
                device_id=self.device_context.device_id,
                target_seconds=float(jitter.target_seconds),
                fill_seconds=float(jitter.fill_seconds),
                fill_fraction=fill_fraction,
                rebuffer_count=rebuffer_count,
                rebuffering=rebuffering,
            )
        )

    def teardown(self, context: WorkerContext) -> None:
        try:
            self.device_context.device.close()
            logger.info(f"Device {self.device_context.device_id} closed successfully")
        except Exception as e:  # noqa: BLE001 - cleanup must not fail
            logger.warning(
                f"Error closing device {self.device_context.device_id}: {e}", exc_info=True
            )
