import logging

from textual.reactive import reactive
from textual.widgets import Static

from tsdr.core.events.events import (
    JitterBufferUpdateEvent,
    StatsUpdateEvent,
)
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.units import format_hz
from tsdr.devices import NetworkDeviceParams
from tsdr.devices.base import HasJitterBuffer
from tsdr.tui.widgets.panel import PanelWidget

logger = logging.getLogger(__name__)


_STATE_COLORS = {
    DeviceState.STOPPED: "dim",
    DeviceState.STARTING: "yellow",
    DeviceState.RUNNING: "green",
    DeviceState.STOPPING: "yellow",
    DeviceState.ERROR: "red",
}


class StatsWidget(Static, PanelWidget):
    """Display device and signal statistics.

    Reactive props:
      focused_device_id: str | None — re-reads engine config on change.
    """

    focused_device_id: reactive[str | None] = reactive(None)

    def __init__(self):
        super().__init__("Statistics: No data")
        self.current_event: StatsUpdateEvent | None = None
        self._channel_bandwidth: float | None = None
        self._device_id: str | None = None
        self._jitter: JitterBufferUpdateEvent | None = None
        self._network_address: str | None = None
        self._identity_label: str | None = None
        self._identity_serial: str | None = None
        self._frequency_range: tuple[float, float] | None = None
        self._frequency_controllable: bool = False
        self._controller_center_frequency: float | None = None
        self._controller_gain: int | None = None
        self._wire_bytes_per_sec: float = 0.0
        self._device_state: DeviceState | None = None

    def on_mount(self) -> None:
        self._read_config()

    def watch_focused_device_id(self, _device_id: str | None) -> None:
        # Reconciler set a new focused device on us; re-read engine config.
        # (Engine has already been told about the focus change via the command site.)
        self._read_config()
        self._update()

    def _read_config(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return
        if device.device_id != self._device_id:
            self._jitter = None
        self._device_id = device.device_id
        if isinstance(device.params, NetworkDeviceParams):
            self._network_address = f"{device.params.host}:{device.params.port}"
        else:
            self._network_address = None
        identity = device.device.identity
        self._identity_label = identity.type_label
        self._identity_serial = identity.serial
        caps = device.device.capabilities
        self._frequency_range = caps.frequency_range
        self._frequency_controllable = caps.frequency_controllable
        self._controller_center_frequency = caps.controller_center_frequency
        self._controller_gain = caps.controller_gain
        self._wire_bytes_per_sec = (
            device.device.wire_bytes_per_sec if isinstance(device.device, HasJitterBuffer) else 0.0
        )
        self._device_state = device.state
        profile = device.demod_profile
        self._channel_bandwidth = profile.channel_bandwidth if profile else None

    def update_stats(self, event: StatsUpdateEvent) -> None:
        self.current_event = event
        self._update()

    def update_jitter_buffer(self, event: JitterBufferUpdateEvent) -> None:
        # Only display state for the focused device.
        if self._device_id is not None and event.device_id != self._device_id:
            return
        self._jitter = event
        self._update()

    def update_config(self) -> None:
        self._read_config()
        self._update()

    def _update(self):
        self.update("\n".join(self._render_stats()))

    def _render_stats(self) -> list[str]:
        """Render statistics as formatted text with Rich markup."""
        if not self._device_id:
            return ["[dim]Statistics: No data[/dim]"]

        lines = []

        lines.append(f"[bold]Device:[/bold] {self._device_id}")
        if self._identity_label is not None:
            id_line = f"  Hardware:      [white]{self._identity_label}[/white]"
            if self._identity_serial is not None:
                id_line += f" [dim]#{self._identity_serial}[/dim]"
            lines.append(id_line)
        if self._device_state is not None:
            state_color = _STATE_COLORS[self._device_state]
            lines.append(
                f"  State:         [{state_color}]{self._device_state.value.upper()}[/{state_color}]"
            )
        if self._frequency_range is not None:
            lo, hi = self._frequency_range
            # Locked range uses short SI ("144.46M") to fit the 42-char sidebar.
            if self._frequency_controllable:
                range_str = f"{format_hz(lo, decimals=3, long_suffix=True)} – {format_hz(hi, decimals=3, long_suffix=True)}"
                lines.append(f"  {'Range:':<15}[white]{range_str}[/white]")
            else:
                range_str = f"{format_hz(lo, decimals=3)} – {format_hz(hi, decimals=3)}"
                lines.append(f"  [yellow]{'Locked:':<15}{range_str}[/yellow]")
        if self._controller_center_frequency is not None:
            tuned_str = format_hz(self._controller_center_frequency, decimals=3, long_suffix=True)
            lines.append(f"  {'Peer tune:':<15}[yellow]{tuned_str}[/yellow]")
        if self._controller_gain is not None:
            lines.append(f"  {'Peer gain:':<15}[yellow]{self._controller_gain}[/yellow]")
        if self._network_address is not None:
            lines.append(f"  Endpoint:      [white]{self._network_address}[/white]")

        event = self.current_event
        if event is not None:
            self._render_event_sections(lines, event)

        if self._network_address is not None:
            self._render_network_section(lines)

        return lines

    def _render_event_sections(self, lines: list[str], event: StatsUpdateEvent) -> None:
        lines.append("[bold cyan]Signal:[/bold cyan]")
        # IQ amplitude (shows ADC utilization and clipping)
        if event.iq_rms is not None:
            # Color-code: green if good (>0.3), yellow if low, red if very low or clipping
            clip = event.iq_clip_pct or 0.0
            if clip > 1.0:
                iq_color = "red"
            elif event.iq_rms > 0.3:
                iq_color = "green"
            elif event.iq_rms > 0.1:
                iq_color = "yellow"
            else:
                iq_color = "red"
            lines.append(
                f"  IQ Amplitude:  [{iq_color}]{event.iq_rms:>5.3f}[/{iq_color}]/"
                f"[{iq_color}]{event.iq_peak:>5.3f}[/{iq_color}] rms/peak"
            )

        lines.append("[bold cyan]Processing:[/bold cyan]")
        lines.append(
            f"  Window:        [white]{event.fft_window}/{event.spectrum_bins} bins[/white]"
        )
        if event.update_rate_fps > 0:
            lines.append(f"  Update Rate:   [white]{event.update_rate_fps:>5} fps[/white]")

        if event.spectrum_bins > 0:
            freq_res = event.sample_rate / event.spectrum_bins
            if freq_res >= 1000:
                freq_res_str = f"{freq_res / 1000:.2f} kHz"
            else:
                freq_res_str = f"{freq_res:.1f} Hz"
            lines.append(f"  Freq Res:      [white]{freq_res_str:>8}[/white]")

        if event.demod_mode != "RAW":
            lines.append("[bold cyan]Audio:[/bold cyan]")
            lines.append(f"  Mode:          [white]{event.demod_mode:>8}[/white]")
            if event.demod_mode == "WFM" and event.stereo is not None:
                label = "[green]STEREO[/green]" if event.stereo else "[dim]MONO[/dim]"
                lines.append(f"  Decode:        {label:>8}")
            if self._channel_bandwidth is not None:
                bw_khz = self._channel_bandwidth / 1000
                lines.append(f"  Bandwidth:     [green]{bw_khz:>7.1f}[/green] kHz")

        lines.append("[bold cyan]Queue:[/bold cyan]")
        queue_util = (event.queue_size / max(event.queue_capacity, 1)) * 100
        if queue_util < 50:
            util_color = "green"
        elif queue_util < 80:
            util_color = "yellow"
        else:
            util_color = "red"
        lines.append(f"  Size:          {event.queue_size:>4} / {event.queue_capacity:>4}")
        lines.append(f"  Utilization:   [{util_color}]{queue_util:>7.1f}[/{util_color}] %")

    def _render_network_section(self, lines: list[str]) -> None:
        # JitterBufferUpdateEvent is coalesced source-side and may not arrive
        # for a moment after mount; render dim zeros until the first one lands
        # so the placeholder isn't mistaken for a buffer-underrun alarm.
        lines.append("[bold cyan]Network:[/bold cyan]")
        if self._wire_bytes_per_sec > 0:
            mb_s = self._wire_bytes_per_sec / 1_000_000
            lines.append(f"  Bandwidth:     [white]{mb_s:>5.2f}[/white] MB/s")
        else:
            lines.append("  Bandwidth:     [dim] 0.00[/dim] MB/s")
        j = self._jitter
        if j is None:
            lines.append("  Buffer Target: [dim] 0.00[/dim] s")
            lines.append("  Fill:          [dim] 0.00[/dim] s  ([dim]  0%[/dim])")
            lines.append("  Rebuffers:     [dim]   0[/dim]")
            return
        pct = j.fill_fraction * 100
        fill_color = "red" if j.rebuffering or pct < 10 else "yellow" if pct < 50 else "green"
        rb_color = "red" if j.rebuffering else "yellow" if j.rebuffer_count > 0 else "green"
        lines.append(f"  Buffer Target: [white]{j.target_seconds:>5.2f}[/white] s")
        lines.append(
            f"  Fill:          [{fill_color}]{j.fill_seconds:>5.2f}[/{fill_color}] s"
            f"  ([{fill_color}]{pct:>3.0f}%[/{fill_color}])"
        )
        lines.append(f"  Rebuffers:     [{rb_color}]{j.rebuffer_count:>4}[/{rb_color}]")
