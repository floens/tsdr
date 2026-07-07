from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from textual.reactive import reactive

from tsdr.core.events.events import (
    AudioOutputStatsEvent,
    JitterBufferUpdateEvent,
    StatsUpdateEvent,
)
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.core.units import format_hz
from tsdr.devices import NetworkDeviceParams
from tsdr.devices.base import HasJitterBuffer
from tsdr.tui.commands._format import format_rate
from tsdr.tui.widgets.section_panel import Section, SectionPanel

if TYPE_CHECKING:
    from tsdr.devices._jitter_buffer import JitterBuffer

logger = logging.getLogger(__name__)


_STATE_COLORS = {
    DeviceState.STOPPED: "dim",
    DeviceState.STARTING: "yellow",
    DeviceState.RUNNING: "green",
    DeviceState.STOPPING: "yellow",
    DeviceState.ERROR: "red",
}

_LABEL_WIDTH = 14
_LEVEL_TC_S = 0.5  # EMA time constant for the IQ level readout


def _row(label: str, value: str) -> str:
    """A `label   value` row; callers pass bare values (no leading padding)."""
    return f"{label:<{_LABEL_WIDTH}} {value}"


def _jitter_snapshot(device_id: str, jitter: JitterBuffer) -> JitterBufferUpdateEvent:
    return JitterBufferUpdateEvent(
        device_id=device_id,
        target_seconds=float(jitter.target_seconds),
        fill_seconds=float(jitter.fill_seconds),
        fill_fraction=float(jitter.fill_fraction),
        rebuffer_count=int(jitter.rebuffer_count),
        rebuffering=bool(jitter.rebuffering),
    )


class StatsWidget(SectionPanel):
    """Display device and signal statistics.

    Signal-quality metrics (SNR, gain, squelch, clipping) live in the tuner strip,
    so this panel does not repeat them.

    Reactive props:
      focused_device_id: str | None — re-reads engine config on change.
    """

    focused_device_id: reactive[str | None] = reactive(None)

    def __init__(self):
        super().__init__()
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
        self._requested_sample_rate: float | None = None
        self._level_dbfs: float | None = None
        self._audio_underflows: int = 0
        self._audio_drops: int = 0

    def on_mount(self) -> None:
        self._read_config()

    def watch_focused_device_id(self, _device_id: str | None) -> None:
        # Reconciler set a new focused device on us; re-read engine config.
        # (Engine has already been told about the focus change via the command site.)
        self._read_config()
        self.refresh()

    def _read_config(self) -> None:
        engine = get_engine()
        device = engine.get_focused_device()
        if device is None:
            return
        if device.device_id != self._device_id:
            self._audio_underflows = 0
            self._audio_drops = 0
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
        # Jitter events are coalesced (fire only on change), so seed from the device
        # or a panel opened mid-stream shows stale zeros until the next change.
        if isinstance(device.device, HasJitterBuffer):
            self._wire_bytes_per_sec = device.device.wire_bytes_per_sec
            self._jitter = _jitter_snapshot(device.device_id, device.device.jitter)
        else:
            self._wire_bytes_per_sec = 0.0
            self._jitter = None
        self._device_state = device.state
        self._requested_sample_rate = device.config.sample_rate
        profile = device.demod_profile
        self._channel_bandwidth = profile.channel_bandwidth if profile else None

    def update_stats(self, event: StatsUpdateEvent) -> None:
        self.current_event = event
        self._update_level(event)
        self.refresh()

    def _update_level(self, event: StatsUpdateEvent) -> None:
        # dt-based alpha so the smoothing is independent of the update rate.
        if event.iq_rms is None:
            return
        level = 20.0 * math.log10(max(event.iq_rms, 1e-9))
        dt = 1.0 / (event.update_rate_fps or 20)
        alpha = math.exp(-dt / _LEVEL_TC_S)
        if self._level_dbfs is None:
            self._level_dbfs = level
        else:
            self._level_dbfs = alpha * self._level_dbfs + (1.0 - alpha) * level

    def update_jitter_buffer(self, event: JitterBufferUpdateEvent) -> None:
        # Only display state for the focused device.
        if self._device_id is not None and event.device_id != self._device_id:
            return
        self._jitter = event
        self.refresh()

    def update_audio_stats(self, event: AudioOutputStatsEvent) -> None:
        if self._device_id is not None and event.device_id != self._device_id:
            return
        self._audio_underflows = event.underflow_count
        self._audio_drops = event.drop_count
        self.refresh()

    def update_config(self) -> None:
        self._read_config()
        self.refresh()

    def build_sections(self) -> list[Section]:
        if not self._device_id:
            return [Section("[dim]Statistics: No data[/dim]")]

        sections = [Section("\n".join(self._device_lines()), min_width=22)]

        event = self.current_event
        if event is not None:
            sections.append(Section("\n".join(self._signal_lines(event)), min_width=20))
            sections.append(Section("\n".join(self._processing_lines(event)), min_width=22))
            audio = self._audio_lines(event)
            if audio:
                sections.append(Section("\n".join(audio), min_width=18))

        if self._network_address is not None:
            sections.append(Section("\n".join(self._network_lines()), min_width=22))

        return sections

    def _device_lines(self) -> list[str]:
        lines = [f"[bold]Device:[/bold] {self._device_id}"]
        if self._identity_label is not None:
            hw = f"[white]{self._identity_label}[/white]"
            if self._identity_serial is not None:
                hw += f" [dim]#{self._identity_serial}[/dim]"
            lines.append(_row("Hardware:", hw))
        if self._device_state is not None:
            color = _STATE_COLORS[self._device_state]
            lines.append(_row("State:", f"[{color}]{self._device_state.value.upper()}[/{color}]"))
        if self._frequency_range is not None:
            lo, hi = self._frequency_range
            # Locked range uses short SI ("144.46M") to fit the 42-char sidebar.
            if self._frequency_controllable:
                rng = f"{format_hz(lo, decimals=3, long_suffix=True)} – {format_hz(hi, decimals=3, long_suffix=True)}"
                lines.append(_row("Range:", f"[white]{rng}[/white]"))
            else:
                rng = f"{format_hz(lo, decimals=3)} – {format_hz(hi, decimals=3)}"
                lines.append(_row("Locked:", f"[yellow]{rng}[/yellow]"))
        if self._controller_center_frequency is not None:
            tuned = format_hz(self._controller_center_frequency, decimals=3, long_suffix=True)
            lines.append(_row("Peer tune:", f"[yellow]{tuned}[/yellow]"))
        if self._controller_gain is not None:
            lines.append(_row("Peer gain:", f"[yellow]{self._controller_gain}[/yellow]"))
        if self._network_address is not None:
            lines.append(_row("Endpoint:", f"[white]{self._network_address}[/white]"))
        return lines

    def _signal_lines(self, event: StatsUpdateEvent) -> list[str]:
        lines = ["[bold]Signal:[/bold]"]
        if self._level_dbfs is not None:
            level = self._level_dbfs
            clip = event.iq_clip_pct or 0.0
            if clip > 1.0 or level > -1.0:  # clipping / no headroom
                color = "red"
            elif level > -10.0:  # healthy ADC level
                color = "green"
            elif level > -20.0:  # low but usable
                color = "yellow"
            else:  # near silent
                color = "red"
            lines.append(_row("Level:", f"[{color}]{level:.1f}[/{color}] dBFS"))
        return lines

    def _processing_lines(self, event: StatsUpdateEvent) -> list[str]:
        lines = [
            "[bold]Processing:[/bold]",
            _row("Sample:", self._sample_rate_value(event)),
            _row("Window:", f"[white]{event.fft_window}/{event.spectrum_bins} bins[/white]"),
        ]
        if event.update_rate_fps > 0:
            lines.append(_row("Update Rate:", f"[white]{event.update_rate_fps} fps[/white]"))
        if event.spectrum_bins > 0:
            freq_res = event.sample_rate / event.spectrum_bins
            res = f"{freq_res / 1000:.2f} kHz" if freq_res >= 1000 else f"{freq_res:.1f} Hz"
            lines.append(_row("Freq Res:", f"[white]{res}[/white]"))
        return lines

    def _sample_rate_value(self, event: StatsUpdateEvent) -> str:
        act_v, act_u = format_rate(event.sample_rate)
        req = self._requested_sample_rate
        if req is None:
            return f"[white]{act_v} {act_u}[/white]"
        req_v, req_u = format_rate(req)
        color = "white" if abs(req - event.sample_rate) < 1.0 else "yellow"
        if act_u == req_u:
            return f"[{color}]{act_v} / {req_v} {act_u}[/{color}]"
        return f"[{color}]{act_v} {act_u} / {req_v} {req_u}[/{color}]"

    def _audio_lines(self, event: StatsUpdateEvent) -> list[str]:
        if event.demod_mode == "RAW":
            return []
        lines = [
            "[bold]Audio:[/bold]",
            _row("Mode:", f"[white]{event.demod_mode}[/white]"),
        ]
        if event.demod_mode == "WFM" and event.stereo is not None:
            label = "[green]STEREO[/green]" if event.stereo else "[dim]MONO[/dim]"
            lines.append(_row("Decode:", label))
        if self._channel_bandwidth is not None:
            bw_khz = self._channel_bandwidth / 1000
            lines.append(_row("Bandwidth:", f"[green]{bw_khz:.1f}[/green] kHz"))
        ur_color = "red" if self._audio_underflows > 0 else "green"
        lines.append(_row("Underruns:", f"[{ur_color}]{self._audio_underflows}[/{ur_color}]"))
        if self._audio_drops > 0:
            lines.append(_row("Drops:", f"[red]{self._audio_drops}[/red]"))
        return lines

    def _network_lines(self) -> list[str]:
        lines = ["[bold]Network:[/bold]"]
        if self._wire_bytes_per_sec > 0:
            mb_s = self._wire_bytes_per_sec / 1_000_000
            lines.append(_row("Bandwidth:", f"[white]{mb_s:.2f}[/white] MB/s"))
        else:
            lines.append(_row("Bandwidth:", "[dim]0.00[/dim] MB/s"))
        j = self._jitter
        if j is None:
            lines.append(_row("Buffer Target:", "[dim]0.00[/dim] s"))
            lines.append(_row("Fill:", "[dim]0.00[/dim] s ([dim]0%[/dim])"))
            lines.append(_row("Rebuffers:", "[dim]0[/dim]"))
            return lines
        pct = j.fill_fraction * 100
        fill_color = "red" if j.rebuffering or pct < 10 else "yellow" if pct < 50 else "green"
        rb_color = "red" if j.rebuffering else "yellow" if j.rebuffer_count > 0 else "green"
        lines.append(_row("Buffer Target:", f"[white]{j.target_seconds:.2f}[/white] s"))
        lines.append(
            _row(
                "Fill:",
                f"[{fill_color}]{j.fill_seconds:.2f}[/{fill_color}] s"
                f" ([{fill_color}]{pct:.0f}%[/{fill_color}])",
            )
        )
        lines.append(_row("Rebuffers:", f"[{rb_color}]{j.rebuffer_count}[/{rb_color}]"))
        return lines
