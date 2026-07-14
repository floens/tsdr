"""Protocol-layer tests for the KiwiSDR device.

Every frame is hand-built; nothing here touches a socket or a live server. The
byte layouts were verified against the KiwiSDR source (``~/git/KiwiSDR``) and a
live public receiver during development, but no captures or server URLs are
committed.
"""

import itertools
import struct

import numpy as np
import pytest
from websocket import WebSocketConnectionClosedException

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.devices import KiwiSDRDevice, KiwiSDRParams, create_device
from tsdr.devices import kiwisdr as k


def _snd_frame(
    *,
    seq: int = 1,
    smeter_dbm: float = -78.0,
    iq_pairs: list[tuple[float, float]] | None = None,
    flags: int = k.SND_FLAG_STEREO | k.SND_FLAG_LITTLE_ENDIAN,
) -> bytes:
    """Build a binary SND packet. STEREO flag -> 20-byte header."""
    header = b"SND" + bytes([flags]) + struct.pack("<I", seq)
    header += struct.pack(">H", int(round((smeter_dbm + 127.0) * 10)))
    if flags & k.SND_FLAG_STEREO:
        header += bytes([255, 0]) + struct.pack("<I", 0) + struct.pack("<I", 0)
    pairs = iq_pairs if iq_pairs is not None else [(0.5, -0.25)]
    body = b"".join(struct.pack("<hh", int(i * 32768), int(q * 32768)) for i, q in pairs)
    return header + body


def _wf_frame(
    *, byte_vals: list[int], x_bin: int = 0, zoom: int = 0, seq: int = 0, wf_flags: int = 0
) -> bytes:
    header = b"W/F " + struct.pack("<III", x_bin, (wf_flags << 16) | zoom, seq)
    return header + bytes(byte_vals)


def _msg(text: str) -> bytes:
    return b"MSG " + text.encode()


def test_decode_snd_iq_ordering_and_scale():
    pkt = k.decode_snd(_snd_frame(seq=7, smeter_dbm=-78.0, iq_pairs=[(0.5, -0.25), (-0.1, 0.2)]))
    assert pkt.seq == 7  # u32 LE
    assert abs(pkt.smeter_dbm + 78.0) < 0.05  # u16 BE
    iq = np.frombuffer(pkt.iq_bytes, dtype=np.complex64)
    assert iq.shape == (2,)
    assert abs(iq[0].real - 0.5) < 1e-3
    assert abs(iq[0].imag + 0.25) < 1e-3


def test_decode_snd_header_length_from_flag_byte():
    # STEREO set -> 20-byte header.
    iq = np.frombuffer(k.decode_snd(_snd_frame(iq_pairs=[(0.1, 0.1)])).iq_bytes, dtype=np.complex64)
    assert iq.size == 1
    # STEREO clear -> 10-byte header, one s16 per sample.
    mono = b"SND" + bytes([0x00]) + struct.pack("<I", 3) + struct.pack(">H", 1270)
    mono += struct.pack("<hh", 100, 200)
    assert np.frombuffer(k.decode_snd(mono).iq_bytes, dtype=np.complex64).size == 1


def test_decode_snd_restart_flag_exposed():
    pkt = k.decode_snd(_snd_frame(flags=k.SND_FLAG_STEREO | k.SND_FLAG_RESTART))
    assert pkt.flags & k.SND_FLAG_RESTART


def test_decode_snd_rejects_short_and_non_snd():
    with pytest.raises(DeviceError):
        k.decode_snd(b"SND\x08")
    with pytest.raises(DeviceError):
        k.decode_snd(b"MSG hello")


def test_decode_wf_db_decode_and_floor():
    wf = k.decode_wf(
        _wf_frame(byte_vals=[255, 55, 155, 100] + [200] * 1020), wf_fft_size=1024, wf_cal=-13
    )
    assert wf.db_bins.size == 1024
    assert wf.db_bins.dtype == np.float32
    assert np.all(wf.db_bins[:4] == -200.0)
    # byte 200 -> -(255-200) + (-13) = -68 dBm.
    assert abs(float(wf.db_bins[4]) - (-68.0)) < 0.01


def test_decode_wf_bin_count_from_argument_not_hardcoded():
    wf = k.decode_wf(_wf_frame(byte_vals=[128] * 512), wf_fft_size=512, wf_cal=0)
    assert wf.db_bins.size == 512


def test_decode_wf_refuses_compression():
    with pytest.raises(DeviceError, match="compress"):
        k.decode_wf(_wf_frame(byte_vals=[0] * 1024, wf_flags=0x0001), wf_fft_size=1024, wf_cal=0)


def test_decode_wf_rejects_short():
    with pytest.raises(DeviceError):
        k.decode_wf(b"W/F " + struct.pack("<III", 0, 0, 0) + bytes(10), wf_fft_size=1024, wf_cal=0)


def test_parse_msg_multi_token_and_uri_decode():
    toks = k.parse_msg(_msg("center_freq=15000000 bandwidth=30000000 ident=a%20b wf_setup"))
    assert toks["center_freq"] == "15000000"
    assert toks["bandwidth"] == "30000000"
    assert toks["ident"] == "a b"
    assert toks["wf_setup"] == ""


def test_parse_msg_uri_encoded_blob_has_no_spaces():
    # load_cfg is a URI-encoded JSON blob; a plain split must not choke.
    toks = k.parse_msg(_msg("load_cfg=%7b%22a%22%3a1%7d badp=0"))
    assert toks["badp"] == "0"
    assert toks["load_cfg"].startswith("{")


_SND_SETUP = [
    "sample_rate=11998.979406",
    "rx_chans=8 chan_no_pwd=0",
    "badp=0",
    "load_cfg=%7b%22x%22%3a1%7d",
    "cfg_loaded",
    "center_freq=15000000 bandwidth=30000000 adc_clk_nom=66666600",
    "audio_init=0 audio_rate=12000",
]


def _collect(order: list[str], ready):
    frames = iter(_msg(s) for s in order)
    return k.collect_handshake(lambda: next(frames), ready, deadline_s=5.0, clock=lambda: 0.0)


@pytest.mark.parametrize(
    "order",
    [
        _SND_SETUP,
        list(reversed(_SND_SETUP)),
        [_SND_SETUP[6], _SND_SETUP[5], _SND_SETUP[2], _SND_SETUP[0]],
    ],
)
def test_collect_handshake_is_order_agnostic(order):
    acc = _collect(order, k._snd_ready)
    assert acc["badp"] == "0"
    assert acc["audio_rate"] == "12000"
    assert acc["bandwidth"] == "30000000"


def test_collect_handshake_waits_for_all_required_keys():
    clock = itertools.count(0.0, 1.0).__next__
    with pytest.raises(DeviceError, match="timed out"):
        k.collect_handshake(
            lambda: _msg("badp=0 bandwidth=30000000"), k._snd_ready, deadline_s=3.0, clock=clock
        )


def test_collect_handshake_wf_ready_on_setup_token():
    acc = _collect(
        ["badp=0", "wf_fft_size=1024 wf_cal=-13 wf_chans=3 zoom_cap=11 wf_setup"], k._wf_ready
    )
    assert acc["wf_fft_size"] == "1024"
    assert acc["wf_cal"] == "-13"


@pytest.mark.parametrize(
    ("token", "match"),
    [
        ("badp=1", "wrong or missing password"),
        ("badp=5", "duplicate IP"),
        ("too_busy=8", "busy"),
        ("redirect=http://other", "redirect"),
        ("wb_only", "wideband"),
        ("ip_limit=1440,1.2.3.4", "time limit"),
        ("down=1 reason_disabled=maintenance", "maintenance"),
    ],
)
def test_collect_handshake_error_paths(token, match):
    frames = iter([_msg(token)])
    with pytest.raises(DeviceError, match=match):
        k.collect_handshake(lambda: next(frames), k._snd_ready, deadline_s=5.0, clock=lambda: 0.0)


def test_collect_handshake_ignores_non_msg_frames():
    frames = iter([_snd_frame(), _msg("badp=0 audio_rate=12000 bandwidth=30000000")])
    acc = k.collect_handshake(lambda: next(frames), k._snd_ready, deadline_s=5.0, clock=lambda: 0.0)
    assert acc["audio_rate"] == "12000"


def test_format_auth_empty_is_hash():
    assert k.format_auth("") == "SET auth t=kiwi p=#"


def test_format_auth_encodes_password():
    assert k.format_auth("p@ss w/d") == "SET auth t=kiwi p=p%40ss%20w%2Fd"


def test_format_tune_khz_and_passband():
    cmd = k.format_tune(10_000_000)
    assert "freq=10000.000" in cmd
    assert "low_cut=-6000 high_cut=6000" in cmd
    assert cmd.startswith("SET mod=iq")


def test_snd_gating_satisfies_cmd_snd_all():
    cmds = " || ".join(k.snd_gating_commands(12000, 15_000_000, "tsdr"))
    assert "AR OK in=12000" in cmds  # CMD_AR_OK
    assert "mod=iq" in cmds and "freq=15000.000" in cmds  # CMD_FREQ|MODE|PASSBAND
    assert "agc=0" in cmds  # CMD_AGC
    assert "little-endian" in cmds
    assert "ident_user=tsdr" in cmds


def test_wf_gating_satisfies_cmd_wf_all():
    cmds = k.wf_gating_commands()
    joined = " || ".join(cmds)
    assert "zoom=0 start=0" in joined  # CMD_ZOOM|START
    assert "maxdb=0 mindb=-100" in joined  # CMD_DB
    assert "wf_speed=4" in joined  # CMD_SPEED
    assert "wf_comp=0" in joined


def test_resolve_endpoint_follows_redirect_keeps_port(monkeypatch):
    calls = []

    def fake_status(host, port, timeout):
        calls.append((host, port))
        if host == "21671.proxy.kiwisdr.com":
            return 307, "http://21671.proxy2.kiwisdr.com/status"
        return 200, None

    monkeypatch.setattr(k, "_http_status", fake_status)
    host, port = k.resolve_endpoint("21671.proxy.kiwisdr.com", 8073)
    assert (host, port) == ("21671.proxy2.kiwisdr.com", 8073)


def test_resolve_endpoint_direct_unchanged(monkeypatch):
    monkeypatch.setattr(k, "_http_status", lambda h, p, t: (200, None))
    assert k.resolve_endpoint("kiwi.example.com", 8073) == ("kiwi.example.com", 8073)


def test_resolve_endpoint_unreachable_falls_back(monkeypatch):
    monkeypatch.setattr(k, "_http_status", lambda h, p, t: (None, None))
    assert k.resolve_endpoint("dead.example.com", 8073) == ("dead.example.com", 8073)


def _device() -> KiwiSDRDevice:
    return KiwiSDRDevice(host="h", port=8073)


def test_snd_setup_uses_exact_sample_rate_and_range():
    dev = _device()
    dev._apply_snd_setup(
        {
            "audio_rate": "12000",
            "bandwidth": "30000000",
            "center_freq": "15000000",
            "sample_rate": "11998.979406",
            "freq_offset": "0.000",
        }
    )
    dev._rebuild_capabilities()
    assert dev.actual_sample_rate == pytest.approx(11998.979406)
    caps = dev.capabilities
    assert caps.frequency_range == (0.0, 30_000_000.0)
    assert caps.sample_rates == (pytest.approx(11998.979406),)
    assert caps.frequency_controllable is True
    assert caps.gain_supported is False
    assert caps.gain_unit == "dB"


def test_snd_setup_falls_back_to_nominal_rate():
    dev = _device()
    dev._apply_snd_setup({"audio_rate": "12000", "bandwidth": "30000000"})
    assert dev.actual_sample_rate == 12000.0


def test_snd_setup_applies_freq_offset():
    dev = _device()
    dev._apply_snd_setup({"audio_rate": "12000", "bandwidth": "30000000", "freq_offset": "125.000"})
    dev._rebuild_capabilities()
    assert dev.capabilities.frequency_range == (125_000.0, 30_125_000.0)


def test_wf_setup_reads_cal_and_detects_no_wf():
    dev = _device()
    dev._apply_wf_setup(
        {
            "wf_fft_size": "1024",
            "wf_cal": "-13",
            "wf_chans": "3",
            "zoom_cap": "11",
            "wf_share": "1",
            "wf_setup": "",
        }
    )
    assert dev._wf_fft_size == 1024
    assert dev._wf_cal == -13
    assert dev._wf_chans == 3
    assert dev._zoom_cap == 11

    dev.close()  # nothing started; must be safe
    nowf = _device()
    nowf._apply_wf_setup({"wf_fft_size": "1024", "wf_cal": "0", "wf_chans": "0", "wf_setup": ""})
    assert nowf._wf_chans == 0


def test_send_wraps_socket_close_as_device_error():
    # A concurrent close() during rapid stop/start can kill the socket mid-send;
    # _send must surface DeviceError, not a raw websocket exception (which would
    # escape set_frequency and crash the io worker).
    dev = _device()

    class _DeadWS:
        def send(self, cmd: str) -> None:
            raise WebSocketConnectionClosedException("socket is already closed.")

    dev._snd_ws = _DeadWS()  # type: ignore[assignment]
    with pytest.raises(DeviceError):
        dev._send(dev._snd_ws, dev._snd_send_lock, "SET keepalive")


def test_factory_creates_kiwisdr_device():
    dev = create_device(KiwiSDRParams(host="kiwi", port=8073, password="pw", user="me"))
    assert isinstance(dev, KiwiSDRDevice)
    assert dev.host == "kiwi" and dev.password == "pw" and dev.user == "me"
    assert dev.get_sample_format().value == "complex64"


def test_factory_rejects_bad_port():
    with pytest.raises(ValueError, match="Port"):
        create_device(KiwiSDRParams(host="kiwi", port=0))


def test_params_describe():
    assert KiwiSDRParams(host="kiwi", port=8073).describe() == "kiwi:8073"


def test_format_wf_view():
    assert k.format_wf_view(9, 7_100_000.0) == "SET zoom=9 cf=7100.000"
    assert k.format_wf_view(0, 15_000_000.0) == "SET zoom=0 cf=15000.000"


def test_covering_zoom_boundaries():
    bw = 30e6
    assert k.covering_zoom(bw, 30e6, 14) == 0
    assert k.covering_zoom(bw, 15e6, 14) == 1
    # Span between two levels picks the wider (covering) one.
    assert k.covering_zoom(bw, 10e6, 14) == 1
    assert k.covering_zoom(bw, 1.0, 14) == 14
    assert k.covering_zoom(bw, 1.0, 11) == 11  # wf_share cap


def test_wf_frame_geometry_zoom0_full_band():
    center, span = k.wf_frame_geometry(0, 0, 30e6, 1024, 0.0)
    assert span == 30e6
    assert center == 15e6


def test_wf_frame_geometry_zoomed():
    # zoom 1 halves the span; start bin walks in bandwidth/(1024<<14) steps.
    hz_per_start = 30e6 / (1024 << 14)
    center, span = k.wf_frame_geometry(100, 1, 30e6, 1024, 0.0)
    assert span == 15e6
    assert center == 100 * hz_per_start + 7.5e6


def test_wf_frame_geometry_applies_freq_offset():
    center, span = k.wf_frame_geometry(0, 0, 30e6, 1024, 100e6)
    assert center == 100e6 + 15e6


def test_wf_frames_queue_and_drain():
    dev = _device()
    dev._bandwidth_hz = 30e6
    dev._wf_fft_size = 1024
    dev._handle_wf_frame(_wf_frame(byte_vals=[100] * 1024, seq=1))
    dev._handle_wf_frame(_wf_frame(byte_vals=[110] * 1024, seq=2, zoom=1))

    frames = dev.drain_spectrum_frames()
    assert [f.seq for f in frames] == [1, 2]
    assert frames[0].span_hz == 30e6
    assert frames[0].center_hz == 15e6
    assert frames[1].span_hz == 15e6
    assert dev.drain_spectrum_frames() == []


def test_wf_frame_queue_drops_oldest_on_overflow():
    dev = _device()
    for seq in range(k._WF_QUEUE_FRAMES + 10):
        dev._handle_wf_frame(_wf_frame(byte_vals=[100] * 1024, seq=seq))
    frames = dev.drain_spectrum_frames()
    assert len(frames) == k._WF_QUEUE_FRAMES
    assert frames[0].seq == 10
    assert frames[-1].seq == k._WF_QUEUE_FRAMES + 9


class _RecordingWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, cmd: str) -> None:
        self.sent.append(cmd)


def test_set_spectrum_view_sends_covering_zoom():
    dev = _device()
    dev._wf_ws = _RecordingWS()  # type: ignore[assignment]
    dev._bandwidth_hz = 30e6
    dev._zoom_cap = 14
    dev.set_spectrum_view(7.1e6, 100e3)
    # 30e6 / 2^8 = 117 kHz covers a 100 kHz view; 2^9 = 58.6 kHz would not.
    assert dev._wf_ws.sent == ["SET zoom=8 cf=7100.000"]


def test_set_spectrum_view_cf_is_baseband():
    dev = _device()
    dev._wf_ws = _RecordingWS()  # type: ignore[assignment]
    dev._bandwidth_hz = 30e6
    dev._freq_offset_hz = 100e6
    dev.set_spectrum_view(107.1e6, 30e6)
    assert dev._wf_ws.sent == ["SET zoom=0 cf=7100.000"]


def test_set_spectrum_view_noop_before_open():
    dev = _device()
    dev.set_spectrum_view(7.1e6, 100e3)  # must not raise


def test_provides_spectrum_capability():
    dev = _device()
    dev._apply_wf_setup({"wf_fft_size": "1024", "wf_chans": "2"})
    dev._rebuild_capabilities()
    assert dev.capabilities.provides_spectrum is True

    nowf = _device()
    nowf._apply_wf_setup({"wf_fft_size": "1024", "wf_chans": "0"})
    nowf._rebuild_capabilities()
    assert nowf.capabilities.provides_spectrum is False


def test_spectrum_view_status_tracks_sent_and_frames():
    dev = _device()
    dev._wf_ws = _RecordingWS()  # type: ignore[assignment]
    dev._bandwidth_hz = 30e6

    assert dev.spectrum_view_status() is None  # nothing sent yet

    dev.set_spectrum_view(7.1e6, 100e3)
    status = dev.spectrum_view_status()
    assert status is not None
    assert (status.requested_zoom, status.requested_center_hz) == (8, 7.1e6)
    assert status.zoom_cap == 14
    assert status.frame_zoom is None  # no frame yet

    dev._handle_wf_frame(_wf_frame(byte_vals=[100] * 1024, zoom=8, x_bin=0))
    status = dev.spectrum_view_status()
    assert status is not None
    assert status.frame_zoom == 8
    assert status.frame_span_hz == 30e6 / 256
    assert status.frame_bins == 1024


def test_wf_interp_matches_kiwi_web_default():
    """Fold mode must stay DROP + CIC comp (13), the KiwiSDR web client's own
    default (openwebrx.js:258). The enum is MAX=0 MIN=1 LAST=2 DROP=3 CMA=4,
    +10 for CIC comp (rx_waterfall.h:207-208) — 13 is NOT "MAX + comp"; a
    past comment mislabeled it. Trace smoothing belongs client-side."""
    interp = next(c for c in k.wf_gating_commands() if c.startswith("SET interp="))
    value = int(interp.removeprefix("SET interp="))
    assert value == 13
    assert value - k._WF_INTERP_CIC_COMP == k._WF_INTERP_DROP


def test_wf_setup_parses_expected_fps():
    dev = _device()
    dev._apply_wf_setup({"wf_fft_size": "1024", "wf_fps": "23", "wf_chans": "2"})
    assert dev._wf_fps_expected == 23.0


def test_spectrum_view_status_carries_frame_rate():
    dev = _device()
    dev._wf_ws = _RecordingWS()  # type: ignore[assignment]
    dev._bandwidth_hz = 30e6
    dev.set_spectrum_view(7.1e6, 100e3)

    status = dev.spectrum_view_status()
    assert status is not None
    assert status.expected_fps == 23.0
    assert status.measured_fps is None  # no full measurement interval yet

    dev._wf_fps_measured = 9.7
    status = dev.spectrum_view_status()
    assert status is not None
    assert status.measured_fps == 9.7
