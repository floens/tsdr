import socket

from tsdr.core.directory.model import PublicDevice
from tsdr.core.directory.probe import parse_kiwi_status, probe_device, tcp_probe


def test_tcp_probe_reachable() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()
    try:
        rtt = tcp_probe(host, port, timeout=2.0)
    finally:
        server.close()
    assert rtt is not None
    assert rtt >= 0.0


def test_tcp_probe_down() -> None:
    # Bind then close to get a port nothing is listening on.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    assert tcp_probe("127.0.0.1", port, timeout=1.0) is None


def test_probe_device_spyserver_reachable() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()
    device = PublicDevice(source="spyserver", id="s", name="x", host=host, port=port, usable=True)
    try:
        result = probe_device(device, timeout=2.0)
    finally:
        server.close()
    assert result.reachable
    assert result.rtt_ms is not None


def test_parse_kiwi_status_active() -> None:
    body = b"status=active\noffline=no\nusers=2\nusers_max=8\nname=Test Kiwi\n"
    result = parse_kiwi_status(body)
    assert result.reachable
    assert result.active is True
    assert (result.users, result.users_max) == (2, 8)


def test_parse_kiwi_status_offline() -> None:
    result = parse_kiwi_status(b"status=active\noffline=yes\nusers=0\nusers_max=4\n")
    assert result.reachable  # HTTP responded
    assert result.active is False


def test_parse_kiwi_status_garbage() -> None:
    result = parse_kiwi_status(b"<html>not a kiwi</html>")
    assert result.reachable  # a 200 body still counts as reachable
    assert result.active is False
    assert result.users is None
