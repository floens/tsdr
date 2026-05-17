"""Manual jitter-injection TCP proxy.

Forwards a single TCP connection from a local port to an upstream host:port
while randomly delaying or batching bytes — useful for stress-testing the
network jitter buffer without OS-level traffic shaping (which is
platform-specific: tc on Linux, Network Link Conditioner on macOS).

Usage:
    python tests/manual/tcp_jitter_proxy.py \\
        --listen 1234 --upstream localhost:1234 \\
        --max-delay 0.5 --stall-prob 0.05

Then add the device against the proxy port:
    sdr add rtl0 --type rtltcp --host localhost --port 1234

Not part of the automated test suite.
"""

import argparse
import random
import socket
import sys
import threading
import time


def _pipe(
    src: socket.socket,
    dst: socket.socket,
    *,
    max_delay: float,
    stall_prob: float,
    stall_min: float,
    stall_max: float,
    label: str,
) -> None:
    """Read from src, sleep a random amount, write to dst. Closes both on EOF."""
    rng = random.Random()
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            # Small jitter on every chunk.
            if max_delay > 0:
                time.sleep(rng.uniform(0, max_delay))
            # Occasional long stall.
            if stall_prob > 0 and rng.random() < stall_prob:
                dur = rng.uniform(stall_min, stall_max)
                print(f"[{label}] STALL {dur:.2f}s", file=sys.stderr)
                time.sleep(dur)
            dst.sendall(chunk)
    except OSError as e:
        print(f"[{label}] error: {e}", file=sys.stderr)
    finally:
        try:
            dst.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _serve(args: argparse.Namespace) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", args.listen))
    listener.listen(1)
    up_host, up_port_s = args.upstream.split(":")
    up_port = int(up_port_s)
    print(
        f"Listening on :{args.listen} → forwarding to {up_host}:{up_port}; "
        f"max_delay={args.max_delay}s, stall_prob={args.stall_prob}, "
        f"stall={args.stall_min}–{args.stall_max}s",
        file=sys.stderr,
    )

    while True:
        client, addr = listener.accept()
        print(f"Client connected from {addr}", file=sys.stderr)
        upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            upstream.connect((up_host, up_port))
        except OSError as e:
            print(f"Upstream connect failed: {e}", file=sys.stderr)
            client.close()
            continue

        # Two pipes: downstream and upstream directions. Downstream gets the
        # heavy jitter (it's the IQ stream); upstream commands are tiny and
        # forwarded without delay.
        t1 = threading.Thread(
            target=_pipe,
            args=(upstream, client),
            kwargs={
                "max_delay": args.max_delay,
                "stall_prob": args.stall_prob,
                "stall_min": args.stall_min,
                "stall_max": args.stall_max,
                "label": "→client",
            },
            daemon=True,
        )
        t2 = threading.Thread(
            target=_pipe,
            args=(client, upstream),
            kwargs={
                "max_delay": 0.0,
                "stall_prob": 0.0,
                "stall_min": 0.0,
                "stall_max": 0.0,
                "label": "→server",
            },
            daemon=True,
        )
        t1.start()
        t2.start()
        # Wait for either direction to close, then tear both down.
        t1.join()
        t2.join()
        client.close()
        upstream.close()
        print("Connection closed", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=int, required=True, help="Local port to listen on")
    parser.add_argument(
        "--upstream",
        required=True,
        help="Upstream host:port (e.g. localhost:1234 for an rtl_tcp server)",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.05,
        help="Maximum per-chunk delay in seconds (jitter floor)",
    )
    parser.add_argument(
        "--stall-prob",
        type=float,
        default=0.02,
        help="Probability per chunk of a long stall (0.0 disables)",
    )
    parser.add_argument(
        "--stall-min", type=float, default=0.3, help="Minimum stall duration in seconds"
    )
    parser.add_argument(
        "--stall-max", type=float, default=2.0, help="Maximum stall duration in seconds"
    )
    args = parser.parse_args()
    try:
        _serve(args)
    except KeyboardInterrupt:
        print("\nShutting down", file=sys.stderr)


if __name__ == "__main__":
    main()
