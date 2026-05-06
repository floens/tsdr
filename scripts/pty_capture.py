#!/usr/bin/env python3
"""PTY proxy that captures all terminal output for debugging kitty graphics issues.

Usage:
    # Capture mode:
    python scripts/pty_capture.py [-o capture.log] -- uv run tsdr

    # Analysis mode:
    python scripts/pty_capture.py --analyze capture.log
"""

import argparse
import fcntl
import os
import pty
import re
import select
import signal
import sys
import termios
import time

# Escape sequence markers (as bytes)
BSU = b"\x1b[?2026h"
ESU = b"\x1b[?2026l"
APC_START = b"\x1b_G"
APC_END = b"\x1b\\"


def annotate(data: bytes) -> str:
    """Scan byte stream and label escape sequences."""
    parts: list[str] = []
    i = 0
    text_run = 0

    def flush_text():
        nonlocal text_run
        if text_run > 0:
            parts.append(f"text({text_run}B)")
            text_run = 0

    while i < len(data):
        # Check for BSU
        if data[i : i + len(BSU)] == BSU:
            flush_text()
            parts.append("[BSU]")
            i += len(BSU)
            continue

        # Check for ESU
        if data[i : i + len(ESU)] == ESU:
            flush_text()
            parts.append("[ESU]")
            i += len(ESU)
            continue

        # Check for Kitty APC: \x1b_G ... \x1b\\
        if data[i : i + len(APC_START)] == APC_START:
            flush_text()
            end = data.find(APC_END, i + len(APC_START))
            if end == -1:
                # Incomplete APC - split across chunks
                remaining = data[i + len(APC_START) :]
                parts.append(f"[KITTY INCOMPLETE {len(remaining)}B]")
                i = len(data)
                continue
            body = data[i + len(APC_START) : end]
            # Split header;payload
            semi = body.find(b";")
            if semi != -1:
                header = body[:semi].decode("ascii", errors="replace")
                payload_len = len(body) - semi - 1
                parts.append(f"[KITTY {header};payload={payload_len}B]")
            else:
                header = body.decode("ascii", errors="replace")
                parts.append(f"[KITTY {header}]")
            i = end + len(APC_END)
            continue

        # Check for CSI: \x1b[ ... (final byte 0x40-0x7E)
        if i + 1 < len(data) and data[i : i + 2] == b"\x1b[":
            flush_text()
            j = i + 2
            while j < len(data) and 0x20 <= data[j] <= 0x3F:
                j += 1
            if j < len(data) and 0x40 <= data[j] <= 0x7E:
                seq = data[i + 2 : j + 1].decode("ascii", errors="replace")
                parts.append(f"[CSI {seq}]")
                i = j + 1
                continue
            # Malformed CSI, treat as text
            text_run += 1
            i += 1
            continue

        # Check for other ESC sequences: \x1b followed by something
        if data[i : i + 1] == b"\x1b" and i + 1 < len(data):
            flush_text()
            # OSC: \x1b] ... ST
            if data[i + 1 : i + 2] == b"]":
                # Find ST (\x1b\\ or \x07)
                st1 = data.find(b"\x1b\\", i + 2)
                st2 = data.find(b"\x07", i + 2)
                ends = [e for e in [st1, st2] if e != -1]
                if ends:
                    end = min(ends)
                    skip = 2 if data[end : end + 1] == b"\x1b" else 1
                    parts.append(f"[OSC {end - i - 2}B]")
                    i = end + skip
                    continue
            # Simple two-byte escape
            parts.append(f"[ESC {chr(data[i + 1])}]")
            i += 2
            continue

        text_run += 1
        i += 1

    flush_text()
    return " ".join(parts)


def log_chunk(f, data: bytes, chunk_num: int) -> None:
    """Write one log entry for a chunk read from the PTY."""
    ts = time.time()
    ann = annotate(data)
    hex_str = data.hex()
    if len(hex_str) > 1024:
        hex_str = hex_str[:1024] + f"...({len(data)}B total)"

    f.write(f"=== CHUNK #{chunk_num} ts={ts:.6f} len={len(data)} ===\n")
    f.write(f"ANN: {ann}\n")
    f.write(f"HEX: {hex_str}\n\n")
    f.flush()


def run_proxy(cmd: list[str], logpath: str) -> int:
    """Run the PTY proxy, return child exit code."""
    # Save terminal state
    saved_attrs = termios.tcgetattr(sys.stdin.fileno())
    win_size = fcntl.ioctl(sys.stdin.fileno(), termios.TIOCGWINSZ, b"\x00" * 8)

    master_fd, slave_fd = pty.openpty()

    # Set slave terminal size to match real terminal
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, win_size)

    child_pid = os.fork()
    if child_pid == 0:
        # Child process
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)
        os.execvp(cmd[0], cmd)
        # unreachable

    # Parent process
    slave_stat = os.fstat(slave_fd)
    os.close(slave_fd)

    # SIGWINCH forwarding
    def on_sigwinch(signum, frame):
        try:
            size = fcntl.ioctl(sys.stdin.fileno(), termios.TIOCGWINSZ, b"\x00" * 8)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, size)
            os.kill(child_pid, signal.SIGWINCH)
        except OSError:
            pass

    signal.signal(signal.SIGWINCH, on_sigwinch)

    # Put stdin in raw mode
    tty_attrs = termios.tcgetattr(sys.stdin.fileno())
    raw = list(tty_attrs)
    # cfmakeraw equivalent
    raw[0] &= ~(
        termios.IGNBRK
        | termios.BRKINT
        | termios.PARMRK
        | termios.ISTRIP
        | termios.INLCR
        | termios.IGNCR
        | termios.ICRNL
        | termios.IXON
    )
    raw[1] &= ~termios.OPOST
    raw[2] &= ~(termios.CSIZE | termios.PARENB)
    raw[2] |= termios.CS8
    raw[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
    raw[6][termios.VMIN] = 1
    raw[6][termios.VTIME] = 0
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, raw)

    chunk_num = 0
    total_bytes_read = 0
    exit_code = 1

    logfile = open(logpath, "w")
    logfile.write(f"# PTY capture started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    logfile.write(f"# Command: {' '.join(cmd)}\n")
    logfile.write(f"# Slave PTY: dev={slave_stat.st_dev} ino={slave_stat.st_ino}\n\n")
    logfile.flush()

    try:
        stdin_fd = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()

        while True:
            try:
                rfds, _, _ = select.select([stdin_fd, master_fd], [], [], 0.5)
            except InterruptedError:
                continue

            if not rfds:
                # Timeout - check if child is still alive
                pid, status = os.waitpid(child_pid, os.WNOHANG)
                if pid != 0:
                    exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
                    break
                continue

            for fd in rfds:
                if fd == stdin_fd:
                    try:
                        data = os.read(stdin_fd, 4096)
                    except OSError:
                        data = b""
                    if not data:
                        break
                    os.write(master_fd, data)

                elif fd == master_fd:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        data = b""
                    if not data:
                        # Child closed PTY
                        pid, status = os.waitpid(child_pid, 0)
                        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
                        break
                    chunk_num += 1
                    total_bytes_read += len(data)
                    log_chunk(logfile, data, chunk_num)
                    os.write(stdout_fd, data)
            else:
                continue
            break

    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, saved_attrs)
        logfile.write(
            f"\n# Capture ended. {chunk_num} chunks, {total_bytes_read} bytes read from PTY master.\n"
        )
        logfile.close()
        try:
            os.close(master_fd)
        except OSError:
            pass

    return exit_code


# --- Analysis mode ---

CHUNK_RE = re.compile(r"^=== CHUNK #(\d+) ts=([\d.]+) len=(\d+) ===$")
ANN_RE = re.compile(r"^ANN: (.*)$")


def analyze(logpath: str) -> None:
    """Parse a capture log and produce analysis."""
    chunks: list[tuple[int, float, int, str]] = []  # num, ts, length, annotation

    with open(logpath) as f:
        num = 0
        ts = 0.0
        length = 0
        for line in f:
            m = CHUNK_RE.match(line.strip())
            if m:
                num = int(m.group(1))
                ts = float(m.group(2))
                length = int(m.group(3))
                continue
            m = ANN_RE.match(line.strip())
            if m:
                chunks.append((num, ts, length, m.group(1)))

    if not chunks:
        print("No chunks found in log.")
        return

    first_ts = chunks[0][1]

    # Kitty commands timeline
    print("=" * 80)
    print("KITTY GRAPHICS COMMANDS")
    print("=" * 80)
    print(f"{'Time':>10}  {'Chunk':>6}  {'Len':>6}  Command")
    print("-" * 80)

    kitty_re = re.compile(r"\[KITTY ([^\]]+)\]")
    bsu_esu_re = re.compile(r"\[(BSU|ESU)\]")
    incomplete_re = re.compile(r"\[KITTY INCOMPLETE")

    in_sync = False
    kitty_count = 0

    for num, ts, length, ann in chunks:
        kitty_matches = kitty_re.findall(ann)
        bsu_esu = bsu_esu_re.findall(ann)
        incomplete = incomplete_re.findall(ann)

        if not kitty_matches and not bsu_esu and not incomplete:
            continue

        rel_t = ts - first_ts
        sync_marker = ""

        for be in bsu_esu:
            if be == "BSU":
                in_sync = True
                sync_marker = " >>SYNC"
            elif be == "ESU":
                in_sync = False
                sync_marker = " <<SYNC"

        sync_ctx = " [IN SYNC]" if in_sync and not sync_marker else ""

        for km in kitty_matches:
            kitty_count += 1
            # Extract action
            action = ""
            if "a=T" in km:
                action = "TRANSMIT"
            elif "a=d" in km:
                action = "DELETE"
            elif "a=p" in km:
                action = "PLACE"
            # Extract image id
            id_m = re.search(r"i=(\d+)", km)
            img_id = id_m.group(1) if id_m else "?"

            short = f"{action:>8} i={img_id}"
            print(f"{rel_t:10.3f}  #{num:>5}  {length:>5}  {short}{sync_ctx}{sync_marker}")
            sync_marker = ""  # only show on first line

        for _ in incomplete:
            print(f"{rel_t:10.3f}  #{num:>5}  {length:>5}  *** INCOMPLETE APC ***{sync_ctx}")

        if not kitty_matches and not incomplete and bsu_esu:
            for be in bsu_esu:
                label = "--- BSU ---" if be == "BSU" else "--- ESU ---"
                print(f"{rel_t:10.3f}  #{num:>5}  {length:>5}  {label}")

    print(f"\nTotal kitty commands: {kitty_count}")

    # Split detection
    print("\n" + "=" * 80)
    print("SPLIT APC DETECTION")
    print("=" * 80)
    splits = [(n, t, _l, a) for n, t, _l, a in chunks if "INCOMPLETE" in a]
    if splits:
        for num, ts, _length, ann in splits:
            print(f"  CHUNK #{num} at t={ts - first_ts:.3f}s: {ann}")
    else:
        print("  No split APCs detected.")

    # Sync block summary
    print("\n" + "=" * 80)
    print("SYNC BLOCKS WITH KITTY COMMANDS")
    print("=" * 80)
    block_num = 0
    block_cmds: list[str] = []
    block_start = 0.0

    for _num, ts, _length, ann in chunks:
        if "[BSU]" in ann:
            block_num += 1
            block_cmds = []
            block_start = ts - first_ts
        for km in kitty_re.findall(ann):
            block_cmds.append(km)
        if "[ESU]" in ann and block_cmds:
            print(f"\n  Block #{block_num} at t={block_start:.3f}s:")
            for cmd in block_cmds:
                print(f"    {cmd}")


def main():
    parser = argparse.ArgumentParser(description="PTY capture for terminal debugging")
    parser.add_argument("--analyze", metavar="LOGFILE", help="Analyze a capture log")
    parser.add_argument("-o", "--output", metavar="FILE", help="Capture output file")
    parser.add_argument("cmd", nargs="*", help="Command to run (after --)")

    args = parser.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return

    if not args.cmd:
        parser.error("Provide a command to run, e.g.: -- uv run tsdr")

    logpath = args.output or f"pty_capture_{int(time.time())}.log"
    print(f"Capturing to: {logpath}", file=sys.stderr)

    exit_code = run_proxy(args.cmd, logpath)
    print(f"\nCapture saved to: {logpath}", file=sys.stderr)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
