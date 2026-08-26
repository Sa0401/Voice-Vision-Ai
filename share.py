"""Start VisionVoice locally and expose it through a temporary Cloudflare Tunnel."""

from __future__ import annotations

import re
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path


PORT = 8501
PUBLIC_URL_FILE = Path("runtime/public-url.txt")
PUBLIC_URL_PATTERN = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com", re.IGNORECASE)


def local_ip() -> str:
    """Return the current LAN address without making an internet request."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        try:
            connection.connect(("10.255.255.255", 1))
            return connection.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def write_public_url(url: str) -> None:
    PUBLIC_URL_FILE.parent.mkdir(exist_ok=True)
    PUBLIC_URL_FILE.write_text(url, encoding="utf-8")


def main() -> int:
    cloudflared = shutil.which("cloudflared")
    if cloudflared is None:
        print("Cloudflare Tunnel is required for a public link.", file=sys.stderr)
        print("Install it on macOS with: brew install cloudflared", file=sys.stderr)
        return 1

    PUBLIC_URL_FILE.unlink(missing_ok=True)
    local_link = f"http://{local_ip()}:{PORT}"
    print(f"Local link:  {local_link}")
    print("Starting a temporary global link…")

    app_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            str(PORT),
        ]
    )
    tunnel_process = subprocess.Popen(
        [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert tunnel_process.stdout is not None
        for line in tunnel_process.stdout:
            print(line, end="")
            match = PUBLIC_URL_PATTERN.search(line)
            if match:
                public_link = match.group(0)
                write_public_url(public_link)
                print(f"\nGlobal link: {public_link}\n")
                print("Share this global link only with trusted people. Stop this command to close it.")
        return tunnel_process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        for process in (tunnel_process, app_process):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in (tunnel_process, app_process):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
