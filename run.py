"""Start the FastAPI server and webcam client together.

Usage:
    python run.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = ROOT_DIR / "api" / "fastapi_server.py"
CLIENT_SCRIPT = ROOT_DIR / "api" / "fastapi_client.py"
HEALTH_URL = "http://127.0.0.1:8000/health"


def wait_for_server(url: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2):
                return
        except (URLError, OSError) as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError(f"Server did not become ready at {url}: {last_error}")


def start_process(script: Path, extra_args: list[str] | None = None) -> subprocess.Popen[str]:
    command = [sys.executable, str(script)]
    if extra_args:
        command.extend(extra_args)

    return subprocess.Popen(command, cwd=str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the emotion recognition server and client")
    parser.add_argument("--server-timeout", type=float, default=60.0)
    parser.add_argument("client_args", nargs=argparse.REMAINDER, help="Arguments forwarded to fastapi_client.py")
    args = parser.parse_args()

    if not SERVER_SCRIPT.is_file():
        raise SystemExit(f"Missing server script: {SERVER_SCRIPT}")
    if not CLIENT_SCRIPT.is_file():
        raise SystemExit(f"Missing client script: {CLIENT_SCRIPT}")

    print(f"[INFO] starting server: {SERVER_SCRIPT}")
    server_process = start_process(SERVER_SCRIPT)

    try:
        wait_for_server(HEALTH_URL, timeout_seconds=args.server_timeout)
        print(f"[INFO] server is ready: {HEALTH_URL}")

        print(f"[INFO] starting client: {CLIENT_SCRIPT}")
        client_process = start_process(CLIENT_SCRIPT, args.client_args)

        try:
            client_code = client_process.wait()
            return client_code
        finally:
            if client_process.poll() is None:
                client_process.terminate()
                client_process.wait(timeout=10)
    except KeyboardInterrupt:
        print("[INFO] stopping processes")
        return 130
    finally:
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())