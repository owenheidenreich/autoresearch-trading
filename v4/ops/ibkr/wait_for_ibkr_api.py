"""Wait for the local IBKR API port and optionally run the entitlement probe."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--auto-ports", default="4000,4002,7497,7496,4001")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--run-entitlement-probe", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + max(float(args.timeout_seconds), 0.0)
    ports = candidate_ports(args)
    connected_port = None
    while time.monotonic() <= deadline:
        connected_port = first_connectable_port(args.host, ports)
        if connected_port is not None:
            payload = {"status": "port_open", "host": args.host, "port": int(connected_port), "candidate_ports": ports}
            print(json.dumps(payload, indent=2, sort_keys=True))
            if args.run_entitlement_probe:
                return run_entitlement_probe(args, port=connected_port)
            return 0
        time.sleep(max(float(args.poll_seconds), 0.5))
    print(
        json.dumps(
            {
                "status": "timeout",
                "host": args.host,
                "candidate_ports": ports,
                "timeout_seconds": args.timeout_seconds,
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 1


def can_connect(host: str, port: int, *, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def first_connectable_port(host: str, ports: list[int]) -> int | None:
    for port in ports:
        if can_connect(host, port, timeout=2.0):
            return port
    return None


def candidate_ports(args: argparse.Namespace) -> list[int]:
    ports: list[int] = []
    for value in [str(args.port), *str(args.auto_ports).split(",")]:
        try:
            port = int(value.strip())
        except ValueError:
            continue
        if port > 0 and port not in ports:
            ports.append(port)
    return ports


def run_entitlement_probe(args: argparse.Namespace, *, port: int) -> int:
    python = args.repo_root / ".venv/bin/python"
    if not python.exists():
        python = Path(sys.executable)
    cmd = [
        str(python),
        "-m",
        "v4.scripts.check_ibkr_live_data_entitlements",
        "--ibkr-host",
        args.host,
        "--ibkr-port",
        str(port),
        "--ibkr-auto-ports",
        "4000,4002,7497,7496,4001",
    ]
    return subprocess.call(cmd, cwd=str(args.repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
