"""Probe local IBKR API stability without creating orders."""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", default="4002,4000,7497,7496,4001")
    parser.add_argument("--client-id", type=int, default=145)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--stable-seconds", type=float, default=10.0)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + max(0.0, float(args.timeout_seconds))
    ports = candidate_ports(args.ports)
    attempts: list[dict[str, Any]] = []
    while time.monotonic() <= deadline:
        first = probe_once(args.host, ports, int(args.client_id))
        attempts.append(first)
        if first.get("connected"):
            stable_seconds = max(0.0, float(args.stable_seconds))
            if stable_seconds:
                time.sleep(stable_seconds)
            second = probe_once(args.host, [int(first["port"])], int(args.client_id) + 1)
            attempts.append(second)
            if second.get("connected"):
                payload = {
                    "status": "pass",
                    "host": args.host,
                    "port": int(first["port"]),
                    "stable_seconds": stable_seconds,
                    "account_count": int(second.get("account_count", 0)),
                    "account_id_redacted": redact_account_id(second.get("primary_account_id")),
                    "attempts_tail": sanitize_attempts(attempts[-6:]),
                    "broker_order_endpoint_called": False,
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
                if float(args.hold_seconds) > 0:
                    return hold_connection(
                        host=args.host,
                        port=int(first["port"]),
                        client_id=int(args.client_id) + 2,
                        hold_seconds=float(args.hold_seconds),
                        poll_seconds=max(5.0, float(args.poll_seconds)),
                    )
                return 0
        time.sleep(max(0.5, float(args.poll_seconds)))
    payload = {
        "status": "blocked",
        "blocked_reason": "ibkr_api_stability_probe_failed",
        "host": args.host,
        "ports": ports,
        "timeout_seconds": float(args.timeout_seconds),
        "attempts_tail": sanitize_attempts(attempts[-10:]),
        "broker_order_endpoint_called": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1


def candidate_ports(text: str) -> list[int]:
    ports: list[int] = []
    for value in str(text).split(","):
        try:
            port = int(value.strip())
        except ValueError:
            continue
        if port > 0 and port not in ports:
            ports.append(port)
    return ports


def probe_once(host: str, ports: list[int], client_id: int) -> dict[str, Any]:
    try:
        from ib_insync import IB  # type: ignore
    except ImportError:
        return {"connected": False, "blocked_reason": "missing_ib_insync"}
    attempts: list[dict[str, Any]] = []
    for port in ports:
        if not socket_open(host, port):
            attempts.append({"port": port, "status": "socket_closed"})
            continue
        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=8)
            accounts = list(ib.managedAccounts() or [])
            ib.disconnect()
            return {
                "connected": True,
                "host": host,
                "port": int(port),
                "account_count": len(accounts),
                "primary_account_id": accounts[0] if accounts else None,
                "attempts": attempts + [{"port": int(port), "status": "connected"}],
            }
        except Exception as exc:
            attempts.append({"port": int(port), "status": "api_failed", "error": str(exc)})
            if ib.isConnected():
                ib.disconnect()
    return {"connected": False, "attempts": attempts}


def socket_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def redact_account_id(value: Any) -> str | None:
    text = "" if value is None else str(value)
    if len(text) < 4:
        return None if not text else "***"
    return f"{text[:2]}***{text[-2:]}"


def sanitize_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for attempt in attempts:
        clean = dict(attempt)
        if "primary_account_id" in clean:
            clean["primary_account_id_redacted"] = redact_account_id(clean.pop("primary_account_id"))
        out.append(clean)
    return out


def hold_connection(*, host: str, port: int, client_id: int, hold_seconds: float, poll_seconds: float) -> int:
    try:
        from ib_insync import IB  # type: ignore
    except ImportError:
        print(json.dumps({"status": "blocked", "blocked_reason": "missing_ib_insync"}))
        return 1
    ib = IB()
    deadline = time.monotonic() + max(0.0, hold_seconds)
    try:
        ib.connect(host, port, clientId=client_id, timeout=8)
        accounts = list(ib.managedAccounts() or [])
        print(
            json.dumps(
                {
                    "status": "holding",
                    "host": host,
                    "port": int(port),
                    "client_id": int(client_id),
                    "hold_seconds": float(hold_seconds),
                    "account_count": len(accounts),
                    "account_id_redacted": redact_account_id(accounts[0] if accounts else None),
                    "broker_order_endpoint_called": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        while time.monotonic() < deadline:
            if not ib.isConnected():
                print(json.dumps({"status": "blocked", "blocked_reason": "ibkr_keepalive_disconnected"}))
                return 1
            ib.sleep(min(max(1.0, poll_seconds), max(1.0, deadline - time.monotonic())))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "blocked", "blocked_reason": "ibkr_keepalive_failed", "error": str(exc)}))
        return 1
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
