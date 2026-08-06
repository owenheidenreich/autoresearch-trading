"""One-shot, zero-network proof that launchd's approved Python can use the repo.

This is infrastructure-only.  It reads owned local inputs, exercises both
Track-A capture entry points in ``--dry-run`` mode, and writes one hashed
receipt.  Socket creation is blocked for the duration of the probe.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import date, datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = "autoresearch.python-direct-documents-probe.v1"
FORBIDDEN_RUNTIME_MODULE_PREFIXES = ("ib_insync", "ibapi")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("receipt_sha256", None)
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _argv(values: Sequence[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = list(values)
    try:
        yield
    finally:
        sys.argv = original


@contextmanager
def _network_blocked() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network disabled by Python-direct permission probe")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection


def _run_capture_dry_runs(
    *, repo_root: Path, session: date, definition_path: Path, scratch_root: Path
) -> dict[str, Any]:
    from v4.scripts import capture_databento_live_opra_definitions as definitions
    from v4.scripts import capture_databento_live_opra_training_twin as market

    outputs: dict[str, Any] = {}
    definition_args = [
        str(definitions.__file__),
        "--session-date",
        session.isoformat(),
        "--duration-seconds",
        "30",
        "--output-dir",
        str(scratch_root / "definitions"),
        "--env-file",
        str(repo_root / "v4/.env"),
        "--dry-run",
    ]
    with _argv(definition_args), redirect_stdout(io.StringIO()) as stdout:
        rc = definitions.main()
    if rc != 0:
        raise RuntimeError(f"definition dry-run returned {rc}")
    outputs["definitions"] = json.loads(stdout.getvalue())

    market_args = [
        str(market.__file__),
        "--session-date",
        session.isoformat(),
        "--definition-path",
        str(definition_path),
        "--duration-seconds",
        "180",
        "--expected-symbol-count",
        "510",
        "--schemas",
        "cbbo-1s",
        "cbbo-1m",
        "ohlcv-1m",
        "trades",
        "--output-dir",
        str(scratch_root / "market"),
        "--env-file",
        str(repo_root / "v4/.env"),
        "--dry-run",
    ]
    with _argv(market_args), redirect_stdout(io.StringIO()) as stdout:
        rc = market.main()
    if rc != 0:
        raise RuntimeError(f"market dry-run returned {rc}")
    outputs["market"] = json.loads(stdout.getvalue())
    return outputs


def run_probe(
    *,
    repo_root: Path,
    receipt_path: Path,
    session: date,
    definition_path: Path,
    require_receipt_in_repo: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    status_path = repo_root / "STATUS.md"
    if not status_path.is_file():
        raise RuntimeError(f"STATUS.md unreadable: {status_path}")
    definition_path = definition_path.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=False)
    if require_receipt_in_repo:
        try:
            receipt_path.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError("probe receipt must be written inside the repository") from exc

    with _network_blocked():
        dry_runs = _run_capture_dry_runs(
            repo_root=repo_root,
            session=session,
            definition_path=definition_path,
            scratch_root=receipt_path.parent / "dry_run_outputs_not_created",
        )

    forbidden_loaded = sorted(
        name
        for name in sys.modules
        if name.startswith(FORBIDDEN_RUNTIME_MODULE_PREFIXES)
    )
    if forbidden_loaded:
        raise RuntimeError(f"broker modules loaded during probe: {forbidden_loaded}")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "interpreter": {
            "argv0": sys.executable,
            "realpath": os.path.realpath(sys.executable),
            "version": sys.version.split()[0],
        },
        "checks": {
            "status_read": {
                "path": str(status_path),
                "sha256": _sha256(status_path),
            },
            "definition_input": {
                "path": str(definition_path),
                "sha256": _sha256(definition_path),
            },
            "definition_dry_run": dry_runs["definitions"],
            "market_dry_run": dry_runs["market"],
            "repo_write": str(receipt_path),
        },
        "hard_stops": {
            "network_allowed": False,
            "broker_or_order_path": False,
            "model_load_fit_or_search": False,
            "paid_data_contact": False,
            "protected_holdout_access": False,
            "runtime_or_launchd_change_beyond_probe": False,
        },
    }
    payload["receipt_sha256"] = _stable_hash(payload)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if _stable_hash(json.loads(receipt_path.read_text())) != payload["receipt_sha256"]:
        raise RuntimeError("probe receipt failed self-verification")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--session-date", type=date.fromisoformat, required=True)
    parser.add_argument("--definition-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_probe(
        repo_root=args.repo_root,
        receipt_path=args.receipt,
        session=args.session_date,
        definition_path=args.definition_path,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
