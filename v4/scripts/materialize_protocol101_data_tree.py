"""Materialize Protocol101 local data files before long acceptance/training runs.

The project data tree currently lives under a macOS/iCloud-synced location in
some environments. Cloud-evicted "dataless" files can make parquet reads block
until macOS downloads the bytes. This helper inventories dataless files and,
when explicitly executed, streams every file once to force local materialization.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = (
    Path("data/raw/databento/opra_spxw_definition"),
    Path("data/raw/databento/opra_spxw_cbbo_1m"),
    Path("data/raw/databento/opra_spxw_ohlcv_1m"),
    Path("data/raw/databento/opra_spxw_statistics"),
    Path("data/vendor/thetadata/index/spx_1m"),
    Path("data/vendor/thetadata/index/vix_1m"),
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_data_materialization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Actually stream files to force local materialization.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for smoke testing. Zero means no cap.")
    parser.add_argument("--per-file-timeout-seconds", type=int, default=20)
    parser.add_argument("--stream-one-file", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def is_dataless(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0


def iter_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(files)


def stream_file_direct(path: Path) -> tuple[int, str]:
    total = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(chunk)
        return total, ""
    except OSError as exc:
        return total, str(exc)


def stream_file(path: Path, *, timeout_seconds: int) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "v4.scripts.materialize_protocol101_data_tree",
        "--stream-one-file",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(int(timeout_seconds), 1),
        )
    except subprocess.TimeoutExpired:
        return 0, f"timeout_after_{int(timeout_seconds)}s"
    if completed.returncode != 0:
        return 0, (completed.stderr or completed.stdout or f"returncode_{completed.returncode}").strip()
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return 0, "invalid_child_stream_output"
    return int(payload.get("bytes_streamed") or 0), str(payload.get("error") or "")


def build_payload(*, roots: list[Path], execute: bool, max_files: int, per_file_timeout_seconds: int = 20) -> dict[str, Any]:
    started = time.monotonic()
    files = iter_files(roots)
    if max_files > 0:
        files = files[: int(max_files)]
    before_dataless = [path for path in files if is_dataless(path)]
    materialized = []
    errors = []
    bytes_streamed = 0
    if execute:
        for path in before_dataless:
            streamed, error = stream_file(path, timeout_seconds=int(per_file_timeout_seconds))
            bytes_streamed += streamed
            if error:
                errors.append({"path": str(path), "error": error})
            elif not is_dataless(path):
                materialized.append(str(path))
    after_dataless = [path for path in files if is_dataless(path)]
    return {
        "schema_version": "Protocol101DataMaterializationV1",
        "status": "pass" if not errors else "fail",
        "execute": bool(execute),
        "roots": [str(path) for path in roots],
        "file_count": len(files),
        "max_files": int(max_files),
        "per_file_timeout_seconds": int(per_file_timeout_seconds),
        "before_dataless_count": len(before_dataless),
        "after_dataless_count": len(after_dataless),
        "materialized_count": len(materialized),
        "bytes_streamed": int(bytes_streamed),
        "duration_seconds": round(time.monotonic() - started, 3),
        "before_dataless_sample": [str(path) for path in before_dataless[:50]],
        "after_dataless_sample": [str(path) for path in after_dataless[:50]],
        "materialized_sample": materialized[:50],
        "errors": errors[:50],
    }


def main() -> int:
    args = parse_args()
    if args.stream_one_file is not None:
        streamed, error = stream_file_direct(args.stream_one_file)
        print(json.dumps({"bytes_streamed": int(streamed), "error": error}, sort_keys=True))
        return 0 if not error else 1
    roots = list(args.root or DEFAULT_ROOTS)
    payload = build_payload(
        roots=roots,
        execute=bool(args.execute),
        max_files=int(args.max_files),
        per_file_timeout_seconds=int(args.per_file_timeout_seconds),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / ("materialization_execute_summary.json" if args.execute else "materialization_inventory_summary.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "out": str(out), "before_dataless_count": payload["before_dataless_count"], "after_dataless_count": payload["after_dataless_count"], "execute": payload["execute"]}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
