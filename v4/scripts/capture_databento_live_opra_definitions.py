"""Capture a bounded intraday replay of current SPXW OPRA definitions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd

from v4.checks.paid_data_guard import add_paid_data_approval_args, require_paid_data_approval
from v4.scripts.capture_databento_live_opra_training_twin import (
    DATASET,
    DEFAULT_APPROVAL_MANIFEST,
    _load_env_file,
    _sha256_path,
    _stable_hash,
)


SCHEMA_VERSION = "autoresearch.databento-live-opra-definition-capture.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-date", type=date.fromisoformat, required=True)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    add_paid_data_approval_args(parser, default_manifest=DEFAULT_APPROVAL_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1.0 <= args.duration_seconds <= 60.0:
        raise SystemExit("--duration-seconds must be between 1 and 60")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_paid_data_approval(
        manifest_path=args.approval_manifest,
        approval_text=args.approval_text,
        approval_env_var=args.approval_env_var,
        operation="bounded Databento Live OPRA SPXW.OPT intraday definition replay",
    )
    _load_env_file(args.env_file)
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("missing DATABENTO_API_KEY")

    import databento as db

    raw_path = args.output_dir / "opra_live_definitions.dbn.zst"
    counts: Counter[str] = Counter()
    errors: list[str] = []

    def callback(record: Any) -> None:
        counts[f"{type(record).__name__}:rtype={int(record.rtype)}"] += 1

    def error_callback(exc: Exception) -> None:
        errors.append(f"{type(exc).__name__}:{exc}")

    started = time.time_ns()
    client = db.Live(
        key=key,
        ts_out=True,
        compression=db.Compression.ZSTD,
        reconnect_policy="none",
    )
    client.subscribe(
        dataset=DATASET,
        schema="definition",
        symbols="SPXW.OPT",
        stype_in="parent",
        start=0,
    )
    client.add_stream(raw_path, exception_callback=error_callback)
    client.add_callback(callback, exception_callback=error_callback)
    client.start()
    client.block_for_close(timeout=float(args.duration_seconds))
    finished = time.time_ns()
    if errors:
        raise RuntimeError(f"definition replay callback errors: {errors[:5]}")
    if not raw_path.exists() or raw_path.stat().st_size <= 0:
        raise RuntimeError("definition replay produced no DBN bytes")

    frame = db.DBNStore.from_file(raw_path).to_df(schema="definition")
    reset = frame.reset_index()
    required = {"raw_symbol", "expiration", "asset"}
    if not required.issubset(reset.columns):
        raise RuntimeError(f"live definition schema missing {sorted(required-set(reset.columns))}")
    expiration = pd.to_datetime(reset["expiration"], utc=True, errors="coerce")
    current = reset[
        expiration.dt.date.eq(args.session_date)
        & reset["asset"].astype(str).eq("SPXW")
        & reset["raw_symbol"].astype(str).str.startswith("SPXW  ")
    ].copy()
    current = current.sort_values("ts_recv").drop_duplicates("raw_symbol", keep="last")
    symbols = tuple(sorted(current["raw_symbol"].astype(str)))
    current_path = args.output_dir / "current_session_definitions.parquet"
    current.to_parquet(current_path, index=False, compression="zstd")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "CAPTURED_CURRENT_SESSION_DEFINITIONS" if symbols else "NO_CURRENT_SESSION_DEFINITIONS",
        "dataset": DATASET,
        "subscription": {
            "schema": "definition",
            "symbols": "SPXW.OPT",
            "stype_in": "parent",
            "start": 0,
            "duration_seconds": float(args.duration_seconds),
        },
        "session_date": args.session_date.isoformat(),
        "capture_started_unix_ns": started,
        "capture_finished_unix_ns": finished,
        "record_counts_by_class": dict(sorted(counts.items())),
        "decoded_definition_rows": int(len(reset)),
        "current_session_definition_count": len(symbols),
        "current_session_symbols_sha256": hashlib.sha256("\n".join(symbols).encode()).hexdigest(),
        "first_current_session_symbol": symbols[0] if symbols else None,
        "last_current_session_symbol": symbols[-1] if symbols else None,
        "raw_dbn": {
            "path": str(raw_path.resolve()),
            "bytes": raw_path.stat().st_size,
            "sha256": _sha256_path(raw_path),
        },
        "current_session_parquet": {
            "path": str(current_path.resolve()),
            "bytes": current_path.stat().st_size,
            "sha256": _sha256_path(current_path),
        },
        "hard_stops": {
            "broker_accessed": False,
            "holdout_open_count": 0,
            "model_loaded_or_fit": False,
            "order_path_accessed": False,
            "paper_runtime_accessed": False,
            "promotion_or_default_changed": False,
        },
    }
    payload["summary_sha256"] = _stable_hash(payload)
    (args.output_dir / "definition_capture_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if symbols else 2


if __name__ == "__main__":
    raise SystemExit(main())
