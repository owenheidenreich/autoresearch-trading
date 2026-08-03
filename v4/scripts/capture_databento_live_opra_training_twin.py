"""Capture a bounded, no-order Databento Live OPRA parity sample.

This recorder is deliberately narrower than a trading runtime.  It subscribes
only to the exact SPXW contracts whose definitions expire on the requested
session, writes the untouched mixed-schema DBN stream, and records local
receipt timing and provenance.  It has no broker, order, model, or holdout
imports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


DATASET = "OPRA.PILLAR"
ALLOWED_SCHEMAS = (
    "cbbo-1s",
    "cbbo-1m",
    "cmbp-1",
    "tcbbo",
    "trades",
    "ohlcv-1m",
    "statistics",
    "status",
)
DEFAULT_APPROVAL_MANIFEST = Path(
    "v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/authorization.json"
)
SUMMARY_SCHEMA = "autoresearch.databento-live-opra-training-twin-capture.v1"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("summary_sha256", None)
    encoded = json.dumps(
        material, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def select_session_symbols(definition_path: Path, session_date: date) -> tuple[str, ...]:
    """Return the exact unique SPXW raw symbols expiring on ``session_date``."""

    required = {"raw_symbol", "expiration", "asset"}
    if definition_path.name.endswith((".dbn", ".dbn.zst")):
        import databento as db

        frame = db.DBNStore.from_file(definition_path).to_df(schema="definition").reset_index()
    else:
        frame = pd.read_parquet(definition_path)
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"definition file missing columns: {sorted(missing)}")
    expiration = pd.to_datetime(frame["expiration"], utc=True, errors="coerce")
    mask = (
        expiration.dt.date.eq(session_date)
        & frame["asset"].astype(str).eq("SPXW")
        & frame["raw_symbol"].astype(str).str.startswith("SPXW  ")
    )
    symbols = tuple(sorted(set(frame.loc[mask, "raw_symbol"].astype(str))))
    if not symbols:
        raise RuntimeError(f"no SPXW definitions expire on {session_date}")
    if len(symbols) > 2_000:
        raise RuntimeError(f"unexpectedly broad SPXW 0DTE symbol set: {len(symbols)}")
    return symbols


def _quantiles(values: Iterable[int]) -> dict[str, int] | None:
    array = np.asarray(tuple(values), dtype=np.int64)
    if not len(array):
        return None
    return {
        "min": int(np.min(array)),
        "p50": int(np.quantile(array, 0.50, method="nearest")),
        "p90": int(np.quantile(array, 0.90, method="nearest")),
        "p99": int(np.quantile(array, 0.99, method="nearest")),
        "max": int(np.max(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-date", type=date.fromisoformat, required=True)
    parser.add_argument("--definition-path", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=90.0)
    parser.add_argument(
        "--schemas", nargs="+", choices=ALLOWED_SCHEMAS, default=list(ALLOWED_SCHEMAS)
    )
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    add_paid_data_approval_args(parser, default_manifest=DEFAULT_APPROVAL_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1.0 <= args.duration_seconds <= 300.0:
        raise SystemExit("--duration-seconds must be between 1 and 300")
    schemas = tuple(dict.fromkeys(args.schemas))
    symbols = select_session_symbols(args.definition_path, args.session_date)
    plan = {
        "dataset": DATASET,
        "schemas": list(schemas),
        "session_date": args.session_date.isoformat(),
        "definition_path": str(args.definition_path.resolve()),
        "definition_sha256": _sha256_path(args.definition_path),
        "symbol_count": len(symbols),
        "symbols_sha256": hashlib.sha256("\n".join(symbols).encode()).hexdigest(),
        "duration_seconds": float(args.duration_seconds),
        "output_dir": str(args.output_dir.resolve()),
        "network": not args.dry_run,
        "broker_or_order_path": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_paid_data_approval(
        manifest_path=args.approval_manifest,
        approval_text=args.approval_text,
        approval_env_var=args.approval_env_var,
        operation=(
            f"bounded Databento Live {DATASET} capture of {len(symbols)} SPXW 0DTE symbols "
            f"for {args.duration_seconds:g} seconds"
        ),
    )
    _load_env_file(args.env_file)
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("missing DATABENTO_API_KEY")

    import databento as db

    raw_path = args.output_dir / "opra_live_mixed.dbn.zst"
    counts: Counter[str] = Counter()
    first_local_ns: dict[str, int] = {}
    last_local_ns: dict[str, int] = {}
    receipt_minus_ts_recv_ns: dict[str, list[int]] = defaultdict(list)
    ts_out_minus_ts_recv_ns: dict[str, list[int]] = defaultdict(list)
    local_minus_ts_out_ns: dict[str, list[int]] = defaultdict(list)
    callback_errors: list[str] = []

    def on_record(record: Any) -> None:
        local_ns = time.time_ns()
        rtype = getattr(record, "rtype", None)
        rtype_value = int(rtype) if rtype is not None else None
        name = (
            f"{type(record).__name__}:rtype={rtype_value}"
            if rtype_value is not None
            else type(record).__name__
        )
        counts[name] += 1
        first_local_ns.setdefault(name, local_ns)
        last_local_ns[name] = local_ns
        ts_recv = getattr(record, "ts_recv", None)
        if isinstance(ts_recv, int) and 0 < ts_recv <= local_ns:
            receipt_minus_ts_recv_ns[name].append(local_ns - ts_recv)
        ts_out = getattr(record, "ts_out", None)
        if isinstance(ts_recv, int) and isinstance(ts_out, int) and ts_recv <= ts_out:
            ts_out_minus_ts_recv_ns[name].append(ts_out - ts_recv)
            if ts_out <= local_ns:
                local_minus_ts_out_ns[name].append(local_ns - ts_out)

    def on_callback_error(exc: Exception) -> None:
        callback_errors.append(f"{type(exc).__name__}:{exc}")

    capture_started_ns = time.time_ns()
    client = db.Live(
        key=key,
        ts_out=True,
        compression=db.Compression.ZSTD,
        reconnect_policy="none",
    )
    for schema in schemas:
        client.subscribe(
            dataset=DATASET,
            schema=schema,
            symbols=symbols,
            stype_in="raw_symbol",
        )
    client.add_stream(raw_path, exception_callback=on_callback_error)
    client.add_callback(on_record, exception_callback=on_callback_error)
    client.start()
    client.block_for_close(timeout=float(args.duration_seconds))
    capture_finished_ns = time.time_ns()

    if callback_errors:
        raise RuntimeError(f"live capture callback errors: {callback_errors[:5]}")
    if not raw_path.exists() or raw_path.stat().st_size <= 0:
        raise RuntimeError("Databento live capture produced no raw DBN bytes")

    classes = sorted(counts)
    result: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "CAPTURED_NO_ORDER_LIVE_SAMPLE",
        "plan": plan,
        "capture_started_unix_ns": capture_started_ns,
        "capture_finished_unix_ns": capture_finished_ns,
        "elapsed_seconds": (capture_finished_ns - capture_started_ns) / 1e9,
        "record_counts_by_class": {name: counts[name] for name in classes},
        "records_total": int(sum(counts.values())),
        "first_local_receipt_unix_ns_by_class": {
            name: first_local_ns[name] for name in classes
        },
        "last_local_receipt_unix_ns_by_class": {
            name: last_local_ns[name] for name in classes
        },
        "local_receipt_minus_ts_recv_ns": {
            name: _quantiles(receipt_minus_ts_recv_ns[name]) for name in classes
        },
        "ts_out_minus_ts_recv_ns": {
            name: _quantiles(ts_out_minus_ts_recv_ns[name]) for name in classes
        },
        "local_receipt_minus_ts_out_ns": {
            name: _quantiles(local_minus_ts_out_ns[name]) for name in classes
        },
        "raw_dbn": {
            "path": str(raw_path.resolve()),
            "bytes": raw_path.stat().st_size,
            "sha256": _sha256_path(raw_path),
        },
        "hard_stops": {
            "broker_accessed": False,
            "order_path_accessed": False,
            "paper_runtime_accessed": False,
            "model_loaded_or_fit": False,
            "holdout_open_count": 0,
            "promotion_or_default_changed": False,
        },
    }
    if result["records_total"] <= 0:
        result["status"] = "EMPTY_LIVE_SAMPLE"
    result["summary_sha256"] = _stable_hash(result)
    summary_path = args.output_dir / "capture_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["records_total"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
