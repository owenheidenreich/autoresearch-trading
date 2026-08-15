"""Estimate the missing SPXW 0DTE CBBO backfill from recorded costs.

This script contacts no vendor.  It identifies non-empty, already-owned 0DTE
OHLCV sessions before the current quote corpus and applies cost measurements
already stored in repository download receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256


SCHEMA = "v5.spxw-0dte-cbbo-backfill-estimate.v1"
CUTOFF = "2025-08-01"


def _recorded_costs(audit_root: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for path in sorted(audit_root.glob("databento*_downloads.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(key in row for key in ("schema", "date", "cost_estimate_usd")):
                values.setdefault(str(row["schema"]), {})[str(row["date"])] = float(
                    row["cost_estimate_usd"]
                )
    return values


def estimate_backfill(
    *,
    sessions: int,
    cbbo_costs: dict[str, float],
    definition_costs: dict[str, float],
) -> dict[str, Any]:
    if sessions <= 0 or not cbbo_costs or not definition_costs:
        raise RuntimeError("backfill estimate lacks sessions or recorded costs")
    cbbo_mean = float(pd.Series(cbbo_costs).mean())
    definition_mean = float(pd.Series(definition_costs).mean())
    cbbo_estimate = cbbo_mean * sessions
    definition_estimate = definition_mean * sessions
    combined = cbbo_estimate + definition_estimate
    return {
        "additional_sessions": int(sessions),
        "recorded_cbbo_cost_sessions": int(len(cbbo_costs)),
        "recorded_definition_cost_sessions": int(len(definition_costs)),
        "recorded_cbbo_daily_mean_usd": cbbo_mean,
        "recorded_definition_daily_mean_usd": definition_mean,
        "estimated_cbbo_usd": cbbo_estimate,
        "estimated_definitions_usd": definition_estimate,
        "estimated_combined_usd": combined,
        "recommended_hard_cap_usd": int(math.ceil(combined / 25.0) * 25 + 25),
    }


def run(*, ohlcv_root: Path, audit_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("refusing to overwrite 0DTE backfill estimate")
    paths = sorted(
        path
        for path in ohlcv_root.glob("*.parquet")
        if path.name.split(".", 1)[0] < CUTOFF
        and pq.ParquetFile(path).metadata.num_rows > 0
    )
    sessions = [path.name.split(".", 1)[0] for path in paths]
    if not sessions or len(sessions) != len(set(sessions)):
        raise RuntimeError("0DTE OHLCV session inventory is empty or duplicated")
    costs = _recorded_costs(audit_root)
    estimate = estimate_backfill(
        sessions=len(sessions),
        cbbo_costs=costs.get("cbbo-1m", {}),
        definition_costs=costs.get("definition", {}),
    )
    manifest = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in paths
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "purpose": "owner decision estimate for the missing same-day SPXW full-ladder quote history",
        "scope": {
            "traded_position": "one long SPXW 0DTE call or put only",
            "context": "SPXW or SPX only",
            "first_session": sessions[0],
            "last_session": sessions[-1],
            "schemas": ["definition", "cbbo-1m"],
            "same_day_expiry_only": True,
            "no_longer_tenor_or_multileg_data": True,
        },
        "estimate": estimate,
        "method_limits": [
            "This is an extrapolation from recorded historical costs, not a fresh vendor quote.",
            "An authorized exact cost preflight must precede any paid request.",
            "The downloader must stop before purchase above the hard cap.",
        ],
        "inputs": {
            "ohlcv_root": str(ohlcv_root),
            "owned_nonempty_sessions": len(paths),
            "ohlcv_manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
            "audit_root": str(audit_root),
        },
        "integrity": {
            "vendor_contacted": False,
            "data_downloaded": False,
            "money_spent": False,
            "reserved_sessions_used": False,
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        ohlcv_root=args.ohlcv_root,
        audit_root=args.audit_root,
        output_path=args.output,
    )
    print(json.dumps({"estimate": payload["estimate"], "receipt_sha256": payload["receipt_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
