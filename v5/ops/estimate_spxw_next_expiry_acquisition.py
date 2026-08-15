"""Estimate next-expiry SPXW quote acquisition from already-recorded costs.

No vendor API is called.  The estimate uses owned 0DTE CBBO row counts, the
next listed expiry in owned definitions, and historical cost estimates already
stored in the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256


SCHEMA = "v5.spxw-next-expiry-acquisition-estimate.v1"
TARGET_SESSIONS = 1_045


def _recorded_costs(audit_root: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for path in sorted(audit_root.glob("databento*_downloads.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            schema = row.get("schema")
            session = row.get("date")
            cost = row.get("cost_estimate_usd")
            if schema is None or session is None or cost is None:
                continue
            values.setdefault(str(schema), {})[str(session)] = float(cost)
    return values


def estimate_costs(
    recent: pd.DataFrame,
    *,
    cbbo_cost_by_session: dict[str, float],
    definition_cost_by_session: dict[str, float],
    target_sessions: int = TARGET_SESSIONS,
) -> dict[str, Any]:
    required = {"session", "rows_0dte", "symbols_0dte", "symbols_next_expiry"}
    if not required.issubset(recent.columns):
        raise RuntimeError("recent inventory columns are incomplete")
    cost_rows = recent[recent["session"].isin(cbbo_cost_by_session)].copy()
    cost_rows["cost_usd"] = cost_rows["session"].map(cbbo_cost_by_session)
    if cost_rows.empty or cost_rows["rows_0dte"].sum() <= 0:
        raise RuntimeError("no recorded CBBO costs align to the owned inventory")
    usd_per_row = float(cost_rows["cost_usd"].sum() / cost_rows["rows_0dte"].sum())
    rows_per_symbol = float(recent["rows_0dte"].sum() / recent["symbols_0dte"].sum())
    recent_quote_estimate = float(
        (recent["symbols_next_expiry"] * rows_per_symbol * usd_per_row).sum()
    )
    quote_estimate = recent_quote_estimate / len(recent) * target_sessions
    if not definition_cost_by_session:
        raise RuntimeError("no recorded definition costs are available")
    definition_daily_mean = float(pd.Series(definition_cost_by_session).mean())
    definition_estimate = definition_daily_mean * target_sessions
    combined = quote_estimate + definition_estimate
    return {
        "target_sessions": int(target_sessions),
        "recent_sessions": int(len(recent)),
        "cbbo_cost_covered_sessions": int(len(cost_rows)),
        "weighted_recorded_cbbo_usd_per_row": usd_per_row,
        "owned_0dte_rows_per_symbol": rows_per_symbol,
        "median_0dte_symbols": float(recent["symbols_0dte"].median()),
        "median_next_expiry_symbols": float(recent["symbols_next_expiry"].median()),
        "estimated_next_expiry_cbbo_usd": quote_estimate,
        "recorded_definition_daily_mean_usd": definition_daily_mean,
        "estimated_definitions_usd": definition_estimate,
        "estimated_combined_usd": combined,
        "recommended_hard_cap_usd": int(math.ceil(combined / 25.0) * 25 + 25),
    }


def run(
    *,
    quote_root: Path,
    definition_root: Path,
    audit_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("refusing to overwrite acquisition estimate")
    costs = _recorded_costs(audit_root)
    rows = []
    quote_paths = sorted(quote_root.glob("*.cbbo-1m.parquet"))
    definition_paths = []
    for quote_path in quote_paths:
        session = quote_path.name.split(".", 1)[0]
        definition_path = definition_root / f"{session}.definition.parquet"
        definition_paths.append(definition_path)
        quotes = pd.read_parquet(quote_path, columns=["instrument_id"])
        definitions = pd.read_parquet(
            definition_path, columns=["instrument_id", "expiration"]
        ).drop_duplicates()
        definitions["expiration"] = pd.to_datetime(
            definitions["expiration"], utc=True
        ).dt.strftime("%Y-%m-%d")
        future = sorted(value for value in definitions["expiration"].unique() if value > session)
        if not future:
            raise RuntimeError(f"{session}: no next listed expiry in definitions")
        rows.append(
            {
                "session": session,
                "rows_0dte": int(len(quotes)),
                "symbols_0dte": int(quotes["instrument_id"].nunique()),
                "symbols_next_expiry": int(
                    definitions.loc[
                        definitions["expiration"].eq(future[0]), "instrument_id"
                    ].nunique()
                ),
            }
        )
    recent = pd.DataFrame(rows)
    estimate = estimate_costs(
        recent,
        cbbo_cost_by_session=costs.get("cbbo-1m", {}),
        definition_cost_by_session=costs.get("definition", {}),
    )
    manifest = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in [*quote_paths, *definition_paths]
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "purpose": "owner decision estimate for next-listed-expiry SPXW CBBO-1m plus definitions over the existing 1,045-session horizon",
        "estimate": estimate,
        "scope": {
            "traded_instrument": "SPXW options only",
            "context": "SPX or a causal SPXW put-call-forward only; no futures or ETF context",
            "expiry": "nearest listed SPXW expiration strictly after each session",
            "schemas": ["definition", "cbbo-1m"],
            "no_statistics_or_trade_print_purchase": True,
        },
        "method_limits": [
            "This is an extrapolation from recorded historical costs, not a fresh vendor quote.",
            "An authorized read-only vendor preflight must run before any download.",
            "The downloader must stop before purchase if the exact estimate exceeds the hard cap.",
        ],
        "inputs": {
            "quote_root": str(quote_root),
            "definition_root": str(definition_root),
            "audit_root": str(audit_root),
            "source_files": len(manifest),
            "source_manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
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
    parser.add_argument("--quote-root", type=Path, required=True)
    parser.add_argument("--definition-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        quote_root=args.quote_root,
        definition_root=args.definition_root,
        audit_root=args.audit_root,
        output_path=args.output,
    )
    print(json.dumps({"estimate": payload["estimate"], "receipt_sha256": payload["receipt_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
