"""Decompose the already-spent iron-fly result into its two credit spreads.

This is failure attribution, not a new economic cell: entry center, entry
minute, exit minute/type, width, and fees all remain exactly those selected by
the frozen iron-fly run.  The component sums must reproduce that run exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import (
    QUOTE_COLUMNS,
    canonical_json,
    file_sha256,
    load_validated_settlements,
    prepare_quotes,
)
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD
from v5.research.defined_risk_iron_fly import _entry_structure, _live


SCHEMA = "v5.defined-risk-iron-fly-leg-attribution.v1"


class LegAttributionError(RuntimeError):
    """The component reconstruction does not match frozen iron-fly evidence."""


def summarize_components(frame: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    fold_indices = np.array_split(np.arange(len(frame)), 5)
    for column in ("call_touch_net_usd", "put_touch_net_usd", "call_mid_gross_usd", "put_mid_gross_usd"):
        folds = [float(frame.iloc[index][column].mean()) for index in fold_indices]
        rows[column] = {
            "mean_per_session_usd": float(frame[column].mean()),
            "mean_per_trade_usd": float(frame.loc[frame["traded"], column].mean()),
            "positive_chronological_folds": int(sum(value > 0.0 for value in folds)),
            "fold_means_usd": folds,
            "worst_trade_usd": float(frame.loc[frame["traded"], column].min()),
        }
    return rows


def run(
    *,
    session_results_path: Path,
    ladder_path: Path,
    settlement_receipt_path: Path,
    raw_quote_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise LegAttributionError("refusing to overwrite leg-attribution receipt")
    results = pd.read_parquet(session_results_path).sort_values("session").reset_index(drop=True)
    ladder = pd.read_parquet(
        ladder_path,
        columns=[
            "session", "minute", "contract_id", "strike", "right", "bid", "ask", "mid",
            "bid_size", "ask_size", "quote_age_ms", "underlying_price",
        ],
    )
    settlements = load_validated_settlements(settlement_receipt_path)
    rows = []
    fallback_paths: set[Path] = set()
    for row in results.itertuples(index=False):
        if not bool(row.traded):
            rows.append(
                {
                    "session": str(row.session),
                    "traded": False,
                    "call_touch_net_usd": 0.0,
                    "put_touch_net_usd": 0.0,
                    "call_mid_gross_usd": 0.0,
                    "put_mid_gross_usd": 0.0,
                }
            )
            continue
        quotes = ladder[ladder["session"].astype(str).eq(str(row.session))]
        entry = quotes[quotes["minute"].astype(str).eq("15:00")]
        legs = _entry_structure(entry, float(entry["underlying_price"].median()))
        if legs is None or float(legs["short_call"]["strike"]) != float(row.center_strike):
            raise LegAttributionError(f"{row.session}: frozen entry structure does not reproduce")
        if row.exit_type == "validated_cash_settlement":
            settlement = settlements[str(row.session)]
            center = float(row.center_strike)
            call_exit = min(max(settlement - center, 0.0), 5.0)
            put_exit = min(max(center - settlement, 0.0), 5.0)
            call_exit_mid = call_exit
            put_exit_mid = put_exit
        else:
            snapshot = quotes[quotes["minute"].astype(str).eq(str(row.exit_minute))]
            ids = {name: str(value["contract_id"]) for name, value in legs.items()}
            if any(snapshot[snapshot["contract_id"].astype(str).eq(value)].empty for value in ids.values()):
                raw_path = raw_quote_root / f"databento_spxw_0dte_{row.session}.parquet"
                fallback_paths.add(raw_path)
                raw = pd.read_parquet(raw_path, columns=list(QUOTE_COLUMNS))
                prepared = prepare_quotes(raw, str(row.session))
                snapshot = _live(
                    prepared[prepared["minute"].astype(str).eq(str(row.exit_minute))]
                )
            exit_legs = {
                name: snapshot[
                    snapshot["contract_id"].astype(str).eq(contract_id)
                ].iloc[-1]
                for name, contract_id in ids.items()
            }
            call_exit = float(exit_legs["short_call"]["ask"] - exit_legs["long_call"]["bid"])
            put_exit = float(exit_legs["short_put"]["ask"] - exit_legs["long_put"]["bid"])
            call_exit_mid = float(exit_legs["short_call"]["mid"] - exit_legs["long_call"]["mid"])
            put_exit_mid = float(exit_legs["short_put"]["mid"] - exit_legs["long_put"]["mid"])
        call_entry = float(legs["short_call"]["bid"] - legs["long_call"]["ask"])
        put_entry = float(legs["short_put"]["bid"] - legs["long_put"]["ask"])
        call_entry_mid = float(legs["short_call"]["mid"] - legs["long_call"]["mid"])
        put_entry_mid = float(legs["short_put"]["mid"] - legs["long_put"]["mid"])
        rows.append(
            {
                "session": str(row.session),
                "traded": True,
                "call_touch_net_usd": (call_entry - call_exit) * 100.0 - 2 * FEES_PER_ROUND_TRIP_USD,
                "put_touch_net_usd": (put_entry - put_exit) * 100.0 - 2 * FEES_PER_ROUND_TRIP_USD,
                "call_mid_gross_usd": (call_entry_mid - call_exit_mid) * 100.0,
                "put_mid_gross_usd": (put_entry_mid - put_exit_mid) * 100.0,
            }
        )
    components = pd.DataFrame(rows)
    touch_error = np.max(
        np.abs(
            components["call_touch_net_usd"]
            + components["put_touch_net_usd"]
            - results["net_touch_usd"]
        )
    )
    mid_error = np.max(
        np.abs(
            components["call_mid_gross_usd"]
            + components["put_mid_gross_usd"]
            - results["gross_mid_usd"]
        )
    )
    if touch_error > 1e-9 or mid_error > 1e-9:
        raise LegAttributionError("credit-spread components do not sum to frozen iron fly")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "purpose": "failure attribution of the already-spent iron-fly result at identical entries and exits",
        "population": {
            "sessions": int(len(components)),
            "trades": int(components["traded"].sum()),
        },
        "components": summarize_components(components),
        "reproduction": {
            "maximum_touch_sum_error_usd": float(touch_error),
            "maximum_mid_sum_error_usd": float(mid_error),
        },
        "inputs": {
            "session_results_path": str(session_results_path),
            "session_results_sha256": file_sha256(session_results_path),
            "ladder_path": str(ladder_path),
            "ladder_sha256": file_sha256(ladder_path),
            "settlement_receipt_path": str(settlement_receipt_path),
            "settlement_receipt_sha256": file_sha256(settlement_receipt_path),
            "raw_fallbacks": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in sorted(fallback_paths)
            ],
        },
        "integrity": {
            "new_entry_or_exit_selected": False,
            "new_economic_cell": False,
            "same_frozen_exit_for_both_components": True,
            "component_sides_not_promotable": True,
            "model_fit": False,
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
    parser.add_argument("--session-results", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--settlement-receipt", type=Path, required=True)
    parser.add_argument("--raw-quote-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        session_results_path=args.session_results,
        ladder_path=args.ladder,
        settlement_receipt_path=args.settlement_receipt,
        raw_quote_root=args.raw_quote_root,
        output_path=args.output,
    )
    print(json.dumps({"components": payload["components"], "receipt_sha256": payload["receipt_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
