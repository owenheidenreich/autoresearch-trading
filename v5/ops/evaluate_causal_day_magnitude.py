"""Read the decisive Job-39 mid-to-mid result before any wider economics.

The signed reopening is spent if the frozen primary cell has non-positive
gross mid-to-mid P&L per trade.  This command therefore reads only that cell,
writes its selected trades and stops.  Matched controls, bid economics and the
exit stage remain unread unless this receipt says ``PRIMARY_PASS``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_architectures import computed_parameter_counts
from v5.research.causal_day_magnitude import select_clock_trades
from v5.research.causal_day_policy_gate import (
    CONSERVATIVE_EFFECTIVE_OBSERVATIONS,
    MEASURED_EFFECTIVE_OBSERVATIONS,
)
from v5.research.knobs import frozen_value


PRIMARY_ARCHITECTURE = "neural_four_head"
PRIMARY_HORIZON = 120
PRIMARY_DEPTH_THRESHOLD = 30
PRIMARY_TRADE_CAP = 2
PRIMARY_RISK_MODE = "ticket_only"
FEES_PER_ROUND_TRIP_USD = 3.08


def _verified_receipt(path: Path, *, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"receipt self-hash mismatch: {path}")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"unexpected receipt schema at {path}")
    return value


def join_scored_candidates(
    predictions: pd.DataFrame, candidates: pd.DataFrame, *, horizon: int
) -> pd.DataFrame:
    keys = ["session", "entry_minute", "contract_id"]
    if predictions.duplicated(keys).any() or candidates.duplicated(keys).any():
        raise RuntimeError("prediction/candidate key is not one-to-one")
    score = f"predicted_depth_{horizon}m"
    candidate_columns = [
        *keys,
        "right",
        "self_delta",
        "entry_ask_usd",
        "entry_mid_usd",
        "spread_usd",
        "moneyness_itm_points",
        f"clock_exit_minute_{horizon}m",
        f"clock_exit_mid_value_{horizon}m",
        f"net_bid_{horizon}m_usd",
        f"net_mid_{horizon}m_usd",
    ]
    missing = sorted(set(candidate_columns) - set(candidates.columns))
    if missing or score not in predictions:
        raise RuntimeError(f"primary inputs are incomplete: {missing or [score]}")
    merged = predictions.merge(
        candidates[candidate_columns], on=keys, how="left", validate="one_to_one"
    )
    if len(merged) != len(predictions) or merged[candidate_columns[3:]].isna().any().any():
        raise RuntimeError("a scored contract has no complete frozen candidate outcome")
    if not merged["session"].astype(str).lt("2026-08-06").all():
        raise RuntimeError("reserved session entered the economic population")
    direct_gross = (
        pd.to_numeric(merged[f"clock_exit_mid_value_{horizon}m"], errors="raise")
        * 100.0
        - pd.to_numeric(merged["entry_mid_usd"], errors="raise")
    )
    stored_gross = (
        pd.to_numeric(merged[f"net_mid_{horizon}m_usd"], errors="raise")
        + FEES_PER_ROUND_TRIP_USD
    )
    if not np.allclose(direct_gross, stored_gross, atol=1e-8, rtol=0.0):
        raise RuntimeError("mid-gross accounting does not reproduce from source prices")
    return merged


def evaluate_primary_cell(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not scored["architecture"].eq(PRIMARY_ARCHITECTURE).all():
        raise RuntimeError("primary predictions contain another architecture")
    if not scored["horizon_minutes"].eq(PRIMARY_HORIZON).all():
        raise RuntimeError("primary predictions contain another horizon")
    if scored["shuffled_label_null"].astype(bool).any():
        raise RuntimeError("primary kill cell must use the real-label model")
    trades = select_clock_trades(
        scored,
        horizon=PRIMARY_HORIZON,
        depth_threshold=PRIMARY_DEPTH_THRESHOLD,
        trade_cap=PRIMARY_TRADE_CAP,
    )
    trade_count = len(trades)
    mean_gross = float(trades["gross_mid_usd"].mean()) if trade_count else None
    passed = bool(trade_count and mean_gross is not None and mean_gross > 0.0)
    fold_means = {}
    for fold in range(1, 6):
        values = (
            trades.loc[trades["fold"].eq(fold), "gross_mid_usd"]
            if trade_count
            else pd.Series(dtype=float)
        )
        fold_means[str(fold)] = float(values.mean()) if len(values) else None
    return trades, {
        "architecture": PRIMARY_ARCHITECTURE,
        "horizon_minutes": PRIMARY_HORIZON,
        "predicted_depth_threshold_points": PRIMARY_DEPTH_THRESHOLD,
        "trade_cap": PRIMARY_TRADE_CAP,
        "risk_mode": PRIMARY_RISK_MODE,
        "metric": "mean gross mid-to-mid dollars per trade before fees",
        "spread_removed": True,
        "fees_removed": True,
        "trades": trade_count,
        "days_traded": int(trades["session"].nunique()) if trade_count else 0,
        "mean_gross_mid_to_mid_usd_per_trade": mean_gross,
        "fold_mean_gross_mid_to_mid_usd_per_trade": fold_means,
        "pass_rule": "strictly greater than zero with at least one trade",
        "passed": passed,
    }


def run(
    *,
    declaration_path: Path,
    fit_receipt_path: Path,
    feature_audit_path: Path,
    candidates_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite primary economics: {out_dir}")
    declaration = _verified_receipt(
        declaration_path, schema="v5.causal-day-trader-fit-declaration.v4"
    )
    fit = _verified_receipt(
        fit_receipt_path, schema="v5.causal-day-magnitude-fit.v1"
    )
    audit = _verified_receipt(
        feature_audit_path, schema="v5.causal-day-fit-feature-audit.v1"
    )
    if audit.get("status") != "PASS_BEFORE_ECONOMICS" or not all(
        audit.get("assertions", {}).values()
    ):
        raise RuntimeError("feature timestamp audit is not fully green")
    if fit.get("economics_read") is not False or fit.get("threshold_tuned") is not False:
        raise RuntimeError("fit receipt says economics or threshold tuning already occurred")
    if fit.get("declaration", {}).get("sha256") != file_sha256(declaration_path):
        raise RuntimeError("fit receipt is not bound to this V4 declaration")

    matches = [
        value
        for value in fit.get("predictions", [])
        if value.get("architecture") == PRIMARY_ARCHITECTURE
        and value.get("horizon_minutes") == PRIMARY_HORIZON
        and value.get("shuffled_label_null") is False
    ]
    if len(matches) != 1:
        raise RuntimeError("fit receipt does not identify one frozen primary prediction")
    prediction_path = Path(matches[0]["path"])
    if file_sha256(prediction_path) != matches[0].get("sha256"):
        raise RuntimeError("primary prediction artifact hash mismatch")
    predictions = pd.read_parquet(prediction_path)
    candidates = pd.read_parquet(candidates_path)
    scored = join_scored_candidates(predictions, candidates, horizon=PRIMARY_HORIZON)
    trades, primary = evaluate_primary_cell(scored)

    out_dir.mkdir(parents=True, exist_ok=False)
    trades_path = out_dir / "primary_selected_trades.parquet"
    trades.to_parquet(trades_path, index=False)
    per_parameter = int(frozen_value("minimum_sessions_per_neural_parameter"))
    counts = computed_parameter_counts()
    status = (
        "PRIMARY_PASS"
        if primary["passed"]
        else "NEGATIVE_STOP_MID_TO_MID_GROSS_NOT_POSITIVE"
    )
    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-magnitude-primary-economics.v1",
        "created_on": "2026-08-14",
        "status": status,
        "declaration": {
            "path": str(declaration_path),
            "sha256": file_sha256(declaration_path),
        },
        "fit_receipt": {
            "path": str(fit_receipt_path),
            "sha256": file_sha256(fit_receipt_path),
        },
        "feature_audit": {
            "path": str(feature_audit_path),
            "sha256": file_sha256(feature_audit_path),
        },
        "primary_kill_condition": primary,
        "evidence_budget": {
            "generous_effective_observations": MEASURED_EFFECTIVE_OBSERVATIONS,
            "generous_parameter_budget": MEASURED_EFFECTIVE_OBSERVATIONS
            // per_parameter,
            "conservative_effective_observations": CONSERVATIVE_EFFECTIVE_OBSERVATIONS,
            "conservative_parameter_budget": CONSERVATIVE_EFFECTIVE_OBSERVATIONS
            // per_parameter,
            "conservative_reported_range": [29, 50],
            "primary_architecture_parameters": counts[PRIMARY_ARCHITECTURE],
        },
        "selected_trades": {
            "path": str(trades_path),
            "sha256": file_sha256(trades_path),
            "rows": len(trades),
        },
        "wider_economics_read": False,
        "matched_control_read": False,
        "exit_policy_fit": False,
        "on_failure": (
            "The reopening is spent as a negative result; do not relabel, search an "
            "operating point, fit an exit, change a threshold/seed, or retry"
        ),
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        declaration_path=args.declaration,
        fit_receipt_path=args.fit_receipt,
        feature_audit_path=args.feature_audit,
        candidates_path=args.candidates,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
