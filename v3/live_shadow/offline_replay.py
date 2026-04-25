"""Offline shadow replay — historical bar to DecisionSnapshot JSONL.

First of the five live-shadow tasks from the original Codex handoff
(now unblocked by the champion adoption documented in
v3/reference/spx_combined_3seed_001_champion_adoption_2026_04_25.md).

Takes the canonical champion artifacts (action-surface dataset +
per-seed chosen_trades + per-window model.pkl) and emits a
DecisionSnapshot JSONL with all per-bar information that a live shadow
session would record. Also runs a feature_parity check against a
one-day rebuild to confirm the bundle is reproducible from the raw v2
data with the documented recipe.

Usage:

    .venv/bin/python -m v3.live_shadow.offline_replay \\
        --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \\
        --chosen-trades v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed42/seed_42/chosen_trades.pkl \\
        --model-dir v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed42/seed_42 \\
        --output v3/artifacts/live_shadow_offline_replay/seed42.jsonl \\
        --parity-day 2024-04-01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.action_surface_dataset import (
    DEFAULT_HISTORY_BARS,
    DEFAULT_TOP_K_CONTRACTS,
    build_action_surface_bundle,
)
from v3.layer2.common import load_export_bundle, load_pickle
from v3.live_shadow.feature_parity import compare_feature_rows
from v3.live_shadow.resolver import ShadowContractResolver
from v3.live_shadow.schema import (
    ContractCandidateSnapshot,
    DecisionSnapshot,
    QuoteSnapshot,
)
from v3.live_shadow.snapshot import append_decision_snapshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--chosen-trades", required=True)
    p.add_argument("--model-dir", required=True, help="Path to seed_NN dir holding window_*/model.pkl")
    p.add_argument("--output", required=True, help="JSONL output path")
    p.add_argument("--parity-day", default="", help="If set, rebuild this day from raw v2 sources and feature_parity check vs the bundle.")
    p.add_argument("--max-trades", type=int, default=0, help="Cap trades for smoke runs (0 = all)")
    return p.parse_args()


def _expiry_from_day(day: str) -> str:
    return day.replace("-", "")


def _moneyness_label(bucket: float) -> str:
    if not np.isfinite(bucket):
        return "unknown"
    b = int(round(float(bucket)))
    return {-1: "ITM", 0: "ATM", 1: "OTM"}.get(b, str(b))


def _risk_band_label(code: float) -> str:
    if not np.isfinite(code):
        return "unknown"
    b = int(round(float(code)))
    return {0: "fill", 1: "tight", 2: "near", 3: "wide", 4: "edge"}.get(b, str(b))


def _maybe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    bundle = load_export_bundle(args.dataset)
    meta = bundle["meta"]
    rows: pd.DataFrame = bundle["rows"]
    top_k = int(meta["top_k_contracts_per_side"])
    contract_strike = bundle["contract_strike"]
    contract_features = bundle["contract_features"]
    tradeable_mask = np.nan_to_num(bundle["action_labels"]["tradeable_mask"], nan=0.0) > 0.5
    contract_feature_names = list(meta.get("contract_feature_names", ()))
    fname = {n: i for i, n in enumerate(contract_feature_names)}
    al = bundle["action_labels"]

    chosen = load_pickle(args.chosen_trades)
    n_total = len(chosen)
    if args.max_trades and args.max_trades > 0:
        chosen = chosen.head(args.max_trades).copy()

    # Index lookup
    rows_idx = rows.reset_index(drop=True)
    key_to_row = {(str(d), int(b)): i for i, (d, b) in enumerate(zip(rows_idx["day"], rows_idx["bar_index"]))}

    # Session id = sha256 of model_dir + dataset + chosen_trades names (deterministic)
    sid_input = f"{args.model_dir}|{args.dataset}|{args.chosen_trades}"
    session_id = hashlib.sha256(sid_input.encode()).hexdigest()[:16]
    dataset_fingerprint = meta.get("dataset_fingerprint")
    feature_schema = list(meta.get("scalar_feature_names", []))
    resolver = ShadowContractResolver()

    # We don't re-run the model here (chosen_trades.pkl already contains the
    # decision); this harness serializes the historical decision into
    # DecisionSnapshot form. A future live-shadow runner will rerun the model
    # against live features and write the same schema for parity comparison.
    n_emitted = 0
    n_unresolved = 0
    n_no_order = 0
    sides_count = {"call": 0, "put": 0, "flat": 0}

    for _, trade in chosen.iterrows():
        day = str(trade["day"])
        bar = int(trade["bar_index"])
        row_i = key_to_row.get((day, bar))
        if row_i is None:
            continue

        # Per-action metadata (24 contract candidates)
        candidates: list[ContractCandidateSnapshot] = []
        for action_idx in range(1, 1 + 2 * top_k):
            slot = action_idx - 1
            side = "call" if slot < top_k else "put"
            strike = float(contract_strike[row_i, slot])
            if not np.isfinite(strike):
                continue
            tradeable = bool(tradeable_mask[row_i, action_idx])
            risk_code = float(contract_features[row_i, slot, fname.get("risk_band", -1)]) if "risk_band" in fname else float("nan")
            money_code = float(contract_features[row_i, slot, fname.get("moneyness_bucket", -1)]) if "moneyness_bucket" in fname else float("nan")
            premium = _maybe_float(al["entry_fill_mid"][row_i, action_idx]) if "entry_fill_mid" in al else None
            spread = _maybe_float(al.get("entry_spread_fraction", np.full_like(al["tradeable_mask"], np.nan))[row_i, action_idx]) if "entry_spread_fraction" in al else None
            cspec = resolver.spec(
                expiry_yyyymmdd=_expiry_from_day(day),
                strike=strike,
                right="C" if side == "call" else "P",
            ) if tradeable else None
            candidates.append(
                ContractCandidateSnapshot(
                    action_id=action_idx,
                    slot=slot,
                    side=side,
                    strike=strike,
                    risk_band=_risk_band_label(risk_code),
                    moneyness=_moneyness_label(money_code),
                    premium=premium,
                    spread_fraction=spread,
                    model_score=None,
                    win_prob=None,
                    stopout_prob=None,
                    quote=QuoteSnapshot(),
                    greeks=None,
                    contract=cspec,
                )
            )

        # Selected action
        chosen_aid = int(trade.get("chosen_action_id", 0))
        chosen_side = str(trade.get("chosen_side", "flat"))
        if chosen_side not in {"call", "put"}:
            chosen_side = "flat"
        sides_count[chosen_side] = sides_count.get(chosen_side, 0) + 1

        if chosen_aid > 0 and np.isfinite(trade.get("chosen_strike", np.nan)):
            chosen_spec = resolver.spec(
                expiry_yyyymmdd=_expiry_from_day(day),
                strike=float(trade["chosen_strike"]),
                right="C" if chosen_side == "call" else "P",
            )
        else:
            chosen_spec = None
            n_unresolved += 1 if chosen_aid > 0 else 0
            if chosen_aid == 0:
                n_no_order += 1

        # Scalar features dict from rows
        scalar_features = {
            name: _maybe_float(rows_idx.iloc[row_i][name]) or 0.0
            for name in feature_schema
            if name in rows_idx.columns
        }

        # Diagnostics: store the calibration scores from chosen_trades for
        # later replay parity.
        diagnostics = {
            "decision_margin": _maybe_float(trade.get("decision_margin")),
            "best_nonflat_score": _maybe_float(trade.get("best_nonflat_score")),
            "flat_score": _maybe_float(trade.get("flat_score")),
            "pred_win_prob": _maybe_float(trade.get("pred_win_prob")),
            "pred_stopout_risk": _maybe_float(trade.get("pred_stopout_risk")),
            "chosen_premium": _maybe_float(trade.get("chosen_premium")),
            "chosen_spread_fraction": _maybe_float(trade.get("chosen_spread_fraction")),
            "chosen_objective_pnl": _maybe_float(trade.get("chosen_objective_pnl")),
            "chosen_time_stop_pnl": _maybe_float(trade.get("chosen_time_stop_pnl")),
            "window_idx": int(trade.get("window_idx", -1)),
            "seed": int(trade.get("seed", -1)),
        }

        snapshot = DecisionSnapshot(
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
            day=day,
            completed_bar_index=bar,
            model_artifact=args.model_dir,
            dataset_fingerprint=dataset_fingerprint,
            feature_schema=feature_schema,
            features=scalar_features,
            candidates=candidates,
            selected_action_id=chosen_aid,
            selected_contract=chosen_spec,
            no_order_reason="" if chosen_aid > 0 else "flat (model abstained)",
            scores={
                "decision_margin": float(trade.get("decision_margin", 0.0) or 0.0),
                "best_nonflat_score": float(trade.get("best_nonflat_score", 0.0) or 0.0),
                "flat_score": float(trade.get("flat_score", 0.0) or 0.0),
            },
            diagnostics=diagnostics,
        )
        append_decision_snapshot(args.output, snapshot)
        n_emitted += 1

    print(f"Wrote {n_emitted} DecisionSnapshots to {args.output}", flush=True)
    print(f"  total chosen trades available: {n_total}")
    print(f"  side distribution emitted: call={sides_count.get('call',0)} put={sides_count.get('put',0)} flat={sides_count.get('flat',0)}")
    print(f"  unresolved (action_id > 0 but no strike): {n_unresolved}")
    print(f"  no_order (action_id == 0 / flat): {n_no_order}")

    # Optional feature_parity check: rebuild the chosen day from raw v2 and
    # compare the bundle's row vs the rebuilt row.
    if args.parity_day:
        print(f"\nFeature parity check on day={args.parity_day}...", flush=True)
        ds = V2Dataset.load()
        cfg = GuardrailConfig()
        execution = meta.get("execution_window", {})
        rebuilt = build_action_surface_bundle(
            ds, cfg, equity=25_000.0,
            days=[args.parity_day],
            history_bars=int(meta.get("history_bars", DEFAULT_HISTORY_BARS)),
            top_k_contracts=top_k,
            execution_start_bar=int(execution.get("start_bar", 15)),
            execution_end_bar=int(execution.get("end_bar", 120)),
            utility_horizon_bars=int(meta.get("utility_horizon_bars", 60)),
            contract_selection_mode=str(meta.get("contract_selection_mode", "risk_band")),
        )
        rebuilt_rows = rebuilt["rows"].reset_index(drop=True)
        # Match each rebuilt row to its position in the full bundle by bar_index.
        full_day_idx = rows_idx[rows_idx["day"].astype(str) == args.parity_day].sort_values("bar_index").reset_index(drop=True)
        if len(full_day_idx) != len(rebuilt_rows):
            print(f"  WARN: row counts differ — full={len(full_day_idx)} rebuilt={len(rebuilt_rows)}")
        n_compared = 0
        n_passed = 0
        max_diff = 0.0
        all_missing: set[str] = set()
        all_mismatch: set[str] = set()
        for i in range(min(len(full_day_idx), len(rebuilt_rows))):
            hist = {
                name: float(full_day_idx.iloc[i][name])
                for name in feature_schema
                if name in full_day_idx.columns
            }
            live = {
                name: float(rebuilt_rows.iloc[i][name])
                for name in feature_schema
                if name in rebuilt_rows.columns
            }
            res = compare_feature_rows(
                feature_names=feature_schema,
                historical=hist,
                live_style=live,
            )
            n_compared += 1
            if res.passed:
                n_passed += 1
            max_diff = max(max_diff, res.max_abs_diff)
            for f in res.missing_features:
                all_missing.add(f)
            for f in res.mismatched_features:
                all_mismatch.add(f)
        print(f"  bars compared: {n_compared}  passed: {n_passed}")
        print(f"  max abs feature diff: {max_diff:.3e}")
        print(f"  missing features (any bar): {sorted(all_missing) or '<none>'}")
        print(f"  mismatched features (any bar): {sorted(all_mismatch) or '<none>'}")
        gate_pass = (n_passed == n_compared) and not all_missing and not all_mismatch
        print(f"  feature_parity gate: {'PASS' if gate_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
