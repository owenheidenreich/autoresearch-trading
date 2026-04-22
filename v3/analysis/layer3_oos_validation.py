"""Out-of-sample validation of Layer-2 + Layer-3 on the 20 trading days
between 2026-03-05 and 2026-04-01 — data that exists in the V2Dataset
cache but was never included in the Layer-2 export bundle's test sets.

This is the OOS retest the prior strategic plan called for, executed on
cached data we already have (no new vendor needed).

Pipeline:
1. Identify days in V2Dataset not in any existing fold's test_days.
2. For each new day:
   a. build_labeled_day -> bars + sidecar
   b. build_export_rows_for_day -> per-bar feature rows (same as bundle)
3. Apply fold-4 Layer-2 entry/side model to predict scores.
4. Apply fold-4 calibration thresholds (no recalibration -- clean OOS).
5. per_day_choice -> chosen trade per day.
6. compute_time_stop_pnl_for_direction -> PnL (Layer-2 alone).
7. Train a "fold-5" Layer-3 HistGB model on ALL 275 existing chosen
   trades from the in-sample Layer-3 dataset.
8. Apply Layer-3 fold-5 model at threshold 0.17 to new chosen trades.
9. Report Layer-2-alone vs Layer-2+Layer-3 OOS metrics.

Verdict criteria:
- HARD PASS: Layer-2+Layer-3 OOS PF >= 1.50 AND beats Layer-2-alone
  OOS by >= 0.20 PF (system generalizes; lift transfers)
- SOFT PASS: Layer-2 alone OOS PF >= 1.0 AND Layer-3 doesn't crash
  it (entry generalizes; exit may or may not)
- FAIL: Layer-2 alone OOS PF < 1.0 (entry doesn't generalize;
  in-sample lift was selection bias)

Run:
    python -m v3.analysis.layer3_oos_validation
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v3.analysis.layer3_train_replay import (
    CORRECTED_BASELINE_DD,
    CORRECTED_BASELINE_PF,
    HEURISTIC_BASELINE_DD,
    HEURISTIC_BASELINE_PF,
    TRADE_STATE_NAMES,
    _build_per_trade_data,
    _flatten_to_rows,
    _replay_with_model,
    _agg,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_export_rows_for_day,
    build_labeled_day,
    compute_time_stop_pnl_for_direction,
    effective_direction,
    finalize_export_dataframe,
    load_export_bundle,
    load_json,
    load_pickle,
    per_day_choice,
    replay_metrics_from_pnls,
    teacher_direction_hint_from_row,
)
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_oos_validation")
DEFAULT_SEED = 42
DEFAULT_LAYER3_THRESHOLD = 0.17  # peak from Stage 4 reality checks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOS validation on cached 2026-03-05+ days.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_LAYER3_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _identify_oos_days(ds: V2Dataset, fold4_test_days: list[str]) -> list[str]:
    all_days = sorted(set(ds.dates))
    last_in_sample = max(fold4_test_days)
    return sorted([d for d in all_days if d > last_in_sample])


def _build_oos_export_rows(
    ds: V2Dataset, cfg: GuardrailConfig, days: list[str], equity: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict]]:
    """Build per-bar export rows for the OOS days, mirroring layer2_dataset.pkl format."""
    all_rows: list[dict[str, Any]] = []
    day_cache: dict[str, tuple] = {}
    for i, day in enumerate(days):
        log, sidecar = build_labeled_day(ds, day, cfg, equity=equity)
        if log is None or sidecar is None:
            print(f"  skipping {day}: no labeled bars/sidecar")
            continue
        # day_fold_id -1 placeholder (these are OOS, no fold)
        rows = build_export_rows_for_day(ds, day, log, day_fold_id=-1, sidecar=sidecar)
        all_rows.extend(rows)
        day_cache[day] = (log, sidecar)
        if (i + 1) % 5 == 0:
            print(f"  built export rows for {i+1}/{len(days)} days; rows so far: {len(all_rows)}")
    df = finalize_export_dataframe(all_rows)
    return df, day_cache


def _apply_layer2_models(
    df: pd.DataFrame, fold_dir: str, feature_names: list[str],
    direction_mode: str, score_mode: str, side_score_weight: float,
    entry_threshold: float, side_threshold: float,
) -> pd.DataFrame:
    entry_model = _load_pkl(os.path.join(fold_dir, "entry_model.pkl"))
    side_model = _load_pkl(os.path.join(fold_dir, "side_model.pkl"))
    X = df[feature_names].to_numpy(dtype=np.float32)
    entry_pred = entry_model.predict(X)
    side_pred = side_model.predict(X)
    out = df.copy()
    out["entry_score"] = entry_pred
    out["side_score"] = side_pred
    out["side_conf"] = np.abs(out["side_score"])
    out["predicted_direction"] = np.where(out["side_score"] > 0, "call", "put")
    # Adjust for surface guardrails (only one direction passable)
    only_call = (out["has_passing_call"] > 0.5) & (out["has_passing_put"] <= 0.5)
    only_put = (out["has_passing_put"] > 0.5) & (out["has_passing_call"] <= 0.5)
    neither = (out["has_passing_call"] <= 0.5) & (out["has_passing_put"] <= 0.5)
    out.loc[only_call, "predicted_direction"] = "call"
    out.loc[only_put, "predicted_direction"] = "put"
    out.loc[only_call | only_put, "side_conf"] = 1.0
    out.loc[neither, "predicted_direction"] = ""
    out.loc[neither, "side_conf"] = 0.0
    out.loc[~((out["has_passing_call"] > 0.5) & (out["has_passing_put"] > 0.5)), "side_score"] = 0.0
    out["effective_direction"] = out.apply(
        effective_direction, axis=1, direction_mode=direction_mode, policy_mode="scalar_side",
    )
    return out


def _select_oos_trades(
    df: pd.DataFrame, day_cache: dict, entry_threshold: float, side_threshold: float,
    score_mode: str, side_score_weight: float, direction_mode: str,
) -> pd.DataFrame:
    trades = []
    for day, day_rows in df.groupby("day"):
        chosen = per_day_choice(
            day_rows, entry_threshold, side_threshold,
            score_mode=score_mode, side_score_weight=side_score_weight,
            policy_mode="scalar_side", direction_mode=direction_mode,
        )
        if chosen is None:
            continue
        log, sidecar = day_cache.get(day, (None, None))
        if log is None or sidecar is None:
            continue
        bar_index = int(chosen["bar_index"])
        bar = next((b for b in log.bars if b.bar_index == bar_index), None)
        if bar is None:
            continue
        direction = str(chosen["effective_direction"])
        pnl = compute_time_stop_pnl_for_direction(bar, sidecar, direction)
        if pnl is None:
            continue
        trades.append({
            "day": day, "bar_index": bar_index, "direction": direction,
            "pnl": float(pnl),
            "entry_value_raw": float(chosen["entry_value_raw"]) if pd.notna(chosen["entry_value_raw"]) else None,
            "oracle_slice_outcome": str(chosen.get("oracle_slice_outcome", "")),
            "fold_idx": -1,  # OOS marker
        })
    return pd.DataFrame(trades)


def _train_fold5_layer3(
    in_sample_trade_data: list[dict], seed: int,
) -> HistGradientBoostingClassifier:
    """Train Layer-3 on ALL 275 in-sample chosen trades (folds 0-4)."""
    X, y, _ = _flatten_to_rows(in_sample_trade_data)
    print(f"  Training fold-5 Layer-3 on {len(in_sample_trade_data)} trades / {len(X)} bar-rows; pos_rate={y.mean():.3f}")
    m = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=seed + 999, early_stopping=False,
    )
    m.fit(X, y)
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds_meta = list(bundle["meta"]["folds"])
    feature_names = list(bundle["meta"]["feature_names"])
    fold4 = folds_meta[-1]
    fold4_test = list(fold4["test_days"])
    fold_dir = os.path.join(args.baseline_run_dir, "folds", str(int(fold4["fold_idx"])))
    print(f"Using fold-{fold4['fold_idx']} model from {fold_dir}")

    manifest = load_pickle(os.path.join(args.baseline_run_dir, "manifest.pkl"))
    direction_mode = manifest.get("direction_mode", "teacher_if_triggered_else_put")
    score_mode = manifest.get("score_mode", "product")
    side_score_weight = float(manifest.get("side_score_weight", 0.15))
    print(f"manifest: direction_mode={direction_mode}, score_mode={score_mode}, side_score_weight={side_score_weight}")

    calib = load_json(os.path.join(fold_dir, "calibration.json"))
    entry_threshold = float(calib["entry_threshold"])
    side_threshold = float(calib["side_threshold"])
    print(f"Calibration thresholds: entry={entry_threshold:.4f}, side={side_threshold:.4f}")

    print()
    print("Loading V2Dataset...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    oos_days = _identify_oos_days(ds, fold4_test)
    print(f"OOS days identified: {len(oos_days)} from {oos_days[0]} to {oos_days[-1]}")

    # === Step 1: Build OOS export rows ===
    print()
    print("Building OOS export rows (matches Layer-2 dataset bundle format)...")
    df_oos, day_cache = _build_oos_export_rows(ds, cfg, oos_days, args.equity)
    print(f"OOS export rows: {len(df_oos)}")

    # === Step 2: Apply fold-4 Layer-2 model + thresholds ===
    print()
    print("Applying fold-4 Layer-2 model (entry + side) ...")
    df_oos_pred = _apply_layer2_models(
        df_oos, fold_dir, feature_names, direction_mode, score_mode, side_score_weight,
        entry_threshold, side_threshold,
    )

    # === Step 3: Per-day choice -> chosen trades ===
    print()
    print("Selecting OOS chosen trades via per_day_choice ...")
    oos_trades = _select_oos_trades(
        df_oos_pred, day_cache, entry_threshold, side_threshold,
        score_mode, side_score_weight, direction_mode,
    )
    print(f"OOS chosen trades: {len(oos_trades)} ({len(oos_trades)/max(len(oos_days),1):.2f} per day)")
    if oos_trades.empty:
        print("ERROR: zero chosen trades on OOS days. Cannot evaluate.")
        return 1

    # === Step 4: Layer-2 alone metrics on OOS ===
    print()
    print("=" * 100)
    print("Layer-2 alone OOS metrics (asymmetric spread, matches in-sample pipeline)")
    print("=" * 100)
    sorted_t = oos_trades.sort_values(["day", "bar_index"])
    pnls = sorted_t["pnl"].astype(float).tolist()
    layer2_alone = replay_metrics_from_pnls(pnls, args.equity)
    layer2_alone["trades"] = float(len(pnls))
    layer2_alone["mean_pnl"] = float(np.mean(pnls))
    layer2_alone["call_pct"] = float((sorted_t["direction"] == "call").mean())
    print(f"  PF={layer2_alone['pf']:.3f}  DD={layer2_alone['max_dd_pct']:.1f}%  mean=${layer2_alone['mean_pnl']:.0f}  trades={int(layer2_alone['trades'])}  call%={layer2_alone['call_pct']*100:.1f}")
    print(f"  In-sample anchor (Layer-2 alone, corrected): PF={CORRECTED_BASELINE_PF:.3f} DD={CORRECTED_BASELINE_DD:.1f}%")
    print(f"  In-sample anchor (Layer-2 alone, fold 4):    PF=1.801 DD=32.2%")

    # === Step 5: Build per-trade Layer-3 data for OOS chosen trades ===
    print()
    print("Building per-trade Layer-3 data for OOS chosen trades...")
    paths_cache: dict[int, dict] = {}
    minute_map_cache: dict[str, dict] = {}
    oos_trade_data: list[dict] = []
    for _, trade in oos_trades.iterrows():
        day = str(trade["day"])
        log, sidecar = day_cache[day]
        # Use fake "fold_idx=-1" -> won't match training fold
        td = _build_per_trade_data(
            trade, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            td["fold_idx"] = -1  # OOS marker
            oos_trade_data.append(td)
    print(f"OOS Layer-3 trade data: {len(oos_trade_data)}")

    # === Step 6: Train a "fold-5" Layer-3 model on ALL 275 in-sample chosen trades ===
    print()
    print("Training fold-5 Layer-3 model on all in-sample chosen trades...")
    in_sample_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    in_sample_trade_data: list[dict] = []
    for _, trade in in_sample_trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=args.equity)
            if log_d is None:
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        td = _build_per_trade_data(
            trade, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            in_sample_trade_data.append(td)
    layer3_model = _train_fold5_layer3(in_sample_trade_data, args.seed)

    # === Step 7: Apply Layer-3 to OOS trades at threshold ===
    print()
    print(f"Applying Layer-3 fold-5 model at threshold {args.threshold} ...")
    sim_rows = _replay_with_model(oos_trade_data, layer3_model, args.threshold)
    sim_df = pd.DataFrame(sim_rows)
    if sim_df.empty:
        print("ERROR: Layer-3 produced zero exits.")
        return 1
    sim_sorted = sim_df.sort_values(["day", "entry_bar"])

    # === Step 8: Layer-2 + Layer-3 OOS metrics ===
    print()
    print("=" * 100)
    print(f"Layer-2 + Layer-3 OOS metrics (corrected exit spread, threshold={args.threshold})")
    print("=" * 100)
    pnls_l3 = sim_sorted["exit_pnl"].astype(float).tolist()
    composed = replay_metrics_from_pnls(pnls_l3, args.equity)
    composed["trades"] = float(len(pnls_l3))
    composed["mean_pnl"] = float(np.mean(pnls_l3))
    composed["call_pct"] = float((sim_sorted["direction"] == "call").mean())
    composed["mean_bars_held"] = float(sim_sorted["bars_held"].mean())
    early_exits = float((sim_sorted["trigger"] == "model").mean())
    composed["early_exit_share"] = early_exits
    print(f"  PF={composed['pf']:.3f}  DD={composed['max_dd_pct']:.1f}%  mean=${composed['mean_pnl']:.0f}  trades={int(composed['trades'])}")
    print(f"  Mean bars held: {composed['mean_bars_held']:.1f}  Early exit share: {early_exits*100:.1f}%")
    print(f"  In-sample anchor (composed): PF=2.228 DD=26.3%")
    print(f"  In-sample anchor (composed, fold 4): PF=3.492 DD=29.1%")

    # === Verdict ===
    print()
    print("=" * 100)
    print("OOS Verdict")
    print("=" * 100)
    pf_l2 = layer2_alone["pf"]
    pf_l3 = composed["pf"]
    delta = pf_l3 - pf_l2

    if pf_l2 < 1.0:
        verdict = (f"FAIL -- Layer-2 alone OOS PF {pf_l2:.3f} < 1.0; entry doesn't generalize. "
                   f"In-sample 1.472 lift was selection bias.")
    elif pf_l3 >= 1.50 and delta >= 0.20:
        verdict = (f"HARD PASS -- Layer-3 OOS PF {pf_l3:.3f} >= 1.50 AND lift over Layer-2-alone "
                   f"{delta:+.3f} >= +0.20. System generalizes; learned exit transfers.")
    elif pf_l2 >= 1.0 and pf_l3 >= pf_l2:
        verdict = (f"SOFT PASS -- Layer-2 alone OOS PF {pf_l2:.3f} >= 1.0 (entry generalizes); "
                   f"Layer-3 PF {pf_l3:.3f} (Δ {delta:+.3f}) doesn't crash it but lift is modest.")
    else:
        verdict = (f"MIXED -- Layer-2 alone OOS PF {pf_l2:.3f} >= 1.0 but Layer-3 hurts "
                   f"(Δ {delta:+.3f}); entry generalizes but exit doesn't.")
    print(f"  {verdict}")

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_oos_days": int(len(oos_days)),
            "n_oos_chosen_trades": int(len(oos_trades)),
            "oos_days_first": oos_days[0] if oos_days else None,
            "oos_days_last": oos_days[-1] if oos_days else None,
            "fold_used": int(fold4["fold_idx"]),
            "entry_threshold": entry_threshold,
            "side_threshold": side_threshold,
            "layer3_threshold": float(args.threshold),
        },
        "layer2_alone_oos": layer2_alone,
        "layer2_layer3_oos": composed,
        "anchors": {
            "in_sample_layer2_alone_corrected_pf": CORRECTED_BASELINE_PF,
            "in_sample_layer2_alone_corrected_dd": CORRECTED_BASELINE_DD,
            "in_sample_composed_pf_thr017": 2.228,
            "in_sample_composed_dd_thr017": 26.3,
            "in_sample_layer2_alone_fold4_pf": 1.801,
            "in_sample_composed_fold4_pf_thr017": 3.492,
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "oos_validation.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    oos_trades.to_csv(os.path.join(args.out_dir, "oos_chosen_trades.csv"), index=False)
    sim_sorted.to_csv(os.path.join(args.out_dir, "oos_layer3_trades.csv"), index=False)
    print()
    print(f"Saved: {out}")
    print(f"Saved: {os.path.join(args.out_dir, 'oos_chosen_trades.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'oos_layer3_trades.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
