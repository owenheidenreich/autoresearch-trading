"""Side-aware calibration: per-side abstention thresholds for the unified policy.

Hypothesis path 3 from the SPX live-readiness next-plan: the model
under-scores puts but the underlying utility may still clear a per-side
threshold tuned on put-only validation data. By calibrating
(decision_margin, min_win_prob, max_stopout_prob) separately for the
best-call slot and the best-put slot, then picking the highest-scoring
surviving side at each bar, we test whether calibration alone can claw
back put winners without a retrain.

This is a postprocessing analysis. It re-runs predict on the loaded
window-K model and rebuilds the daily selection with side-aware gates,
comparing the new chosen-trade PnL on the simulated-L3 oracle exits
against production.

Caveat: the simulated-L3 oracle was trained on a call-heavy chosen-trade
set, so put-side simulated exit PnL is extrapolation. Treat dollar
magnitudes as upper-bound; the SIGN of the result (does side-aware
calibration recover any put winners) is robust because it's measured on
the same hybrid_live label the model trained against.

Usage:

    .venv/bin/python -m v3.analysis.side_aware_calibration \\
        --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \\
        --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42.npz \\
        --model v3/artifacts/layer2_unified_policy_spx_live_hybrid_001_seed42/seed_42 \\
        --window 5 \\
        --out v3/artifacts/side_aware_calibration/seed42_w5.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from v3.harness.rolling_windows import generate_rolling_windows
from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle, load_pickle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--simulated-l3-oracle", required=True)
    p.add_argument("--model", required=True, help="Path to seed_NN dir holding window_*/{model.pkl,calibration.json}")
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _per_row_best_side(
    util: np.ndarray,
    win_prob: np.ndarray,
    stopout_prob: np.ndarray,
    label_target: np.ndarray,
    l3_pnl: np.ndarray,
    tradeable: np.ndarray,
    top_k: int,
) -> dict[str, np.ndarray]:
    """For each row, find best-call and best-put action info."""
    n_rows = util.shape[0]
    rows = np.arange(n_rows)
    util_tokens = util[:, 1:].astype(np.float64).copy()
    util_tokens[~tradeable[:, 1:]] = -np.inf
    flat = util[:, 0].astype(np.float64)

    call_util = util_tokens[:, :top_k]
    put_util = util_tokens[:, top_k:]
    best_call_slot = call_util.argmax(axis=1)
    best_put_slot = put_util.argmax(axis=1)
    best_call_score = call_util[rows, best_call_slot]
    best_put_score = put_util[rows, best_put_slot]
    best_call_action = best_call_slot + 1
    best_put_action = best_put_slot + 1 + top_k

    # Per-side margin = score - flat
    bc_margin = best_call_score - flat
    bp_margin = best_put_score - flat

    # Per-side win, stopout (at the best-side action)
    bc_win = win_prob[rows, best_call_action] if win_prob is not None else np.full(n_rows, np.nan)
    bp_win = win_prob[rows, best_put_action] if win_prob is not None else np.full(n_rows, np.nan)
    bc_stop = stopout_prob[rows, best_call_action] if stopout_prob is not None else np.full(n_rows, np.nan)
    bp_stop = stopout_prob[rows, best_put_action] if stopout_prob is not None else np.full(n_rows, np.nan)

    # Per-side label target (hybrid_live) and L3 exit PnL at the best-side action
    bc_label = label_target[rows, best_call_action]
    bp_label = label_target[rows, best_put_action]
    bc_l3pnl = l3_pnl[rows, best_call_action]
    bp_l3pnl = l3_pnl[rows, best_put_action]

    return {
        "best_call_action": best_call_action,
        "best_put_action": best_put_action,
        "best_call_score": best_call_score,
        "best_put_score": best_put_score,
        "best_call_margin": bc_margin,
        "best_put_margin": bp_margin,
        "best_call_win": bc_win,
        "best_put_win": bp_win,
        "best_call_stopout": bc_stop,
        "best_put_stopout": bp_stop,
        "best_call_label": bc_label,
        "best_put_label": bp_label,
        "best_call_l3_pnl": bc_l3pnl,
        "best_put_l3_pnl": bp_l3pnl,
        "tradeable_call": np.isfinite(best_call_score),
        "tradeable_put": np.isfinite(best_put_score),
    }


def _calibrate_one_side(
    df: pd.DataFrame,
    side: str,
    pnl_col: str = "label_pnl",
) -> dict[str, Any]:
    """Calibrate (decision_margin, min_win_prob, max_stopout_prob) for one side.

    df has one row per bar with: day, bar_index, margin, win, stopout, l3_pnl, label_pnl.
    Picks one trade per day (highest margin) among rows passing the gates.
    Maximizes objective_pf with in-band trade-share and minimum-trades preference.
    """
    total_days = int(df["day"].nunique())
    pos_margins = df.loc[df["margin"] >= 0.0, "margin"].to_numpy(dtype=np.float64)
    if pos_margins.size == 0:
        return {
            "side": side,
            "decision_margin": float("inf"),
            "min_win_prob": 1.0,
            "max_stopout_prob": 0.0,
            "objective_pf": 0.0,
            "trade_share": 0.0,
            "trades": 0,
        }
    quantiles = (
        np.quantile(pos_margins, np.linspace(0.0, 0.95, 16))
        if pos_margins.size >= 5
        else np.array([], dtype=np.float64)
    )
    margin_grid = np.unique(np.concatenate(([0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35], quantiles)))
    margin_grid = margin_grid[margin_grid >= 0.0]
    win_grid = [0.0, 0.20, 0.30, 0.40, 0.50]
    stopout_grid = [1.0, 0.80, 0.70, 0.55, 0.45]
    min_trades_required = max(4, int(round(0.05 * total_days)))

    best_payload: dict[str, Any] | None = None
    best_key = None
    any_qualified = False
    for margin in margin_grid:
        for min_win in win_grid:
            for max_stop in stopout_grid:
                gate = (
                    (df["margin"] >= margin)
                    & (df["win"].fillna(0.0) >= min_win)
                    & (df["stopout"].fillna(1.0) <= max_stop)
                )
                eligible = df[gate]
                if eligible.empty:
                    pf_metric = 0.0
                    n_trades = 0
                    mean_pnl = 0.0
                    trade_share = 0.0
                else:
                    idx = eligible.groupby("day")["margin"].idxmax()
                    chosen = eligible.loc[idx]
                    pnls = chosen[pnl_col].to_numpy(dtype=np.float64)
                    pos = pnls[pnls > 0].sum()
                    neg = -pnls[pnls < 0].sum()
                    pf_metric = float(pos / neg) if neg > 1e-9 else (float("inf") if pos > 0 else 0.0)
                    n_trades = int(len(chosen))
                    mean_pnl = float(pnls.mean()) if len(pnls) else 0.0
                    trade_share = float(n_trades / max(total_days, 1))
                in_band = 0.10 <= trade_share <= 0.60
                enough_trades = n_trades >= min_trades_required
                pf_qualified = enough_trades and pf_metric >= 1.0 and pf_metric < float("inf")
                if pf_qualified:
                    any_qualified = True
                key = (
                    1 if (pf_qualified and in_band) else 0,
                    1 if pf_qualified else 0,
                    1 if enough_trades else 0,
                    pf_metric if pf_metric != float("inf") else 1e9,
                    mean_pnl,
                    -abs(trade_share - 0.30),
                    n_trades,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_payload = {
                        "side": side,
                        "decision_margin": float(margin),
                        "min_win_prob": float(min_win),
                        "max_stopout_prob": float(max_stop),
                        "objective_pf": float(pf_metric) if pf_metric != float("inf") else 1e9,
                        "trade_share": float(trade_share),
                        "trades": int(n_trades),
                        "mean_pnl": float(mean_pnl),
                        "pf_qualified": bool(pf_qualified),
                        "in_band": bool(in_band),
                    }
    # Strict fail-out: if no pf-qualified config exists, return an
    # impossible threshold so the OOS gate rejects everything for this side.
    if not any_qualified:
        return {
            "side": side,
            "decision_margin": float("inf"),
            "min_win_prob": 1.0,
            "max_stopout_prob": 0.0,
            "objective_pf": 0.0,
            "trade_share": 0.0,
            "trades": 0,
            "mean_pnl": 0.0,
            "pf_qualified": False,
            "in_band": False,
            "fail_out": True,
            "best_unqualified_pf": float(best_payload.get("objective_pf", 0.0)) if best_payload else 0.0,
        }
    return best_payload or {}


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bundle = load_export_bundle(args.dataset)
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    top_k = int(meta["top_k_contracts_per_side"])
    al = bundle["action_labels"]
    tradeable = np.nan_to_num(al["tradeable_mask"], nan=0.0) > 0.5

    z = np.load(args.simulated_l3_oracle, allow_pickle=True)
    l3_pnl = z["l3_exit_pnl"]
    l3_exit_bar = z["l3_exit_bar"]

    # Compute hybrid_live label per row
    print("Recomputing hybrid_live target utility per row...", flush=True)
    n_rows = len(rows)
    n_actions = l3_pnl.shape[1]
    hybrid_target = np.full((n_rows, n_actions), np.nan, dtype=np.float64)
    entry_mid = al["entry_fill_mid"]
    entry_spread = al["entry_spread_fraction"]
    entry_bar = al["entry_fill_bar"]
    stopout_risk = al["stopout_risk"]
    for r in range(n_rows):
        if not tradeable[r, 1:].any():
            hybrid_target[r, 0] = 0.0
            continue
        for a in range(n_actions):
            if a == 0:
                hybrid_target[r, a] = 0.0
                continue
            pnl = l3_pnl[r, a] if np.isfinite(l3_pnl[r, a]) else None
            em = float(entry_mid[r, a])
            if not np.isfinite(em) or em <= 0:
                continue
            eb = int(entry_bar[r, a]) if np.isfinite(entry_bar[r, a]) else 0
            xb = int(l3_exit_bar[r, a]) if np.isfinite(l3_exit_bar[r, a]) and l3_exit_bar[r, a] >= 0 else None
            hybrid_target[r, a] = hybrid_live_utility(
                pnl=pnl,
                entry_mid=em,
                spread_fraction=float(entry_spread[r, a]),
                stopout_risk=float(stopout_risk[r, a]),
                entry_bar=eb,
                exit_bar=xb,
            )

    # Resolve window
    days_arr = rows["day"].astype(str).to_numpy()
    unique_days = sorted(rows["day"].astype(str).unique().tolist())
    windows = generate_rolling_windows(unique_days)
    target = next((w for w in windows if int(w.window_idx) == int(args.window)), None)
    if target is None:
        raise RuntimeError(f"Window {args.window} not found")
    val_days_set = set(target.val_days)
    oos_days_set = set(target.oos_days)
    val_mask = pd.Series(days_arr).isin(val_days_set).to_numpy()
    oos_mask = pd.Series(days_arr).isin(oos_days_set).to_numpy()

    # Predict using window model
    model_path = os.path.join(args.model, f"window_{int(args.window):02d}", "model.pkl")
    print(f"Loading model: {model_path}", flush=True)
    predictor = load_pickle(model_path)
    scalar = rows.loc[:, meta["scalar_feature_names"]].to_numpy(dtype=np.float32)
    seq = bundle["sequence_features"]
    seq_mask = bundle["sequence_mask"]
    contracts = bundle["contract_features"]
    contract_mask = bundle["contract_mask"]
    pred = predictor.predict(scalar, seq, seq_mask, contracts, contract_mask, batch_size=2048)
    util = pred["utility"]
    win_prob = pred.get("win_prob")
    stopout_prob = pred.get("stopout_prob")

    side_data = _per_row_best_side(util, win_prob, stopout_prob, hybrid_target, l3_pnl, tradeable, top_k)

    # Build per-side per-row frames
    def make_side_frame(side: str, cohort_mask: np.ndarray) -> pd.DataFrame:
        if side == "call":
            margin = side_data["best_call_margin"]
            win = side_data["best_call_win"]
            stop = side_data["best_call_stopout"]
            label_pnl = side_data["best_call_label"]
            l3 = side_data["best_call_l3_pnl"]
            avail = side_data["tradeable_call"]
            action = side_data["best_call_action"]
        else:
            margin = side_data["best_put_margin"]
            win = side_data["best_put_win"]
            stop = side_data["best_put_stopout"]
            label_pnl = side_data["best_put_label"]
            l3 = side_data["best_put_l3_pnl"]
            avail = side_data["tradeable_put"]
            action = side_data["best_put_action"]
        keep = cohort_mask & avail & np.isfinite(margin) & np.isfinite(label_pnl)
        if keep.sum() == 0:
            return pd.DataFrame()
        df = pd.DataFrame({
            "day": days_arr[keep],
            "bar_index": rows["bar_index"].to_numpy()[keep],
            "side": side,
            "action_id": action[keep],
            "margin": margin[keep].astype(np.float64),
            "win": win[keep].astype(np.float64),
            "stopout": stop[keep].astype(np.float64),
            "label_pnl": label_pnl[keep].astype(np.float64),
            "l3_pnl": l3[keep].astype(np.float64),
            "score": (
                side_data["best_call_score"][keep] if side == "call" else side_data["best_put_score"][keep]
            ).astype(np.float64),
        })
        return df

    val_call = make_side_frame("call", val_mask)
    val_put = make_side_frame("put", val_mask)
    oos_call = make_side_frame("call", oos_mask)
    oos_put = make_side_frame("put", oos_mask)

    # Calibrate per side on val
    print(f"Calibrating side-aware gates on val ({len(val_call)} call rows, {len(val_put)} put rows)...", flush=True)
    cal_call = _calibrate_one_side(val_call, "call")
    cal_put = _calibrate_one_side(val_put, "put")
    # Global calibration (recompute the same way for fair comparison)
    val_global = pd.concat([val_call, val_put], ignore_index=True)
    cal_global = _calibrate_one_side(val_global, "global")

    print(f"  call calibration: {cal_call}", flush=True)
    print(f"  put  calibration: {cal_put}", flush=True)
    print(f"  global ref:      {cal_global}", flush=True)

    # Apply on OOS: side-aware selection
    def apply_gate(df: pd.DataFrame, cal: dict[str, Any]) -> pd.DataFrame:
        if df.empty or not cal:
            return df.iloc[:0]
        gate = (
            (df["margin"] >= cal["decision_margin"])
            & (df["win"].fillna(0.0) >= cal["min_win_prob"])
            & (df["stopout"].fillna(1.0) <= cal["max_stopout_prob"])
        )
        return df[gate].copy()

    oos_call_pass = apply_gate(oos_call, cal_call)
    oos_put_pass = apply_gate(oos_put, cal_put)

    # Side-aware selection: per day, pick highest-score-surviving side, else flat
    combined = pd.concat([oos_call_pass, oos_put_pass], ignore_index=True)
    if combined.empty:
        side_aware_chosen = combined
    else:
        idx = combined.groupby("day")["score"].idxmax()
        side_aware_chosen = combined.loc[idx].sort_values(["day", "bar_index"]).reset_index(drop=True)

    # Global-calibrated selection (for ablation)
    oos_call_g = apply_gate(oos_call, cal_global)
    oos_put_g = apply_gate(oos_put, cal_global)
    combined_g = pd.concat([oos_call_g, oos_put_g], ignore_index=True)
    if combined_g.empty:
        global_chosen = combined_g
    else:
        idx_g = combined_g.groupby("day")["score"].idxmax()
        global_chosen = combined_g.loc[idx_g].sort_values(["day", "bar_index"]).reset_index(drop=True)

    def trade_metrics(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"trades": 0, "trade_share_oos": 0.0, "pf_label": 0.0, "pf_l3": 0.0,
                    "mean_label_pnl": 0.0, "mean_l3_pnl": 0.0, "total_label_pnl": 0.0, "total_l3_pnl": 0.0,
                    "calls": 0, "puts": 0}
        pl = df["label_pnl"].to_numpy()
        l3 = df["l3_pnl"].to_numpy()
        pf_label = float(pl[pl > 0].sum() / max(-pl[pl < 0].sum(), 1e-9)) if (pl < 0).any() else float("inf") if (pl > 0).any() else 0.0
        pf_l3 = float(l3[l3 > 0].sum() / max(-l3[l3 < 0].sum(), 1e-9)) if (l3 < 0).any() else float("inf") if (l3 > 0).any() else 0.0
        return {
            "trades": int(len(df)),
            "trade_share_oos": float(len(df) / max(len(set(oos_days_set)), 1)),
            "pf_label": pf_label,
            "pf_l3": pf_l3,
            "mean_label_pnl": float(pl.mean()),
            "mean_l3_pnl": float(l3.mean()),
            "total_label_pnl": float(pl.sum()),
            "total_l3_pnl": float(l3.sum()),
            "calls": int((df["side"] == "call").sum()),
            "puts": int((df["side"] == "put").sum()),
        }

    side_aware_metrics = trade_metrics(side_aware_chosen)
    global_metrics = trade_metrics(global_chosen)

    # Compare to production (load chosen_trades.pkl for this seed/window)
    prod_chosen_path = os.path.join(args.model, "chosen_trades.pkl")
    production_metrics = None
    if os.path.exists(prod_chosen_path):
        prod = load_pickle(prod_chosen_path)
        prod_w = prod[prod["window_idx"] == int(args.window)].copy() if "window_idx" in prod.columns else prod.copy()
        if not prod_w.empty:
            # For production we have chosen_objective_pnl (= hybrid_live label PnL of chosen trade)
            pl_prod = prod_w["chosen_objective_pnl"].to_numpy(dtype=np.float64)
            pf_prod = float(pl_prod[pl_prod > 0].sum() / max(-pl_prod[pl_prod < 0].sum(), 1e-9)) if (pl_prod < 0).any() else float("inf")
            production_metrics = {
                "trades": int(len(prod_w)),
                "calls": int((prod_w["chosen_side"] == "call").sum()),
                "puts": int((prod_w["chosen_side"] == "put").sum()),
                "pf_label": pf_prod,
                "mean_label_pnl": float(pl_prod.mean()),
                "total_label_pnl": float(pl_prod.sum()),
            }

    out = {
        "window": int(args.window),
        "n_oos_days": len(oos_days_set),
        "calibration": {
            "side_aware_call": cal_call,
            "side_aware_put": cal_put,
            "global_reference": cal_global,
        },
        "selection_results": {
            "side_aware": side_aware_metrics,
            "global_recalibrated": global_metrics,
            "production": production_metrics,
        },
        "side_aware_chosen_trades": side_aware_chosen.to_dict("records"),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    print(f"Wrote {args.out}", flush=True)
    print()
    print("=== W{} comparison ===".format(args.window))
    print(f"{'variant':<25} {'trades':>7} {'calls':>6} {'puts':>5} {'pf_label':>10} {'mean$/trade':>12} {'total$':>10}")
    print("-" * 90)
    for label, m in [
        ("production (orig calib)", production_metrics),
        ("global recalibrated",     global_metrics),
        ("side-aware",              side_aware_metrics),
    ]:
        if m is None:
            print(f"  {label:<25} (n/a)")
            continue
        c = m.get("calls", 0)
        p = m.get("puts", 0)
        print(f"  {label:<25} {m['trades']:>7} {c:>6} {p:>5} {m['pf_label']:>10.3f} {m.get('mean_label_pnl',0):>12.2f} {m.get('total_label_pnl',0):>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
