"""W5 day-level decomposition + side-substitution counterfactual.

Per the SPX live-readiness next-plan: don't run another promotion until we
know whether W5's failure is "puts existed but the model under-scored them"
vs "no put opportunities existed in W5" vs "calibration filtered them out".

For each W5 OOS day of the spx_live_hybrid_001 seed-42 run, this:

1. Pulls the daily chosen trade (if any) from the production chosen_trades.pkl,
   joining to the L3 routed exit decision at thr=0.20.
2. Computes per-row label-best-call and label-best-put utility under
   hybrid_live (the training target), and the oracle-best action overall.
3. Reports the model's predicted best-call vs best-put score at the chosen
   bar and the calibration margin used.
4. Adds three counterfactuals (W5-only, simulated-L3 exits):
   - production: what the model + L3 routed actually delivered at thr=0.20
   - label-side-swap: at each chosen bar, if label-best-put beats
     label-best-call by >= margin dollars, swap to the best put (for a
     sweep of margins); else keep the production trade
   - oracle-best: per chosen bar, take the action with the highest hybrid_live
     target utility (upper bound)

The L3 oracle was trained on the call-heavy champion chosen-trade set, so the
put-side simulated exit PnL is extrapolation. We surface that caveat in the
output rather than claiming the swap PnL as ground-truth realizable.

Usage:

    .venv/bin/python -m v3.analysis.w5_day_diagnostic \
        --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \
        --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42.npz \
        --chosen-trades v3/artifacts/layer2_unified_policy_spx_live_hybrid_001_seed42/seed_42/chosen_trades.pkl \
        --layer3-trades v3/artifacts/layer3_unified_spx_live_hybrid_routed_seed42/layer3_trades_thr_0p20.csv \
        --window 5 \
        --out v3/artifacts/w5_diagnostic/seed42_w5.json
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
    p.add_argument("--chosen-trades", required=True)
    p.add_argument("--layer3-trades", required=True, help="CSV from layer3_unified_*/layer3_trades_thr_0pXX.csv")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--margins", default="50,100,200,500", help="Comma-separated label-edge margins (dollars) for the swap counterfactual.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _per_side_best(values: np.ndarray, tradeable: np.ndarray, top_k: int) -> dict[str, np.ndarray]:
    """Per-row best call and best put indices + values across the contract slots."""
    tokens = values[:, 1:].astype(np.float64).copy()
    tokens[~tradeable[:, 1:]] = -np.inf
    call = tokens[:, :top_k]
    put = tokens[:, top_k:]
    best_call_slot = call.argmax(axis=1)
    best_put_slot = put.argmax(axis=1)
    rows = np.arange(values.shape[0])
    best_call_val = call[rows, best_call_slot]
    best_put_val = put[rows, best_put_slot]
    return {
        "best_call_val": np.where(np.isfinite(best_call_val), best_call_val, np.nan),
        "best_put_val": np.where(np.isfinite(best_put_val), best_put_val, np.nan),
        "best_call_action_id": (best_call_slot + 1).astype(int),
        "best_put_action_id": (best_put_slot + 1 + top_k).astype(int),
    }


def _hybrid_live_for_row(
    row_pnl: np.ndarray,
    entry_mid: np.ndarray,
    entry_spread: np.ndarray,
    stopout_risk: np.ndarray,
    entry_bar: np.ndarray,
    exit_bar: np.ndarray,
) -> np.ndarray:
    """Compute hybrid_live_utility for a single row across all 25 actions."""
    out = np.full(row_pnl.shape, np.nan, dtype=np.float64)
    out[0] = 0.0
    for a in range(1, len(row_pnl)):
        pnl = row_pnl[a] if np.isfinite(row_pnl[a]) else None
        em = float(entry_mid[a])
        if not np.isfinite(em) or em <= 0:
            continue
        eb = int(entry_bar[a]) if np.isfinite(entry_bar[a]) else 0
        xb = int(exit_bar[a]) if np.isfinite(exit_bar[a]) and exit_bar[a] >= 0 else None
        out[a] = hybrid_live_utility(
            pnl=pnl,
            entry_mid=em,
            spread_fraction=float(entry_spread[a]),
            stopout_risk=float(stopout_risk[a]),
            entry_bar=eb,
            exit_bar=xb,
        )
    return out


def main() -> int:
    args = parse_args()
    margins = [float(x) for x in args.margins.split(",") if x.strip()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bundle = load_export_bundle(args.dataset)
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    top_k = int(meta["top_k_contracts_per_side"])
    al = bundle["action_labels"]
    tradeable_mask = np.nan_to_num(al["tradeable_mask"], nan=0.0) > 0.5
    contract_strike = bundle["contract_strike"]

    z = np.load(args.simulated_l3_oracle, allow_pickle=True)
    l3_pnl = z["l3_exit_pnl"]
    l3_exit_bar = z["l3_exit_bar"]

    # Build hybrid_live per-row utility (the training target). Recompute rather
    # than rely on stored utility_raw because that was computed with time_stop.
    print("Recomputing hybrid_live target utility per row (this is the training target)...", flush=True)
    n_rows = len(rows)
    n_actions = l3_pnl.shape[1]
    hybrid_target = np.full((n_rows, n_actions), np.nan, dtype=np.float64)
    entry_mid = al["entry_fill_mid"]
    entry_spread = al["entry_spread_fraction"]
    entry_bar = al["entry_fill_bar"]
    stopout_risk = al["stopout_risk"]
    for r in range(n_rows):
        if not tradeable_mask[r, 1:].any():
            hybrid_target[r, 0] = 0.0
            continue
        hybrid_target[r] = _hybrid_live_for_row(
            l3_pnl[r],
            entry_mid[r],
            entry_spread[r],
            stopout_risk[r],
            entry_bar[r],
            l3_exit_bar[r],
        )

    label_best = _per_side_best(hybrid_target, tradeable_mask, top_k)

    # Find W5 OOS days
    days_arr = rows["day"].astype(str).to_numpy()
    unique_days = sorted(rows["day"].astype(str).unique().tolist())
    windows = generate_rolling_windows(unique_days)
    target_window = next((w for w in windows if int(w.window_idx) == int(args.window)), None)
    if target_window is None:
        raise RuntimeError(f"Window {args.window} not found")
    oos_days = sorted(target_window.oos_days)
    print(f"W{args.window} OOS: {len(oos_days)} days, {oos_days[0]} .. {oos_days[-1]}", flush=True)

    # Production chosen trades for this window
    chosen = load_pickle(args.chosen_trades)
    chosen_w = chosen[chosen["window_idx"] == int(args.window)].copy()
    print(f"Production chosen trades in W{args.window}: {len(chosen_w)}", flush=True)

    # L3 routed trades csv (default thr_0p20)
    l3_trades = pd.read_csv(args.layer3_trades)
    l3_w = l3_trades[l3_trades["window_idx"] == int(args.window)].copy() if "window_idx" in l3_trades.columns else l3_trades.copy()
    print(f"L3 routed trades for W{args.window}: {len(l3_w)}", flush=True)

    # Index lookup: (day, bar_index) -> dataset row index
    rows_idx = rows.reset_index(drop=True)
    key_to_row = {(str(d), int(b)): i for i, (d, b) in enumerate(zip(rows_idx["day"], rows_idx["bar_index"]))}

    # Build per-day decomposition
    out_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    n_chosen_with_label = 0
    n_swap_at: dict[float, int] = {m: 0 for m in margins}
    swap_pnl_delta_at: dict[float, float] = {m: 0.0 for m in margins}
    swap_pnl_total_at: dict[float, float] = {m: 0.0 for m in margins}
    oracle_pnl_total = 0.0
    production_pnl_total = float(l3_w["exit_pnl"].sum()) if "exit_pnl" in l3_w.columns else float("nan")

    for day in oos_days:
        ct = chosen_w[chosen_w["day"] == day]
        l3 = l3_w[l3_w["day"] == day] if "day" in l3_w.columns else pd.DataFrame()
        rec: dict[str, Any] = {"day": day}
        if len(ct) == 0:
            rec["chosen"] = None
        else:
            r = ct.iloc[0]
            rec["bar_index"] = int(r["bar_index"])
            rec["chosen_side"] = str(r["chosen_side"])
            rec["chosen_strike"] = float(r["chosen_strike"]) if np.isfinite(r["chosen_strike"]) else None
            rec["chosen_objective_pnl"] = float(r["chosen_objective_pnl"])
            rec["chosen_time_stop_pnl"] = float(r["chosen_time_stop_pnl"])
            rec["decision_margin"] = float(r["decision_margin"])
            rec["best_nonflat_score"] = float(r["best_nonflat_score"])
            rec["flat_score"] = float(r["flat_score"])
            rec["pred_win_prob"] = float(r["pred_win_prob"])
            rec["pred_stopout_risk"] = float(r["pred_stopout_risk"])
            rec["sigma_pos"] = float(r["sigma_pos"])
            rec["omar_mid_pos_units"] = float(r["omar_mid_pos_units"])
            rec["vwap"] = float(r["vwap"])
            rec["underlying_close"] = float(r["underlying_close"])
            rec["iv_percentile"] = float(r["iv_percentile"])
            rec["first15_range_pct"] = float(r["first15_range_pct"])
            rec["best_forward_pnl_call_session_end"] = float(r["best_forward_pnl_call"])
            rec["best_forward_pnl_put_session_end"] = float(r["best_forward_pnl_put"])

            row_idx = key_to_row.get((day, int(r["bar_index"])))
            if row_idx is None:
                rec["label_best_call"] = None
                rec["label_best_put"] = None
            else:
                rec["label_best_call"] = float(label_best["best_call_val"][row_idx]) if np.isfinite(label_best["best_call_val"][row_idx]) else None
                rec["label_best_put"] = float(label_best["best_put_val"][row_idx]) if np.isfinite(label_best["best_put_val"][row_idx]) else None
                rec["label_best_call_action_id"] = int(label_best["best_call_action_id"][row_idx])
                rec["label_best_put_action_id"] = int(label_best["best_put_action_id"][row_idx])
                rec["label_best_call_strike"] = float(contract_strike[row_idx, label_best["best_call_action_id"][row_idx] - 1]) if label_best["best_call_action_id"][row_idx] > 0 else None
                rec["label_best_put_strike"] = float(contract_strike[row_idx, label_best["best_put_action_id"][row_idx] - 1]) if label_best["best_put_action_id"][row_idx] > 0 else None
                rec["label_put_minus_call"] = (
                    rec["label_best_put"] - rec["label_best_call"]
                    if rec["label_best_put"] is not None and rec["label_best_call"] is not None
                    else None
                )
                rec["label_best_side"] = (
                    "put"
                    if (rec["label_best_put"] or -np.inf) > (rec["label_best_call"] or -np.inf)
                    else "call"
                )

                # Counterfactual swap target PnL: hybrid_live target at best_put_action_id
                # (this is what the training target was telling us puts could deliver,
                # using the simulated-L3 oracle exits)
                bp_aid = int(label_best["best_put_action_id"][row_idx])
                if bp_aid > 0:
                    rec["swap_to_best_put_target_pnl"] = float(hybrid_target[row_idx, bp_aid]) if np.isfinite(hybrid_target[row_idx, bp_aid]) else None
                    rec["swap_to_best_put_l3_exit_pnl"] = float(l3_pnl[row_idx, bp_aid]) if np.isfinite(l3_pnl[row_idx, bp_aid]) else None
                else:
                    rec["swap_to_best_put_target_pnl"] = None
                    rec["swap_to_best_put_l3_exit_pnl"] = None

                # Oracle-best
                row_target = hybrid_target[row_idx]
                trade_actions = np.where(tradeable_mask[row_idx])[0]
                if len(trade_actions) > 0:
                    oracle_aid = int(trade_actions[np.nanargmax(row_target[trade_actions])])
                    rec["oracle_best_action_id"] = oracle_aid
                    rec["oracle_best_target_pnl"] = float(hybrid_target[row_idx, oracle_aid]) if np.isfinite(hybrid_target[row_idx, oracle_aid]) else None
                    rec["oracle_best_side"] = "flat" if oracle_aid == 0 else ("call" if oracle_aid <= top_k else "put")
                    rec["oracle_best_l3_exit_pnl"] = float(l3_pnl[row_idx, oracle_aid]) if np.isfinite(l3_pnl[row_idx, oracle_aid]) else None
                    if rec["oracle_best_l3_exit_pnl"] is not None:
                        oracle_pnl_total += rec["oracle_best_l3_exit_pnl"]

            n_chosen_with_label += 1

            # L3 routed actual outcome for this trade
            if len(l3) > 0:
                lt = l3.iloc[0]
                rec["l3_routed_direction"] = str(lt["direction"]) if "direction" in lt else None
                rec["l3_routed_exit_pnl"] = float(lt["exit_pnl"]) if "exit_pnl" in lt else None
                rec["l3_routed_exit_bar"] = int(lt["exit_bar"]) if "exit_bar" in lt else None
                rec["l3_routed_bars_held"] = int(lt["bars_held"]) if "bars_held" in lt else None
                rec["l3_routed_trigger"] = str(lt["trigger"]) if "trigger" in lt else None

            # Counterfactual swap: at each margin threshold, decide if we'd swap
            for m in margins:
                if (
                    rec.get("label_put_minus_call") is not None
                    and rec["label_put_minus_call"] >= m
                    and rec.get("swap_to_best_put_l3_exit_pnl") is not None
                    and rec.get("l3_routed_exit_pnl") is not None
                ):
                    n_swap_at[m] += 1
                    delta = rec["swap_to_best_put_l3_exit_pnl"] - rec["l3_routed_exit_pnl"]
                    swap_pnl_delta_at[m] += delta
                    swap_pnl_total_at[m] += rec["swap_to_best_put_l3_exit_pnl"]
                else:
                    if rec.get("l3_routed_exit_pnl") is not None:
                        swap_pnl_total_at[m] += rec["l3_routed_exit_pnl"]
        out_rows.append(rec)

    # Aggregates
    summary: dict[str, Any] = {
        "window": int(args.window),
        "n_oos_days": len(oos_days),
        "oos_days_range": [oos_days[0], oos_days[-1]],
        "n_chosen_trades": int(len(chosen_w)),
        "production_pnl_total": production_pnl_total,
        "production_per_trade_avg": production_pnl_total / max(len(l3_w), 1),
        "oracle_best_pnl_total_simulated_l3": oracle_pnl_total,
        "oracle_best_pnl_per_chosen_trade_avg": oracle_pnl_total / max(n_chosen_with_label, 1),
        "swap_counterfactuals": {
            f"margin_{int(m)}": {
                "n_swapped": int(n_swap_at[m]),
                "swap_pnl_delta_total": float(swap_pnl_delta_at[m]),
                "scenario_pnl_total": float(swap_pnl_total_at[m]),
                "swap_per_trade_avg": (float(swap_pnl_delta_at[m]) / max(n_swap_at[m], 1)),
            }
            for m in margins
        },
        "chosen_side_distribution": {
            "call": int((chosen_w["chosen_side"] == "call").sum()),
            "put": int((chosen_w["chosen_side"] == "put").sum()),
        },
    }

    # Also: across W5 OOS bars (not just chosen), what fraction have label_best_side = put?
    w5_mask = pd.Series(days_arr).isin(set(oos_days)).to_numpy()
    if w5_mask.sum() > 0:
        # Use the same hybrid_live label
        bc = label_best["best_call_val"][w5_mask]
        bp = label_best["best_put_val"][w5_mask]
        finite = np.isfinite(bc) & np.isfinite(bp)
        bc, bp = bc[finite], bp[finite]
        w5_label_best_put = (bp > bc).sum()
        w5_label_best_call = (bc > bp).sum()
        # Also: fraction of W5 bars where label says put beats call by various margins
        edge_dist = {}
        for m in margins:
            edge_dist[f"n_bars_label_put_beats_call_by_{int(m)}"] = int(((bp - bc) >= m).sum())
        summary["w5_oos_bar_label_distribution"] = {
            "n_bars_with_both_sides_finite": int(finite.sum()),
            "n_bars_label_best_put": int(w5_label_best_put),
            "n_bars_label_best_call": int(w5_label_best_call),
            "label_put_minus_call_mean": float((bp - bc).mean()),
            "label_put_minus_call_median": float(np.median(bp - bc)),
            "edge_distribution": edge_dist,
        }

    summary["per_day_decomposition"] = out_rows

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    print(f"Wrote {args.out}", flush=True)

    # Concise stdout summary
    print()
    print(f"=== W{args.window} OOS bar-level label distribution (across {summary.get('w5_oos_bar_label_distribution',{}).get('n_bars_with_both_sides_finite','?')} tradeable bars) ===")
    if "w5_oos_bar_label_distribution" in summary:
        d = summary["w5_oos_bar_label_distribution"]
        print(f"  n_bars truth=call: {d['n_bars_label_best_call']}  truth=put: {d['n_bars_label_best_put']}")
        print(f"  label_put_minus_call: mean={d['label_put_minus_call_mean']:.2f} median={d['label_put_minus_call_median']:.2f}")
        for k, v in d["edge_distribution"].items():
            print(f"  {k}: {v}")
    print()
    print(f"=== W{args.window} chosen-trade summary ===")
    print(f"  n_chosen={summary['n_chosen_trades']}  call={summary['chosen_side_distribution']['call']} put={summary['chosen_side_distribution']['put']}")
    print(f"  production L3 routed PnL total (thr=0.20): ${production_pnl_total:.2f} ({production_pnl_total / max(len(l3_w), 1):.2f}/trade)")
    print(f"  oracle-best PnL total (sim L3 exits): ${oracle_pnl_total:.2f} ({oracle_pnl_total / max(n_chosen_with_label, 1):.2f}/trade)")
    print()
    print(f"=== swap-to-best-put counterfactual ===")
    print(f"  {'margin($)':>10} {'n_swap':>7} {'delta_total$':>14} {'scenario_total$':>17}")
    for m in margins:
        s = summary["swap_counterfactuals"][f"margin_{int(m)}"]
        print(f"  {m:>10.0f} {s['n_swapped']:>7} {s['swap_pnl_delta_total']:>14.2f} {s['scenario_pnl_total']:>17.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
