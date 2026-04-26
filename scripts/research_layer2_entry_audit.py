"""Layer-2 entry audit: is regime spread propagated from entry decisions?

Tests whether the L3 oracle's cross-cell PF spread is L2's fault (regime-
skewed entry policy) or genuinely an L3 problem (exit policy generalizing
unevenly across regimes).

Method:
  - Stratify the layer2_action_surface_dataset rows by trend × vol regime
    (same 9-cell scheme as research_regime_stratified_eval.py).
  - Per cell, count:
    * tradeable_bars: number of bars where any contract was tradeable
      (universe context — how often the regime actually appeared)
    * picked_entries: number of bars L2 actually picked (chosen_action_id > 0)
    * pick_rate: picked_entries / tradeable_bars
    * call_share / put_share: side bias of picks
    * mean_entry_score: predicted dollar/return scores at picks
    * realized_pf: PF of these entries with the BASELINE oracle (isolating
      L2 effect from L3 exit policy)

Decision criteria:
  - If pick_rate is roughly uniform (within 2x) across cells AND realized PF
    varies wildly cell-to-cell → spread is L3 problem → proceed to Angle A
  - If pick_rate varies > 5x across cells, OR if entry scores skew strongly
    by regime → regime spread is propagated from entry → STOP L3 ladder,
    pivot to L2 work
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle


SEEDS = [42, 43, 44, 45, 46]
BASE_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_balanced_fresh.npz"
CHOSEN_PATTERN = "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{seed}/seed_{seed}/chosen_trades.pkl"
SPX_1MIN = "/Users/gduby/.cache/autoresearch-trading/data/spx_1min.pkl"


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def hl(pnl_raw, em, ef, eb, so, exit_bar):
    return hybrid_live_utility(
        pnl_raw if np.isfinite(pnl_raw) else None,
        entry_mid=em, spread_fraction=ef if np.isfinite(ef) else 0.0,
        stopout_risk=so if np.isfinite(so) else 0.0,
        entry_bar=int(eb),
        exit_bar=int(exit_bar) if exit_bar >= 0 else None,
        session_end_bar=375,
    )


def build_daily_returns():
    with open(SPX_1MIN, "rb") as f:
        spx = pickle.load(f)
    daily = spx.groupby("date")["spx_close"].last().sort_index()
    daily.index = daily.index.astype(str)
    return daily.pct_change(20).to_dict()


def trend_regime(ret20d):
    if not np.isfinite(ret20d): return "unknown"
    if ret20d <= -0.02: return "bear"
    if ret20d >= 0.02: return "bull"
    return "chop"


def vol_regime(iv_pct):
    if not np.isfinite(iv_pct): return "unknown"
    if iv_pct < 0.33: return "low"
    if iv_pct < 0.66: return "mid"
    return "high"


def main():
    print("=== Layer-2 Entry Audit (regime-stratified) ===\n")

    print("Loading SPX daily returns + action-surface dataset...")
    daily_returns = build_daily_returns()
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    al = bundle["action_labels"]
    rows["__row__"] = np.arange(len(rows))
    print(f"  {len(rows)} bar-rows in action-surface dataset")
    print()

    # ----- Universe context: tradeable_bars per regime cell -----
    # A bar is "tradeable" if the action_labels.tradeable_mask has any
    # tradeable contract for that bar (excluding the flat action token 0).
    tm = al["tradeable_mask"][:, 1:]  # drop flat action
    tm = np.nan_to_num(tm, nan=0.0) > 0.5
    any_tradeable = tm.any(axis=1)

    rows["any_tradeable"] = any_tradeable
    rows["day_str"] = rows["day"].astype(str)
    rows["ret_20d"] = rows["day_str"].map(daily_returns)
    # iv_percentile is in the bundle's row-level columns
    if "iv_percentile" in rows.columns:
        rows["iv_pct"] = rows["iv_percentile"]
    else:
        # Fallback — pull from al if present
        rows["iv_pct"] = np.nan
    rows["trend"] = rows["ret_20d"].map(trend_regime)
    rows["vol"] = rows["iv_pct"].map(vol_regime)

    universe = rows[rows["any_tradeable"]].copy()
    print(f"Tradeable bars (universe): {len(universe)}")
    print()

    print("=== Universe distribution by regime cell ===")
    print(f"{'trend':>6} {'vol':>6} {'tradeable_bars':>15}")
    universe_counts = {}
    for t in ["bear", "chop", "bull", "unknown"]:
        for v in ["low", "mid", "high", "unknown"]:
            n = ((universe["trend"] == t) & (universe["vol"] == v)).sum()
            if n > 0:
                universe_counts[(t, v)] = int(n)
                print(f"{t:>6} {v:>6} {n:>15}")
    print()

    # ----- Per-seed L2 picks: assemble across all 5 seeds -----
    print("=== Loading 5 seeds of L2 chosen trades + baseline oracle ===")
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    pick_records = []
    for seed in SEEDS:
        cp = CHOSEN_PATTERN.format(seed=seed)
        bp = BASE_PATTERN.format(seed=seed)
        if not os.path.exists(cp) or not os.path.exists(bp):
            print(f"  seed {seed}: missing files, skip")
            continue
        fw = pd.read_pickle(cp)
        fw = fw[fw["chosen_action_id"] > 0].reset_index(drop=True)
        oracle = np.load(bp, allow_pickle=True)
        ex_pnl, ex_bar = oracle["l3_exit_pnl"], oracle["l3_exit_bar"]

        for _, row in fw.iterrows():
            key = (row["day"], row["bar_index"])
            r = key_to_row.get(key)
            if r is None: continue
            a = int(row["chosen_action_id"])
            if a <= 0: continue
            em = float(al["entry_fill_mid"][r, a])
            ef = float(al["entry_spread_fraction"][r, a])
            eb_ = float(al["entry_fill_bar"][r, a])
            so = float(al["stopout_risk"][r, a])
            if not (np.isfinite(em) and np.isfinite(eb_)): continue

            day = str(row["day"])
            ret20 = daily_returns.get(day, np.nan)
            iv_pct = float(row.get("iv_percentile", np.nan))

            pnl, bar = float(ex_pnl[r, a]), int(ex_bar[r, a])
            pick_records.append({
                "seed": seed,
                "day": day,
                "side": row["chosen_side"],
                "trend": trend_regime(ret20),
                "vol": vol_regime(iv_pct),
                "chosen_objective_pnl": float(row.get("chosen_objective_pnl", np.nan)),
                "pred_dollar_score": float(row.get("pred_dollar_score", np.nan)),
                "pred_return_score": float(row.get("pred_return_score", np.nan)),
                "pred_clean_entry_prob": float(row.get("pred_clean_entry_prob", np.nan)),
                "pred_win_prob": float(row.get("pred_win_prob", np.nan)),
                "decision_margin": float(row.get("decision_margin", np.nan)),
                "base_hl": hl(pnl, em, ef, eb_, so, bar),
            })

    df = pd.DataFrame(pick_records)
    print(f"  Total L2 picks across {len(set(df['seed']))} seeds: {len(df)}")
    print()

    # ----- Per-cell picks + pick-rate -----
    print("=== L2 entry distribution + pick-rate by regime cell ===")
    print(f"{'trend':>6} {'vol':>6} {'tradeable':>10} {'picked':>7} {'pick_rate':>10} {'%calls':>7} {'mean_dollar':>12} {'realized_pf':>12}")
    cell_results = []
    for t in ["bear", "chop", "bull"]:
        for v in ["low", "mid", "high"]:
            uni = universe_counts.get((t, v), 0)
            sub = df[(df["trend"] == t) & (df["vol"] == v)]
            picked = len(sub)
            # pick_rate is picks across all 5 seeds; normalize by 5 to get
            # per-seed pick_rate (universe is shared across seeds since L2
            # sees the same action surface)
            pick_rate = (picked / 5) / max(uni, 1) if uni > 0 else 0.0
            pct_calls = (sub["side"] == "call").mean() if picked else float("nan")
            mean_dollar = sub["pred_dollar_score"].mean() if picked else float("nan")
            realized = pf(sub["base_hl"]) if picked else float("nan")
            print(f"{t:>6} {v:>6} {uni:>10} {picked:>7} {pick_rate:>10.4f} {pct_calls:>7.2%} "
                  f"{mean_dollar:>12.3f} {realized:>12.3f}")
            cell_results.append({
                "trend": t, "vol": v,
                "tradeable_bars_universe": int(uni),
                "picked_total": int(picked),
                "picked_per_seed": float(picked / 5),
                "pick_rate": float(pick_rate),
                "pct_calls": float(pct_calls) if np.isfinite(pct_calls) else None,
                "mean_pred_dollar_score": float(mean_dollar) if np.isfinite(mean_dollar) else None,
                "mean_pred_return_score": float(sub["pred_return_score"].mean()) if picked else None,
                "mean_pred_clean_entry": float(sub["pred_clean_entry_prob"].mean()) if picked else None,
                "mean_decision_margin": float(sub["decision_margin"].mean()) if picked else None,
                "realized_pf_baseline": float(realized) if np.isfinite(realized) else None,
            })
    print()

    # ----- Summary: pick-rate spread across cells -----
    pick_rates = [c["pick_rate"] for c in cell_results if c["pick_rate"] > 0]
    if pick_rates:
        rate_min, rate_max = min(pick_rates), max(pick_rates)
        rate_ratio = rate_max / max(rate_min, 1e-9)
        print(f"Pick-rate range across non-empty cells:")
        print(f"  min: {rate_min:.4f}  max: {rate_max:.4f}  ratio (max/min): {rate_ratio:.2f}x")
        print()

    # ----- Entry-quality drift: do predicted scores vary by regime? -----
    print("=== Entry quality drift (mean predicted scores by trend) ===")
    for t in ["bear", "chop", "bull"]:
        sub = df[df["trend"] == t]
        if not len(sub): continue
        print(f"  {t:>5} (n={len(sub):>4}): "
              f"dollar={sub['pred_dollar_score'].mean():>+7.3f} "
              f"return={sub['pred_return_score'].mean():>+7.3f} "
              f"clean_entry={sub['pred_clean_entry_prob'].mean():>5.3f} "
              f"win={sub['pred_win_prob'].mean():>5.3f} "
              f"margin={sub['decision_margin'].mean():>+7.3f}")
    print()

    # ----- Decision -----
    print("=== Decision ===")
    rate_skew = (rate_ratio if pick_rates else 0)
    realized_pfs = [c["realized_pf_baseline"] for c in cell_results
                    if c["realized_pf_baseline"] is not None and c["picked_total"] >= 30]
    if realized_pfs:
        pf_min, pf_max = min(realized_pfs), max(realized_pfs)
        pf_spread = pf_max - pf_min
        print(f"  Pick-rate ratio max/min: {rate_skew:.2f}x")
        print(f"  Realized PF spread (baseline oracle, n>=30 cells): {pf_spread:.3f} ({pf_min:.2f} - {pf_max:.2f})")
        print()

        # Heuristic decision
        if rate_skew >= 5.0:
            print("  L2 PICKS at WIDELY DIFFERENT RATES across regimes (>5x).")
            print("  -> Most regime spread is propagated from entry, not exit.")
            print("  -> STOP the L3 ladder; pivot to L2 entry-policy work.")
            verdict = "L2_BOTTLENECK"
        elif rate_skew >= 2.0:
            print(f"  L2 picks at moderately different rates ({rate_skew:.2f}x). Borderline.")
            print("  -> Proceed to Angle A with awareness; some regime spread is L2.")
            verdict = "BORDERLINE"
        else:
            print("  L2 pick-rate is roughly uniform across regimes (<2x ratio).")
            print("  -> Spread is genuinely an L3 problem; proceed to Angle A.")
            verdict = "L3_PROBLEM"
    else:
        verdict = "INSUFFICIENT_DATA"
        print("  Insufficient data per cell for clean decision.")

    summary = {
        "n_seeds": len(set(df["seed"])),
        "n_picks_total": int(len(df)),
        "tradeable_bars_universe": int(len(universe)),
        "cells": cell_results,
        "pick_rate_ratio_max_min": float(rate_skew) if pick_rates else None,
        "verdict": verdict,
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/layer2_entry_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv("v3/artifacts/research/layer2_entry_audit_picks.csv", index=False)
    print(f"\nWrote v3/artifacts/research/layer2_entry_audit.json + picks.csv")


if __name__ == "__main__":
    main()
