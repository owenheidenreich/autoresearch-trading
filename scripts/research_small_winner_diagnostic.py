"""Diagnose why the L3 oracle BLOWS UP small-winner trades.

Phase 1 (this script): characterize the failure pattern using the existing
oracle_exit_timing_audit.csv (1717 trades with best/pred/time_stop pnl per
trade) AND fw_trade_trajectories.pkl (61 trades with per-bar pnl).

Questions:
  1. Where is the failure concentrated? (amplitude bucket × side × duration)
  2. Is this a "model misses signal" failure (under-predicts target=1 at peak)
     or a "target is misshapen" failure (peak signal is diluted across bars)?
  3. For small-amp trades, where is the unique peak located? How many bars
     have target=1? What's the avg pnl at target=1 bars?

If small-amp trades have many target=1 bars at low pnl while big-amp trades
have a sharp single peak, the target itself is the problem and any classifier
will struggle.
"""
from __future__ import annotations

import json
import pickle
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def main():
    print("=== Small-winner failure diagnostic ===\n")

    audit = pd.read_csv("v3/artifacts/research/oracle_exit_timing_audit.csv")
    print(f"Audit rows: {len(audit)}  (1717 expected from autoresearch summary)")
    print(f"Columns: {list(audit.columns)}\n")

    # ------------------------------------------------------------
    # Phase 1: Characterize failure by amplitude bucket
    # ------------------------------------------------------------
    print("=" * 80)
    print("PHASE 1: Failure characterization by best_exit_pnl bucket\n")

    audit["pnl_loss"] = audit["best_exit_pnl"] - audit["pred_exit_pnl"]  # money left on table
    audit["pred_held_to_end"] = (
        np.abs(audit["pred_exit_pnl"] - audit["time_stop_pnl"]) < 1.0
    )

    buckets = [
        (0, 200, "$0-200 (small)"),
        (200, 500, "$200-500"),
        (500, 1000, "$500-1000"),
        (1000, 5000, "$1k-5k"),
        (5000, 1e9, "$5k+"),
    ]

    print(f"{'bucket':>20} {'n':>4} {'mean_capture':>13} {'frac_exit_at_end':>17} {'mean_loss_$':>12} {'mean_pred_$':>12}")
    for lo, hi, label in buckets:
        sub = audit[(audit["best_exit_pnl"] >= lo) & (audit["best_exit_pnl"] < hi)]
        if len(sub) == 0:
            continue
        cap = sub["captured_of_best"].mean()
        frac_end = sub["pred_held_to_end"].mean()
        loss = sub["pnl_loss"].mean()
        pred_avg = sub["pred_exit_pnl"].mean()
        print(f"{label:>20} {len(sub):>4} {cap:>13.3f} {frac_end:>17.1%} ${loss:>11.0f} ${pred_avg:>11.0f}")
    print()

    # Exit quality distribution by bucket
    print("Exit quality by amplitude bucket:")
    print(f"{'bucket':>20} {'n':>4}", end="")
    for q in ["good (≥85%)", "ok (50-85%)", "weak (0-50%)", "negative-on-winner", "no-trade-window"]:
        print(f" {q[:18]:>18}", end="")
    print()
    for lo, hi, label in buckets:
        sub = audit[(audit["best_exit_pnl"] >= lo) & (audit["best_exit_pnl"] < hi)]
        if len(sub) == 0:
            continue
        print(f"{label:>20} {len(sub):>4}", end="")
        vc = sub["exit_quality"].value_counts(normalize=True).to_dict()
        for q in ["good (≥85%)", "ok (50-85%)", "weak (0-50%)", "negative-on-winner", "no-trade-window"]:
            print(f" {vc.get(q, 0.0):>18.1%}", end="")
        print()
    print()

    # By side
    print("By chosen side:")
    for side in ["call", "put"]:
        sub = audit[audit["side"] == side]
        print(f"  {side} (n={len(sub)}): mean capture={sub['captured_of_best'].mean():.3f}, "
              f"frac_held_to_end={sub['pred_held_to_end'].mean():.1%}")
    print()

    # ------------------------------------------------------------
    # Phase 2: Target structure on per-bar trajectories
    # ------------------------------------------------------------
    print("=" * 80)
    print("PHASE 2: Target structure analysis on per-bar trajectories\n")

    with open("v3/artifacts/research/fw_trade_trajectories.pkl", "rb") as f:
        trajs = pickle.load(f)
    print(f"Loaded {len(trajs)} trajectories with per-bar pnl.\n")

    # For each trajectory, compute target signal as the oracle's training target does:
    #   target[i] = int(pnls[i] >= max(pnls[i+1:]))   (i.e. >= suffix_max[i])
    # Then categorize:
    #   - peak_bar: bar with highest pnl
    #   - n_target_eq_1: how many bars have target=1
    #   - mean_pnl_at_target_1: average pnl among target=1 bars
    #   - peak_pnl: highest pnl in trajectory

    rows = []
    for t in trajs:
        bars = t.get("pnl_per_bar", [])
        if len(bars) < 3:
            continue
        pnls = np.array([b["pnl"] for b in bars], dtype=np.float64)
        n = len(pnls)
        suffix_max = np.full(n, -np.inf)
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(pnls[i + 1], suffix_max[i + 1])
        suffix_max[-1] = pnls[-1]
        target = (pnls >= suffix_max).astype(int)

        peak_pnl = float(pnls.max())
        peak_bar = int(np.argmax(pnls))
        n_target_1 = int(target.sum())
        n_target_1_post_peak = int(target[peak_bar:].sum())
        n_target_1_pre_peak = int(target[:peak_bar].sum())
        target_1_pnls = pnls[target == 1]
        mean_pnl_at_t1 = float(np.mean(target_1_pnls)) if len(target_1_pnls) > 0 else float("nan")

        rows.append({
            "trade_id": t.get("trade_id", "?"),
            "side": t.get("chosen_side", "?"),
            "n_bars": n,
            "peak_pnl": peak_pnl,
            "peak_bar_offset": peak_bar,
            "n_target_1": n_target_1,
            "n_target_1_post_peak": n_target_1_post_peak,
            "n_target_1_pre_peak": n_target_1_pre_peak,
            "frac_target_1": n_target_1 / n,
            "mean_pnl_at_target_1": mean_pnl_at_t1,
        })

    df = pd.DataFrame(rows)
    print(f"Trajectories analyzed: {len(df)}")
    print(f"  by peak_pnl bucket:")

    target_buckets = [
        (-1e9, 0, "loser"),
        (0, 200, "small win"),
        (200, 500, "$200-500"),
        (500, 1000, "$500-1k"),
        (1000, 1e9, "$1k+"),
    ]

    print(f"\n{'bucket':>15} {'n':>4} {'mean_n_bars':>12} {'mean_n_t=1':>11} {'mean_frac_t=1':>14} "
          f"{'mean_pnl@t=1':>13} {'peak_bar_avg':>13}")
    for lo, hi, label in target_buckets:
        sub = df[(df["peak_pnl"] >= lo) & (df["peak_pnl"] < hi)]
        if len(sub) == 0:
            continue
        print(f"{label:>15} {len(sub):>4} {sub['n_bars'].mean():>12.1f} "
              f"{sub['n_target_1'].mean():>11.1f} {sub['frac_target_1'].mean():>14.3f} "
              f"${sub['mean_pnl_at_target_1'].mean():>12.0f} "
              f"{sub['peak_bar_offset'].mean():>13.1f}")
    print()

    # Critical question: is the target signal SHARP or DILUTED for small wins?
    print("Target signal sharpness by amplitude bucket:")
    print(f"  (sharper = peak is the only target=1 bar; diluted = many post-peak bars also target=1)")
    print(f"\n{'bucket':>15} {'n':>4} {'frac_unique_peak':>17} {'avg_n_t=1':>11} {'avg_post_peak_t=1':>17}")
    for lo, hi, label in target_buckets:
        sub = df[(df["peak_pnl"] >= lo) & (df["peak_pnl"] < hi)]
        if len(sub) == 0:
            continue
        # frac_unique_peak = fraction of trades where target=1 fires at exactly 1 bar (the peak)
        frac_unique = (sub["n_target_1"] == 1).mean()
        avg_total = sub["n_target_1"].mean()
        avg_post = sub["n_target_1_post_peak"].mean()
        print(f"{label:>15} {len(sub):>4} {frac_unique:>17.1%} {avg_total:>11.1f} {avg_post:>17.1f}")
    print()

    # --- Cross-cut: when target=1 fires, what's the pnl distribution? ---
    print("Distribution of pnl values at target=1 bars (across all trajectories):")
    all_t1_pnls = []
    for t in trajs:
        bars = t.get("pnl_per_bar", [])
        if len(bars) < 3: continue
        pnls = np.array([b["pnl"] for b in bars], dtype=np.float64)
        n = len(pnls)
        suffix_max = np.full(n, -np.inf)
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(pnls[i + 1], suffix_max[i + 1])
        suffix_max[-1] = pnls[-1]
        target = (pnls >= suffix_max).astype(int)
        all_t1_pnls.extend(pnls[target == 1].tolist())
    all_t1_pnls = np.asarray(all_t1_pnls)
    print(f"  n target=1 bars: {len(all_t1_pnls)}")
    print(f"  pct profitable:  {(all_t1_pnls > 0).mean():.1%}")
    print(f"  pct loss-bars:   {(all_t1_pnls < 0).mean():.1%}")
    print(f"  mean / median:   ${all_t1_pnls.mean():.0f} / ${np.median(all_t1_pnls):.0f}")
    print(f"  quartiles:       {np.percentile(all_t1_pnls, [25, 50, 75])}")
    print()

    # ------------------------------------------------------------
    # Phase 3: Counterfactual: what if target=1 only at profitable bars?
    # ------------------------------------------------------------
    print("=" * 80)
    print("PHASE 3: Counterfactual target reshape\n")
    print("Current target: target[i] = int(pnls[i] >= suffix_max[i])")
    print("Proposed (H3e): target[i] = int(pnls[i] >= suffix_max[i] AND pnls[i] > 0)")
    print()

    n_old_t1 = 0
    n_new_t1 = 0
    n_old_t1_loss = 0
    n_new_t1_loss = 0
    new_t1_pnls = []

    rows_h3e = []
    for t in trajs:
        bars = t.get("pnl_per_bar", [])
        if len(bars) < 3: continue
        pnls = np.array([b["pnl"] for b in bars], dtype=np.float64)
        n = len(pnls)
        suffix_max = np.full(n, -np.inf)
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(pnls[i + 1], suffix_max[i + 1])
        suffix_max[-1] = pnls[-1]

        target_old = (pnls >= suffix_max).astype(int)
        target_new = ((pnls >= suffix_max) & (pnls > 0)).astype(int)

        n_old_t1 += int(target_old.sum())
        n_new_t1 += int(target_new.sum())
        n_old_t1_loss += int(((target_old == 1) & (pnls < 0)).sum())
        n_new_t1_loss += int(((target_new == 1) & (pnls < 0)).sum())
        new_t1_pnls.extend(pnls[target_new == 1].tolist())

        peak_pnl = float(pnls.max())
        rows_h3e.append({
            "peak_pnl": peak_pnl,
            "n_t1_old": int(target_old.sum()),
            "n_t1_new": int(target_new.sum()),
        })

    h3e_df = pd.DataFrame(rows_h3e)

    print(f"Across all {len(trajs)} trajectories:")
    print(f"  Old target=1 bars: {n_old_t1}  ({n_old_t1_loss} on losses, {100*n_old_t1_loss/n_old_t1:.1f}%)")
    print(f"  New target=1 bars: {n_new_t1}  ({n_new_t1_loss} on losses, {100*n_new_t1_loss/max(n_new_t1,1):.1f}%)")
    if new_t1_pnls:
        new_arr = np.asarray(new_t1_pnls)
        print(f"  New target=1 pnl: mean ${new_arr.mean():.0f}, median ${np.median(new_arr):.0f}")
        print(f"  All new t=1 are profitable? {(new_arr > 0).all()}")
    print()

    print("Per-bucket: old vs new target=1 count:")
    print(f"{'bucket':>15} {'n_traj':>7} {'old_avg':>9} {'new_avg':>9} {'delta':>9}")
    for lo, hi, label in target_buckets:
        sub = h3e_df[(h3e_df["peak_pnl"] >= lo) & (h3e_df["peak_pnl"] < hi)]
        if len(sub) == 0: continue
        print(f"{label:>15} {len(sub):>7} {sub['n_t1_old'].mean():>9.1f} "
              f"{sub['n_t1_new'].mean():>9.1f} {sub['n_t1_new'].mean() - sub['n_t1_old'].mean():>+9.1f}")
    print()

    # The losers bucket: what does the new target do?
    print("Crucial test on LOSER trajectories (peak_pnl < 0):")
    losers = h3e_df[h3e_df["peak_pnl"] < 0]
    print(f"  n loser trajectories: {len(losers)}")
    if len(losers) > 0:
        print(f"  old target=1 avg/trade: {losers['n_t1_old'].mean():.1f}  (these are all losses!)")
        print(f"  new target=1 avg/trade: {losers['n_t1_new'].mean():.1f}  (should be 0 — no profitable bars)")
    print()

    # Save
    os.makedirs("v3/artifacts/research", exist_ok=True)
    df.to_csv("v3/artifacts/research/small_winner_diagnostic_traj.csv", index=False)
    h3e_df.to_csv("v3/artifacts/research/h3e_target_counterfactual.csv", index=False)
    print(f"Wrote v3/artifacts/research/small_winner_diagnostic_traj.csv")
    print(f"Wrote v3/artifacts/research/h3e_target_counterfactual.csv")


if __name__ == "__main__":
    main()
