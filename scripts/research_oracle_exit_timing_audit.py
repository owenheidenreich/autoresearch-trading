"""Leak-free audit of L3 oracle exit-bar quality.

For each chosen forward-walk trade, compare the oracle's predicted exit_bar
to the actual optimal exit_bar (the bar with max realized pnl from entry to
session end). This is descriptive analysis on per-trade outcomes, NOT a
classifier — no leakage risk.

Questions:
  1. Is oracle's predicted exit_bar systematically EARLIER than optimal?
  2. How often does oracle pick within N bars of optimal?
  3. What's the pnl gap between oracle's exit and optimal exit?
  4. Is the early-exit bias regime-dependent (high-vol vs low-vol)?
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from v3.layer2.common import load_export_bundle


def main():
    print("=== L3 Oracle Exit-Timing Audit (FW + OOS combined) ===\n")

    # Load extended bundle to get action_labels with mfe/mae
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]

    # Need: per-row, per-action: oracle's predicted exit_bar, oracle's predicted exit_pnl,
    # actual time-stop pnl, actual best exit pnl, actual best exit bar

    # We have:
    #   oracle.l3_exit_pnl[N, 25] - predicted pnl at oracle's predicted exit
    #   oracle.l3_exit_bar[N, 25] - oracle's predicted exit bar
    #   al['utility_raw'][N, 25] - time-stop pnl (hold to bar 120)
    #   al['best_exit_pnl'][N, 25] - perfect-info best exit pnl (oracle ceiling)

    # For each seed, find chosen-trade rows and join to oracle predictions
    seeds = [42, 43, 44, 45, 46]

    key_to_row = (
        rows[["day", "bar_index", "__row__"]]
        .drop_duplicates(subset=["day", "bar_index"])
        .set_index(["day", "bar_index"])["__row__"]
        .to_dict()
    )

    all_records = []
    for s in seeds:
        oracle = np.load(
            f"v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{s}_balanced_fresh.npz",
            allow_pickle=True,
        )
        l3_exit_pnl = oracle["l3_exit_pnl"]
        l3_exit_bar = oracle["l3_exit_bar"]
        # OOS chosen trades
        df_oos = pd.read_pickle(
            f"v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{s}/seed_{s}/chosen_trades.pkl"
        )
        # Forward walk chosen trades
        try:
            df_fw = pd.read_pickle(f"v3/artifacts/forward_walk/forward_walk_chosen_seed{s}.pkl")
            df_fw = df_fw.copy()
        except:
            df_fw = pd.DataFrame()

        for src, df in [("oos", df_oos), ("fw", df_fw)]:
            if df.empty: continue
            for _, row in df.iterrows():
                if row.get("chosen_action_id", 0) <= 0: continue
                key = (row["day"], row["bar_index"])
                r = key_to_row.get(key)
                if r is None: continue
                a = int(row["chosen_action_id"])

                pred_exit_pnl = float(l3_exit_pnl[r, a])
                pred_exit_bar = int(l3_exit_bar[r, a])
                time_stop_pnl = float(al["utility_raw"][r, a])
                best_exit_pnl = float(al["best_exit_pnl"][r, a])

                if not (np.isfinite(pred_exit_pnl) and np.isfinite(time_stop_pnl) and np.isfinite(best_exit_pnl)):
                    continue

                all_records.append({
                    "seed": s, "src": src,
                    "day": row["day"], "bar": row["bar_index"], "side": row["chosen_side"],
                    "pred_exit_pnl": pred_exit_pnl,
                    "pred_exit_bar": pred_exit_bar,
                    "time_stop_pnl": time_stop_pnl,
                    "best_exit_pnl": best_exit_pnl,
                    "entry_bar": int(row.get("chosen_action_id", 0)),  # not actual entry, but a sanity
                })

    audit_df = pd.DataFrame(all_records)
    print(f"Audit rows: {len(audit_df)} ({(audit_df['src']=='oos').sum()} OOS, {(audit_df['src']=='fw').sum()} FW)")
    print()

    # Q1: oracle pred exit bar — distribution
    print("=== Q1: oracle's predicted exit_bar distribution ===")
    print(audit_df["pred_exit_bar"].describe())
    print()

    # Q2: oracle pred pnl vs actual outcomes
    print("=== Q2: oracle's predicted pnl vs realized ===")
    audit_df["pred_minus_time_stop"] = audit_df["pred_exit_pnl"] - audit_df["time_stop_pnl"]
    audit_df["best_minus_time_stop"] = audit_df["best_exit_pnl"] - audit_df["time_stop_pnl"]
    audit_df["pred_minus_best"] = audit_df["pred_exit_pnl"] - audit_df["best_exit_pnl"]
    audit_df["captured_of_best"] = (
        audit_df["pred_exit_pnl"] / audit_df["best_exit_pnl"]
    ).where(audit_df["best_exit_pnl"] > 100, np.nan)
    print(f"  pred_exit_pnl - time_stop_pnl (oracle's lift over time-stop):")
    print(f"    mean ${audit_df['pred_minus_time_stop'].mean():.0f}, median ${audit_df['pred_minus_time_stop'].median():.0f}")
    print(f"    > 0 (oracle better): {(audit_df['pred_minus_time_stop'] > 0).mean()*100:.0f}%")
    print()
    print(f"  pred_exit_pnl - best_exit_pnl (oracle leaves on table; should be <= 0):")
    print(f"    mean ${audit_df['pred_minus_best'].mean():.0f}, median ${audit_df['pred_minus_best'].median():.0f}")
    print(f"    captures fraction of best (only positive-best): mean {audit_df['captured_of_best'].mean():.3f}, median {audit_df['captured_of_best'].median():.3f}")
    print()

    # Q3: per-side analysis
    print("=== Q3: oracle quality by side ===")
    for side in ["call", "put"]:
        sub = audit_df[audit_df["side"] == side]
        if len(sub) == 0: continue
        cap = sub["captured_of_best"].dropna()
        print(f"  {side}: n={len(sub)}, mean captured {cap.mean():.3f}, "
              f"oracle vs time-stop median diff ${sub['pred_minus_time_stop'].median():.0f}")
    print()

    # Q4: oracle exit relative to best exit (categorize each trade)
    audit_df["exit_quality"] = "—"
    pos_best = audit_df["best_exit_pnl"] > 100
    audit_df.loc[pos_best & (audit_df["captured_of_best"] >= 0.85), "exit_quality"] = "good (≥85%)"
    audit_df.loc[pos_best & (audit_df["captured_of_best"].between(0.50, 0.85)), "exit_quality"] = "ok (50-85%)"
    audit_df.loc[pos_best & (audit_df["captured_of_best"].between(0.0, 0.50)), "exit_quality"] = "early-exit (0-50%)"
    audit_df.loc[pos_best & (audit_df["captured_of_best"] < 0), "exit_quality"] = "negative-on-winner"
    audit_df.loc[~pos_best, "exit_quality"] = "no-best-exists"

    print("=== Q4: exit quality bucket distribution ===")
    print(audit_df["exit_quality"].value_counts().to_string())
    print()

    # Q5: are big winners (high best_exit_pnl) more likely to be early-exited?
    print("=== Q5: capture rate by best_exit_pnl magnitude ===")
    for lo, hi in [(0, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 1e6)]:
        sub = audit_df[(audit_df["best_exit_pnl"] >= lo) & (audit_df["best_exit_pnl"] < hi)]
        if len(sub) == 0: continue
        cap = sub["captured_of_best"].dropna()
        if len(cap) == 0: continue
        print(f"  best ${lo}-${hi}: n={len(sub)}, mean cap {cap.mean():.3f}, median cap {cap.median():.3f}")
    print()

    # Save
    os.makedirs("v3/artifacts/research", exist_ok=True)
    audit_df.to_csv("v3/artifacts/research/oracle_exit_timing_audit.csv", index=False)
    print(f"Wrote v3/artifacts/research/oracle_exit_timing_audit.csv")


if __name__ == "__main__":
    main()
