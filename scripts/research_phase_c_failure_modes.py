"""Phase C: qualitative analysis of specific failure modes.

After H3a / H3e / sideblind / C-Pre / L2 Phase 1 / TCN / H3f, the chop x low
cell remains the persistent floor. Every intervention improves it (TCN
+0.83, H3a +0.05, regret +0.14) but no intervention generalizes
elsewhere. This suggests chop x low has different distributional structure
than other cells.

This script pulls the WORST trades in chop x low (and other floor cells)
to look for human-readable patterns. The goal is to generate hypotheses
for what features or model changes would specifically attack these
failure modes.

Method:
  1. Compute per-trade hybrid_live PnL (with baseline oracle exits)
  2. Stratify by cell, identify worst-N trades per cell
  3. For each worst trade, print entry features + oracle's exit choice
     + best-possible exit (regret)
  4. Compare to BEST trades in same cell to find distinctive patterns

Output: a CSV of worst-and-best trades for human review + summary stats.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle


SEEDS = [42, 43, 44, 45, 46]
BASE_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_balanced_fresh.npz"
CHOSEN_PATTERN = "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{seed}/seed_{seed}/chosen_trades.pkl"


def hl(pnl_raw, em, ef, eb, so, exit_bar):
    return hybrid_live_utility(
        pnl_raw if np.isfinite(pnl_raw) else None,
        entry_mid=em, spread_fraction=ef if np.isfinite(ef) else 0.0,
        stopout_risk=so if np.isfinite(so) else 0.0,
        entry_bar=int(eb),
        exit_bar=int(exit_bar) if exit_bar >= 0 else None,
        session_end_bar=375,
    )


def gather() -> pd.DataFrame:
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    records = []
    for seed in SEEDS:
        cp = CHOSEN_PATTERN.format(seed=seed)
        bp = BASE_PATTERN.format(seed=seed)
        if not (os.path.exists(cp) and os.path.exists(bp)):
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
            if int(row["bar_index"]) >= 120: continue

            pnl, bar = float(ex_pnl[r, a]), int(ex_bar[r, a])
            hl_val = hl(pnl, em, ef, eb_, so, bar)
            if not np.isfinite(hl_val): continue

            best = float(row["best_forward_pnl_call"]) if row["chosen_side"] == "call" \
                   else float(row["best_forward_pnl_put"])
            time_stop = float(row.get("fwd_pnl_time_stop", float("nan")))

            records.append({
                "seed": seed,
                "day": str(row["day"]),
                "bar_index": int(row["bar_index"]),
                "side": row["chosen_side"],
                "strike": float(row.get("chosen_strike", np.nan)),
                "hl": hl_val,
                "oracle_exit_pnl": pnl,
                "oracle_exit_bar": bar,
                "time_stop_pnl": time_stop,
                "best_forward_pnl": best,
                "regret_vs_best": best - pnl,
                # Entry features (causal):
                "vwap": float(row.get("vwap", np.nan)),
                "vwap_slope": float(row.get("vwap_slope", np.nan)),
                "volume_ratio": float(row.get("volume_ratio", np.nan)),
                "first15_range_pct": float(row.get("first15_range_pct", np.nan)),
                "iv_percentile": float(row.get("iv_percentile", np.nan)),
                "atm_iv": float(row.get("atm_iv", np.nan)),
                "sigma_pos": float(row.get("sigma_pos", np.nan)),
                "omar_retest_dist_norm": float(row.get("omar_retest_dist_norm", np.nan)),
                "omar_range_pct": float(row.get("omar_range_pct", np.nan)),
                "last10_range_over_omar": float(row.get("last10_range_over_omar", np.nan)),
                "decision_margin": float(row.get("decision_margin", np.nan)),
                "pred_dollar_score": float(row.get("pred_dollar_score", np.nan)),
                "pred_return_score": float(row.get("pred_return_score", np.nan)),
                "pred_win_prob": float(row.get("pred_win_prob", np.nan)),
                "pred_clean_entry_prob": float(row.get("pred_clean_entry_prob", np.nan)),
                "pred_stopout_risk": float(row.get("pred_stopout_risk", np.nan)),
                "orc_triggered": bool(row.get("orc_triggered", 0)),
                "failed_break_triggered": bool(row.get("failed_break_triggered", 0)),
            })
    df = pd.DataFrame(records).dropna(subset=["sigma_pos", "iv_percentile"])
    df["sigma_b"] = pd.qcut(df["sigma_pos"], 3, labels=False, duplicates="drop")
    df["iv_b"] = pd.qcut(df["iv_percentile"], 3, labels=False, duplicates="drop")
    df["cell"] = "s" + df["sigma_b"].astype(int).astype(str) + "_iv" + df["iv_b"].astype(int).astype(str)
    return df


def main():
    print("=== Phase C: failure-mode analysis ===\n")
    df = gather()
    print(f"Total trades: {len(df)}\n")

    # ----- Per-cell summary -----
    print("Per-cell summary (sigma_pos × iv_percentile):")
    print(f"{'cell':>6} {'n':>4} {'mean_hl':>10} {'med_hl':>10} {'pf':>7} "
          f"{'win_rate':>10} {'mean_oracle$':>14} {'mean_best$':>12}")
    for cell, g in df.groupby("cell"):
        if len(g) < 30:
            continue
        pos_sum = g["hl"][g["hl"] > 0].sum()
        neg_sum = g["hl"][g["hl"] < 0].sum()
        pf_v = pos_sum / abs(neg_sum) if neg_sum < 0 else float("inf")
        win_rate = (g["hl"] > 0).mean()
        print(f"{cell:>6} {len(g):>4} ${g['hl'].mean():>9.0f} ${g['hl'].median():>9.0f} "
              f"{pf_v:>7.2f} {win_rate:>10.1%} ${g['oracle_exit_pnl'].mean():>13.0f} "
              f"${g['best_forward_pnl'].mean():>11.0f}")
    print()

    # ----- Worst trades in chop × low (the persistent floor cell) -----
    floor_cells = ["s0_iv1", "s0_iv0"]  # chop × low neighborhood; sigma extreme + low/mid IV
    for floor_cell in floor_cells:
        sub = df[df["cell"] == floor_cell].copy()
        if len(sub) < 30:
            continue
        print(f"\n=== Cell {floor_cell}: {len(sub)} trades ===")
        worst = sub.nsmallest(10, "hl")
        best = sub.nlargest(10, "hl")
        print(f"\nWORST 10 trades in {floor_cell}:")
        cols = ["seed", "day", "side", "hl", "oracle_exit_pnl", "best_forward_pnl",
                "regret_vs_best", "decision_margin", "pred_dollar_score", "pred_win_prob",
                "vwap_slope", "sigma_pos", "iv_percentile", "first15_range_pct",
                "orc_triggered"]
        print(worst[cols].to_string(index=False))
        print(f"\nBEST 10 trades in {floor_cell}:")
        print(best[cols].to_string(index=False))
        print()
        # Mean comparisons
        print(f"Mean comparison (worst 10 vs best 10) in {floor_cell}:")
        cmp_cols = ["pred_dollar_score", "pred_return_score", "pred_win_prob",
                    "pred_clean_entry_prob", "pred_stopout_risk", "decision_margin",
                    "vwap_slope", "first15_range_pct", "omar_range_pct",
                    "last10_range_over_omar"]
        for c in cmp_cols:
            if c in worst.columns:
                w = worst[c].mean()
                b = best[c].mean()
                if np.isfinite(w) and np.isfinite(b):
                    diff = b - w
                    print(f"  {c:>30}: best {b:>+8.4f}  worst {w:>+8.4f}  diff {diff:>+8.4f}")

    # ----- Save full dataset for offline review -----
    os.makedirs("v3/artifacts/research", exist_ok=True)
    df.to_csv("v3/artifacts/research/phase_c_trade_review.csv", index=False)
    print(f"\nWrote v3/artifacts/research/phase_c_trade_review.csv (full {len(df)} trades)")

    # ----- Side-bias per cell -----
    print("\n=== Side bias per cell ===")
    for cell, g in df.groupby("cell"):
        if len(g) < 30: continue
        n = len(g)
        n_call = (g["side"] == "call").sum()
        n_put = (g["side"] == "put").sum()
        # PF by side
        for side in ["call", "put"]:
            sub = g[g["side"] == side]
            if len(sub) < 15: continue
            pos = sub["hl"][sub["hl"] > 0].sum()
            neg = sub["hl"][sub["hl"] < 0].sum()
            pf_s = pos / abs(neg) if neg < 0 else float("inf")
            print(f"  {cell:>6} {side:>4}: n={len(sub):>4}  PF={pf_s:>5.2f}  mean_hl=${sub['hl'].mean():>+5.0f}")

    # ----- Time-of-day patterns in worst trades -----
    print("\n=== Bar-index distribution of worst 100 trades vs all ===")
    worst_100 = df.nsmallest(100, "hl")
    print(f"All trades   bar_index: median={df['bar_index'].median():.0f}, "
          f"mean={df['bar_index'].mean():.1f}, q25-q75 [{df['bar_index'].quantile(0.25):.0f}, {df['bar_index'].quantile(0.75):.0f}]")
    print(f"Worst 100    bar_index: median={worst_100['bar_index'].median():.0f}, "
          f"mean={worst_100['bar_index'].mean():.1f}, q25-q75 [{worst_100['bar_index'].quantile(0.25):.0f}, {worst_100['bar_index'].quantile(0.75):.0f}]")


if __name__ == "__main__":
    main()
