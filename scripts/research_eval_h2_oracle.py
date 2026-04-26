"""Evaluate H2 (regret-weighted oracle) vs original on seed 42 forward walk.

Apples-to-apples: same dataset, same model checkpoint, same FW trades.
Differs only in the oracle's exit predictions (regret-weighted training).

Output:
  - Per-trade comparison: orig pnl vs H2 pnl (using hybrid_live_utility scoring)
  - Aggregate PF/sum delta
  - Decision: H2 wins → recommend full rebuild; H2 loses → revert
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from v3.layer2.action_surface_dataset import hybrid_live_utility
from v3.layer2.common import load_export_bundle


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def main():
    print("=== H2 evaluation: regret-weighted vs original (seed 42) ===\n")

    h2_path = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_h2regret.npz"
    orig_path = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_balanced_fresh.npz"

    if not os.path.exists(h2_path):
        print(f"  H2 oracle not built yet: {h2_path}")
        return

    fw_df = pd.read_pickle("v3/artifacts/forward_walk/forward_walk_chosen_seed42.pkl")
    fw_df = fw_df.copy()
    finite = np.isfinite(fw_df["fwd_pnl_hybrid_with_oracle"]) & np.isfinite(fw_df["fwd_pnl_time_stop"])
    fw_df = fw_df[finite].reset_index(drop=True)
    print(f"FW trades: {len(fw_df)}")

    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    orig = np.load(orig_path, allow_pickle=True)
    h2 = np.load(h2_path, allow_pickle=True)
    orig_pnl, orig_bar = orig["l3_exit_pnl"], orig["l3_exit_bar"]
    h2_pnl, h2_bar = h2["l3_exit_pnl"], h2["l3_exit_bar"]

    records = []
    for _, row in fw_df.iterrows():
        key = (row["day"], row["bar_index"])
        r = key_to_row.get(key)
        if r is None: continue
        a = int(row["chosen_action_id"])
        if a <= 0: continue
        op = float(orig_pnl[r, a]); ob = int(orig_bar[r, a])
        h2p = float(h2_pnl[r, a]); h2b = int(h2_bar[r, a])

        entry_mid = float(al["entry_fill_mid"][r, a])
        entry_sf = float(al["entry_spread_fraction"][r, a])
        entry_bar = float(al["entry_fill_bar"][r, a])
        stopout = float(al["stopout_risk"][r, a])
        if not np.isfinite(entry_mid) or not np.isfinite(entry_bar): continue

        op_hl = hybrid_live_utility(
            op if np.isfinite(op) else None,
            entry_mid=entry_mid, spread_fraction=entry_sf if np.isfinite(entry_sf) else 0.0,
            stopout_risk=stopout if np.isfinite(stopout) else 0.0,
            entry_bar=int(entry_bar), exit_bar=ob if ob >= 0 else None, session_end_bar=375,
        )
        h2_hl = hybrid_live_utility(
            h2p if np.isfinite(h2p) else None,
            entry_mid=entry_mid, spread_fraction=entry_sf if np.isfinite(entry_sf) else 0.0,
            stopout_risk=stopout if np.isfinite(stopout) else 0.0,
            entry_bar=int(entry_bar), exit_bar=h2b if h2b >= 0 else None, session_end_bar=375,
        )
        records.append({
            "day": row["day"], "side": row["chosen_side"], "action_id": a,
            "orig_bar": ob, "orig_pnl": op, "orig_hl": op_hl,
            "h2_bar": h2b, "h2_pnl": h2p, "h2_hl": h2_hl,
            "fwd_pnl_time_stop": float(row["fwd_pnl_time_stop"]),
            "fwd_reported": float(row["fwd_pnl_hybrid_with_oracle"]),
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("  No comparable trades")
        return

    print()
    print(f"{'metric':>25} {'orig':>12} {'h2':>12} {'delta':>10}")
    print(f"{'PF (hybrid_live)':>25} {pf(df['orig_hl']):>12.3f} {pf(df['h2_hl']):>12.3f} {pf(df['h2_hl'])-pf(df['orig_hl']):>+10.3f}")
    print(f"{'sum $ (hybrid_live)':>25} ${df['orig_hl'].sum():>11.0f} ${df['h2_hl'].sum():>11.0f} ${df['h2_hl'].sum()-df['orig_hl'].sum():>+9.0f}")
    print(f"{'mean exit_bar':>25} {df['orig_bar'].mean():>12.1f} {df['h2_bar'].mean():>12.1f} {df['h2_bar'].mean()-df['orig_bar'].mean():>+10.1f}")
    print()

    print("Per-trade detail (where H2 differs from orig):")
    df['delta'] = df['h2_hl'] - df['orig_hl']
    df_diff = df[df['delta'].abs() > 50].sort_values('delta')
    if df_diff.empty:
        print("  no material differences")
    else:
        print(f"{'day':>12} {'side':>5} {'orig_bar':>9} {'h2_bar':>7} {'orig_hl':>9} {'h2_hl':>9} {'delta':>9}")
        for _, r in df_diff.iterrows():
            print(f"{str(r['day']):>12} {r['side']:>5} {r['orig_bar']:>9} {r['h2_bar']:>7} {r['orig_hl']:>9.0f} {r['h2_hl']:>9.0f} {r['delta']:>+9.0f}")
    print()

    summary = {
        "n_trades": int(len(df)),
        "orig_pf_hl": float(pf(df["orig_hl"])),
        "h2_pf_hl": float(pf(df["h2_hl"])),
        "orig_sum": float(df["orig_hl"].sum()),
        "h2_sum": float(df["h2_hl"].sum()),
        "delta_pf": float(pf(df["h2_hl"]) - pf(df["orig_hl"])),
        "delta_sum": float(df["h2_hl"].sum() - df["orig_hl"].sum()),
        "mean_orig_bar": float(df["orig_bar"].mean()),
        "mean_h2_bar": float(df["h2_bar"].mean()),
        "h2_better_count": int((df["delta"] > 50).sum()),
        "h2_worse_count": int((df["delta"] < -50).sum()),
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/h2_seed42_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv("v3/artifacts/research/h2_seed42_per_trade.csv", index=False)
    print(f"Wrote v3/artifacts/research/h2_seed42_eval.json + per_trade.csv")
    print()

    # Recommendation
    if pf(df["h2_hl"]) > pf(df["orig_hl"]) + 0.05 and df["h2_hl"].sum() > df["orig_hl"].sum():
        print("✓ H2 WINS: PF lifted >+0.05 AND sum > orig. Recommend rebuilding all 5 seeds.")
    elif pf(df["h2_hl"]) < pf(df["orig_hl"]) - 0.05 or df["h2_hl"].sum() < df["orig_hl"].sum() - 1000:
        print("✗ H2 LOSES: PF or sum is meaningfully worse. Recommend git revert.")
    else:
        print("≈ H2 NEUTRAL: marginal change. Recommend git revert; not worth deployment risk.")


if __name__ == "__main__":
    main()
