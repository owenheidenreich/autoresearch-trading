"""Compare relaxed-target oracle vs original oracle on seed 42 forward walk.

Apples-to-apples test: same seed, same model checkpoint, same FW trades.
Difference is only the L3 oracle's exit predictions (relaxed vs original
target). The model's chosen actions don't change — we just look at how
the new oracle's predicted exit pnl differs.

Output: side-by-side trade comparison + aggregate PF/sum.
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
    print("=== Compare relaxed-target oracle vs original (seed 42) ===\n")

    # Load FW trades for seed 42
    fw_df = pd.read_pickle("v3/artifacts/forward_walk/forward_walk_chosen_seed42.pkl")
    fw_df = fw_df.copy()
    finite = np.isfinite(fw_df["fwd_pnl_hybrid_with_oracle"]) & np.isfinite(fw_df["fwd_pnl_time_stop"])
    fw_df = fw_df[finite].reset_index(drop=True)
    print(f"FW trades: {len(fw_df)}")

    # Bundle for joining
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    # Load both oracles
    orig_path = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_balanced_fresh.npz"
    relx_path = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_relaxed85.npz"
    if not os.path.exists(relx_path):
        print(f"  {relx_path} not yet built")
        return
    orig = np.load(orig_path, allow_pickle=True)
    relx = np.load(relx_path, allow_pickle=True)
    orig_pnl, orig_bar = orig["l3_exit_pnl"], orig["l3_exit_bar"]
    relx_pnl, relx_bar = relx["l3_exit_pnl"], relx["l3_exit_bar"]

    # For each chosen trade in FW, compute hybrid_live_utility under each oracle
    records = []
    for _, row in fw_df.iterrows():
        key = (row["day"], row["bar_index"])
        r = key_to_row.get(key)
        if r is None: continue
        a = int(row["chosen_action_id"])
        if a <= 0: continue
        orig_p = float(orig_pnl[r, a])
        orig_b = int(orig_bar[r, a])
        relx_p = float(relx_pnl[r, a])
        relx_b = int(relx_bar[r, a])

        entry_mid = float(al["entry_fill_mid"][r, a])
        entry_sf = float(al["entry_spread_fraction"][r, a])
        entry_bar = float(al["entry_fill_bar"][r, a])
        stopout = float(al["stopout_risk"][r, a])
        if not np.isfinite(entry_mid) or not np.isfinite(entry_bar):
            continue

        # Compute hybrid_live_utility for each oracle's prediction
        orig_pnl_hl = hybrid_live_utility(
            orig_p if np.isfinite(orig_p) else None,
            entry_mid=entry_mid,
            spread_fraction=entry_sf if np.isfinite(entry_sf) else 0.0,
            stopout_risk=stopout if np.isfinite(stopout) else 0.0,
            entry_bar=int(entry_bar),
            exit_bar=orig_b if orig_b >= 0 else None,
            session_end_bar=375,
        )
        relx_pnl_hl = hybrid_live_utility(
            relx_p if np.isfinite(relx_p) else None,
            entry_mid=entry_mid,
            spread_fraction=entry_sf if np.isfinite(entry_sf) else 0.0,
            stopout_risk=stopout if np.isfinite(stopout) else 0.0,
            entry_bar=int(entry_bar),
            exit_bar=relx_b if relx_b >= 0 else None,
            session_end_bar=375,
        )
        records.append({
            "day": row["day"], "side": row["chosen_side"], "action_id": a,
            "orig_pred_exit_pnl": orig_p, "orig_pred_exit_bar": orig_b,
            "relx_pred_exit_pnl": relx_p, "relx_pred_exit_bar": relx_b,
            "orig_pnl_hybrid_live": orig_pnl_hl,
            "relx_pnl_hybrid_live": relx_pnl_hl,
            "fwd_pnl_time_stop": float(row["fwd_pnl_time_stop"]),
            "fwd_pnl_reported_hybrid_oracle": float(row["fwd_pnl_hybrid_with_oracle"]),
        })

    df = pd.DataFrame(records)
    print(f"Comparable trades: {len(df)}\n")

    print(f"{'orig_pf_hybrid':>16} {'relx_pf_hybrid':>16} {'orig_sum':>10} {'relx_sum':>10}")
    print(f"{pf(df['orig_pnl_hybrid_live']):>16.3f} {pf(df['relx_pnl_hybrid_live']):>16.3f} "
          f"{df['orig_pnl_hybrid_live'].sum():>10.0f} {df['relx_pnl_hybrid_live'].sum():>10.0f}")
    print()

    # Side-by-side per trade
    print(f"=== Per-trade comparison ===")
    print(f"{'day':>12} {'side':>5} {'orig_bar':>9} {'relx_bar':>9} {'orig_pnl':>9} {'relx_pnl':>9} {'time_stop':>10} {'reported':>9}")
    df_sorted = df.sort_values(by="day")
    total_orig = 0; total_relx = 0
    for _, row in df_sorted.iterrows():
        op = row["orig_pnl_hybrid_live"]
        rp = row["relx_pnl_hybrid_live"]
        ob = row["orig_pred_exit_bar"]
        rb = row["relx_pred_exit_bar"]
        ts = row["fwd_pnl_time_stop"]
        rep = row["fwd_pnl_reported_hybrid_oracle"]
        flag = ""
        if not np.isnan(op) and not np.isnan(rp):
            if rp > op + 50: flag = " ↑"
            elif rp < op - 50: flag = " ↓"
        total_orig += op if np.isfinite(op) else 0
        total_relx += rp if np.isfinite(rp) else 0
        print(f"{str(row['day']):>12} {row['side']:>5} {ob:>9} {rb:>9} {op:>9.0f} {rp:>9.0f} {ts:>10.0f} {rep:>9.0f}{flag}")
    print()
    print(f"Total orig: ${total_orig:.0f}, relx: ${total_relx:.0f}, delta: ${total_relx-total_orig:+.0f}")

    os.makedirs("v3/artifacts/research", exist_ok=True)
    df.to_csv("v3/artifacts/research/relaxed_oracle_seed42_compare.csv", index=False)
    summary = {
        "n_trades": int(len(df)),
        "orig_pf": float(pf(df["orig_pnl_hybrid_live"])),
        "relx_pf": float(pf(df["relx_pnl_hybrid_live"])),
        "orig_sum": float(df["orig_pnl_hybrid_live"].sum()),
        "relx_sum": float(df["relx_pnl_hybrid_live"].sum()),
        "delta_sum": float(df["relx_pnl_hybrid_live"].sum() - df["orig_pnl_hybrid_live"].sum()),
        "trades_relaxed_better": int(((df["relx_pnl_hybrid_live"] - df["orig_pnl_hybrid_live"]) > 50).sum()),
        "trades_relaxed_worse": int(((df["relx_pnl_hybrid_live"] - df["orig_pnl_hybrid_live"]) < -50).sum()),
    }
    with open("v3/artifacts/research/relaxed_oracle_seed42_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written: {summary}")


if __name__ == "__main__":
    main()
