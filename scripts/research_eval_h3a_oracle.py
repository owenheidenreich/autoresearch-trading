"""Evaluate H3a (3 new trade_state features) oracle vs baseline on FULL OOS.

Apples-to-apples comparison:
  baseline: simulated_l3_oracle_spx_live_0945_1130_seed42_balanced_fresh.npz
            (7-feature trade_state, current production)
  h3a:      simulated_l3_oracle_spx_live_0945_1130_seed42_h3a.npz
            (10-feature trade_state: +realized_vol_10bar, +pnl_velocity_5bar, +mfe_decay_rate)

Reports:
  - Aggregate PF, sum, mean per oracle (hybrid_live scoring)
  - Per-side PF breakdown (calls/puts)
  - Per-bucket capture-of-best (the H3a decision metric):
      best_exit_pnl bucket -> (oracle_pnl - best_exit_pnl) / max(|best|, $1)
      Baseline shows -1.43 in $0-200 bucket (oracle BLOWS UP small winners).
  - H3a decision: PASS if agg PF +0.10 OR small-winner ($0-200) capture >= 0.

The full-OOS row set is taken from forward_walk_chosen_seed42.pkl (~359 trades);
extended via merging with the layer2 chosen_trades.pkl if needed for 1664-trade
coverage. Match the row set actually used in research_eval_h2_oracle.py for
direct comparability.
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


def hl_score(pnl_raw, entry_features, exit_bar):
    return hybrid_live_utility(
        pnl_raw if np.isfinite(pnl_raw) else None,
        entry_mid=entry_features["entry_mid"],
        spread_fraction=entry_features["entry_sf"] if np.isfinite(entry_features["entry_sf"]) else 0.0,
        stopout_risk=entry_features["stopout_risk"] if np.isfinite(entry_features["stopout_risk"]) else 0.0,
        entry_bar=int(entry_features["entry_bar"]),
        exit_bar=int(exit_bar) if exit_bar >= 0 else None,
        session_end_bar=375,
    )


def main():
    print("=== H3a evaluation: new trade_state features vs baseline (seed 42) ===\n")

    base_path = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_balanced_fresh.npz"
    h3a_path = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_h3a.npz"

    if not os.path.exists(h3a_path):
        print(f"  H3a oracle not yet built: {h3a_path}")
        return 1
    if not os.path.exists(base_path):
        print(f"  Baseline oracle missing: {base_path}")
        return 1

    # Full OOS chosen trades from the combined champion (apples-to-apples
    # baseline). Forward_walk pickle was the 5-trade subset that misled the
    # initial H2 eval; the full chosen_trades.pkl is the correct full-OOS source.
    fw_df = pd.read_pickle(
        "v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed42/seed_42/chosen_trades.pkl"
    ).copy()
    fw_df = fw_df[fw_df["chosen_action_id"] > 0].reset_index(drop=True)
    print(f"OOS trades (chosen, non-flat): {len(fw_df)}")
    print(f"  day range: {fw_df['day'].min()} -> {fw_df['day'].max()}")

    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    base = np.load(base_path, allow_pickle=True)
    h3a = np.load(h3a_path, allow_pickle=True)
    base_pnl, base_bar = base["l3_exit_pnl"], base["l3_exit_bar"]
    h3a_pnl, h3a_bar = h3a["l3_exit_pnl"], h3a["l3_exit_bar"]

    # Best-exit-pnl per row/action: max over all bars in the trajectory.
    # We don't have raw trajectories here for capture-of-best; defer to a
    # secondary step that loads fw_trade_trajectories.pkl. For now, compare
    # base vs h3a directly.
    records = []
    for _, row in fw_df.iterrows():
        key = (row["day"], row["bar_index"])
        r = key_to_row.get(key)
        if r is None: continue
        a = int(row["chosen_action_id"])
        if a <= 0: continue
        bp = float(base_pnl[r, a]); bb = int(base_bar[r, a])
        hp = float(h3a_pnl[r, a]); hb = int(h3a_bar[r, a])

        em = float(al["entry_fill_mid"][r, a])
        ef = float(al["entry_spread_fraction"][r, a])
        eb = float(al["entry_fill_bar"][r, a])
        so = float(al["stopout_risk"][r, a])
        if not np.isfinite(em) or not np.isfinite(eb): continue

        ef_dict = {"entry_mid": em, "entry_sf": ef, "entry_bar": eb, "stopout_risk": so}
        bs_hl = hl_score(bp, ef_dict, bb)
        hs_hl = hl_score(hp, ef_dict, hb)
        records.append({
            "day": row["day"], "side": row["chosen_side"], "action_id": a,
            "base_bar": bb, "base_pnl": bp, "base_hl": bs_hl,
            "h3a_bar": hb, "h3a_pnl": hp, "h3a_hl": hs_hl,
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("  No comparable trades")
        return 1

    print()
    print(f"{'metric':>25} {'base':>12} {'h3a':>12} {'delta':>10}")
    print(f"{'PF (hybrid_live)':>25} {pf(df['base_hl']):>12.3f} {pf(df['h3a_hl']):>12.3f} {pf(df['h3a_hl'])-pf(df['base_hl']):>+10.3f}")
    print(f"{'sum $':>25} ${df['base_hl'].sum():>11.0f} ${df['h3a_hl'].sum():>11.0f} ${df['h3a_hl'].sum()-df['base_hl'].sum():>+9.0f}")
    print(f"{'mean exit_bar':>25} {df['base_bar'].mean():>12.1f} {df['h3a_bar'].mean():>12.1f} {df['h3a_bar'].mean()-df['base_bar'].mean():>+10.1f}")
    print()

    # Per-side breakdown
    for side in ["call", "put"]:
        sub = df[df["side"] == side]
        if len(sub) == 0: continue
        print(f"{'side='+side+' (n='+str(len(sub))+')':>25} PF base {pf(sub['base_hl']):.3f}  h3a {pf(sub['h3a_hl']):.3f}  delta {pf(sub['h3a_hl'])-pf(sub['base_hl']):+.3f}")
    print()

    # Per-bucket capture-of-best: addresses the small-winner failure that
    # H3a was designed to fix. Capture = (oracle_pnl - best) / max(|best|, 1).
    # Baseline reported -1.43 in $0-200 bucket (oracle BLOWS UP small winners).
    # Pass criterion: H3a small-winner capture >= 0.
    best = np.where(
        fw_df["chosen_side"].values == "call",
        fw_df["best_forward_pnl_call"].values,
        fw_df["best_forward_pnl_put"].values,
    )
    df_aug = df.copy()
    df_aug["best"] = best[df.index.values]
    df_aug["base_capture"] = (df_aug["base_pnl"] - df_aug["best"]) / np.clip(np.abs(df_aug["best"]), 1.0, None)
    df_aug["h3a_capture"] = (df_aug["h3a_pnl"] - df_aug["best"]) / np.clip(np.abs(df_aug["best"]), 1.0, None)
    buckets = [(0, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 1e9)]
    print(f"{'bucket':>15} {'n':>4} {'base_capture':>14} {'h3a_capture':>13} {'delta':>8}")
    bucket_results = {}
    for lo_b, hi_b in buckets:
        sel = (df_aug["best"] >= lo_b) & (df_aug["best"] < hi_b)
        sub = df_aug[sel]
        if len(sub) == 0: continue
        bc = float(sub["base_capture"].mean())
        hc = float(sub["h3a_capture"].mean())
        label = f"${lo_b:.0f}-{hi_b:.0f}" if hi_b < 1e9 else f"${lo_b:.0f}+"
        print(f"{label:>15} {len(sub):>4} {bc:>14.3f} {hc:>13.3f} {hc-bc:>+8.3f}")
        bucket_results[f"${lo_b}-{hi_b}"] = {"n": int(len(sub)), "base": bc, "h3a": hc, "delta": float(hc-bc)}
    print()

    # Bootstrap CI on delta-PF for statistical significance
    rng = np.random.default_rng(42)
    n = len(df)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        sub = df.iloc[idx]
        deltas.append(pf(sub["h3a_hl"]) - pf(sub["base_hl"]))
    deltas = np.asarray(deltas)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    p_negative = float((deltas <= 0).mean())
    print(f"Bootstrap delta-PF 95% CI: [{lo:+.3f}, {hi:+.3f}]  (P[delta<=0] = {p_negative:.3f})")
    print()

    # Per-bucket capture-of-best: requires per-trade peak from trajectories
    # where available. Skip if trajectories pickle isn't matched to this OOS
    # set; report the calls-vs-puts asymmetry instead (the H2 hint that
    # H3a was supposed to address).

    # Decision gate
    delta_pf = pf(df["h3a_hl"]) - pf(df["base_hl"])
    delta_sum = df["h3a_hl"].sum() - df["base_hl"].sum()

    print(f"=== Decision gate ===")
    print(f"  delta PF: {delta_pf:+.3f}  (gate: +0.10)")
    print(f"  delta sum: ${delta_sum:+.0f}")
    print()

    summary = {
        "n_trades": int(len(df)),
        "base_pf": float(pf(df["base_hl"])),
        "h3a_pf": float(pf(df["h3a_hl"])),
        "base_sum": float(df["base_hl"].sum()),
        "h3a_sum": float(df["h3a_hl"].sum()),
        "delta_pf": float(delta_pf),
        "delta_sum": float(delta_sum),
        "mean_base_bar": float(df["base_bar"].mean()),
        "mean_h3a_bar": float(df["h3a_bar"].mean()),
        "h3a_better_count": int((df["h3a_hl"] - df["base_hl"] > 50).sum()),
        "h3a_worse_count": int((df["h3a_hl"] - df["base_hl"] < -50).sum()),
    }

    summary["bootstrap_ci_lo"] = lo
    summary["bootstrap_ci_hi"] = hi
    summary["bootstrap_p_neg"] = p_negative
    summary["per_bucket_capture"] = bucket_results

    # Plan pass condition: agg PF +0.10 OR small-winner ($0-200) capture-of-best >= 0
    small_bucket_h3a = bucket_results.get("$0-200", {}).get("h3a", float("-inf"))
    small_bucket_base = bucket_results.get("$0-200", {}).get("base", float("-inf"))
    small_bucket_delta = small_bucket_h3a - small_bucket_base
    summary["small_winner_capture_h3a"] = small_bucket_h3a
    summary["small_winner_capture_delta"] = small_bucket_delta

    # Verdict: combine PF gate with significance check + small-winner gate
    pf_significant = lo > 0  # 95% CI excludes zero
    pf_passes = delta_pf >= 0.10
    small_passes = small_bucket_h3a >= 0
    small_lifts = small_bucket_delta >= 0.50  # meaningful absolute lift on the failing bucket

    if (pf_passes and pf_significant) or small_passes:
        verdict = "PASS"
        print("✓ H3a PASSES decision gate.")
        print("  Recommend rebuilding all 5 seeds + forward-walk gate.")
    elif pf_passes and small_lifts:
        verdict = "PASS-WEAK"
        print("≈ H3a WEAK PASS: PF lift +0.10 with wide CI, but small-winner capture lifts >= +0.50.")
        print("  Worth proceeding to 5-seed; small-winner improvement is the actual target.")
    elif delta_pf <= -0.10:
        verdict = "FAIL"
        print("✗ H3a FAILS: PF meaningfully worse. Recommend git revert.")
    else:
        verdict = "NEUTRAL"
        print("≈ H3a NEUTRAL: marginal change, no small-winner improvement. Recommend git revert.")
    summary["verdict"] = verdict

    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/h3a_seed42_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv("v3/artifacts/research/h3a_seed42_per_trade.csv", index=False)
    print(f"\nWrote v3/artifacts/research/h3a_seed42_eval.json + per_trade.csv")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
