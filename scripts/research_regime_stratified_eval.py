"""Regime-stratified diagnostic on H3a vs baseline 5-seed OOS results.

Asks the question the user surfaced: is the model regime-adaptive, or
regime-specialized? Slices the existing 376-trade-per-seed OOS sample by
two axes:

  - Trend regime (20-day SPX return at trade's day): bear / chop / bull
  - Vol regime (VIX at entry bar):                   low / mid / high

For each of 9 (trend x vol) cells, reports:
  - n trades (baseline + h3a same set; only oracle predictions differ)
  - PF baseline, PF h3a, delta
  - sum$ baseline, h3a, delta

If H3a's lift concentrates in 1-2 cells: oracle is regime-specialized.
If lift is uniform: oracle is regime-agnostic, structural change (H3c
sequence model) is the next move.

Uses the same eval logic as research_eval_h3a_5seed.py (apples-to-apples
hybrid_live scoring).
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
H3A_PATTERN = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{seed}_h3a.npz"
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
    """Compute trailing 20-day SPX return for each trading day."""
    with open(SPX_1MIN, "rb") as f:
        spx = pickle.load(f)
    daily = spx.groupby("date")["spx_close"].last().sort_index()
    daily.index = daily.index.astype(str)
    # 20-day rolling return (today / 20 days ago - 1)
    rolling_20d = daily.pct_change(20)
    return rolling_20d.to_dict()


def trend_regime(ret20d: float) -> str:
    if not np.isfinite(ret20d):
        return "unknown"
    if ret20d <= -0.02:
        return "bear"
    elif ret20d >= 0.02:
        return "bull"
    else:
        return "chop"


def vol_regime(iv_pct: float) -> str:
    """Classify by iv_percentile (already normalized to [0, 1])."""
    if not np.isfinite(iv_pct):
        return "unknown"
    if iv_pct < 0.33:
        return "low"
    elif iv_pct < 0.66:
        return "mid"
    else:
        return "high"


def gather_records(bundle, key_to_row, al, daily_returns):
    """Combine all 5 seeds into a single dataframe with regime labels."""
    all_records = []
    for seed in SEEDS:
        bp = BASE_PATTERN.format(seed=seed)
        hp = H3A_PATTERN.format(seed=seed)
        cp = CHOSEN_PATTERN.format(seed=seed)
        if not all(os.path.exists(p) for p in [bp, hp, cp]):
            continue
        fw = pd.read_pickle(cp)
        fw = fw[fw["chosen_action_id"] > 0].reset_index(drop=True)
        base = np.load(bp, allow_pickle=True)
        h3a = np.load(hp, allow_pickle=True)
        bp_arr, bb_arr = base["l3_exit_pnl"], base["l3_exit_bar"]
        hp_arr, hb_arr = h3a["l3_exit_pnl"], h3a["l3_exit_bar"]

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

            bp_pnl = float(bp_arr[r, a]); bb_bar = int(bb_arr[r, a])
            hp_pnl = float(hp_arr[r, a]); hb_bar = int(hb_arr[r, a])

            all_records.append({
                "seed": seed,
                "day": day,
                "side": row["chosen_side"],
                "iv_percentile": iv_pct,
                "ret_20d": ret20,
                "trend": trend_regime(ret20),
                "vol": vol_regime(iv_pct),
                "base_hl": hl(bp_pnl, em, ef, eb_, so, bb_bar),
                "h3a_hl": hl(hp_pnl, em, ef, eb_, so, hb_bar),
            })
    return pd.DataFrame(all_records)


def main():
    print("=== Regime-stratified diagnostic: H3a vs baseline ===\n")
    print("Loading SPX daily returns for trend regime classification...")
    daily_returns = build_daily_returns()
    print(f"  {len(daily_returns)} daily returns computed.\n")

    print("Loading bundle...")
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    df = gather_records(bundle, key_to_row, al, daily_returns)
    print(f"Total trades across 5 seeds: {len(df)}")
    print(f"  trend distribution: {df['trend'].value_counts().to_dict()}")
    print(f"  vol distribution:   {df['vol'].value_counts().to_dict()}")
    print()

    # Aggregate
    print("=== Aggregate (sanity check vs prior 5-seed eval) ===")
    print(f"  base mean PF:   {pf(df['base_hl']):.3f}")
    print(f"  h3a mean PF:    {pf(df['h3a_hl']):.3f}")
    print(f"  delta:          {pf(df['h3a_hl']) - pf(df['base_hl']):+.3f}")
    print()

    # Per-trend
    print("=== By TREND regime (20-day SPX return at trade day) ===")
    print(f"  {'regime':>8} {'n':>5} {'base_pf':>9} {'h3a_pf':>9} {'delta':>9} {'base_sum':>10} {'h3a_sum':>10}")
    for r_label in ["bear", "chop", "bull", "unknown"]:
        sub = df[df["trend"] == r_label]
        if len(sub) == 0: continue
        bp_ = pf(sub["base_hl"]); hp_ = pf(sub["h3a_hl"])
        bs = sub["base_hl"].sum(); hs = sub["h3a_hl"].sum()
        print(f"  {r_label:>8} {len(sub):>5} {bp_:>9.3f} {hp_:>9.3f} {hp_-bp_:>+9.3f} ${bs:>9.0f} ${hs:>9.0f}")
    print()

    # Per-vol
    print("=== By VOL regime (iv_percentile at entry) ===")
    print(f"  {'regime':>8} {'n':>5} {'base_pf':>9} {'h3a_pf':>9} {'delta':>9} {'base_sum':>10} {'h3a_sum':>10}")
    for r_label in ["low", "mid", "high", "unknown"]:
        sub = df[df["vol"] == r_label]
        if len(sub) == 0: continue
        bp_ = pf(sub["base_hl"]); hp_ = pf(sub["h3a_hl"])
        bs = sub["base_hl"].sum(); hs = sub["h3a_hl"].sum()
        print(f"  {r_label:>8} {len(sub):>5} {bp_:>9.3f} {hp_:>9.3f} {hp_-bp_:>+9.3f} ${bs:>9.0f} ${hs:>9.0f}")
    print()

    # Bootstrap helper for delta-PF significance
    def boot_delta_ci(sub, n_iter=2000):
        rng = np.random.default_rng(42)
        n = len(sub)
        if n < 5:
            return float("nan"), float("nan"), float("nan")
        deltas = []
        bh = sub["base_hl"].values
        hh = sub["h3a_hl"].values
        for _ in range(n_iter):
            idx = rng.integers(0, n, size=n)
            deltas.append(pf(hh[idx]) - pf(bh[idx]))
        deltas = np.asarray(deltas)
        return (float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
                float((deltas <= 0).mean()))

    # Cross-tab: trend x vol
    print("=== Cross-tab: TREND x VOL (the regime adaptability question) ===")
    print(f"  {'trend':>8} {'vol':>6} {'n':>5} {'base_pf':>9} {'h3a_pf':>9} {'delta':>9}  {'95%CI_delta':>20} {'p<=0':>6}")
    cells = []
    for t in ["bear", "chop", "bull"]:
        for v in ["low", "mid", "high"]:
            sub = df[(df["trend"] == t) & (df["vol"] == v)]
            if len(sub) < 5:
                continue
            bp_ = pf(sub["base_hl"]); hp_ = pf(sub["h3a_hl"])
            ci_lo, ci_hi, p_neg = boot_delta_ci(sub)
            print(f"  {t:>8} {v:>6} {len(sub):>5} {bp_:>9.3f} {hp_:>9.3f} {hp_-bp_:>+9.3f}  [{ci_lo:+6.2f}, {ci_hi:+6.2f}]  {p_neg:>5.2f}")
            cells.append({"trend": t, "vol": v, "n": int(len(sub)),
                          "base_pf": float(bp_), "h3a_pf": float(hp_),
                          "delta_pf": float(hp_-bp_),
                          "ci_lo": ci_lo, "ci_hi": ci_hi, "p_neg": p_neg})
    print()

    # Where does H3a help vs hurt?
    print("=== Profitability landscape ===")
    # Cells where baseline is profitable (PF > 1)
    prof_cells = [c for c in cells if c["base_pf"] > 1.0]
    unprof_cells = [c for c in cells if c["base_pf"] <= 1.0]
    print(f"  Baseline-profitable cells (PF>1):    {len(prof_cells)}/{len(cells)}")
    for c in sorted(prof_cells, key=lambda x: x["base_pf"]):
        print(f"    {c['trend']:>5} x {c['vol']:>4}  n={c['n']:>3}  base_pf={c['base_pf']:5.2f}  h3a={c['h3a_pf']:5.2f}  delta {c['delta_pf']:+.2f}")
    if unprof_cells:
        print(f"  Baseline-unprofitable cells (PF<=1): {len(unprof_cells)}/{len(cells)}")
        for c in sorted(unprof_cells, key=lambda x: x["base_pf"]):
            print(f"    {c['trend']:>5} x {c['vol']:>4}  n={c['n']:>3}  base_pf={c['base_pf']:5.2f}  h3a={c['h3a_pf']:5.2f}  delta {c['delta_pf']:+.2f}")
    print()

    # H3a impact by cell
    helped = [c for c in cells if c["delta_pf"] > 0.10]
    hurt = [c for c in cells if c["delta_pf"] < -0.10]
    neutral = [c for c in cells if abs(c["delta_pf"]) <= 0.10]
    print(f"  H3a helped (delta>+0.10):  {len(helped)}/{len(cells)} cells")
    print(f"  H3a neutral (|delta|<0.10): {len(neutral)}/{len(cells)} cells")
    print(f"  H3a hurt (delta<-0.10):    {len(hurt)}/{len(cells)} cells")
    print()

    # Interpretation: separate "is profitable" from "is adaptive"
    print("=== Interpretation ===")
    print()
    h3a_significant_hurt = [c for c in cells if c["ci_hi"] < 0 and c["n"] >= 30]
    h3a_significant_help = [c for c in cells if c["ci_lo"] > 0 and c["n"] >= 30]

    # Profitability check (PF > 1 everywhere)
    h3a_unprof = [c for c in cells if c["h3a_pf"] <= 1.0 and c["n"] >= 30]
    print(f"  Profitability (n>=30 cells, PF>1):")
    print(f"    baseline cells PF<=1: {len([c for c in cells if c['base_pf']<=1.0 and c['n']>=30])}")
    print(f"    h3a cells PF<=1:     {len(h3a_unprof)}")
    print()
    # Stability check (PF variance across cells)
    h3a_pfs = [c["h3a_pf"] for c in cells if c["n"] >= 30]
    base_pfs = [c["base_pf"] for c in cells if c["n"] >= 30]
    print(f"  Cross-cell PF spread (n>=30 cells):")
    print(f"    baseline: min {min(base_pfs):.2f} -> max {max(base_pfs):.2f}  range {max(base_pfs)-min(base_pfs):.2f}")
    print(f"    h3a:      min {min(h3a_pfs):.2f} -> max {max(h3a_pfs):.2f}  range {max(h3a_pfs)-min(h3a_pfs):.2f}")
    print()
    # Regime-conditional findings
    print(f"  H3a significantly HELPED (CI excludes 0): {len(h3a_significant_help)} cells")
    for c in h3a_significant_help:
        print(f"    {c['trend']} x {c['vol']:>4} (n={c['n']:>3})  delta {c['delta_pf']:+.2f}  CI [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")
    print(f"  H3a significantly HURT  (CI excludes 0): {len(h3a_significant_hurt)} cells")
    for c in h3a_significant_hurt:
        print(f"    {c['trend']} x {c['vol']:>4} (n={c['n']:>3})  delta {c['delta_pf']:+.2f}  CI [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")
    print()
    if h3a_significant_hurt and h3a_significant_help:
        print("  VERDICT: H3a is REGIME-CONDITIONAL — helps in some regimes, hurts in others.")
        print("  -> Next move: add multi-day regime features (trend, vol percentile) to trade_state")
        print("     so the oracle can learn to behave differently per regime.")
    elif h3a_significant_help and not h3a_significant_hurt:
        print("  VERDICT: H3a uniformly helpful where significant. Regime adaptability OK.")
        print("  -> Next gap is representation depth; H3c sequence model is the move.")
    else:
        print("  VERDICT: No significant H3a effect anywhere. Lift is noise.")
    print()

    summary = {
        "by_trend": {r: {"n": int(len(df[df["trend"]==r])),
                          "base_pf": float(pf(df[df["trend"]==r]["base_hl"])),
                          "h3a_pf": float(pf(df[df["trend"]==r]["h3a_hl"]))}
                     for r in ["bear","chop","bull","unknown"] if len(df[df["trend"]==r]) > 0},
        "by_vol":   {r: {"n": int(len(df[df["vol"]==r])),
                          "base_pf": float(pf(df[df["vol"]==r]["base_hl"])),
                          "h3a_pf": float(pf(df[df["vol"]==r]["h3a_hl"]))}
                     for r in ["low","mid","high","unknown"] if len(df[df["vol"]==r]) > 0},
        "cells": cells,
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/regime_stratified_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv("v3/artifacts/research/regime_stratified_per_trade.csv", index=False)
    print(f"Wrote v3/artifacts/research/regime_stratified_eval.json + per_trade.csv")


if __name__ == "__main__":
    main()
