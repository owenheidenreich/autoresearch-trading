"""Find the right intraday-context axes to reweight along for Step C.

User's framing: macro regime (20-day SPX return × IV percentile) is wrong
for 0DTE — intraday context is what matters. A vol spike at 11am during a
bull tape is the SAME training class as a vol spike at 11am during a bear
tape.

Empirical question: among candidate intraday axes available at trade-entry
time (causal, no future info), which ones best stratify the cross-cell PF
spread? The axes with the largest cross-bucket PF range are the ones the
oracle conditions on differently — those are the right targets for C3 DRO
reweighting.

Method:
  1. For each candidate axis, bucket trades into 3 (low/mid/high tertiles)
     or natural buckets.
  2. Per bucket: count, mean realized hybrid_live PF (with baseline oracle).
  3. Cross-bucket spread = max - min.
  4. Rank axes by spread.
  5. Cross-tab the top 2-3 axes to identify independence vs redundancy.
  6. Predictability test: can the model's per-trade error be predicted from
     entry features alone? (If not, reweighting fights noise.)

Output: a recommendation for which axis (or 2-axis crosstab) to anchor the
C3 reweighting cells around.
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


def compute_intraday_spx_vol(df_spx: pd.DataFrame, lookback: int = 30) -> dict:
    """Per (date, bar_of_day) → 30-bar trailing realized vol (bps/min std of log returns)."""
    df = df_spx.copy().sort_values(["date", "time"]).reset_index(drop=True)
    df["log_ret"] = np.log(df["spx_close"]).diff()
    df["bar_of_day"] = df.groupby("date").cumcount() + 1  # 1..N within each day
    df["realized_vol_30bar"] = df.groupby("date")["log_ret"].transform(
        lambda x: x.rolling(lookback, min_periods=10).std() * 1e4  # bps
    )
    df["date_str"] = df["date"].astype(str)
    out = {(r.date_str, int(r.bar_of_day)): float(r.realized_vol_30bar)
           for r in df.itertuples(index=False)
           if pd.notna(r.realized_vol_30bar)}
    return out


def time_of_day_bucket(bar_index: int) -> str:
    """0DTE-aligned within the spx_live_0945_1130 execution window (bars 30-119).
    Boundary entries at bar 120+ produce NaN hl scores — exclude them via
    'edge' label that callers filter."""
    if bar_index >= 120:
        return "edge"  # boundary: 96 trades all at exactly bar 120, NaN hl
    if bar_index < 50:
        return "early"  # 9:30-10:20 ET (bars 30-49)
    if bar_index < 80:
        return "mid_morning"  # 10:20-10:50 ET (bars 50-79)
    return "late_morning"  # 10:50-11:30 ET (bars 80-119)


def gather_picks() -> pd.DataFrame:
    bundle = load_export_bundle("v3/artifacts/layer2_action_surface_dataset.pkl")
    rows = bundle["rows"].reset_index(drop=True)
    rows["__row__"] = np.arange(len(rows))
    al = bundle["action_labels"]
    key_to_row = (rows[["day", "bar_index", "__row__"]]
                  .drop_duplicates(subset=["day", "bar_index"])
                  .set_index(["day", "bar_index"])["__row__"].to_dict())

    print("Computing intraday SPX vol cache (30-bar trailing)...")
    with open(SPX_1MIN, "rb") as f:
        spx = pickle.load(f)
    intraday_vol = compute_intraday_spx_vol(spx, lookback=30)
    print(f"  cached vol for {len(intraday_vol)} (date, bar) pairs\n")

    records = []
    for seed in SEEDS:
        cp = CHOSEN_PATTERN.format(seed=seed)
        bp = BASE_PATTERN.format(seed=seed)
        if not os.path.exists(cp) or not os.path.exists(bp):
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

            pnl, bar = float(ex_pnl[r, a]), int(ex_bar[r, a])
            day = str(row["day"])
            bar_idx = int(row["bar_index"])
            iv_30 = intraday_vol.get((day, bar_idx + 1))  # bar_of_day is 1-indexed

            records.append({
                "seed": seed,
                "day": day,
                "bar_index": bar_idx,
                "side": row["chosen_side"],
                "base_hl": hl(pnl, em, ef, eb_, so, bar),
                # Causal entry-time intraday features:
                "tod_bucket": time_of_day_bucket(bar_idx),
                "intraday_vol_30bar": iv_30 if iv_30 is not None else float("nan"),
                "vwap_slope": float(row.get("vwap_slope", np.nan)),
                "volume_ratio": float(row.get("volume_ratio", np.nan)),
                "first15_range_pct": float(row.get("first15_range_pct", np.nan)),
                "atm_iv": float(row.get("atm_iv", np.nan)),
                "iv_percentile": float(row.get("iv_percentile", np.nan)),
                "sigma_pos": float(row.get("sigma_pos", np.nan)),
                "omar_retest_dist_norm": float(row.get("omar_retest_dist_norm", np.nan)),
                "omar_range_pct": float(row.get("omar_range_pct", np.nan)),
                "last10_range_over_omar": float(row.get("last10_range_over_omar", np.nan)),
                "decision_margin": float(row.get("decision_margin", np.nan)),
                "orc_triggered": bool(row.get("orc_triggered", 0)),
                "failed_break_triggered": bool(row.get("failed_break_triggered", 0)),
                "inside_first15": bool(row.get("inside_first15", 0)),
                "late_window_40_120_flag": bool(row.get("late_window_40_120_flag", 0)),
                "best_forward_pnl_chosen_side": float(
                    row["best_forward_pnl_call"] if row["chosen_side"] == "call"
                    else row["best_forward_pnl_put"]
                ),
            })
    return pd.DataFrame(records)


def axis_spread(df: pd.DataFrame, col: str, n_buckets: int = 3,
                is_categorical: bool = False) -> dict:
    """For a candidate axis, bucket trades and compute per-bucket PF spread."""
    if is_categorical:
        groups = df.groupby(col)
    else:
        # Tertile bucketing
        valid = df[col].dropna()
        if len(valid) < 30:
            return None
        try:
            df["__bucket__"] = pd.qcut(df[col], n_buckets, labels=False, duplicates="drop")
        except Exception:
            return None
        groups = df.groupby("__bucket__")

    cells = []
    for label, group in groups:
        if len(group) < 30:
            continue
        cells.append({
            "label": str(label),
            "n": int(len(group)),
            "pf": float(pf(group["base_hl"])),
            "mean_pnl": float(group["base_hl"].mean()),
        })

    if "__bucket__" in df.columns:
        df.drop(columns=["__bucket__"], inplace=True, errors="ignore")
    if len(cells) < 2:
        return None
    pfs = [c["pf"] for c in cells]
    return {
        "axis": col,
        "n_cells": len(cells),
        "min_pf": float(min(pfs)),
        "max_pf": float(max(pfs)),
        "spread": float(max(pfs) - min(pfs)),
        "min_n": int(min(c["n"] for c in cells)),
        "cells": cells,
    }


def crosstab_spread(df: pd.DataFrame, col1: str, col2: str,
                    n_buckets: int = 3, cat1: bool = False, cat2: bool = False) -> dict:
    """Cross-tab two axes."""
    df = df.copy()
    if not cat1:
        try:
            df["__b1__"] = pd.qcut(df[col1], n_buckets, labels=False, duplicates="drop")
        except Exception:
            return None
    else:
        df["__b1__"] = df[col1]
    if not cat2:
        try:
            df["__b2__"] = pd.qcut(df[col2], n_buckets, labels=False, duplicates="drop")
        except Exception:
            return None
    else:
        df["__b2__"] = df[col2]
    cells = []
    for (b1, b2), group in df.groupby(["__b1__", "__b2__"]):
        if len(group) < 20:
            continue
        cells.append({
            "b1": str(b1),
            "b2": str(b2),
            "n": int(len(group)),
            "pf": float(pf(group["base_hl"])),
        })
    if len(cells) < 2:
        return None
    pfs = [c["pf"] for c in cells]
    return {
        "axis1": col1, "axis2": col2,
        "n_cells": len(cells),
        "min_pf": float(min(pfs)),
        "max_pf": float(max(pfs)),
        "spread": float(max(pfs) - min(pfs)),
        "min_n": int(min(c["n"] for c in cells)),
        "cells": cells,
    }


def predictability_test(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Can per-trade outcome be predicted from entry features?
    Logistic regression: P(trade was profitable) ~ features. AUC ≥ 0.6 means
    reweighting has signal; AUC ≤ 0.55 means noise dominates.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sub = df.dropna(subset=feature_cols).copy()
    if len(sub) < 100:
        return {"auc": float("nan"), "note": "insufficient data"}
    X = sub[feature_cols].values
    y = (sub["base_hl"] > 0).astype(int).values
    if y.std() < 0.05:  # almost all wins or all losses
        return {"auc": float("nan"), "note": "y has no variance"}
    Xs = StandardScaler().fit_transform(X)
    # 5-fold cross-val AUC by hand
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(sub))
    folds = np.array_split(idx, 5)
    aucs = []
    for i in range(5):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(5) if j != i])
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xs[train], y[train])
        prob = clf.predict_proba(Xs[test])[:, 1]
        try:
            aucs.append(roc_auc_score(y[test], prob))
        except Exception:
            continue
    if not aucs:
        return {"auc": float("nan"), "note": "AUC could not be computed"}
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n": int(len(sub)),
        "win_rate": float(y.mean()),
    }


def main():
    print("=== Intraday axis search for Step C reweighting ===\n")
    df = gather_picks()
    n_total = len(df)
    df = df[df["tod_bucket"] != "edge"].copy()
    n_dropped = n_total - len(df)
    df = df[np.isfinite(df["base_hl"])].copy()
    print(f"Trades aggregated across 5 seeds: {n_total}")
    print(f"Dropped {n_dropped} edge boundary trades (bar_index >= 120, NaN hl scoring)")
    print(f"After NaN filtering: {len(df)}")
    print(f"Aggregate baseline PF: {pf(df['base_hl']):.3f}\n")

    # ----- Per-axis spread analysis -----
    print("=" * 80)
    print("Per-axis cross-bucket PF spread (using baseline oracle exits)\n")

    candidate_axes = [
        ("tod_bucket", True),                 # categorical: morning/midday/afternoon
        ("intraday_vol_30bar", False),
        ("vwap_slope", False),
        ("volume_ratio", False),
        ("first15_range_pct", False),
        ("atm_iv", False),
        ("iv_percentile", False),
        ("sigma_pos", False),
        ("omar_retest_dist_norm", False),
        ("omar_range_pct", False),
        ("last10_range_over_omar", False),
        ("decision_margin", False),
        ("orc_triggered", True),
        ("failed_break_triggered", True),
        ("inside_first15", True),
        ("late_window_40_120_flag", True),
    ]

    results = []
    for col, is_cat in candidate_axes:
        r = axis_spread(df, col, n_buckets=3, is_categorical=is_cat)
        if r is not None:
            results.append(r)

    results.sort(key=lambda r: r["spread"], reverse=True)
    print(f"{'axis':>30} {'n_cells':>8} {'min_n':>6} {'min_pf':>8} {'max_pf':>8} {'spread':>8}")
    for r in results:
        print(f"{r['axis']:>30} {r['n_cells']:>8} {r['min_n']:>6} {r['min_pf']:>8.3f} {r['max_pf']:>8.3f} {r['spread']:>8.3f}")
    print()

    # ----- Top axes detail -----
    print("=" * 80)
    print("Top 5 axes detail\n")
    for r in results[:5]:
        print(f"AXIS: {r['axis']}  (spread={r['spread']:.3f})")
        for c in sorted(r["cells"], key=lambda x: x["pf"]):
            print(f"  {c['label']:>15}  n={c['n']:>4}  PF={c['pf']:>6.3f}  mean_pnl=${c['mean_pnl']:>7.0f}")
        print()

    # ----- Cross-tab the top 2 + add tod_bucket (always informative for 0DTE) -----
    print("=" * 80)
    print("Cross-tabs of top axes\n")
    top_axes = [(r["axis"], any(c[0] == r["axis"] and c[1] for c in candidate_axes))
                for r in results[:3]]
    crosstab_pairs = []
    if any(a[0] == "tod_bucket" for a in top_axes):
        # tod is already in top — cross with the next non-tod top
        non_tod = [a for a in top_axes if a[0] != "tod_bucket"][:2]
        for a in non_tod:
            crosstab_pairs.append(("tod_bucket", True, a[0], a[1]))
    else:
        # Cross top with tod_bucket (tod is always informative for 0DTE)
        for a in top_axes[:2]:
            crosstab_pairs.append(("tod_bucket", True, a[0], a[1]))
    # Cross top1 with top2
    if len(top_axes) >= 2:
        crosstab_pairs.append((top_axes[0][0], top_axes[0][1], top_axes[1][0], top_axes[1][1]))

    for ax1, cat1, ax2, cat2 in crosstab_pairs:
        ct = crosstab_spread(df, ax1, ax2, n_buckets=3, cat1=cat1, cat2=cat2)
        if ct is None:
            print(f"({ax1} x {ax2}: insufficient data)")
            continue
        print(f"CROSS-TAB: {ax1} x {ax2}")
        print(f"  n_cells={ct['n_cells']}  min_n={ct['min_n']}  spread={ct['spread']:.3f}")
        for c in sorted(ct["cells"], key=lambda x: x["pf"]):
            print(f"    {c['b1']:>10} x {c['b2']:>4}  n={c['n']:>4}  PF={c['pf']:>6.3f}")
        print()

    # ----- Predictability test: can features predict trade outcome? -----
    print("=" * 80)
    print("Predictability test: can per-trade outcome be predicted from entry features?\n")
    print("If AUC > 0.60: reweighting has signal to work with.")
    print("If AUC < 0.55: per-trade variance is dominated by noise.\n")

    feature_subsets = [
        ("intraday only",
         ["intraday_vol_30bar", "vwap_slope", "volume_ratio", "first15_range_pct",
          "atm_iv", "iv_percentile", "sigma_pos", "omar_retest_dist_norm",
          "omar_range_pct", "last10_range_over_omar"]),
        ("decision_margin only",
         ["decision_margin"]),
        ("L2 model + intraday",
         ["decision_margin", "intraday_vol_30bar", "vwap_slope", "volume_ratio",
          "first15_range_pct", "atm_iv", "iv_percentile", "sigma_pos"]),
    ]
    for label, cols in feature_subsets:
        rep = predictability_test(df, cols)
        if "auc_mean" in rep:
            print(f"  {label:>25}  AUC = {rep['auc_mean']:.3f} ± {rep['auc_std']:.3f}  "
                  f"(n={rep['n']}, win_rate={rep['win_rate']:.2%})")
        else:
            print(f"  {label:>25}  {rep['note']}")
    print()

    # ----- Recommendation -----
    print("=" * 80)
    print("Recommendation\n")
    top = results[0]
    print(f"Best single-axis stratifier: {top['axis']} (spread={top['spread']:.3f})")
    print(f"  Cell PFs: {[c['pf'] for c in sorted(top['cells'], key=lambda x: x['pf'])]}")
    print()

    summary = {
        "n_trades_total": int(len(df)),
        "aggregate_pf": float(pf(df["base_hl"])),
        "axes": results,
        "feature_predictability": {label: predictability_test(df, cols)
                                   for label, cols in feature_subsets},
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/intraday_axis_search.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv("v3/artifacts/research/intraday_axis_search_picks.csv", index=False)
    print(f"Wrote v3/artifacts/research/intraday_axis_search.json + picks.csv")


if __name__ == "__main__":
    main()
