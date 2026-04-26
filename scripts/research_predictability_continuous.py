"""Continuous-magnitude predictability test for cell-conditional reweighting.

Replaces the binary win/loss AUC test (which conflates +$5000 and +$50 wins)
with three magnitude-aware analyses:

1. Variance decomposition. How much of total trade-pnl variance is
   between-cell (cell-conditioning can exploit) vs within-cell (noise)?
   - If between-cell variance is small (<5%), reweighting has little to work with.
   - If between-cell variance is large (>15%), cells are economically distinct.

2. Continuous regression on hl outcome. Predict signed-log10(|hl|+1) from
   entry features. Report 5-fold CV R-squared. Compare to:
   - linear regression with all features
   - random-forest baseline (catches non-linearities)
   - cell-id-only regression (just dummy variables)

3. Per-cell predictability. Train regression separately within each of the 9
   sigma_pos × iv_percentile cells. Compare per-cell R-squareds. If R-squared varies
   a lot across cells, features have cell-conditional signal — exactly what
   cell-weighted training can exploit. If R-squared is uniformly tiny, no
   per-cell structure to exploit; reweighting just shifts noise.

Outcome target: signed_log_hl = sign(hl) * log10(|hl| + 1). Compresses
outliers (10000 vs 100) while preserving sign and ordering.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

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


def signed_log(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.sign(x) * np.log10(abs(x) + 1.0))


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
            if int(row["bar_index"]) >= 120: continue  # drop boundary

            pnl, bar = float(ex_pnl[r, a]), int(ex_bar[r, a])
            hl_val = hl(pnl, em, ef, eb_, so, bar)
            if not np.isfinite(hl_val): continue

            # Cell ID: sigma_pos × iv_percentile (3 buckets each via tertile)
            records.append({
                "seed": seed,
                "day": row["day"],
                "side": row["chosen_side"],
                "hl": hl_val,
                "signed_log_hl": signed_log(hl_val),
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
                "bar_index": int(row["bar_index"]),
            })
    df = pd.DataFrame(records).dropna()
    # Tertile cell IDs
    df["sigma_b"] = pd.qcut(df["sigma_pos"], 3, labels=False, duplicates="drop")
    df["iv_b"] = pd.qcut(df["iv_percentile"], 3, labels=False, duplicates="drop")
    df["cell_id"] = df["sigma_b"].astype(int) * 3 + df["iv_b"].astype(int)
    return df


def variance_decomposition(df: pd.DataFrame, target_col: str, cell_col: str) -> dict:
    """Total variance of target. Decompose into between-cell + within-cell."""
    y = df[target_col].values
    total_var = float(np.var(y))
    if total_var == 0:
        return {"total": 0.0, "between": 0.0, "within": 0.0, "frac_between": 0.0}
    overall_mean = y.mean()
    between = 0.0
    within = 0.0
    for _, group in df.groupby(cell_col):
        n = len(group)
        cell_mean = group[target_col].mean()
        between += n * (cell_mean - overall_mean) ** 2
        within += ((group[target_col] - cell_mean) ** 2).sum()
    between /= len(df)
    within /= len(df)
    return {
        "total": total_var,
        "between": float(between),
        "within": float(within),
        "frac_between": float(between / total_var),
    }


def cv_r2(X: np.ndarray, y: np.ndarray, model_factory, n_folds: int = 5,
          rng_seed: int = 42) -> tuple[float, float]:
    """Cross-validated R² with given model."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, n_folds)
    r2s = []
    for i in range(n_folds):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        if len(train) < 10 or len(test) < 5:
            continue
        m = model_factory()
        m.fit(X[train], y[train])
        pred = m.predict(X[test])
        try:
            r2s.append(r2_score(y[test], pred))
        except Exception:
            continue
    if not r2s:
        return float("nan"), float("nan")
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    print("=== Continuous-magnitude predictability test ===\n")
    df = gather()
    print(f"Trades (after dropping bar_index>=120 + NaN hl): {len(df)}")
    print(f"Aggregate hl: mean=${df['hl'].mean():.0f}, std=${df['hl'].std():.0f}, "
          f"range [${df['hl'].min():.0f}, ${df['hl'].max():.0f}]")
    print(f"signed_log_hl: mean={df['signed_log_hl'].mean():.3f}, "
          f"std={df['signed_log_hl'].std():.3f}")
    print(f"Cells (sigma_pos × iv_percentile): {df['cell_id'].nunique()}")
    print()

    # 1. Variance decomposition
    print("=" * 80)
    print("Variance decomposition: how much trade-pnl variance is between-cell?\n")
    for tgt in ["hl", "signed_log_hl"]:
        vd = variance_decomposition(df, tgt, "cell_id")
        print(f"  Target = {tgt}:")
        print(f"    Total variance: {vd['total']:.4f}")
        print(f"    Between-cell:   {vd['between']:.4f}  ({vd['frac_between']:.1%} of total)")
        print(f"    Within-cell:    {vd['within']:.4f}")
        print()

    # 2. Continuous regression
    print("=" * 80)
    print("Continuous regression on signed_log_hl (5-fold CV R²)\n")
    feat_cols = ["vwap_slope", "volume_ratio", "first15_range_pct",
                 "atm_iv", "iv_percentile", "sigma_pos",
                 "omar_retest_dist_norm", "omar_range_pct",
                 "last10_range_over_omar", "decision_margin", "bar_index"]
    Xs = StandardScaler().fit_transform(df[feat_cols].values)
    y = df["signed_log_hl"].values

    mean_r2, std_r2 = cv_r2(Xs, y, LinearRegression)
    print(f"  Linear regression (all entry features):     R² = {mean_r2:+.4f} ± {std_r2:.4f}")

    mean_r2, std_r2 = cv_r2(Xs, y, lambda: RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=2, random_state=42))
    print(f"  Random forest (catches non-linearity):     R² = {mean_r2:+.4f} ± {std_r2:.4f}")

    # Cell-only baseline (categorical dummies)
    cell_dummies = pd.get_dummies(df["cell_id"], prefix="cell").values.astype(np.float64)
    mean_r2, std_r2 = cv_r2(cell_dummies, y, LinearRegression)
    print(f"  Cell ID only (9 cell dummies):             R² = {mean_r2:+.4f} ± {std_r2:.4f}")

    # Cell + features
    Xc = np.hstack([Xs, cell_dummies])
    mean_r2, std_r2 = cv_r2(Xc, y, LinearRegression)
    print(f"  Cell ID + features:                        R² = {mean_r2:+.4f} ± {std_r2:.4f}")
    print()

    # 3. Per-cell regression — does R² vary across cells?
    print("=" * 80)
    print("Per-cell regression: does feature signal differ across cells?\n")
    print(f"  {'cell':>8} {'n':>5} {'mean_hl':>10} {'lin_R²':>10}")
    cell_r2s = []
    for cell, group in df.groupby("cell_id"):
        if len(group) < 50:
            continue
        Xs_c = StandardScaler().fit_transform(group[feat_cols].values)
        y_c = group["signed_log_hl"].values
        mean_r2, _ = cv_r2(Xs_c, y_c, LinearRegression, n_folds=3)
        print(f"  cell {cell:>3}  {len(group):>5} {group['hl'].mean():>10.0f}   {mean_r2:>+10.4f}")
        cell_r2s.append(mean_r2)
    print()
    if cell_r2s:
        print(f"  Per-cell R² range: [{min(cell_r2s):+.4f}, {max(cell_r2s):+.4f}]")
        print(f"  Mean: {np.mean(cell_r2s):+.4f}, std: {np.std(cell_r2s):.4f}")
        print()
        if max(cell_r2s) - min(cell_r2s) > 0.05:
            print("  → R² VARIES meaningfully across cells. Cell-conditional structure")
            print("    exists; reweighting can exploit it. Stronger evidence for C3.")
        else:
            print("  → R² is uniformly low across cells. Within-cell variation is noise.")
            print("    Reweighting won't change much beyond shifting attention.")
    print()

    # Save
    summary = {
        "n_trades": int(len(df)),
        "variance_decomposition": {
            tgt: variance_decomposition(df, tgt, "cell_id")
            for tgt in ["hl", "signed_log_hl"]
        },
        "per_cell_r2": cell_r2s,
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/predictability_continuous.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote v3/artifacts/research/predictability_continuous.json")


if __name__ == "__main__":
    main()
