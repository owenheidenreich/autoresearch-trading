"""Iteration 3: time-series CV + feature importance + scaled LR.

Iteration 2 found GB classifier hits PF 2.380 (+27% over baseline 1.877).
But random 5-fold CV may leak temporal info (training on W12 trades to
predict W2 trades is unrealistic). Iteration 3:

1. ROLLING TIME-SERIES CV: train on windows [0..k], predict on window k+1.
   This is the honest test — does the classifier work prospectively?
2. Feature importance: which features actually drive the GB classifier?
3. Scaled LR: with proper StandardScaler, does LR catch up to GB?
4. Reduced feature set: can we get the lift with fewer features (more
   robust to live-data drift)?

Output: v3/artifacts/research/oracle_gate_iter3.json
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def pf(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    pos = p[p > 0].sum()
    neg = p[p < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def main():
    seeds = [42, 43, 44, 45, 46]
    rows = []
    for s in seeds:
        df = pd.read_pickle(
            f"v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{s}/seed_{s}/chosen_trades.pkl"
        )
        df = df.copy()
        df["seed"] = s
        finite = np.isfinite(df["chosen_objective_pnl"]) & np.isfinite(df["chosen_time_stop_pnl"])
        df = df[finite].reset_index(drop=True)
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    oracle_pnl = all_df["chosen_objective_pnl"].astype(float).values
    no_oracle_pnl = all_df["chosen_time_stop_pnl"].astype(float).values
    y = (oracle_pnl > no_oracle_pnl).astype(int)

    feature_cols = [
        "sigma_pos", "iv_percentile", "vix", "atm_iv", "vwap_slope",
        "volume_ratio", "first15_range_pct", "decision_margin",
        "pred_win_prob", "pred_clean_entry_prob", "pred_stopout_risk",
        "side_margin_raw", "time_stop_margin_raw",
        "late_window_40_120_flag",
        "bars_since_break_above_first15", "bars_since_break_below_first15",
    ]
    feature_cols = [c for c in feature_cols if c in all_df.columns]
    X = all_df[feature_cols].fillna(0).values

    print(f"OOS trades: {len(all_df)} | features: {len(feature_cols)}")
    print(f"  oracle better in {y.mean()*100:.1f}%")
    print(f"  always-oracle baseline:    PF {pf(oracle_pnl):.3f}, sum {oracle_pnl.sum():.0f}")
    print()

    # ---- 1. ROLLING TIME-SERIES CV ----
    print("=== 1. Rolling time-series CV ===")
    print("  train on windows [0..k], predict on window k+1")
    print(f"  {'fold':>4} {'train_w':>10} {'pred_w':>7} {'n_train':>8} {'n_pred':>7} {'GB_PF':>7} {'GB_sum':>8} {'always_or_PF':>13}")
    cv_pnl_gb = np.full(len(X), np.nan)
    cv_pnl_lr = np.full(len(X), np.nan)
    fold_results = []
    windows = sorted(all_df["window_idx"].unique())
    for i in range(2, len(windows)):
        train_w = windows[:i]
        pred_w = windows[i]
        tr_mask = all_df["window_idx"].isin(train_w).to_numpy()
        te_mask = (all_df["window_idx"] == pred_w).to_numpy()
        if te_mask.sum() < 5 or tr_mask.sum() < 30:
            continue
        gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
        gb.fit(X[tr_mask], y[tr_mask])
        prob_gb = gb.predict_proba(X[te_mask])[:, 1]
        # Threshold 0.5
        use_oracle = prob_gb > 0.5
        pnl_gb = np.where(use_oracle, oracle_pnl[te_mask], no_oracle_pnl[te_mask])
        cv_pnl_gb[te_mask] = pnl_gb

        sc = StandardScaler().fit(X[tr_mask])
        X_tr_s, X_te_s = sc.transform(X[tr_mask]), sc.transform(X[te_mask])
        lr = LogisticRegression(max_iter=5000, C=0.5)
        lr.fit(X_tr_s, y[tr_mask])
        prob_lr = lr.predict_proba(X_te_s)[:, 1]
        use_oracle_lr = prob_lr > 0.5
        pnl_lr = np.where(use_oracle_lr, oracle_pnl[te_mask], no_oracle_pnl[te_mask])
        cv_pnl_lr[te_mask] = pnl_lr

        fold_results.append({
            "fold": int(i), "train_windows": list(map(int, train_w)),
            "pred_window": int(pred_w),
            "n_train": int(tr_mask.sum()), "n_pred": int(te_mask.sum()),
            "gb_pf_rule": float(pf(pnl_gb)), "gb_sum_rule": float(pnl_gb.sum()),
            "lr_pf_rule": float(pf(pnl_lr)), "lr_sum_rule": float(pnl_lr.sum()),
            "always_oracle_pf": float(pf(oracle_pnl[te_mask])),
            "always_oracle_sum": float(oracle_pnl[te_mask].sum()),
        })
        print(f"  {i:>4} {str(train_w[0])+'..'+str(train_w[-1]):>10} {pred_w:>7} {tr_mask.sum():>8} {te_mask.sum():>7} {pf(pnl_gb):>7.3f} {pnl_gb.sum():>8.0f} {pf(oracle_pnl[te_mask]):>13.3f}")

    valid = ~np.isnan(cv_pnl_gb)
    aggregated_oracle = oracle_pnl[valid]
    aggregated_gb = cv_pnl_gb[valid]
    aggregated_lr = cv_pnl_lr[valid]
    print()
    print(f"  Aggregated across rolling-TS CV ({valid.sum()} trades):")
    print(f"    Always oracle:  PF {pf(aggregated_oracle):.3f}, sum {aggregated_oracle.sum():.0f}")
    print(f"    GB rule:        PF {pf(aggregated_gb):.3f}, sum {aggregated_gb.sum():.0f}  (delta {aggregated_gb.sum()-aggregated_oracle.sum():+.0f})")
    print(f"    LR rule:        PF {pf(aggregated_lr):.3f}, sum {aggregated_lr.sum():.0f}  (delta {aggregated_lr.sum()-aggregated_oracle.sum():+.0f})")
    print()

    # ---- 2. Feature importance ----
    print("=== 2. Feature importance (GB on full data) ===")
    gb_full = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
    gb_full.fit(X, y)
    imp = sorted(zip(feature_cols, gb_full.feature_importances_), key=lambda r: r[1], reverse=True)
    for f, w in imp:
        print(f"  {f:35s}: {w:.4f}")
    print()

    # ---- 3. Reduced feature set ----
    print("=== 3. Reduced feature set: top 5 by importance ===")
    top5 = [f for f, _ in imp[:5]]
    print(f"  features: {top5}")
    Xr = all_df[top5].fillna(0).values
    cv_pnl_r = np.full(len(Xr), np.nan)
    for i in range(2, len(windows)):
        train_w = windows[:i]; pred_w = windows[i]
        tr_mask = all_df["window_idx"].isin(train_w).to_numpy()
        te_mask = (all_df["window_idx"] == pred_w).to_numpy()
        if te_mask.sum() < 5 or tr_mask.sum() < 30:
            continue
        gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
        gb.fit(Xr[tr_mask], y[tr_mask])
        prob = gb.predict_proba(Xr[te_mask])[:, 1]
        use_oracle = prob > 0.5
        cv_pnl_r[te_mask] = np.where(use_oracle, oracle_pnl[te_mask], no_oracle_pnl[te_mask])
    valid = ~np.isnan(cv_pnl_r)
    print(f"  rolling-TS CV with 5 features: PF {pf(cv_pnl_r[valid]):.3f}, sum {cv_pnl_r[valid].sum():.0f}")
    print()

    # ---- 4. Top 3 features ----
    print("=== 4. Top 3 features ===")
    top3 = [f for f, _ in imp[:3]]
    print(f"  features: {top3}")
    Xr = all_df[top3].fillna(0).values
    cv_pnl_r = np.full(len(Xr), np.nan)
    for i in range(2, len(windows)):
        train_w = windows[:i]; pred_w = windows[i]
        tr_mask = all_df["window_idx"].isin(train_w).to_numpy()
        te_mask = (all_df["window_idx"] == pred_w).to_numpy()
        if te_mask.sum() < 5 or tr_mask.sum() < 30: continue
        gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
        gb.fit(Xr[tr_mask], y[tr_mask])
        prob = gb.predict_proba(Xr[te_mask])[:, 1]
        cv_pnl_r[te_mask] = np.where(prob > 0.5, oracle_pnl[te_mask], no_oracle_pnl[te_mask])
    valid = ~np.isnan(cv_pnl_r)
    print(f"  rolling-TS CV with 3 features: PF {pf(cv_pnl_r[valid]):.3f}, sum {cv_pnl_r[valid].sum():.0f}")
    print()

    # ---- 5. Stability check: different random seeds for GB ----
    print("=== 5. Stability: GB with different random_state seeds ===")
    pfs = []
    for rs in [0, 1, 7, 13, 42, 100]:
        cv_pnl_s = np.full(len(X), np.nan)
        for i in range(2, len(windows)):
            train_w = windows[:i]; pred_w = windows[i]
            tr_mask = all_df["window_idx"].isin(train_w).to_numpy()
            te_mask = (all_df["window_idx"] == pred_w).to_numpy()
            if te_mask.sum() < 5 or tr_mask.sum() < 30: continue
            gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=rs)
            gb.fit(X[tr_mask], y[tr_mask])
            prob = gb.predict_proba(X[te_mask])[:, 1]
            cv_pnl_s[te_mask] = np.where(prob > 0.5, oracle_pnl[te_mask], no_oracle_pnl[te_mask])
        valid = ~np.isnan(cv_pnl_s)
        pf_s = pf(cv_pnl_s[valid])
        pfs.append(pf_s)
        print(f"  seed {rs:>3}: PF {pf_s:.3f}")
    print(f"  mean PF across seeds: {np.mean(pfs):.3f} ± {np.std(pfs):.3f}")
    print()

    out = {
        "n_oos": int(len(all_df)),
        "n_features": len(feature_cols),
        "always_oracle_pf": float(pf(oracle_pnl)),
        "rolling_ts_cv_gb_pf": float(pf(aggregated_gb)),
        "rolling_ts_cv_lr_pf": float(pf(aggregated_lr)),
        "rolling_ts_cv_gb_sum": float(aggregated_gb.sum()),
        "rolling_ts_cv_gb_lift_pct": float((aggregated_gb.sum() - aggregated_oracle.sum()) / aggregated_oracle.sum() * 100),
        "feature_importance_top10": [{"feature": f, "importance": float(w)} for f, w in imp[:10]],
        "fold_results": fold_results,
        "stability_pfs": [float(x) for x in pfs],
        "stability_mean": float(np.mean(pfs)),
        "stability_std": float(np.std(pfs)),
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/oracle_gate_iter3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote v3/artifacts/research/oracle_gate_iter3.json")


if __name__ == "__main__":
    main()
