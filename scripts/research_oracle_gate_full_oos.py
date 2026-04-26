"""Validate the Rule B oracle-gate on the FULL 13-window OOS dataset.

The 53-trade forward-walk validation was strong but small. The rolling-
window OOS evaluation has ~390 trades per seed × 5 seeds = ~1950 trades
across 13 disjoint regime windows. This script:

  1. Pulls each seed's OOS chosen_trades.
  2. For each trade, classifies as oracle_better / no_oracle_better.
  3. Audits which features discriminate, by window and aggregate.
  4. Tests Rule B (sigma_pos < -1.0 OR iv_percentile > 0.80) PF lift on
     this much-bigger sample.
  5. Sweeps thresholds to find the best (T1, T2) on the OOS set, then
     reports out-of-fold validation.

Two PnL columns to compare:
  - chosen_objective_pnl    = oracle-augmented hybrid_live exit
  - chosen_time_stop_pnl    = naive time-stop exit (no oracle)

Output:
  - Console report by window + aggregate
  - JSON to v3/artifacts/research/oracle_gate_oos_audit.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def pf(p: np.ndarray) -> float:
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
        finite = np.isfinite(df["chosen_objective_pnl"]) & np.isfinite(
            df["chosen_time_stop_pnl"]
        )
        df = df[finite].reset_index(drop=True)
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    all_df["oracle_pnl"] = all_df["chosen_objective_pnl"].astype(float)
    all_df["no_oracle_pnl"] = all_df["chosen_time_stop_pnl"].astype(float)
    all_df["oracle_better"] = (all_df["oracle_pnl"] > all_df["no_oracle_pnl"]).astype(int)

    print(f"Total OOS trades (5 seeds × 13 windows, finite): {len(all_df)}")
    print(f"Oracle better count: {all_df['oracle_better'].sum()} of {len(all_df)} ({100*all_df['oracle_better'].mean():.1f}%)")
    print()

    # Aggregate baselines
    oracle_pnl = all_df["oracle_pnl"].values
    no_oracle_pnl = all_df["no_oracle_pnl"].values
    perfect = np.maximum(oracle_pnl, no_oracle_pnl)
    print("=== Aggregate baselines (whole OOS) ===")
    print(f"  Always oracle:      sum={oracle_pnl.sum():>10.0f}  PF={pf(oracle_pnl):.3f}")
    print(f"  Always no-oracle:   sum={no_oracle_pnl.sum():>10.0f}  PF={pf(no_oracle_pnl):.3f}")
    print(f"  Perfect selector:   sum={perfect.sum():>10.0f}  PF={pf(perfect):.3f}")
    print()

    # Per-window oracle vs no_oracle
    print("=== Per-window oracle vs no_oracle ===")
    print(f"{'win':>4} {'n':>5} {'oracle_pf':>10} {'no_oracle_pf':>13} {'%oracle_btr':>12} {'PnL_oracle':>11} {'PnL_no_or':>11}")
    per_window = []
    for w, sub in all_df.groupby("window_idx"):
        op = sub["oracle_pnl"].values
        np_ = sub["no_oracle_pnl"].values
        ob = (op > np_).mean()
        per_window.append({
            "window": int(w), "n": int(len(sub)),
            "pf_oracle": pf(op), "pf_no_oracle": pf(np_),
            "pct_oracle_better": float(ob),
            "sum_oracle": float(op.sum()), "sum_no_oracle": float(np_.sum()),
        })
        print(f"{int(w):>4} {len(sub):>5} {pf(op):>10.3f} {pf(np_):>13.3f} {ob:>12.2%} {op.sum():>11.0f} {np_.sum():>11.0f}")
    print()

    # Test Rule B (forward-walk-derived) on full OOS
    print("=== Rule B (sigma_pos < T1 OR iv_percentile > T2) on full OOS ===")
    sigma = all_df["sigma_pos"].values
    iv = all_df["iv_percentile"].values
    # Forward-walk-derived thresholds
    print("Forward-walk thresholds (-1.0, 0.80) applied to OOS:")
    let_run = (sigma < -1.0) | (iv > 0.80)
    pnl = np.where(let_run, no_oracle_pnl, oracle_pnl)
    print(f"  let_run={let_run.sum()} of {len(all_df)} ({100*let_run.mean():.1f}%)")
    print(f"  Rule B PF: {pf(pnl):.3f} (vs always-oracle {pf(oracle_pnl):.3f})")
    print(f"  sum: {pnl.sum():.0f} (vs {oracle_pnl.sum():.0f}, delta {pnl.sum()-oracle_pnl.sum():+.0f})")
    print()

    # Threshold grid sweep on full OOS
    print("=== Sweep on full OOS ===")
    best_t1, best_t2, best_pf = None, None, -np.inf
    grid_results = []
    T1_grid = np.linspace(-2.5, 1.5, 21)
    T2_grid = np.linspace(0.50, 0.95, 19)
    for T1 in T1_grid:
        for T2 in T2_grid:
            let_run = (sigma < T1) | (iv > T2)
            pnl = np.where(let_run, no_oracle_pnl, oracle_pnl)
            this_pf = pf(pnl)
            grid_results.append({"T1": float(T1), "T2": float(T2), "pf": float(this_pf), "let_run": int(let_run.sum())})
            if this_pf > best_pf:
                best_pf = this_pf
                best_t1, best_t2 = float(T1), float(T2)
    print(f"  best in-sample (T1, T2) = ({best_t1:+.2f}, {best_t2:.2f}) → PF {best_pf:.3f}")
    print()

    # 5-fold CV on the threshold rule
    print("=== 5-fold CV: thresholds picked on 4 folds, applied to held-out fold ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_pnls = []
    for fold, (tr, te) in enumerate(kf.split(all_df)):
        # Find best threshold on training
        sigma_tr = sigma[tr]; iv_tr = iv[tr]
        oracle_tr = oracle_pnl[tr]; no_or_tr = no_oracle_pnl[tr]
        best_pf_tr = -np.inf; best_T1_tr = -1.0; best_T2_tr = 0.80
        for T1 in T1_grid:
            for T2 in T2_grid:
                let_run = (sigma_tr < T1) | (iv_tr > T2)
                pnl_tr = np.where(let_run, no_or_tr, oracle_tr)
                if pf(pnl_tr) > best_pf_tr:
                    best_pf_tr = pf(pnl_tr)
                    best_T1_tr, best_T2_tr = float(T1), float(T2)
        let_run_te = (sigma[te] < best_T1_tr) | (iv[te] > best_T2_tr)
        pnl_te = np.where(let_run_te, no_oracle_pnl[te], oracle_pnl[te])
        cv_pnls.extend(pnl_te.tolist())
        print(f"  fold {fold}: train-best (T1, T2) = ({best_T1_tr:+.2f}, {best_T2_tr:.2f}) train_PF={best_pf_tr:.3f}, held-out PF={pf(pnl_te):.3f}")
    cv_pnls = np.array(cv_pnls)
    print(f"  CV total: PF={pf(cv_pnls):.3f}  sum={cv_pnls.sum():.0f}")
    print()

    # Per-feature audit (which features have discriminating power on full OOS?)
    print("=== Feature discrimination audit on full OOS ===")
    candidate_features = [
        "sigma_pos", "iv_percentile", "vix", "atm_iv", "vwap_slope",
        "volume_ratio", "first15_range_pct", "decision_margin",
        "pred_win_prob", "pred_clean_entry_prob", "pred_stopout_risk",
        "side_margin_raw", "time_stop_margin_raw",
        "late_window_40_120_flag",
    ]
    ob_grp = all_df.groupby("oracle_better")
    feature_audit = []
    for feat in candidate_features:
        if feat not in all_df.columns:
            continue
        m_t = ob_grp[feat].mean().get(1, np.nan)
        m_f = ob_grp[feat].mean().get(0, np.nan)
        s_t = ob_grp[feat].std().get(1, np.nan)
        s_f = ob_grp[feat].std().get(0, np.nan)
        diff = m_t - m_f
        # standardized diff (Cohen's d-like)
        pooled_s = float(np.sqrt(((s_t**2 + s_f**2) / 2) or 1e-9))
        cohend = float(diff / pooled_s) if pooled_s > 0 else 0.0
        feature_audit.append({"feature": feat, "mean_oracle_better": float(m_t),
                              "mean_no_oracle_better": float(m_f),
                              "diff": float(diff), "cohens_d": cohend})
        print(f"  {feat:30s}: oracle_better={m_t:>+.3f}  no_oracle_btr={m_f:>+.3f}  diff={diff:>+.3f}  d={cohend:>+.3f}")
    print()
    feature_audit.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)
    print("Top 5 by |Cohen's d|:")
    for r in feature_audit[:5]:
        print(f"  {r['feature']:30s} d={r['cohens_d']:+.3f}")
    print()

    # Train an LR classifier on top features and CV-evaluate
    top_features = [r["feature"] for r in feature_audit[:6]]
    print(f"=== LR classifier on top 6 features: {top_features} ===")
    X = all_df[top_features].fillna(0).values
    y = all_df["oracle_better"].values
    cv_pred = np.zeros(len(all_df))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        cv_pred[te] = clf.predict_proba(X[te])[:, 1]
    print(f"  CV ROC-AUC-ish (frac with prob>0.5 same as truth): {((cv_pred>0.5)==y).mean():.3f}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        use_oracle = cv_pred > thr
        pnl = np.where(use_oracle, oracle_pnl, no_oracle_pnl)
        print(f"  P(oracle_better)>{thr}: use_oracle={use_oracle.sum()}/{len(all_df)} | sum={pnl.sum():.0f}, PF={pf(pnl):.3f}")
    print()

    # Save
    out = {
        "n_oos_trades": int(len(all_df)),
        "pct_oracle_better": float(all_df["oracle_better"].mean()),
        "always_oracle_pf": pf(oracle_pnl),
        "always_no_oracle_pf": pf(no_oracle_pnl),
        "perfect_selector_pf": pf(perfect),
        "rule_b_forward_walk_thresholds": {"T1": -1.0, "T2": 0.80, "pf": float(pf(np.where((sigma < -1.0) | (iv > 0.80), no_oracle_pnl, oracle_pnl)))},
        "best_in_sample_thresholds": {"T1": best_t1, "T2": best_t2, "pf": float(best_pf)},
        "cv_total_pf": float(pf(cv_pnls)),
        "per_window": per_window,
        "feature_audit": feature_audit[:8],
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/oracle_gate_oos_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote v3/artifacts/research/oracle_gate_oos_audit.json")


if __name__ == "__main__":
    main()
