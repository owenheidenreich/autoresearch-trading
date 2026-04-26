"""Iteration 2: deeper exit-decision research on the full OOS set.

Iteration 1 falsified Rule B (sigma_pos < -1.0 OR iv_percentile > 0.80) — it
was overfit to the 53-trade forward-walk regime. On 1664 OOS trades,
always-oracle (PF 1.877) beats every threshold variant.

But:
  - LR on top features hit PF 1.909 at threshold 0.5 — small but real.
  - time_stop_margin_raw (the model's predicted time-stop pnl margin)
    has the strongest single-feature Cohen's d = 0.273.
  - Per-window analysis suggests oracle's value varies dramatically
    (W5 oracle PF 4.62, W12 oracle PF 0.68).

This iteration:
  1. Examines the time_stop_margin_raw signal in detail (continuous gate).
  2. Tries an ENSEMBLE of two classifiers (LR + GB) on a richer feature
     set, using nested 5-fold CV to get an unbiased estimate.
  3. Tests per-window adaptive thresholding (learn one threshold per
     window) vs one global threshold.
  4. Tests soft-blend exit: instead of binary oracle-or-no_oracle, use
     a probability-weighted blend.

Output: v3/artifacts/research/oracle_gate_iter2.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


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

    print(f"OOS trades: {len(all_df)}, oracle better in {y.mean()*100:.1f}%")
    print(f"Always oracle:    PF {pf(oracle_pnl):.3f}, sum {oracle_pnl.sum():.0f}")
    print(f"Always no-oracle: PF {pf(no_oracle_pnl):.3f}, sum {no_oracle_pnl.sum():.0f}")
    perfect = np.maximum(oracle_pnl, no_oracle_pnl)
    print(f"Perfect ceiling:  PF {pf(perfect):.3f}, sum {perfect.sum():.0f}")
    print()

    # ---- 1. time_stop_margin_raw as continuous gate ----
    print("=== 1. time_stop_margin_raw continuous gate ===")
    print("    'low time_stop_margin → likely loser → oracle helps' hypothesis")
    margin = all_df["time_stop_margin_raw"].values
    quantiles = np.percentile(margin, [10, 25, 40, 50, 60, 75, 90])
    print(f"    margin quantiles 10/25/40/50/60/75/90: {quantiles}")
    print()
    for q in [10, 25, 40, 50, 60]:
        cutoff = np.percentile(margin, q)
        # If margin < cutoff (predicted weak trade) → keep oracle. Else → use no_oracle (let strong trades run)
        keep_oracle = margin < cutoff
        pnl = np.where(keep_oracle, oracle_pnl, no_oracle_pnl)
        print(f"    margin < {q}th pctile ({cutoff:>+8.2f}) → oracle, else no_oracle: keep_oracle={keep_oracle.sum()}, PF {pf(pnl):.3f}")
    print()
    for q in [40, 50, 60, 75, 90]:
        cutoff = np.percentile(margin, q)
        keep_oracle = margin > cutoff  # opposite framing
        pnl = np.where(keep_oracle, oracle_pnl, no_oracle_pnl)
        print(f"    margin > {q}th pctile ({cutoff:>+8.2f}) → oracle, else no_oracle: keep_oracle={keep_oracle.sum()}, PF {pf(pnl):.3f}")
    print()

    # ---- 2. Richer feature set, ensemble of LR + GB, nested CV ----
    print("=== 2. Ensemble LR+GB with nested 5-fold CV ===")
    feature_cols = [
        "sigma_pos", "iv_percentile", "vix", "atm_iv", "vwap_slope",
        "volume_ratio", "first15_range_pct", "decision_margin",
        "pred_win_prob", "pred_clean_entry_prob", "pred_stopout_risk",
        "side_margin_raw", "time_stop_margin_raw",
        "late_window_40_120_flag", "atm_iv", "iv_percentile",
        "bars_since_break_above_first15", "bars_since_break_below_first15",
    ]
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in all_df.columns]
    print(f"  using {len(feature_cols)} features: {feature_cols}")
    X = all_df[feature_cols].fillna(0).values

    cv_pred_lr = np.zeros(len(X))
    cv_pred_gb = np.zeros(len(X))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        lr = LogisticRegression(max_iter=2000, C=0.5)
        lr.fit(X[tr], y[tr])
        cv_pred_lr[te] = lr.predict_proba(X[te])[:, 1]
        gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
        gb.fit(X[tr], y[tr])
        cv_pred_gb[te] = gb.predict_proba(X[te])[:, 1]
    cv_pred_ens = 0.5 * cv_pred_lr + 0.5 * cv_pred_gb
    for label, pred in [("LR", cv_pred_lr), ("GB", cv_pred_gb), ("LR+GB ens", cv_pred_ens)]:
        print(f"  {label} (cross-validated):")
        for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
            use_oracle = pred > thr
            pnl = np.where(use_oracle, oracle_pnl, no_oracle_pnl)
            acc = ((pred > 0.5) == y).mean()
            print(f"    P>{thr}: keep_oracle={use_oracle.sum():>4}/{len(X)}  PF={pf(pnl):.3f}  sum={pnl.sum():>10.0f}  acc(@0.5)={acc:.3f}")
    print()

    # ---- 3. Soft-blend by probability ----
    print("=== 3. Soft blend: pnl = p*oracle + (1-p)*no_oracle ===")
    print("  (in production this would mean: weight your hold time toward oracle vs time-stop)")
    for label, pred in [("LR", cv_pred_lr), ("GB", cv_pred_gb), ("LR+GB ens", cv_pred_ens)]:
        soft = pred * oracle_pnl + (1 - pred) * no_oracle_pnl
        print(f"  {label}: soft-blend PF {pf(soft):.3f}, sum {soft.sum():.0f}")
    print()

    # ---- 4. Per-window adaptive threshold (does each window benefit from a different cutoff?) ----
    print("=== 4. Per-window adaptive threshold ===")
    print("  for each window, find best in-sample threshold of LR_pred")
    cv_pnl_per_window = np.zeros(len(X))
    rows_per_window = []
    for w, sub_idx in all_df.groupby("window_idx").indices.items():
        sub_idx = np.array(sub_idx)
        if len(sub_idx) < 10:
            cv_pnl_per_window[sub_idx] = oracle_pnl[sub_idx]  # default to oracle
            continue
        oracle_w = oracle_pnl[sub_idx]
        no_or_w = no_oracle_pnl[sub_idx]
        pred_w = cv_pred_ens[sub_idx]
        best_pf = pf(oracle_w); best_thr = 1.1  # i.e., never override
        for thr in np.linspace(0.30, 0.70, 21):
            use_oracle = pred_w > thr
            pnl_w = np.where(use_oracle, oracle_w, no_or_w)
            if pf(pnl_w) > best_pf:
                best_pf = pf(pnl_w); best_thr = float(thr)
        # apply
        use_oracle = pred_w > best_thr
        cv_pnl_per_window[sub_idx] = np.where(use_oracle, oracle_w, no_or_w)
        rows_per_window.append({"window": int(w), "n": len(sub_idx),
                                "best_thr": best_thr, "pf_after": float(best_pf),
                                "pf_always_oracle": float(pf(oracle_w))})
    print(f"  per-window adaptive (in-sample): total PF {pf(cv_pnl_per_window):.3f}, sum {cv_pnl_per_window.sum():.0f}")
    print(f"{'win':>4} {'n':>5} {'thr':>6} {'pf_adaptive':>12} {'pf_oracle':>10}")
    for r in rows_per_window:
        print(f"  {r['window']:>4} {r['n']:>5} {r['best_thr']:>6.2f} {r['pf_after']:>12.3f} {r['pf_always_oracle']:>10.3f}")
    print()

    out = {
        "n_oos": int(len(all_df)),
        "always_oracle_pf": pf(oracle_pnl),
        "perfect_ceiling_pf": pf(perfect),
        "lr_cv_pf_at_05": float(pf(np.where(cv_pred_lr > 0.5, oracle_pnl, no_oracle_pnl))),
        "gb_cv_pf_at_05": float(pf(np.where(cv_pred_gb > 0.5, oracle_pnl, no_oracle_pnl))),
        "ens_cv_pf_at_05": float(pf(np.where(cv_pred_ens > 0.5, oracle_pnl, no_oracle_pnl))),
        "soft_blend_ens_pf": float(pf(cv_pred_ens * oracle_pnl + (1 - cv_pred_ens) * no_oracle_pnl)),
        "per_window_adaptive_pf": float(pf(cv_pnl_per_window)),
        "per_window_rows": rows_per_window,
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/oracle_gate_iter2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote v3/artifacts/research/oracle_gate_iter2.json")


if __name__ == "__main__":
    main()
