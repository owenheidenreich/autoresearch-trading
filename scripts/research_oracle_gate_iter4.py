"""Iteration 4: train GB classifier on full OOS, evaluate on forward-walk
hold-out (the strongest available test).

Iteration 3 showed rolling-TS-CV PF 1.870 vs baseline 1.831 (+2%). But
the forward-walk window (post-2026-02-24) is the cleanest hold-out we
have — none of those days were in any training fold. Test:

  1. Train GB on all 1664 OOS trades.
  2. Apply to forward-walk's 53 trades.
  3. Compare to always-oracle and forward-walk-derived Rule B.

Also: try thresholds beyond 0.5; some classes (oracle better) are 65%
of OOS, so threshold tuning may help.

Output: v3/artifacts/research/oracle_gate_iter4.json
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

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

    # OOS training set
    oos_rows = []
    for s in seeds:
        df = pd.read_pickle(
            f"v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{s}/seed_{s}/chosen_trades.pkl"
        )
        df = df.copy()
        df["seed"] = s
        finite = np.isfinite(df["chosen_objective_pnl"]) & np.isfinite(df["chosen_time_stop_pnl"])
        df = df[finite].reset_index(drop=True)
        oos_rows.append(df)
    oos_df = pd.concat(oos_rows, ignore_index=True)

    # Forward-walk hold-out (post-2026-02-24)
    fw_rows = []
    for s in seeds:
        df = pd.read_pickle(f"v3/artifacts/forward_walk/forward_walk_chosen_seed{s}.pkl")
        if df.empty:
            continue
        df = df.copy()
        df["seed"] = s
        finite = np.isfinite(df["fwd_pnl_hybrid_with_oracle"]) & np.isfinite(df["fwd_pnl_time_stop"])
        df = df[finite].reset_index(drop=True)
        fw_rows.append(df)
    fw_df = pd.concat(fw_rows, ignore_index=True)

    # The forward-walk DF has both `chosen_objective_pnl` (NaN-padded oracle from earlier run)
    # and `fwd_pnl_hybrid_with_oracle` (the freshly recomputed oracle pnl). Use the latter.
    if "chosen_objective_pnl" in fw_df.columns:
        fw_df = fw_df.drop(columns=["chosen_objective_pnl"])
    fw_df["chosen_objective_pnl"] = fw_df["fwd_pnl_hybrid_with_oracle"].astype(float)
    if "chosen_time_stop_pnl" not in fw_df.columns:
        fw_df["chosen_time_stop_pnl"] = fw_df["fwd_pnl_time_stop"].astype(float)

    feature_cols = [
        "sigma_pos", "iv_percentile", "vix", "atm_iv", "vwap_slope",
        "volume_ratio", "first15_range_pct", "decision_margin",
        "pred_win_prob", "pred_clean_entry_prob", "pred_stopout_risk",
        "side_margin_raw", "time_stop_margin_raw",
        "late_window_40_120_flag",
        "bars_since_break_above_first15", "bars_since_break_below_first15",
    ]
    feature_cols = [c for c in feature_cols if c in oos_df.columns and c in fw_df.columns]

    print(f"OOS train: {len(oos_df)} trades, features {len(feature_cols)}")
    print(f"Forward-walk hold-out: {len(fw_df)} trades")
    print()

    X_oos = oos_df[feature_cols].fillna(0).values
    y_oos = (oos_df["chosen_objective_pnl"] > oos_df["chosen_time_stop_pnl"]).astype(int).values
    X_fw = fw_df[feature_cols].fillna(0).values
    oracle_fw = fw_df["chosen_objective_pnl"].astype(float).values
    no_oracle_fw = fw_df["chosen_time_stop_pnl"].astype(float).values

    # Forward-walk baselines
    print("=== Forward-walk hold-out baselines ===")
    print(f"  Always oracle:    PF {pf(oracle_fw):.3f}, sum {oracle_fw.sum():.0f}")
    print(f"  Always no-oracle: PF {pf(no_oracle_fw):.3f}, sum {no_oracle_fw.sum():.0f}")
    print(f"  Perfect ceiling:  PF {pf(np.maximum(oracle_fw, no_oracle_fw)):.3f}, sum {np.maximum(oracle_fw, no_oracle_fw).sum():.0f}")
    print()

    # Train multiple GB seeds, ensemble predict
    print("=== Train GB on full OOS, predict on forward walk ===")
    probs = []
    for rs in [0, 1, 7, 13, 42, 100]:
        gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=rs)
        gb.fit(X_oos, y_oos)
        prob = gb.predict_proba(X_fw)[:, 1]
        probs.append(prob)
    prob_ens = np.mean(probs, axis=0)
    print(f"  Ensemble of 6 GB seeds, threshold sweep on forward walk:")
    for thr in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        use_oracle = prob_ens > thr
        pnl = np.where(use_oracle, oracle_fw, no_oracle_fw)
        print(f"    thr={thr:.2f}: keep_oracle={use_oracle.sum():>3}/{len(prob_ens)}  PF {pf(pnl):.3f}  sum {pnl.sum():>8.0f}  delta_vs_baseline {pnl.sum()-oracle_fw.sum():+.0f}")
    print()

    # Per-trade detail: where does the classifier disagree with always-oracle?
    print("=== Per-trade classifier output on forward walk ===")
    print(f"  {'seed':>5} {'day':>12} {'side':>5} {'P(oracle_better)':>17} {'oracle_pnl':>11} {'no_or_pnl':>10} {'rule_choice':>13}")
    for i in range(len(fw_df)):
        seed = fw_df.iloc[i]["seed"]
        day = fw_df.iloc[i]["day"]
        side = fw_df.iloc[i]["chosen_side"]
        p = prob_ens[i]
        op = oracle_fw[i]
        np_ = no_oracle_fw[i]
        choice = "oracle" if p > 0.5 else "no_oracle"
        flag = " ✓" if (op > np_) == (p > 0.5) else " ✗"
        print(f"  {seed:>5} {str(day):>12} {side:>5} {p:>17.3f} {op:>11.0f} {np_:>10.0f} {choice:>13}{flag}")
    print()

    # Class accuracy on forward walk
    truth = (oracle_fw > no_oracle_fw).astype(int)
    acc = ((prob_ens > 0.5) == truth).mean()
    print(f"Forward-walk classification accuracy at threshold 0.5: {acc:.3f}")
    print(f"  (truth: {truth.sum()}/{len(truth)} = {truth.mean()*100:.0f}% oracle better)")
    print(f"  (predicted: {(prob_ens>0.5).sum()}/{len(prob_ens)} = {(prob_ens>0.5).mean()*100:.0f}% predicted oracle better)")
    print()

    # Final report
    print("=== Final assessment ===")
    print("OOS (1664 trades): rolling-TS-CV PF 1.870 vs baseline 1.831 (+2%)")
    best_thr = None; best_pf = pf(oracle_fw)
    for thr in np.linspace(0.30, 0.70, 9):
        use_oracle = prob_ens > thr
        pnl = np.where(use_oracle, oracle_fw, no_oracle_fw)
        if pf(pnl) > best_pf:
            best_pf = pf(pnl); best_thr = float(thr)
    if best_thr:
        print(f"Forward walk (53 trades): best threshold {best_thr:.2f} → PF {best_pf:.3f} vs baseline {pf(oracle_fw):.3f}")
    else:
        print(f"Forward walk (53 trades): no threshold beats baseline ({pf(oracle_fw):.3f})")

    out = {
        "n_oos": int(len(oos_df)),
        "n_fw": int(len(fw_df)),
        "fw_baseline_oracle_pf": float(pf(oracle_fw)),
        "fw_perfect_ceiling_pf": float(pf(np.maximum(oracle_fw, no_oracle_fw))),
        "fw_classifier_accuracy_05": float(acc),
        "fw_threshold_sweep": [
            {
                "thr": float(t),
                "pf": float(pf(np.where(prob_ens > t, oracle_fw, no_oracle_fw))),
                "sum": float(np.where(prob_ens > t, oracle_fw, no_oracle_fw).sum()),
                "keep_oracle": int((prob_ens > t).sum()),
            }
            for t in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        ],
    }
    os.makedirs("v3/artifacts/research", exist_ok=True)
    with open("v3/artifacts/research/oracle_gate_iter4.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote v3/artifacts/research/oracle_gate_iter4.json")


if __name__ == "__main__":
    main()
