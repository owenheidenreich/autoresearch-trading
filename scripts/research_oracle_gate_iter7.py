"""Iteration 7: regression-based exit gate.

Iter 4 trained a binary classifier (oracle better? yes/no) and gated by P>0.45.
This iteration trains a regression that predicts (oracle_pnl - no_oracle_pnl),
the EXPECTED $ LIFT from using oracle. Deploy decision: if predicted lift > 0,
use oracle; else use no_oracle.

Hypothesis: regression has more information than classification (uses
magnitude, not just sign). Should yield finer per-trade decisions, especially
distinguishing "marginal oracle wins" from "big oracle wins".

Validation: train on full 1664 OOS, predict on 53 FW hold-out. Compare to
the iter-4 binary-classifier baseline.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


def pf(p):
    p = np.asarray(p, dtype=float); p = p[np.isfinite(p)]
    pos = p[p > 0].sum(); neg = p[p < 0].sum()
    return pos / abs(neg) if neg < 0 else float("inf") if pos > 0 else 0.0


def main():
    seeds = [42, 43, 44, 45, 46]

    oos_rows = []
    for s in seeds:
        df = pd.read_pickle(f"v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{s}/seed_{s}/chosen_trades.pkl")
        df = df.copy(); df["seed"] = s
        df = df[np.isfinite(df["chosen_objective_pnl"]) & np.isfinite(df["chosen_time_stop_pnl"])].reset_index(drop=True)
        oos_rows.append(df)
    oos_df = pd.concat(oos_rows, ignore_index=True)

    fw_rows = []
    for s in seeds:
        df = pd.read_pickle(f"v3/artifacts/forward_walk/forward_walk_chosen_seed{s}.pkl")
        if df.empty: continue
        df = df.copy(); df["seed"] = s
        df = df[np.isfinite(df["fwd_pnl_hybrid_with_oracle"]) & np.isfinite(df["fwd_pnl_time_stop"])].reset_index(drop=True)
        fw_rows.append(df)
    fw_df = pd.concat(fw_rows, ignore_index=True)

    feature_cols = [
        "sigma_pos","iv_percentile","vix","atm_iv","vwap_slope","volume_ratio",
        "first15_range_pct","decision_margin","pred_win_prob","pred_clean_entry_prob",
        "pred_stopout_risk","side_margin_raw","time_stop_margin_raw",
        "late_window_40_120_flag","bars_since_break_above_first15","bars_since_break_below_first15",
    ]
    feature_cols = [c for c in feature_cols if c in oos_df.columns and c in fw_df.columns]

    X_oos = oos_df[feature_cols].fillna(0).values
    y_lift = (oos_df["chosen_objective_pnl"] - oos_df["chosen_time_stop_pnl"]).astype(float).values
    y_binary = (y_lift > 0).astype(int)
    X_fw = fw_df[feature_cols].fillna(0).values
    of = fw_df["fwd_pnl_hybrid_with_oracle"].astype(float).values
    nf = fw_df["fwd_pnl_time_stop"].astype(float).values

    print(f"OOS train: {len(oos_df)} trades; lift target stats: mean ${y_lift.mean():.0f}, std ${y_lift.std():.0f}, median ${np.median(y_lift):.0f}")
    print(f"FW hold-out: {len(fw_df)} trades")
    print(f"Always oracle baseline: PF {pf(of):.3f}, sum ${of.sum():.0f}")
    print()

    # Train regressor
    print("=== GB regression: predicts expected lift ($) ===")
    pred_lift_seeds = []
    for rs in [0, 1, 7, 13, 42, 100]:
        reg = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=rs, loss="huber")
        reg.fit(X_oos, y_lift)
        pred_lift_seeds.append(reg.predict(X_fw))
    pred_lift_ens = np.mean(pred_lift_seeds, axis=0)
    print(f"  predicted lift on FW: mean ${pred_lift_ens.mean():.0f}, range [${pred_lift_ens.min():.0f}, ${pred_lift_ens.max():.0f}]")
    print()

    # Apply: if predicted lift > 0 → use oracle, else use no_oracle
    print("Threshold sweep on predicted lift (oracle if pred_lift > T):")
    for T in [-500, -200, -100, -50, 0, 50, 100, 200]:
        use_oracle = pred_lift_ens > T
        pnl = np.where(use_oracle, of, nf)
        print(f"  pred_lift > ${T:>+5}: keep_oracle={use_oracle.sum()}/{len(of)}  PF {pf(pnl):.3f}  sum ${pnl.sum():.0f}")
    print()

    # Compare to iter 4 (classifier)
    print("=== Comparison: classifier (iter 4) vs regression (iter 7) ===")
    cls_seeds = []
    for rs in [0, 1, 7, 13, 42, 100]:
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=rs)
        clf.fit(X_oos, y_binary)
        cls_seeds.append(clf.predict_proba(X_fw)[:, 1])
    cls_prob = np.mean(cls_seeds, axis=0)
    # Best PF for each
    best_cls_pf = -np.inf
    for thr in np.linspace(0.30, 0.70, 21):
        use_oracle = cls_prob > thr
        pnl = np.where(use_oracle, of, nf)
        if pf(pnl) > best_cls_pf: best_cls_pf = pf(pnl); best_cls_thr = float(thr)
    best_reg_pf = -np.inf
    for T in np.linspace(-500, 500, 21):
        use_oracle = pred_lift_ens > T
        pnl = np.where(use_oracle, of, nf)
        if pf(pnl) > best_reg_pf: best_reg_pf = pf(pnl); best_reg_T = float(T)
    print(f"  classifier best:  thr={best_cls_thr:.2f}  PF {best_cls_pf:.3f}")
    print(f"  regression best:  T=${best_reg_T:+.0f}  PF {best_reg_pf:.3f}")
    print()

    # Try BLEND: rule = "use oracle iff classifier says yes AND regression > 0"
    print("=== Hybrid: classifier AND regression must both agree ===")
    for cls_thr in [0.40, 0.50]:
        for reg_T in [-100, 0, 100]:
            use_oracle = (cls_prob > cls_thr) & (pred_lift_ens > reg_T)
            pnl = np.where(use_oracle, of, nf)
            print(f"  cls>{cls_thr} AND reg>${reg_T:+.0f}: keep_oracle={use_oracle.sum()}, PF {pf(pnl):.3f}, sum ${pnl.sum():.0f}")
    print()

    # OR: use classifier OR regression (more permissive)
    print("=== Hybrid: classifier OR regression says use oracle ===")
    for cls_thr in [0.40, 0.50]:
        for reg_T in [0, 100, 200]:
            use_oracle = (cls_prob > cls_thr) | (pred_lift_ens > reg_T)
            pnl = np.where(use_oracle, of, nf)
            print(f"  cls>{cls_thr} OR reg>${reg_T:+.0f}: keep_oracle={use_oracle.sum()}, PF {pf(pnl):.3f}, sum ${pnl.sum():.0f}")
    print()

    # Per-trade detail: where regression gives different answer than classifier?
    print("=== Disagreement: cls says oracle but reg says no_oracle (or vice versa) ===")
    cls_oracle = cls_prob > 0.45
    reg_oracle = pred_lift_ens > 0
    disagree = cls_oracle != reg_oracle
    if disagree.sum():
        print(f"  {disagree.sum()} trades disagree:")
        for i in np.where(disagree)[0][:10]:
            row = fw_df.iloc[i]
            print(f"    seed {row['seed']} {row['day']} {row['chosen_side']:>4}: cls_p={cls_prob[i]:.2f}, reg_lift=${pred_lift_ens[i]:>+5.0f}, oracle_pnl=${of[i]:>+6.0f}, no_or=${nf[i]:>+6.0f}, true_better={'oracle' if of[i]>nf[i] else 'no_oracle'}")
    else:
        print("  no disagreement at these thresholds")


if __name__ == "__main__":
    main()
