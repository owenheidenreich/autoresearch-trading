"""Iteration 8: ENTRY skip gate.

The exit gate (Iter 4) decides oracle vs no_oracle for trades that are taken.
Iter 8 asks: should we have taken the trade AT ALL?

Train a binary classifier that predicts P(trade is profitable) at entry-time
features. Filter out low-probability trades. The hypothesis is that the
existing pred_win_prob is too crude (we saw earlier it doesn't discriminate
well). A regression-trained classifier on richer features may do better.

Validation: train on full 1664 OOS, predict on 53 FW hold-out. Compare
"all trades taken" baseline to "trade only if P(profit) > T" filter.

The 'profit' label can be defined two ways:
  1. oracle_pnl > 0 (with-oracle profitable)
  2. no_oracle_pnl > 0 (time-stop profitable)
  3. either > 0 (some exit choice would have been profitable)

We'll test all 3 definitions to see which has the strongest signal.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold


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
    X_fw = fw_df[feature_cols].fillna(0).values

    of = fw_df["fwd_pnl_hybrid_with_oracle"].astype(float).values
    nf = fw_df["fwd_pnl_time_stop"].astype(float).values
    of_oos = oos_df["chosen_objective_pnl"].astype(float).values
    nf_oos = oos_df["chosen_time_stop_pnl"].astype(float).values

    print(f"OOS train: {len(oos_df)} trades, FW hold-out: {len(fw_df)}")
    print(f"FW always-oracle baseline:    PF {pf(of):.3f}, sum ${of.sum():.0f}")
    print(f"FW always-no-oracle baseline: PF {pf(nf):.3f}, sum ${nf.sum():.0f}")
    print()

    # Loaded oracle gate to combine with skip
    from v3.live_shadow.oracle_gate import OracleGate
    gate = OracleGate.load()
    fw_gate_probs = gate.classifier.predict_proba(X_fw)[:, 1]
    fw_use_oracle = fw_gate_probs > gate.threshold
    fw_pnl_with_gate = np.where(fw_use_oracle, of, nf)
    print(f"FW with oracle-gate (no skip): PF {pf(fw_pnl_with_gate):.3f}, sum ${fw_pnl_with_gate.sum():.0f}")
    print()

    # Test 3 profit-label definitions
    print("=== Three profitable-label definitions ===")
    for label_name, mk_label in [
        ("oracle_pnl > 0", lambda o, n: (o > 0).astype(int)),
        ("no_oracle_pnl > 0", lambda o, n: (n > 0).astype(int)),
        ("max(oracle, no_oracle) > 0", lambda o, n: (np.maximum(o, n) > 0).astype(int)),
    ]:
        print(f"\n--- Label: {label_name} ---")
        y = mk_label(of_oos, nf_oos)
        print(f"  OOS positive rate: {y.mean()*100:.1f}%")

        # Train classifier on OOS
        probs_seeds = []
        for rs in [0, 1, 7, 13, 42]:
            clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=rs)
            clf.fit(X_oos, y)
            probs_seeds.append(clf.predict_proba(X_fw)[:, 1])
        probs = np.mean(probs_seeds, axis=0)

        # Apply skip gate (if prob too low, skip trade entirely)
        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
            keep = probs > thr
            n_kept = keep.sum()
            if n_kept == 0:
                print(f"  skip<{thr:.2f}: kept=0 (all skipped)")
                continue
            # With oracle-gate also applied (compose)
            pnl_kept = np.where(fw_use_oracle[keep], of[keep], nf[keep])
            print(f"  skip<{thr:.2f}: kept={n_kept}/{len(of)}  PF {pf(pnl_kept):.3f}  sum ${pnl_kept.sum():.0f}")

    # Final: best skip+gate combined recipe
    print("\n=== Final: best skip+gate combo on FW ===")
    # Use oracle_pnl > 0 label, threshold 0.40
    y = (of_oos > 0).astype(int)
    probs_seeds = []
    for rs in [0, 1, 7, 13, 42]:
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=rs)
        clf.fit(X_oos, y)
        probs_seeds.append(clf.predict_proba(X_fw)[:, 1])
    probs = np.mean(probs_seeds, axis=0)
    print(f"\nGrid: skip threshold × oracle-gate threshold")
    print(f"{'skip_T':>7} {'gate_T':>7} {'kept':>5} {'PF':>7} {'sum':>9} {'use_oracle':>10}")
    for skip_T in [0.30, 0.35, 0.40, 0.45]:
        for gate_T in [0.40, 0.45, 0.50]:
            keep = probs > skip_T
            if keep.sum() == 0: continue
            X_kept = X_fw[keep]
            gp = gate.classifier.predict_proba(X_kept)[:, 1]
            uo = gp > gate_T
            pnl = np.where(uo, of[keep], nf[keep])
            print(f"  {skip_T:.2f}    {gate_T:.2f}   {keep.sum():>4}   {pf(pnl):>7.3f}   ${pnl.sum():>7.0f}    {uo.sum():>3}/{keep.sum():<3}")


if __name__ == "__main__":
    main()
