"""Layer-2 per-fold regime diagnostic (Check 1 from the reality-checks plan).

The winning run `v3/artifacts/layer2_shared_enc_fixedq_detach/` posted
aggregate PF 1.455, DD 36.9%, 275 trades. Per-fold breakdown shows
dangerous concentration: fold 0 is underwater (PF 0.860), fold 2 is
marginal (1.038), folds 3 and 4 are unusually strong (2.095 and 1.801).
Before paper trading, diagnose WHY.

For each fold this script reports:
  - Trade-level metrics: PF, DD, WR, mean PnL, call%
  - Feature distribution on the full eligible-bar universe for that
    fold's TEST window (medians + IQRs)
  - Entry/side score distribution on OOF predictions for the fold
  - Oracle outcome mix on chosen trades

Then compares fold 0 (worst) against folds 3/4 (best), flagging any
feature whose median differs by >1σ (using the full-sample pooled
std as the normalizing scale).

Disqualifying signal: a large feature-median gap between the losing
and winning folds = the model has a regime dependency; paper trading
gets paused for a regime-detection workstream.

Run:
    python -m v3.analysis.layer2_per_fold_diagnostic \
        --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.layer2.common import load_export_bundle, load_pickle, DEFAULT_DATASET_PATH, replay_metrics_from_pnls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-2 per-fold regime diagnostic.")
    p.add_argument("--run-dir", default="v3/artifacts/layer2_shared_enc_fixedq_detach")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    p.add_argument("--equity", type=float, default=25_000.0)
    return p.parse_args()


def _trade_metrics(pnls: list[float], equity: float, n_days: int, directions: list[str]) -> dict:
    m = replay_metrics_from_pnls(pnls, equity)
    if directions:
        m["call_pct"] = float(sum(1 for d in directions if d == "call") / len(directions))
    else:
        m["call_pct"] = 0.0
    m["trades_per_day"] = float(len(pnls) / max(n_days, 1))
    return m


def _describe(series: pd.Series, percentiles=(0.25, 0.5, 0.75)) -> dict[str, float]:
    s = series.dropna().astype(float)
    if s.empty:
        return {"n": 0, "median": float("nan"), "mean": float("nan"), "std": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "n": int(s.size),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def main() -> int:
    args = parse_args()

    # --- Load artifacts -----------------------------------------------------
    trades_path = os.path.join(args.run_dir, "layer2_trades.csv")
    trades = pd.read_csv(trades_path)
    print(f"Loaded {len(trades)} trades from {trades_path}")

    oof_path = os.path.join(args.run_dir, "oof_predictions.pkl")
    oof = load_pickle(oof_path) if os.path.exists(oof_path) else None
    if oof is not None:
        print(f"Loaded {len(oof)} OOF predictions (cols: {list(oof.columns)[:8]}...)")

    bundle = load_export_bundle(args.dataset)
    df = bundle["rows"]
    folds = list(bundle["meta"]["folds"])
    feature_names = list(bundle["meta"]["feature_names"])
    # W2a + select context features that distinguish regime; a focused
    # list so the diagnostic output fits on a screen.
    regime_features = [
        "vix", "atm_iv", "iv_percentile",
        "first15_range_pct", "omar_range_pct",
        "sigma_pos", "omar_retest_dist_norm", "last10_range_over_omar",
        "late_window_40_120_flag", "inside_first15", "last10_break_state",
    ]

    # --- Per-fold metrics ---------------------------------------------------
    fold_idx_to_days = {int(f["fold_idx"]): list(f["test_days"]) for f in folds}

    rows = []
    fold_features: dict[int, dict[str, dict[str, float]]] = {}
    fold_scores: dict[int, dict[str, float]] = {}
    fold_outcomes: dict[int, dict[str, int]] = {}

    for fold_idx, test_days in sorted(fold_idx_to_days.items()):
        tr = trades[trades["fold_idx"] == fold_idx]
        pnls = tr["pnl"].astype(float).tolist()
        dirs = tr["direction"].astype(str).tolist()
        n_days = len(test_days)
        m = _trade_metrics(pnls, args.equity, n_days, dirs)

        # Feature distribution over the FULL eligible-bar universe of this
        # fold's test window (not just chosen trades).
        test_df = df[df["day"].isin(test_days)]
        feat_summ = {fn: _describe(test_df[fn]) for fn in regime_features if fn in test_df.columns}
        fold_features[fold_idx] = feat_summ

        # Score distribution on OOF predictions for this fold.
        if oof is not None and "fold_idx" in oof.columns:
            fold_oof = oof[oof["fold_idx"] == fold_idx]
            fold_scores[fold_idx] = {
                "entry_score": _describe(fold_oof["entry_score"]),
                "side_score": _describe(fold_oof["side_score"]),
                "side_conf": _describe(fold_oof["side_conf"]),
            }
        else:
            fold_scores[fold_idx] = {}

        # Oracle outcome mix for CHOSEN trades.
        outcome_series = tr["oracle_slice_outcome"].fillna("").astype(str)
        fold_outcomes[fold_idx] = outcome_series.value_counts().to_dict()

        # Date range label
        date_range = f"{test_days[0]} .. {test_days[-1]}" if test_days else "?"

        rows.append({
            "fold": fold_idx,
            "dates": date_range,
            "n_trades": int(len(tr)),
            "pf": m["pf"],
            "dd_pct": m["max_dd_pct"],
            "wr": float((tr["pnl"] > 0).mean()) if len(tr) else float("nan"),
            "mean_pnl": float(tr["pnl"].mean()) if len(tr) else float("nan"),
            "call_pct": m["call_pct"],
            "tpd": m["trades_per_day"],
        })

    # --- Print per-fold summary --------------------------------------------
    print()
    print("=" * 110)
    print("Per-fold trade metrics")
    print("=" * 110)
    print(f"{'fold':<5}{'dates':<32}{'n_tr':>6}{'PF':>8}{'DD%':>7}{'WR':>7}{'mean$':>10}{'call%':>8}{'TPD':>7}")
    for r in rows:
        print(
            f"{r['fold']:<5}{r['dates']:<32}{r['n_trades']:>6}"
            f"{r['pf']:>8.3f}{r['dd_pct']:>7.1f}"
            f"{100 * r['wr']:>6.1f}%"
            f"{r['mean_pnl']:>10.1f}"
            f"{100 * r['call_pct']:>7.1f}%"
            f"{r['tpd']:>7.3f}"
        )

    # --- Feature distribution comparison -----------------------------------
    print()
    print("=" * 110)
    print("Feature distribution by fold (eligible-bar universe of each fold's test window)")
    print("=" * 110)
    # Header with fold numbers
    fold_ids = sorted(fold_features.keys())
    header = f"{'feature':<32}" + "".join([f"  f{i:>1}_med" + f"  f{i:>1}_std" for i in fold_ids])
    # Simplify: print median + std only
    print(f"{'feature':<32}" + "".join([f"   f{i}_median     f{i}_std" for i in fold_ids]))
    for fn in regime_features:
        if not all(fn in fold_features[i] for i in fold_ids):
            continue
        cells = []
        for i in fold_ids:
            d = fold_features[i][fn]
            cells.append(f" {d['median']:>10.3f} {d['std']:>10.3f}")
        print(f"{fn:<32}" + "".join(cells))

    # --- Score distribution comparison -------------------------------------
    if oof is not None:
        print()
        print("=" * 110)
        print("OOF score distribution by fold (entry_score and side_score across the fold's test set)")
        print("=" * 110)
        for score_name in ("entry_score", "side_score", "side_conf"):
            print(f"\n{score_name}:")
            for i in fold_ids:
                d = fold_scores[i].get(score_name, {})
                if not d:
                    continue
                print(
                    f"  fold {i}: n={d['n']}  median={d['median']:+.3f}  mean={d['mean']:+.3f}  "
                    f"std={d['std']:.3f}  p25={d['p25']:+.3f}  p75={d['p75']:+.3f}"
                )

    # --- Oracle outcome mix -------------------------------------------------
    print()
    print("=" * 110)
    print("Oracle outcome mix on chosen trades, per fold")
    print("=" * 110)
    all_outcomes = sorted({k for d in fold_outcomes.values() for k in d.keys()})
    header = f"{'fold':<6}" + "".join([f"{o or 'empty':>18}" for o in all_outcomes])
    print(header)
    for i in fold_ids:
        cells = []
        total = sum(fold_outcomes[i].values())
        for o in all_outcomes:
            n = fold_outcomes[i].get(o, 0)
            pct = (100 * n / total) if total > 0 else 0.0
            cells.append(f"{n:>4} ({pct:>5.1f}%)   ")
        print(f"{i:<6}" + "".join(cells)[: 18 * len(all_outcomes)])

    # --- Flag large shifts between losing fold(s) and winning fold(s) ------
    print()
    print("=" * 110)
    print("Losers vs winners — feature median shifts (>1 pooled-σ flagged)")
    print("=" * 110)
    # Identify losing / winning folds by PF
    loss_folds = [r["fold"] for r in rows if r["pf"] < 1.0]
    win_folds = [r["fold"] for r in rows if r["pf"] >= 1.5]
    if not loss_folds or not win_folds:
        print(f"(no cleanly separated loser vs winner set: loss={loss_folds}, win={win_folds})")
    else:
        print(f"loss folds: {loss_folds}  win folds: {win_folds}")
        print(f"{'feature':<32}{'loss_median':>14}{'win_median':>13}{'pooled_std':>13}{'abs_shift_σ':>13}{'flag':>6}")
        shifts = []
        for fn in regime_features:
            loss_vals = []
            win_vals = []
            for i in loss_folds:
                if fn in fold_features[i]:
                    test_df = df[df["day"].isin(fold_idx_to_days[i])]
                    loss_vals += test_df[fn].dropna().astype(float).tolist()
            for i in win_folds:
                if fn in fold_features[i]:
                    test_df = df[df["day"].isin(fold_idx_to_days[i])]
                    win_vals += test_df[fn].dropna().astype(float).tolist()
            if not loss_vals or not win_vals:
                continue
            loss_med = float(np.median(loss_vals))
            win_med = float(np.median(win_vals))
            pooled = float(np.std(loss_vals + win_vals))
            shift = abs(loss_med - win_med) / max(pooled, 1e-9)
            flag = " *" if shift >= 1.0 else ""
            shifts.append((shift, fn, loss_med, win_med, pooled, flag))
        shifts.sort(reverse=True)
        for shift, fn, loss_med, win_med, pooled, flag in shifts:
            print(f"{fn:<32}{loss_med:>14.3f}{win_med:>13.3f}{pooled:>13.3f}{shift:>13.3f}{flag:>6}")
        if any(s[0] >= 1.0 for s in shifts):
            print()
            print("⚠ HARD STOP per plan §2a: at least one feature's median differs by >1 pooled-σ")
            print("  between losing and winning folds. The model has a regime dependency.")
            print("  Paper trading paused; diagnose regime before going further.")
        else:
            print()
            print("✓ No single feature crosses the >1σ bar. Regime separation is subtler than")
            print("  single-feature drift; check the score distribution (above) for model-side drift.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
