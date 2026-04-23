"""Phase 2A verification — A3 per-fold walk-forward + A3 OOS bootstrap CI.

The Phase 2A experiment found that dropping mfe_norm from the
augmented L3 model lifts OOS PF from 2.169 → 2.847. This script
runs two follow-up checks:

1. **Per-fold A3 walk-forward**: train A3 on V0 chosen + teacher
   from prior folds (no mfe_norm), test on V1 trades from current
   fold. Threshold 0.19. Fold 0 = time_of_day_90 fallback. Confirms
   no fold catastrophic regression.

2. **A3 OOS bootstrap CI**: resample the 20 A3 OOS PnLs N=10,000
   times, report 95% CI. Compare to the baseline V1+L3 bootstrap
   (95% CI [0.520, 12.645]).

Acceptance:
- Per-fold: no fold below 1.0; aggregate >= 1.50
- Bootstrap: 95% CI lower bound shifted right vs baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v3.analysis.layer3_train_replay import (
    TRADE_STATE_NAMES,
    _agg,
    _build_per_trade_data,
    _replay_fold0_fallback,
)
from v3.analysis.layer3_v31_cleanup import _build_in_sample_trade_data
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_labeled_day,
    load_export_bundle,
    replay_metrics_from_pnls,
)
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_V1_IS_TRADES = os.path.join(
    "v3", "artifacts", "layer2_directional_variants", "in_sample_trades_V1.csv",
)
DEFAULT_V1_OOS_TRADES = os.path.join(
    "v3", "artifacts", "layer2_directional_variants", "oos_trades_V1.csv",
)
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "a3_verification")
DEFAULT_THRESHOLD = 0.19
DEFAULT_SEED = 42
N_X_SIM = 89
TS_OFFSET = N_X_SIM
MFE_NORM_TS_INDEX = 3
N_BOOTSTRAP = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--v1-is-trades", default=DEFAULT_V1_IS_TRADES)
    p.add_argument("--v1-oos-trades", default=DEFAULT_V1_OOS_TRADES)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--fold0-fallback-bars", type=int, default=90)
    return p.parse_args()


def build_X(trade_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X_rows, y_rows = [], []
    for td in trade_data:
        for b in td["per_bar"]:
            feats = np.concatenate([b["l2_feats"], b["trade_state"]])
            X_rows.append(feats)
            y_rows.append(b["target"])
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int8)


def drop_mfe_norm(X: np.ndarray, mfe_norm_idx: int) -> np.ndarray:
    keep = [i for i in range(X.shape[1]) if i != mfe_norm_idx]
    return X[:, keep]


def replay_a3(
    trade_data: list[dict],
    model: HistGradientBoostingClassifier,
    threshold: float,
    mfe_norm_idx: int,
) -> list[dict]:
    out = []
    for td in trade_data:
        feats_full = np.stack([
            np.concatenate([b["l2_feats"], b["trade_state"]])
            for b in td["per_bar"]
        ])
        feats = drop_mfe_norm(feats_full, mfe_norm_idx)
        probs = model.predict_proba(feats)[:, 1]
        exit_idx = None
        trigger = "model"
        for i in range(len(probs)):
            if probs[i] >= threshold:
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = len(probs) - 1
            trigger = "time_stop_fallback"
        bar = td["per_bar"][exit_idx]
        out.append({
            "fold_idx": td["fold_idx"], "day": td["day"],
            "entry_bar": td["entry_bar"], "exit_bar": bar["bar"],
            "direction": td["direction"],
            "exit_pnl": bar["current_pnl"],
            "trigger": trigger,
            "bars_held": bar["bar"] - td["entry_bar"],
        })
    return out


def profit_factor(pnls: np.ndarray) -> float:
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    if neg == 0.0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def bootstrap_pfs(pnls: np.ndarray, n_iter: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = pnls.shape[0]
    out = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        out[i] = profit_factor(pnls[idx])
    return out


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset...", flush=True)
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds_meta = list(bundle["meta"]["folds"])
    fold_test_days = {int(f["fold_idx"]): set(f["test_days"]) for f in folds_meta}

    minutes_to_close_idx = ds.feature_names.index("minutes_to_close")
    mfe_norm_idx = TS_OFFSET + MFE_NORM_TS_INDEX
    print(f"  minutes_to_close idx (unused): {minutes_to_close_idx}", flush=True)
    print(f"  mfe_norm idx (will drop):      {mfe_norm_idx}", flush=True)

    # === Build per-trade data ===
    print(flush=True)
    print("Building chosen + teacher per-trade data...", flush=True)
    in_sample_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    teacher_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "teacher_baseline_trades.csv"))
    day_cache: dict = {}
    paths_cache: dict = {}
    minute_map_cache: dict = {}
    chosen_data = _build_in_sample_trade_data(
        in_sample_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    teacher_data = _build_in_sample_trade_data(
        teacher_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  chosen: {len(chosen_data)}, teacher: {len(teacher_data)}", flush=True)

    print(flush=True)
    print("Building V1 in-sample per-trade data (test set for walk-forward)...", flush=True)
    v1_is_trades = pd.read_csv(args.v1_is_trades)
    v1_is_data = []
    for _, trade in v1_is_trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=args.equity)
            if log_d is None:
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        td = _build_per_trade_data(
            trade, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            v1_is_data.append(td)
    print(f"  V1 in-sample test trades built: {len(v1_is_data)}", flush=True)

    print(flush=True)
    print("Building V1 OOS per-trade data...", flush=True)
    v1_oos_trades = pd.read_csv(args.v1_oos_trades)
    v1_oos_trades["day"] = v1_oos_trades["day"].astype(str)
    v1_oos_data = []
    for _, trade in v1_oos_trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=args.equity)
            if log_d is None:
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        fake_row = pd.Series({
            "day": day,
            "bar_index": int(trade["bar_index"]),
            "direction": str(trade["direction"]),
            "fold_idx": int(trade.get("fold_idx", -1)),
            "pnl": float(trade["pnl"]),
        })
        td = _build_per_trade_data(
            fake_row, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            v1_oos_data.append(td)
    print(f"  V1 OOS test trades built: {len(v1_oos_data)}", flush=True)

    # === Per-fold A3 walk-forward ===
    print(flush=True)
    print("=" * 100)
    print(f"Per-fold A3 walk-forward (drop mfe_norm, threshold={args.threshold:.2f})")
    print("=" * 100, flush=True)
    fold_results: dict[int, dict] = {}
    all_simulated = []

    # V1-alone per-fold baselines for comparison
    v1_alone_per_fold: dict[int, dict] = {}
    for fi, sub in v1_is_trades.groupby("fold_idx"):
        sub_sorted = sub.sort_values(["day", "bar_index"])
        pnls = sub_sorted["pnl"].astype(float).tolist()
        m = replay_metrics_from_pnls(pnls, args.equity)
        m["trades"] = float(len(pnls))
        m["mean_pnl"] = float(np.mean(pnls)) if pnls else 0.0
        v1_alone_per_fold[int(fi)] = m

    for fold_idx in sorted(set(td["fold_idx"] for td in v1_is_data)):
        train_chosen = [td for td in chosen_data if td["fold_idx"] < fold_idx]
        train_teacher = [td for td in teacher_data if td["fold_idx"] < fold_idx]
        train_pool = train_chosen + train_teacher
        test_pool = [td for td in v1_is_data if td["fold_idx"] == fold_idx]

        if len(train_pool) == 0:
            print(flush=True)
            print(f"Fold {fold_idx}: time_of_day_{args.fold0_fallback_bars} fallback", flush=True)
            sim_rows = _replay_fold0_fallback(test_pool, "time_of_day_90", args.fold0_fallback_bars)
        else:
            X_train_full, y_train = build_X(train_pool)
            X_train = drop_mfe_norm(X_train_full, mfe_norm_idx)
            print(flush=True)
            print(
                f"Fold {fold_idx}: training A3 on {len(train_pool)} trades "
                f"({len(train_chosen)} V0 + {len(train_teacher)} teacher) "
                f"/ {len(X_train)} bar-rows / {X_train.shape[1]} features (no mfe_norm)",
                flush=True,
            )
            model = HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.05, max_depth=4,
                max_iter=200, min_samples_leaf=50,
                random_state=args.seed + fold_idx, early_stopping=False,
            )
            model.fit(X_train, y_train)
            sim_rows = replay_a3(test_pool, model, args.threshold, mfe_norm_idx)
            sim_df_tmp = pd.DataFrame(sim_rows)
            print(
                f"  test n={len(test_pool)}; early_exit_share="
                f"{float((sim_df_tmp['trigger']=='model').mean()):.2%}; "
                f"mean bars held={sim_df_tmp['bars_held'].mean():.1f}",
                flush=True,
            )

        sim_df = pd.DataFrame(sim_rows)
        n_test_days = len(fold_test_days.get(int(fold_idx), set()))
        m = _agg(sim_df, args.equity, n_test_days)
        fold_results[int(fold_idx)] = m
        all_simulated.append(sim_df)

    full_df = pd.concat(all_simulated, ignore_index=True) if all_simulated else pd.DataFrame()
    overall_is = _agg(full_df, args.equity, sum(len(d) for d in fold_test_days.values()))

    print(flush=True)
    print("Per-fold A3 vs V1+L3-baseline (Phase 1B) vs V1-alone")
    print(f"{'fold':<6}{'A3 PF':>10}{'A3 DD%':>10}{'A3 mean$':>12}{'A3 trades':>12}"
          f"{'V1 PF':>10}{'A3 vs V1':>12}", flush=True)
    for fi in sorted(fold_results):
        r = fold_results[fi]
        v1r = v1_alone_per_fold.get(fi, {"pf": 0.0})
        delta = r["pf"] - v1r["pf"]
        print(
            f"{fi:<6}{r['pf']:>10.3f}{r['max_dd_pct']:>10.1f}"
            f"{r['mean_pnl']:>12.0f}{int(r['trades']):>12}"
            f"{v1r['pf']:>10.3f}{delta:>+12.3f}",
            flush=True,
        )

    print(flush=True)
    print(f"Aggregate A3 in-sample: PF={overall_is['pf']:.3f} "
          f"DD={overall_is['max_dd_pct']:.1f}% mean=${overall_is['mean_pnl']:.0f} "
          f"trades={int(overall_is['trades'])}", flush=True)

    # === A3 OOS replay + bootstrap ===
    print(flush=True)
    print("=" * 100)
    print(f"A3 OOS replay + bootstrap CI (N={args.n_bootstrap})")
    print("=" * 100, flush=True)

    X_aug_full, y_aug = build_X(chosen_data + teacher_data)
    X_aug = drop_mfe_norm(X_aug_full, mfe_norm_idx)
    print(f"  Augmented A3 training: {len(chosen_data) + len(teacher_data)} trades, "
          f"{len(X_aug)} bar-rows, {X_aug.shape[1]} features", flush=True)
    model_aug = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=args.seed + 1000, early_stopping=False,
    )
    model_aug.fit(X_aug, y_aug)
    a3_oos_sim = replay_a3(v1_oos_data, model_aug, args.threshold, mfe_norm_idx)
    a3_oos_df = pd.DataFrame(a3_oos_sim).sort_values(["day", "entry_bar"])
    a3_oos_pnls = a3_oos_df["exit_pnl"].astype(float).values
    a3_oos_metrics = replay_metrics_from_pnls(a3_oos_pnls.tolist(), args.equity)
    print(f"  A3 OOS observed: PF={a3_oos_metrics['pf']:.3f} "
          f"DD={a3_oos_metrics['max_dd_pct']:.1f}% "
          f"mean=${float(np.mean(a3_oos_pnls)):.0f} "
          f"trades={len(a3_oos_pnls)}", flush=True)

    pfs = bootstrap_pfs(a3_oos_pnls, args.n_bootstrap, args.seed)
    finite = pfs[np.isfinite(pfs)]
    boot_summary = {
        "n_iter": int(pfs.size),
        "n_finite": int(finite.size),
        "n_inf": int((~np.isfinite(pfs)).sum()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p2_5": float(np.percentile(finite, 2.5)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p97_5": float(np.percentile(finite, 97.5)),
    }
    frac_pf_ge_1 = float((finite >= 1.0).mean())
    frac_pf_ge_1_5 = float((finite >= 1.5).mean())
    frac_pf_ge_2 = float((finite >= 2.0).mean())

    print(flush=True)
    print(f"  Bootstrap distribution:", flush=True)
    print(f"    median (p50) = {boot_summary['p50']:.4f}", flush=True)
    print(f"    p2.5 (95% CI lower) = {boot_summary['p2_5']:.4f}", flush=True)
    print(f"    p97.5 (95% CI upper) = {boot_summary['p97_5']:.4f}", flush=True)
    print(f"    Fraction PF >= 1.0  = {frac_pf_ge_1:.4f}", flush=True)
    print(f"    Fraction PF >= 1.5  = {frac_pf_ge_1_5:.4f}", flush=True)
    print(f"    Fraction PF >= 2.0  = {frac_pf_ge_2:.4f}", flush=True)

    # === Verdict ===
    fold_pfs = [fold_results[fi]["pf"] for fi in sorted(fold_results)]
    n_folds_above_1 = sum(1 for pf in fold_pfs if pf >= 1.0)
    n_folds_above_1_2 = sum(1 for pf in fold_pfs if pf >= 1.20)
    min_fold_pf = float(min(fold_pfs))
    min_fold_idx = int(np.argmin(fold_pfs))

    print(flush=True)
    print("=" * 100)
    print("A3 verification verdict")
    print("=" * 100, flush=True)
    print(f"  Per-fold (Phase 1B equivalent for A3):", flush=True)
    print(f"    Min fold PF:           {min_fold_pf:.3f} (fold {min_fold_idx})", flush=True)
    print(f"    n folds >= 1.0:        {n_folds_above_1}/{len(fold_pfs)}", flush=True)
    print(f"    n folds >= 1.20:       {n_folds_above_1_2}/{len(fold_pfs)}", flush=True)
    print(f"    Aggregate PF:          {overall_is['pf']:.3f}", flush=True)
    print(flush=True)
    print(f"  Bootstrap on A3 OOS (N={args.n_bootstrap}):", flush=True)
    print(f"    Observed PF:           {a3_oos_metrics['pf']:.3f}", flush=True)
    print(f"    Median:                {boot_summary['p50']:.3f}", flush=True)
    print(f"    95% CI:                [{boot_summary['p2_5']:.3f}, {boot_summary['p97_5']:.3f}]", flush=True)
    print(f"    Fraction PF >= 1.0:    {frac_pf_ge_1:.3f}", flush=True)
    print(f"    Fraction PF >= 1.5:    {frac_pf_ge_1_5:.3f}", flush=True)
    print(f"    Fraction PF >= 2.0:    {frac_pf_ge_2:.3f}", flush=True)

    if min_fold_pf >= 1.0 and overall_is["pf"] >= 1.50 and boot_summary["p2_5"] >= 0.7:
        verdict = "STRONG-PASS — A3 is robust per-fold, aggregate solid, bootstrap left tail acceptable"
    elif min_fold_pf >= 1.0 and overall_is["pf"] >= 1.50:
        verdict = "PASS — A3 robust per-fold AND aggregate; bootstrap tail still wide (small-N)"
    elif min_fold_pf < 1.0 and min_fold_pf >= 0.7 and overall_is["pf"] >= 1.50:
        verdict = (f"PARTIAL — A3 has fold {min_fold_idx} below 1.0 ({min_fold_pf:.3f}) but no "
                   f"catastrophic regression; aggregate {overall_is['pf']:.3f} good")
    else:
        verdict = (f"FAIL — A3 fold {min_fold_idx} = {min_fold_pf:.3f} OR aggregate "
                   f"{overall_is['pf']:.3f} below threshold")
    print(flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_chosen": int(len(chosen_data)),
            "n_teacher": int(len(teacher_data)),
            "n_v1_is": int(len(v1_is_data)),
            "n_v1_oos": int(len(v1_oos_data)),
            "threshold": float(args.threshold),
            "fold0_fallback_bars": int(args.fold0_fallback_bars),
            "seed": int(args.seed),
            "mfe_norm_idx": int(mfe_norm_idx),
        },
        "per_fold_a3_walk_forward": fold_results,
        "per_fold_v1_alone": v1_alone_per_fold,
        "aggregate_a3_in_sample": overall_is,
        "a3_oos_observed": a3_oos_metrics,
        "a3_oos_bootstrap": boot_summary,
        "a3_oos_fraction_pf_ge_1": frac_pf_ge_1,
        "a3_oos_fraction_pf_ge_1_5": frac_pf_ge_1_5,
        "a3_oos_fraction_pf_ge_2": frac_pf_ge_2,
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "a3_verification.json")
    with open(out, "w") as f:
        json.dump(
            payload, f, indent=2, sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )
    full_df.to_csv(os.path.join(args.out_dir, "a3_in_sample_walk_forward_trades.csv"), index=False)
    a3_oos_df.to_csv(os.path.join(args.out_dir, "a3_oos_trades.csv"), index=False)
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    print(f"Saved: {os.path.join(args.out_dir, 'a3_in_sample_walk_forward_trades.csv')}", flush=True)
    print(f"Saved: {os.path.join(args.out_dir, 'a3_oos_trades.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
