"""Phase 1B — In-sample V1+L3 per-fold walk-forward.

For each in-sample fold k in 0..4:
  - Training set: V0 chosen trades + teacher trades from folds 0..k-1
    (the augmented training distribution from v3.1 cleanup, but
    chronologically restricted)
  - Test set: V1 chosen trades from fold k (puts only — V1 forces
    direction = put on the model's chosen entries)
  - Fold 0 has no prior data → time_of_day_90 fallback (matches the
    Stage 3 per-fold protocol)
  - Apply augmented L3 at threshold 0.19 (Stage B optimal)

Reports per-fold PF/DD/mean$/trades + aggregate.

Acceptance gate:
  - 4 of 5 folds with PF >= 1.20 AND aggregate PF >= 1.50 → confirmed
Disqualifying:
  - Any fold PF < 1.0 OR aggregate PF < 1.40 → V1+L3 OOS-specific
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v3.analysis.layer3_train_replay import (
    TRADE_STATE_NAMES,
    _agg,
    _build_per_trade_data,
    _flatten_to_rows,
    _replay_fold0_fallback,
    _replay_with_model,
)
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
DEFAULT_V1_TRADES = os.path.join(
    "v3", "artifacts", "layer2_directional_variants", "in_sample_trades_V1.csv",
)
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "v1_l3_in_sample_walk_forward")
DEFAULT_THRESHOLD = 0.19
DEFAULT_SEED = 42
DEFAULT_FOLD0_FALLBACK_BARS = 90


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--v1-trades", default=DEFAULT_V1_TRADES)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--fold0-fallback-bars", type=int, default=DEFAULT_FOLD0_FALLBACK_BARS)
    return p.parse_args()


def _build_trade_data_list(
    trades: pd.DataFrame, ds: V2Dataset, cfg: GuardrailConfig, equity: float,
    day_cache: dict, paths_cache: dict, minute_map_cache: dict,
    label: str = "",
) -> list[dict]:
    out = []
    skipped = 0
    for i, trade in trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=equity)
            if log_d is None:
                skipped += 1
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            skipped += 1
            continue
        td = _build_per_trade_data(
            trade, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is None:
            skipped += 1
            continue
        out.append(td)
        if (i + 1) % 50 == 0:
            print(f"    {label} built {i+1}/{len(trades)} trades...", flush=True)
    if skipped:
        print(f"    {label} skipped {skipped} of {len(trades)} trades", flush=True)
    return out


def _split_v1_pnls_by_fold(v1_test_data: list[dict]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for td in v1_test_data:
        out.setdefault(td["fold_idx"], []).append(td["csv_pnl"])
    return out


def _compute_v1_baseline_per_fold(v1_trades: pd.DataFrame, equity: float) -> dict[int, dict]:
    """V1 alone (no L3) per-fold PF for direct comparison."""
    out = {}
    for fold_idx, sub in v1_trades.groupby("fold_idx"):
        sub_sorted = sub.sort_values(["day", "bar_index"])
        pnls = sub_sorted["pnl"].astype(float).tolist()
        m = replay_metrics_from_pnls(pnls, equity)
        m["trades"] = float(len(pnls))
        m["mean_pnl"] = float(np.mean(pnls)) if pnls else 0.0
        out[int(fold_idx)] = m
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

    # === Load trade tables ===
    print(flush=True)
    print("Loading trade tables...", flush=True)
    chosen_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    teacher_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "teacher_baseline_trades.csv"))
    v1_trades = pd.read_csv(args.v1_trades)
    print(f"  layer2_trades.csv: {len(chosen_trades)} V0 chosen", flush=True)
    print(f"  teacher_baseline_trades.csv: {len(teacher_trades)} teacher", flush=True)
    print(f"  in_sample_trades_V1.csv: {len(v1_trades)} V1 chosen", flush=True)

    # === Build per-trade data once ===
    day_cache: dict = {}
    paths_cache: dict = {}
    minute_map_cache: dict = {}

    print(flush=True)
    print("Building V0-chosen per-trade data (training pool)...", flush=True)
    chosen_data = _build_trade_data_list(
        chosen_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
        label="V0-chosen",
    )
    print(f"  V0-chosen trades built: {len(chosen_data)}", flush=True)

    print(flush=True)
    print("Building teacher per-trade data (training pool)...", flush=True)
    teacher_data = _build_trade_data_list(
        teacher_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
        label="teacher",
    )
    print(f"  Teacher trades built: {len(teacher_data)}", flush=True)

    print(flush=True)
    print("Building V1-chosen per-trade data (test set)...", flush=True)
    v1_test_data = _build_trade_data_list(
        v1_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
        label="V1-test",
    )
    print(f"  V1 test trades built: {len(v1_test_data)}", flush=True)

    # === V1-alone per-fold baseline (from CSV pnl, no L3) ===
    v1_alone = _compute_v1_baseline_per_fold(v1_trades, args.equity)

    # === Per-fold walk-forward training and replay ===
    print(flush=True)
    print("=" * 100)
    print(f"Per-fold walk-forward (threshold={args.threshold:.2f})")
    print("=" * 100, flush=True)

    fold_results: dict[int, dict] = {}
    all_simulated = []

    for fold_idx in sorted(set(td["fold_idx"] for td in v1_test_data)):
        # Training pool: V0 chosen + teacher trades from prior folds (chronological)
        train_chosen = [td for td in chosen_data if td["fold_idx"] < fold_idx]
        train_teacher = [td for td in teacher_data if td["fold_idx"] < fold_idx]
        train_pool = train_chosen + train_teacher
        test_pool = [td for td in v1_test_data if td["fold_idx"] == fold_idx]

        if len(train_pool) == 0:
            print(flush=True)
            print(f"Fold {fold_idx}: no prior training data → time_of_day_{args.fold0_fallback_bars} fallback", flush=True)
            sim_rows = _replay_fold0_fallback(test_pool, "time_of_day_90", args.fold0_fallback_bars)
        else:
            X_train, y_train, _ = _flatten_to_rows(train_pool)
            print(flush=True)
            print(
                f"Fold {fold_idx}: training on {len(train_pool)} trades "
                f"({len(train_chosen)} V0 + {len(train_teacher)} teacher) "
                f"/ {len(X_train)} bar-rows; pos_rate={y_train.mean():.3f}",
                flush=True,
            )
            model = HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.05, max_depth=4,
                max_iter=200, min_samples_leaf=50,
                random_state=args.seed + fold_idx, early_stopping=False,
            )
            model.fit(X_train, y_train)
            sim_rows = _replay_with_model(test_pool, model, args.threshold)
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

    # === Aggregate ===
    full_df = pd.concat(all_simulated, ignore_index=True) if all_simulated else pd.DataFrame()
    total_days = sum(len(d) for d in fold_test_days.values())
    overall = _agg(full_df, args.equity, total_days)

    print(flush=True)
    print("=" * 100)
    print("Per-fold table — V1+L3 vs V1-alone (in-sample)")
    print("=" * 100)
    print(f"{'fold':<6}{'V1+L3 PF':>10}{'V1+L3 DD%':>11}{'V1+L3 mean$':>14}{'V1+L3 trades':>14}"
          f"{'V1 PF':>10}{'delta PF':>10}", flush=True)
    for fi in sorted(fold_results):
        r = fold_results[fi]
        v1r = v1_alone.get(fi, {"pf": 0.0})
        delta = r["pf"] - v1r["pf"]
        print(
            f"{fi:<6}{r['pf']:>10.3f}{r['max_dd_pct']:>11.1f}"
            f"{r['mean_pnl']:>14.1f}{int(r['trades']):>14}"
            f"{v1r['pf']:>10.3f}{delta:>+10.3f}",
            flush=True,
        )

    print(flush=True)
    print(f"Aggregate V1+L3 in-sample: PF={overall['pf']:.3f} "
          f"DD={overall['max_dd_pct']:.1f}% mean=${overall['mean_pnl']:.1f} "
          f"trades={int(overall['trades'])}", flush=True)
    v1_full_pnls = v1_trades.sort_values(["day", "bar_index"])["pnl"].astype(float).tolist()
    v1_full = replay_metrics_from_pnls(v1_full_pnls, args.equity)
    print(f"Aggregate V1-alone in-sample: PF={v1_full['pf']:.3f} "
          f"DD={v1_full['max_dd_pct']:.1f}%", flush=True)

    # === Verdict ===
    fold_pfs = [fold_results[fi]["pf"] for fi in sorted(fold_results)]
    n_folds_above_1_2 = sum(1 for pf in fold_pfs if pf >= 1.20)
    min_fold_pf = float(min(fold_pfs))
    min_fold_idx = int(np.argmin(fold_pfs))
    agg_pf = float(overall["pf"])

    print(flush=True)
    print("=" * 100)
    print("Phase 1B verdict")
    print("=" * 100, flush=True)
    print(f"  - 4-of-5 folds >= 1.20 PF gate:   {n_folds_above_1_2} folds pass "
          f"({'PASS' if n_folds_above_1_2 >= 4 else 'FAIL'})", flush=True)
    print(f"  - Aggregate PF >= 1.50 gate:       {agg_pf:.3f} "
          f"({'PASS' if agg_pf >= 1.50 else 'FAIL'})", flush=True)
    print(f"  - Min fold PF < 1.0 disqualifier:  {min_fold_pf:.3f} (fold {min_fold_idx}) "
          f"({'DISQUALIFIED' if min_fold_pf < 1.0 else 'OK'})", flush=True)
    print(f"  - Aggregate PF < 1.40 disqualifier:{agg_pf:.3f} "
          f"({'DISQUALIFIED' if agg_pf < 1.40 else 'OK'})", flush=True)

    if min_fold_pf < 1.0 or agg_pf < 1.40:
        verdict = (
            f"DISQUALIFIED — V1+L3 in-sample weak "
            f"(min fold PF {min_fold_pf:.3f}, aggregate {agg_pf:.3f}). "
            f"OOS lift was likely window-specific."
        )
    elif n_folds_above_1_2 >= 4 and agg_pf >= 1.50:
        verdict = (
            f"CONFIRMED — V1+L3 robust in-sample "
            f"({n_folds_above_1_2}/5 folds >= 1.20, aggregate {agg_pf:.3f}). "
            f"Combined with OOS PF 2.169, V1+L3 is the validated champion."
        )
    else:
        verdict = (
            f"WEAK — V1+L3 doesn't disqualify but doesn't fully confirm "
            f"({n_folds_above_1_2}/5 folds >= 1.20, aggregate {agg_pf:.3f}). "
            f"Phase 1A bootstrap CI's wide lower bound is harder to dismiss."
        )
    print(f"\n  {verdict}", flush=True)

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "v1_trades_file": args.v1_trades,
            "n_v0_chosen": int(len(chosen_data)),
            "n_teacher": int(len(teacher_data)),
            "n_v1_test": int(len(v1_test_data)),
            "threshold": float(args.threshold),
            "fold0_fallback_bars": int(args.fold0_fallback_bars),
            "seed": int(args.seed),
        },
        "overall_v1_l3": overall,
        "overall_v1_alone": v1_full,
        "per_fold_v1_l3": fold_results,
        "per_fold_v1_alone": v1_alone,
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "v1_l3_in_sample_walk_forward.json")
    with open(out, "w") as f:
        json.dump(
            payload, f, indent=2, sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
        )
    full_df.to_csv(os.path.join(args.out_dir, "v1_l3_in_sample_trades.csv"), index=False)
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    print(f"Saved: {os.path.join(args.out_dir, 'v1_l3_in_sample_trades.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
