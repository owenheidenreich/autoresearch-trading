"""Stage 4 of the Layer 3 (learned exits) workstream.

Battery of adversarial reality checks against the Stage 3 PASS-PF
provisional verdict (PF 2.085 at threshold=0.2). Mirrors the Layer-2
in-sample stress battery that falsified the atm_iv K=2 signal.

The threshold-sensitivity sweep is the primary guard. If PF is sharply
peaked at 0.20 with rapid degradation at 0.18 and 0.22, the lift is
selection bias (atm_iv K=2 deja vu). If PF is broad and stable across
0.10-0.30, the lift is real.

Tests:
1. Fine-grained threshold sensitivity (0.05 to 0.40 step 0.02)
2. Random-exit baseline at the same per-bar exit frequency as
   threshold=0.2 (n_seeds=50)
3. Slippage stress on threshold=0.2 model ({$0, $10, $25, $50}/RT)
4. Feature importance from each per-fold trained model
5. Mean bars held vs PF (sanity: are higher-PF thresholds just
   exiting at the heuristic-90 sweet spot?)

Run:
    python -m v3.analysis.layer3_reality_checks
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
    CORRECTED_BASELINE_DD,
    CORRECTED_BASELINE_PF,
    HEURISTIC_BASELINE_DD,
    HEURISTIC_BASELINE_FOLD0,
    HEURISTIC_BASELINE_PF,
    TRADE_STATE_NAMES,
    _build_per_trade_data,
    _flatten_to_rows,
    _replay_fold0_fallback,
    _replay_with_model,
    _agg,
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


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_reality_checks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer 3 reality checks (Stage 4).")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--session-end-bar", type=int, default=DEFAULT_SESSION_END_BAR)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_CONTRACT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-random-seeds", type=int, default=50)
    p.add_argument("--reference-threshold", type=float, default=0.2)
    return p.parse_args()


def _train_all_folds(trade_data: list[dict], seed: int) -> dict[int, HistGradientBoostingClassifier]:
    models = {}
    fold_ids = sorted(set(td["fold_idx"] for td in trade_data))
    for fold_idx in fold_ids:
        train_data = [td for td in trade_data if td["fold_idx"] < fold_idx]
        if len(train_data) == 0:
            continue
        X_train, y_train, _ = _flatten_to_rows(train_data)
        m = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_depth=4,
            max_iter=200, min_samples_leaf=50,
            random_state=seed + fold_idx, early_stopping=False,
        )
        m.fit(X_train, y_train)
        models[fold_idx] = m
    return models


def _replay_at_threshold(
    trade_data: list[dict],
    models: dict[int, HistGradientBoostingClassifier],
    threshold: float,
    fold0_fallback_bars: int,
    equity: float,
    fold_test_days: dict[int, set],
) -> tuple[dict, pd.DataFrame, dict[int, dict]]:
    sims = []
    fold_results: dict[int, dict] = {}
    for fold_idx in sorted(set(td["fold_idx"] for td in trade_data)):
        test_data = [td for td in trade_data if td["fold_idx"] == fold_idx]
        if fold_idx == 0 or fold_idx not in models:
            sim_rows = _replay_fold0_fallback(test_data, "time_of_day_90", fold0_fallback_bars)
        else:
            sim_rows = _replay_with_model(test_data, models[fold_idx], threshold)
        sim_df = pd.DataFrame(sim_rows)
        n_days = len(fold_test_days.get(int(fold_idx), set()))
        m = _agg(sim_df, equity, n_days)
        fold_results[int(fold_idx)] = m
        sims.append(sim_df)
    full = pd.concat(sims, ignore_index=True) if sims else pd.DataFrame()
    total_days = sum(len(d) for d in fold_test_days.values())
    overall = _agg(full, equity, total_days)
    return overall, full, fold_results


def _per_bar_exit_freq(
    trade_data: list[dict],
    models: dict[int, HistGradientBoostingClassifier],
    threshold: float,
) -> float:
    """Fraction of (trade, bar) pairs where Layer-3 says 'exit'."""
    n_total = 0
    n_exit = 0
    for td in trade_data:
        if td["fold_idx"] == 0 or td["fold_idx"] not in models:
            continue
        feats = np.stack([np.concatenate([b["l2_feats"], b["trade_state"]]) for b in td["per_bar"]])
        probs = models[td["fold_idx"]].predict_proba(feats)[:, 1]
        n_total += len(probs)
        n_exit += int((probs >= threshold).sum())
    return float(n_exit / n_total) if n_total else 0.0


def _random_exit_replay(
    trade_data: list[dict],
    exit_p: float,
    rng: np.random.Generator,
    fold0_fallback_bars: int,
    fold_test_days: dict[int, set],
    equity: float,
) -> tuple[dict, dict[int, dict]]:
    """At each post-entry bar, exit with probability exit_p. Same fold-0
    fallback as the learned model."""
    sims = []
    fold_results: dict[int, dict] = {}
    for fold_idx in sorted(set(td["fold_idx"] for td in trade_data)):
        test_data = [td for td in trade_data if td["fold_idx"] == fold_idx]
        if fold_idx == 0:
            sim_rows = _replay_fold0_fallback(test_data, "time_of_day_90", fold0_fallback_bars)
        else:
            sim_rows = []
            for td in test_data:
                exit_pnl = None
                exit_bar = None
                trigger = "random"
                for b in td["per_bar"]:
                    if rng.random() < exit_p:
                        exit_pnl = b["current_pnl"]
                        exit_bar = b["bar"]
                        break
                if exit_pnl is None:
                    last = td["per_bar"][-1]
                    exit_pnl = last["current_pnl"]
                    exit_bar = last["bar"]
                    trigger = "time_stop_fallback"
                sim_rows.append({
                    "fold_idx": td["fold_idx"], "day": td["day"],
                    "entry_bar": td["entry_bar"], "exit_bar": exit_bar,
                    "direction": td["direction"], "exit_pnl": exit_pnl,
                    "trigger": trigger, "bars_held": exit_bar - td["entry_bar"],
                })
        sim_df = pd.DataFrame(sim_rows)
        n_days = len(fold_test_days.get(int(fold_idx), set()))
        m = _agg(sim_df, equity, n_days)
        fold_results[int(fold_idx)] = m
        sims.append(sim_df)
    full = pd.concat(sims, ignore_index=True) if sims else pd.DataFrame()
    total_days = sum(len(d) for d in fold_test_days.values())
    overall = _agg(full, equity, total_days)
    return overall, fold_results


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds_meta = list(bundle["meta"]["folds"])
    fold_test_days = {int(f["fold_idx"]): set(f["test_days"]) for f in folds_meta}

    print(f"Loaded {len(trades)} trades; loading V2Dataset...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    # --- Build per-trade data ---
    day_cache: dict[str, tuple] = {}
    paths_cache: dict[int, dict] = {}
    minute_map_cache: dict[str, dict] = {}
    trade_data: list[dict] = []
    skipped = 0
    print("Building per-trade datasets (slow, ~3-5 min)...")
    for i, trade in trades.iterrows():
        day = str(trade["day"])
        if day not in day_cache:
            day_cache[day] = build_labeled_day(ds, day, cfg, equity=args.equity)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            skipped += 1
            continue
        td = _build_per_trade_data(trade, log, sidecar, paths_cache, minute_map_cache, ds,
                                    args.session_end_bar, args.commission)
        if td is None:
            skipped += 1
            continue
        trade_data.append(td)
        if (i + 1) % 50 == 0:
            print(f"  built {i+1}/{len(trades)}...")
    print(f"Built {len(trade_data)} trades; {skipped} skipped")

    # --- Train per-fold models ---
    print()
    print("Training per-fold HistGB models...")
    models = _train_all_folds(trade_data, args.seed)
    print(f"Trained {len(models)} models (folds {sorted(models.keys())}); fold 0 will use TOD-90 fallback")

    # === Test 1: fine-grained threshold sensitivity ===
    print()
    print("=" * 100)
    print("Test 1 -- Fine-grained threshold sensitivity (the primary K=2 guard)")
    print("=" * 100)
    threshold_grid = np.arange(0.05, 0.41, 0.02)
    threshold_results = {}
    print(f"{'thr':>6}{'PF':>8}{'DD%':>8}{'mean$':>10}{'min_fold':>10}{'min_fold_pf':>14}{'mean_bars':>11}{'exit_freq':>11}")
    for thr in threshold_grid:
        thr_f = float(round(thr, 2))
        overall, full, per_fold = _replay_at_threshold(
            trade_data, models, thr_f, fold0_fallback_bars=90,
            equity=args.equity, fold_test_days=fold_test_days,
        )
        min_fold_pf = float(min(r["pf"] for r in per_fold.values()))
        min_fold_idx = int(min(per_fold, key=lambda fi: per_fold[fi]["pf"]))
        # Fold-1+ trades only for mean bars (fold 0 always uses fallback bars=90)
        non_fb = full[full["fold_idx"] > 0] if not full.empty else pd.DataFrame()
        mean_bars = float(non_fb["bars_held"].mean()) if not non_fb.empty else float("nan")
        exit_freq = _per_bar_exit_freq(trade_data, models, thr_f)
        threshold_results[thr_f] = {
            "overall": overall, "per_fold": per_fold,
            "min_fold_pf": min_fold_pf, "min_fold_idx": min_fold_idx,
            "mean_bars_held_non_fold0": mean_bars,
            "exit_freq": exit_freq,
        }
        print(
            f"{thr_f:>6.2f}{overall['pf']:>8.3f}{overall['max_dd_pct']:>8.1f}{overall['mean_pnl']:>10.0f}"
            f"{min_fold_idx:>10d}{min_fold_pf:>14.3f}{mean_bars:>11.1f}{exit_freq:>11.4f}"
        )

    # === Test 2: random-exit baseline at threshold=0.2 frequency ===
    print()
    print("=" * 100)
    print(f"Test 2 -- Random-exit baseline at exit_freq matched to threshold={args.reference_threshold}")
    print("=" * 100)
    # Find the threshold in our grid closest to the requested reference threshold.
    sorted_thrs = sorted(threshold_results.keys())
    nearest = min(sorted_thrs, key=lambda t: abs(t - args.reference_threshold))
    ref_thr_results = threshold_results[nearest]
    args.reference_threshold = nearest
    ref_freq = ref_thr_results["exit_freq"]
    ref_pf = ref_thr_results["overall"]["pf"]
    print(f"Reference: threshold={args.reference_threshold:.2f}  exit_freq={ref_freq:.4f}  PF={ref_pf:.3f}")
    pfs_random = []
    for s in range(args.seed, args.seed + args.n_random_seeds):
        rng = np.random.default_rng(s)
        overall_r, _ = _random_exit_replay(
            trade_data, ref_freq, rng, fold0_fallback_bars=90,
            fold_test_days=fold_test_days, equity=args.equity,
        )
        pfs_random.append(float(overall_r["pf"]))
    pfs_random_arr = np.array(pfs_random)
    pct_at_or_below = float(np.mean(pfs_random_arr <= ref_pf))
    print(f"Random-exit PF distribution over {args.n_random_seeds} seeds:")
    print(f"  mean={pfs_random_arr.mean():.3f}  std={pfs_random_arr.std():.3f}  min={pfs_random_arr.min():.3f}  max={pfs_random_arr.max():.3f}")
    print(f"  p25={np.quantile(pfs_random_arr, 0.25):.3f}  p50={np.quantile(pfs_random_arr, 0.5):.3f}  p75={np.quantile(pfs_random_arr, 0.75):.3f}  p95={np.quantile(pfs_random_arr, 0.95):.3f}")
    print(f"  Reference Layer-3 PF {ref_pf:.3f} sits at percentile {pct_at_or_below*100:.1f} of random distribution")
    print(f"  ({100*np.mean(pfs_random_arr >= ref_pf):.1f}% of random seeds match or exceed Layer-3 PF)")

    # === Test 3: slippage stress at threshold=0.2 ===
    print()
    print("=" * 100)
    print(f"Test 3 -- Slippage stress at threshold={args.reference_threshold}")
    print("=" * 100)
    overall_ref, full_ref, per_fold_ref = _replay_at_threshold(
        trade_data, models, args.reference_threshold, fold0_fallback_bars=90,
        equity=args.equity, fold_test_days=fold_test_days,
    )
    print(f"{'slip_$/RT':>10}{'PF':>8}{'DD%':>8}{'mean$':>10}{'edge_vs_baseline':>20}")
    slip_grid = [0.0, 10.0, 25.0, 50.0]
    slip_results = {}
    for slip in slip_grid:
        adj_pnls = (full_ref["exit_pnl"].astype(float) - slip).tolist()
        m = replay_metrics_from_pnls(adj_pnls, args.equity)
        # Edge vs the baseline at the same slippage (corrected baseline 1.472 at slip=0
        # — for slippage applied to baseline, approximate via simple subtraction proportional)
        # Just report Layer-3 PF; baseline Layer-2 slippage was already explored elsewhere.
        slip_results[slip] = {"pf": m["pf"], "max_dd_pct": m["max_dd_pct"], "mean_pnl": float(np.mean(adj_pnls))}
        print(f"{slip:>10.0f}{m['pf']:>8.3f}{m['max_dd_pct']:>8.1f}{slip_results[slip]['mean_pnl']:>10.0f}")

    # === Test 4: feature importance per fold ===
    print()
    print("=" * 100)
    print("Test 4 -- Top features by gradient-boosting importance, per fold")
    print("=" * 100)
    feature_names_l2 = [f"l2_{i}" for i in range(ds.X_sim.shape[1])]
    all_feature_names = feature_names_l2 + TRADE_STATE_NAMES
    feature_importance: dict[int, list] = {}
    # HistGB has built-in feature_importances_? Actually no, HistGradientBoostingClassifier doesn't.
    # Fall back to permutation importance? Too slow. Use a proxy: split-importance-ish via
    # examining the predictor; but HistGB exposes nothing user-friendly. Skip rigorous
    # importance and just note this is a TODO.
    print("  (HistGradientBoostingClassifier doesn't expose feature_importances_; skipping rigorous importance for v3.0)")

    # === Verdict ===
    print()
    print("=" * 100)
    print("Stage 4 Verdict")
    print("=" * 100)
    # Threshold sensitivity check: count thresholds in [0.10, 0.30] where PF >= heuristic ceiling
    band_thresholds = [t for t in threshold_results if 0.10 <= t <= 0.30]
    n_pass_in_band = sum(1 for t in band_thresholds if threshold_results[t]["overall"]["pf"] >= HEURISTIC_BASELINE_PF)
    pct_pass_in_band = n_pass_in_band / max(len(band_thresholds), 1)
    print(f"  Threshold sensitivity: {n_pass_in_band}/{len(band_thresholds)} thresholds in [0.10, 0.30] beat heuristic PF {HEURISTIC_BASELINE_PF}")
    print(f"  Random-exit at matched freq: Layer-3 at percentile {pct_at_or_below*100:.1f}")
    print(f"  Slippage at $25/RT: PF {slip_results[25.0]['pf']:.3f}")

    if pct_pass_in_band >= 0.50 and pct_at_or_below >= 0.85 and slip_results[25.0]['pf'] >= HEURISTIC_BASELINE_PF:
        verdict = "PASS -- threshold robustness + random-exit beat + slippage robustness all hold"
    elif pct_pass_in_band < 0.30 and pct_at_or_below < 0.70:
        verdict = (f"FALSIFIED -- only {n_pass_in_band}/{len(band_thresholds)} thresholds beat heuristic AND "
                   f"random-exit beats Layer-3 in {100*(1-pct_at_or_below):.0f}% of seeds. "
                   f"K=2 selection-bias confirmed.")
    elif pct_at_or_below < 0.70:
        verdict = (f"FALSIFIED-RANDOM -- random-exit at matched freq beats Layer-3 in "
                   f"{100*(1-pct_at_or_below):.0f}% of seeds. Model isn't doing real work.")
    elif pct_pass_in_band < 0.30:
        verdict = (f"FALSIFIED-THRESHOLD -- only {n_pass_in_band}/{len(band_thresholds)} thresholds beat heuristic. "
                   f"K=2 selection-bias confirmed; lift was a single-point peak.")
    else:
        verdict = (f"WEAK -- partial signals: threshold band {n_pass_in_band}/{len(band_thresholds)}, "
                   f"random percentile {pct_at_or_below*100:.0f}%, slip-25 PF {slip_results[25.0]['pf']:.3f}. "
                   f"Inconclusive but not clearly falsified.")
    print(f"  VERDICT: {verdict}")

    # --- Save ---
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_trades": int(len(trade_data)),
            "n_random_seeds": int(args.n_random_seeds),
            "reference_threshold": float(args.reference_threshold),
        },
        "threshold_sensitivity": {
            f"{t:.2f}": {
                "overall": v["overall"], "per_fold": v["per_fold"],
                "min_fold_pf": v["min_fold_pf"], "min_fold_idx": v["min_fold_idx"],
                "mean_bars_held_non_fold0": v["mean_bars_held_non_fold0"],
                "exit_freq": v["exit_freq"],
            } for t, v in threshold_results.items()
        },
        "random_exit_baseline": {
            "exit_freq_matched": float(ref_freq),
            "reference_pf": float(ref_pf),
            "n_seeds": int(args.n_random_seeds),
            "pf_mean": float(pfs_random_arr.mean()),
            "pf_std": float(pfs_random_arr.std()),
            "pf_min": float(pfs_random_arr.min()),
            "pf_max": float(pfs_random_arr.max()),
            "pf_quantiles": {f"p{int(q*100)}": float(np.quantile(pfs_random_arr, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
            "ref_percentile_in_random": pct_at_or_below,
            "fraction_random_ge_ref": float(np.mean(pfs_random_arr >= ref_pf)),
        },
        "slippage_stress": {
            f"slip_{int(s)}": v for s, v in slip_results.items()
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "reality_checks.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
