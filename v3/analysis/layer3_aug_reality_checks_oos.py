"""Stage B of the post-OOS research workstream: Stage 4-style reality
checks on the AUGMENTED Layer-3 model evaluated on OOS.

The Stage 4 reality checks (in-sample) confirmed the chosen-only model
held up in-sample. But the augmented model's +0.509 PF OOS lift hasn't
been pressure-tested. This script does:

1. Threshold sensitivity sweep on augmented model evaluated on the 20
   cached OOS days. Span [0.05, 0.40] step 0.02. Looking for K=2-style
   single-point peak (lift evaporates one decile away from threshold
   0.17) vs broad lift band.
2. Random-exit baseline at matched OOS exit frequency. 50 seeds. If
   augmented OOS PF (1.537) sits at p100 like in-sample did, real
   signal. If at p70-80, weak.

Verdict gates:
- PASS: PF >= 1.50 in >= 5/10 thresholds in [0.10, 0.30] AND augmented
  PF percentile vs random >= 90.
- FALSIFIED: <3 thresholds beat 1.50 OR percentile < 70.
- WEAK: middle case.

Run:
    python -m v3.analysis.layer3_aug_reality_checks_oos
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
    _flatten_to_rows,
    _replay_with_model,
)
from v3.analysis.layer3_v31_cleanup import (
    _build_in_sample_trade_data,
    _build_oos_trade_data,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import replay_metrics_from_pnls


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_aug_reality_checks_oos")
DEFAULT_REFERENCE_THRESHOLD = 0.17
DEFAULT_SEED = 42
N_RANDOM_SEEDS = 50

# Anchors
HEURISTIC_PF_FOR_OOS_NA = None  # We don't have a direct heuristic OOS anchor; use 1.50 absolute
LAYER2_ALONE_OOS_PF = 0.869
CHOSEN_ONLY_OOS_PF = 1.028
AUGMENTED_OOS_PF_REF = 1.537


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmented Layer-3 reality checks on OOS.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--reference-threshold", type=float, default=DEFAULT_REFERENCE_THRESHOLD)
    p.add_argument("--n-random-seeds", type=int, default=N_RANDOM_SEEDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _replay_pf_dd(trade_data: list[dict], model, threshold: float, equity: float) -> tuple[float, float, float, list[dict]]:
    sim = _replay_with_model(trade_data, model, threshold)
    if not sim:
        return 0.0, 0.0, 0.0, []
    df = pd.DataFrame(sim).sort_values(["day", "entry_bar"])
    pnls = df["exit_pnl"].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    mean_bars = float(df["bars_held"].mean()) if len(df) else 0.0
    return float(m["pf"]), float(m["max_dd_pct"]), mean_bars, sim


def _per_bar_exit_freq(trade_data: list[dict], model, threshold: float) -> float:
    n_total = 0
    n_exit = 0
    for td in trade_data:
        feats = np.stack([np.concatenate([b["l2_feats"], b["trade_state"]]) for b in td["per_bar"]])
        probs = model.predict_proba(feats)[:, 1]
        n_total += len(probs)
        n_exit += int((probs >= threshold).sum())
    return float(n_exit / n_total) if n_total else 0.0


def _random_exit_replay(trade_data: list[dict], exit_p: float, rng: np.random.Generator,
                        equity: float) -> float:
    sim = []
    for td in trade_data:
        exit_pnl = None
        for b in td["per_bar"]:
            if rng.random() < exit_p:
                exit_pnl = b["current_pnl"]
                break
        if exit_pnl is None:
            exit_pnl = td["per_bar"][-1]["current_pnl"]
        sim.append({"day": td["day"], "entry_bar": td["entry_bar"], "exit_pnl": exit_pnl})
    if not sim:
        return 0.0
    df = pd.DataFrame(sim).sort_values(["day", "entry_bar"])
    m = replay_metrics_from_pnls(df["exit_pnl"].astype(float).tolist(), equity)
    return float(m["pf"])


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading V2Dataset...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    print("Building in-sample chosen + teacher trade data...")
    in_sample_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "layer2_trades.csv"))
    teacher_trades = pd.read_csv(os.path.join(args.baseline_run_dir, "teacher_baseline_trades.csv"))
    day_cache: dict = {}
    paths_cache: dict = {}
    minute_map_cache: dict = {}
    in_sample_data = _build_in_sample_trade_data(
        in_sample_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    teacher_data = _build_in_sample_trade_data(
        teacher_trades, ds, cfg, args.equity, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  in-sample chosen={len(in_sample_data)}, teacher={len(teacher_data)}")

    print("Building OOS trade data...")
    oos_data, _ = _build_oos_trade_data(
        ds, cfg, args, day_cache, paths_cache, minute_map_cache,
    )
    print(f"  OOS={len(oos_data)}")

    print()
    print("Training augmented model (chosen + teacher)...")
    augmented_data = in_sample_data + teacher_data
    X_aug, y_aug, _ = _flatten_to_rows(augmented_data)
    model_aug = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=args.seed + 1000, early_stopping=False,
    )
    model_aug.fit(X_aug, y_aug)
    print(f"  augmented trained on {len(X_aug)} bar-rows")

    # === Test 1: Threshold sensitivity on OOS ===
    print()
    print("=" * 100)
    print("Test 1 -- Augmented model threshold sensitivity on OOS")
    print("=" * 100)
    threshold_grid = np.arange(0.05, 0.41, 0.02)
    threshold_results = {}
    print(f"{'thr':>6}{'OOS_PF':>10}{'OOS_DD%':>10}{'mean_bars':>11}{'exit_freq':>11}")
    for thr in threshold_grid:
        thr_f = float(round(thr, 2))
        pf, dd, mean_bars, _ = _replay_pf_dd(oos_data, model_aug, thr_f, args.equity)
        exit_freq = _per_bar_exit_freq(oos_data, model_aug, thr_f)
        threshold_results[thr_f] = {
            "pf": pf, "max_dd_pct": dd, "mean_bars_held": mean_bars,
            "exit_freq": exit_freq,
        }
        print(f"{thr_f:>6.2f}{pf:>10.3f}{dd:>10.1f}{mean_bars:>11.1f}{exit_freq:>11.4f}")

    # === Test 2: Random-exit baseline on OOS at matched freq ===
    print()
    print("=" * 100)
    print(f"Test 2 -- Random-exit baseline at matched OOS exit_freq (threshold={args.reference_threshold})")
    print("=" * 100)
    ref_pf, ref_dd, _, _ = _replay_pf_dd(oos_data, model_aug, args.reference_threshold, args.equity)
    ref_freq = _per_bar_exit_freq(oos_data, model_aug, args.reference_threshold)
    print(f"Reference: thr={args.reference_threshold}  exit_freq={ref_freq:.4f}  PF={ref_pf:.3f}")

    pfs_random = []
    for s in range(args.seed, args.seed + args.n_random_seeds):
        rng = np.random.default_rng(s + 50000)
        pf_r = _random_exit_replay(oos_data, ref_freq, rng, args.equity)
        pfs_random.append(pf_r)
    pfs_random_arr = np.array(pfs_random)

    print(f"Random-exit OOS PF distribution over {args.n_random_seeds} seeds:")
    print(f"  mean={pfs_random_arr.mean():.3f}  std={pfs_random_arr.std():.3f}  min={pfs_random_arr.min():.3f}  max={pfs_random_arr.max():.3f}")
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles = {f"p{int(q*100)}": float(np.quantile(pfs_random_arr, q)) for q in qs}
    for k, v in quantiles.items():
        print(f"  {k}={v:.3f}")
    pct_at_or_below = float(np.mean(pfs_random_arr <= ref_pf))
    pct_ge = float(np.mean(pfs_random_arr >= ref_pf))
    print(f"  Augmented OOS PF {ref_pf:.3f} sits at percentile {pct_at_or_below*100:.1f} of random distribution")
    print(f"  ({pct_ge*100:.1f}% of random seeds match or exceed augmented OOS PF)")

    # === Verdict ===
    print()
    print("=" * 100)
    print("Stage B Verdict")
    print("=" * 100)
    band_thrs = [t for t in threshold_results if 0.10 <= t <= 0.30]
    n_pass_band = sum(1 for t in band_thrs if threshold_results[t]["pf"] >= 1.50)
    n_above_layer2 = sum(1 for t in band_thrs if threshold_results[t]["pf"] >= LAYER2_ALONE_OOS_PF)
    print(f"  Threshold sensitivity: {n_pass_band}/{len(band_thrs)} thresholds in [0.10, 0.30] beat 1.50 OOS")
    print(f"  Threshold sensitivity: {n_above_layer2}/{len(band_thrs)} thresholds beat Layer-2-alone OOS (0.869)")
    print(f"  Random-exit percentile: Layer-3 augmented at p{pct_at_or_below*100:.0f}")

    if n_pass_band >= 5 and pct_at_or_below >= 0.90:
        verdict = (f"PASS -- {n_pass_band}/{len(band_thrs)} thresholds beat 1.50 AND "
                   f"random-exit percentile {pct_at_or_below*100:.0f} >= 90. Augmented OOS lift is robust.")
    elif n_pass_band < 3 or pct_at_or_below < 0.70:
        verdict = (f"FALSIFIED -- {n_pass_band}/{len(band_thrs)} thresholds beat 1.50; "
                   f"random-exit percentile {pct_at_or_below*100:.0f}. Augmented OOS lift is selection bias.")
    else:
        verdict = (f"WEAK -- {n_pass_band}/{len(band_thrs)} thresholds beat 1.50; "
                   f"random-exit percentile {pct_at_or_below*100:.0f}. Partial signal.")
    print(f"  VERDICT: {verdict}")

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_oos": int(len(oos_data)),
            "reference_threshold": float(args.reference_threshold),
            "n_random_seeds": int(args.n_random_seeds),
        },
        "threshold_sensitivity_oos": {f"{t:.2f}": v for t, v in threshold_results.items()},
        "random_exit_baseline_oos": {
            "exit_freq_matched": float(ref_freq),
            "reference_pf": float(ref_pf),
            "pf_mean": float(pfs_random_arr.mean()),
            "pf_std": float(pfs_random_arr.std()),
            "pf_min": float(pfs_random_arr.min()),
            "pf_max": float(pfs_random_arr.max()),
            "pf_quantiles": quantiles,
            "ref_percentile_in_random": pct_at_or_below,
            "fraction_random_ge_ref": pct_ge,
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "aug_reality_checks_oos.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
