"""Random-suppression ablation for the atm_iv payoff-gating probe.

The payoff-gating probe found atm_iv bottom-2-decile suppression lifts
PF 1.455 -> 1.530 on the detach-side baseline. The headline +0.075 PF
could be either:

(a) atm_iv is identifying low-payoff fallback bars (real signal), OR
(b) suppressing ANY 66 of 176 fallback bars happens to tighten the
    equity curve enough on this sample to lift PF by ~0.075 (no signal,
    just trade-count effect).

This script tests (b) directly. For each random seed, it suppresses 66
of the 176 fallback bars uniformly at random, recomputes PF/DD/TPD
chronologically composed with the unchanged 99 teacher trades, and
reports the distribution. If atm_iv-gated PF (1.530) sits inside the
random-suppression distribution at, say, 1 sigma, the SOFT verdict
becomes a hard falsification.

Disqualifying signal:
- atm_iv PF percentile within random-suppression distribution >= 70th
  percentile -> atm_iv may be doing real work
- atm_iv PF percentile within random-suppression distribution <= 50th
  percentile -> atm_iv is no better than dropping bars at random

Run:
    python -m v3.analysis.layer2_random_suppression_ablation \
        --baseline-run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    load_export_bundle,
    replay_metrics_from_pnls,
    teacher_direction_hint_from_row,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer2_random_suppression_ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-suppression ablation for atm_iv gating.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--n-suppress", type=int, default=66, help="How many fallback bars to suppress per seed.")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--atm-iv-pf", type=float, default=1.530, help="Reference atm_iv-gated PF for percentile comparison.")
    return p.parse_args()


def _load_chosen(baseline_run_dir: str) -> pd.DataFrame:
    """Load trades.csv and join with the export bundle to determine which
    rows had a teacher trigger (route_source) vs a model-only fallback."""
    trades = pd.read_csv(os.path.join(baseline_run_dir, "layer2_trades.csv"))
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    rows = bundle["rows"]
    join_cols = ["day", "bar_index", "fold_id",
                 "orc_buy_call", "orc_buy_put",
                 "failed_break_buy_call", "failed_break_buy_put",
                 "teacher_any_triggered"]
    chosen = trades.merge(
        rows[join_cols],
        left_on=["day", "bar_index", "fold_idx"],
        right_on=["day", "bar_index", "fold_id"],
        how="left",
        validate="one_to_one",
    )
    chosen["teacher_direction"] = chosen.apply(teacher_direction_hint_from_row, axis=1)
    chosen["route_source"] = np.where(chosen["teacher_direction"] != "", "teacher", "fallback")
    return chosen


def _compose_metrics_chrono(
    teacher_trades: pd.DataFrame,
    kept_fallback: pd.DataFrame,
    total_days: int,
    equity: float,
) -> dict[str, float]:
    """PF/DD on chronologically-ordered teacher + kept_fallback trades."""
    t_view = teacher_trades[["day", "bar_index", "fold_idx", "pnl"]].copy()
    f_view = kept_fallback[["day", "bar_index", "fold_idx", "pnl"]].copy()
    combined = pd.concat([t_view, f_view], ignore_index=True)
    combined = combined.sort_values(["day", "bar_index"]).reset_index(drop=True)
    m = replay_metrics_from_pnls(combined["pnl"].astype(float).tolist(), equity)
    m["trades"] = float(len(combined))
    m["trades_per_day"] = float(len(combined) / max(total_days, 1))
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chosen = _load_chosen(args.baseline_run_dir)
    teacher = chosen[chosen["route_source"] == "teacher"].copy()
    fallback = chosen[chosen["route_source"] == "fallback"].copy().reset_index(drop=True)
    n_fallback = len(fallback)
    if args.n_suppress >= n_fallback:
        raise ValueError(f"n_suppress={args.n_suppress} must be < fallback count {n_fallback}")

    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds = list(bundle["meta"]["folds"])
    total_days = sum(len(f["test_days"]) for f in folds)

    baseline = _compose_metrics_chrono(teacher, fallback, total_days, args.equity)
    print(f"Baseline (no suppression): PF={baseline['pf']:.3f}  DD={baseline['max_dd_pct']:.1f}%  TPD={baseline['trades_per_day']:.3f}")
    print(f"Reference (atm_iv gated):  PF={args.atm_iv_pf:.3f} (target for percentile comparison)")
    print(f"Drawing {args.n_seeds} random suppressions of {args.n_suppress} of {n_fallback} fallback bars...")

    pfs = np.zeros(args.n_seeds, dtype=float)
    dds = np.zeros(args.n_seeds, dtype=float)
    tpds = np.zeros(args.n_seeds, dtype=float)
    for i in range(args.n_seeds):
        rng = np.random.default_rng(args.start_seed + i)
        keep_idx = rng.choice(n_fallback, size=n_fallback - args.n_suppress, replace=False)
        keep_idx = np.sort(keep_idx)
        kept = fallback.iloc[keep_idx]
        m = _compose_metrics_chrono(teacher, kept, total_days, args.equity)
        pfs[i] = float(m["pf"]) if np.isfinite(m["pf"]) else float("nan")
        dds[i] = float(m["max_dd_pct"])
        tpds[i] = float(m["trades_per_day"])

    finite = np.isfinite(pfs)
    pfs_f = pfs[finite]
    print()
    print("=" * 80)
    print(f"Random-suppression PF distribution (n={int(finite.sum())} valid seeds)")
    print("=" * 80)
    print(f"  mean={np.mean(pfs_f):.3f}  std={np.std(pfs_f):.3f}  min={np.min(pfs_f):.3f}  max={np.max(pfs_f):.3f}")
    qs = [0.05, 0.10, 0.25, 0.50, 0.70, 0.75, 0.90, 0.95]
    pcts = np.quantile(pfs_f, qs)
    print("  Quantiles:")
    for q, v in zip(qs, pcts):
        print(f"    p{int(q*100):>2d}={v:.3f}")

    pct_at_or_below = float(np.mean(pfs_f <= args.atm_iv_pf))
    print()
    print(f"  atm_iv reference PF {args.atm_iv_pf:.3f} sits at percentile {pct_at_or_below*100:.1f} of the random distribution")

    # Tail comparison: how rare is a +0.075 PF lift over baseline at random?
    lift_atm_iv = args.atm_iv_pf - baseline["pf"]
    lifts = pfs_f - baseline["pf"]
    pct_lift_ge = float(np.mean(lifts >= lift_atm_iv))
    print(f"  atm_iv lift over baseline: {lift_atm_iv:+.3f}  -- {pct_lift_ge*100:.1f}% of random seeds match or exceed this")

    # Verdict
    print()
    print("=" * 80)
    print("Verdict (per probe header)")
    print("=" * 80)
    if pct_at_or_below >= 0.70:
        verdict = f"PLAUSIBLE -- atm_iv at p{int(pct_at_or_below*100)} of random; signal might be real"
    elif pct_at_or_below <= 0.50:
        verdict = f"FALSIFIED -- atm_iv at p{int(pct_at_or_below*100)} of random; gate is no better than random suppression"
    else:
        verdict = f"INCONCLUSIVE -- atm_iv at p{int(pct_at_or_below*100)} of random; in the wash"
    print(f"  {verdict}")

    # Save artifact
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_seeds": int(args.n_seeds),
            "n_suppress": int(args.n_suppress),
            "n_fallback": int(n_fallback),
            "atm_iv_pf_reference": float(args.atm_iv_pf),
        },
        "baseline": {k: float(v) for k, v in baseline.items()},
        "random_suppression": {
            "pf_mean": float(np.mean(pfs_f)),
            "pf_std": float(np.std(pfs_f)),
            "pf_min": float(np.min(pfs_f)),
            "pf_max": float(np.max(pfs_f)),
            "pf_quantiles": {f"p{int(q*100)}": float(v) for q, v in zip(qs, pcts)},
            "atm_iv_percentile_in_random": pct_at_or_below,
            "fraction_random_lifts_ge_atm_iv": pct_lift_ge,
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "random_suppression_ablation.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
