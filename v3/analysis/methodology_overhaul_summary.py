"""Phase R6 — Methodology overhaul summary + final champion.

Aggregates R1-R5 findings into a deployment-ready verdict:
  - Confirmed champion: V0 (model's chosen direction) + time-stop exit
  - Cross-window aggregate: PF 1.132 on 780 OOS days
  - Previous "V1 + A3 @ 0.19" claim (OOS PF 2.847 on 20 days) reframed
    as a regime-specific artifact, not a robust finding
  - Layer-3 re-evaluation (proper per-window walk-forward retraining)
    flagged as future work — the existing L3 model has leakage across
    rolling OOS windows so can't be honestly applied here

Outputs a single summary.json + synthesis markdown doc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from v3.layer2.common import load_json, save_json


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "methodology_overhaul_summary")
DEFAULT_EVAL_JSON = os.path.join("v3", "artifacts", "rolling_directional_eval", "eval.json")
DEFAULT_COND_JSON = os.path.join(
    "v3", "artifacts", "conditional_directional_rule", "conditional_eval.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-json", default=DEFAULT_EVAL_JSON)
    p.add_argument("--cond-json", default=DEFAULT_COND_JSON)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eval_data = load_json(args.eval_json)
    cond_data = load_json(args.cond_json)

    # Extract key stats
    variants = eval_data["aggregate_per_variant"]
    per_win = eval_data["per_window_per_variant"]

    v0 = variants["V0"]
    v1 = variants["V1"]
    v2 = variants["V2"]

    cond_agg = cond_data["aggregate"]

    # Per-window V0 PF distribution
    v0_pfs = [r["pf"] for r in per_win if r["variant"] == "V0"]
    v1_pfs = [r["pf"] for r in per_win if r["variant"] == "V1"]
    v0_arr = np.array(v0_pfs)
    v1_arr = np.array(v1_pfs)

    summary = {
        "champion": {
            "directional_rule": "V0 (model's chosen direction)",
            "exit_rule": "time-stop (session end)",
            "sizing": "1 contract",
            "aggregate_pf": v0["agg_pf"],
            "aggregate_dd_pct": v0["agg_dd_pct"],
            "aggregate_mean_pnl": v0["agg_mean_pnl"],
            "aggregate_trades": v0["agg_trades"],
            "per_window_pf_mean": v0["per_window_pf_mean"],
            "per_window_pf_median": v0["per_window_pf_median"],
            "per_window_pf_std": v0["per_window_pf_std"],
            "per_window_pf_min": v0["per_window_pf_min"],
            "per_window_pf_max": v0["per_window_pf_max"],
            "beats_v1_in_n_of_13_windows": v0["beat_V1_count"],
        },
        "vs_previous_claim": {
            "previous_champion": "V1 + A3 @ 0.19 (always_put + augmented L3 minus mfe_norm)",
            "previous_oos_pf_claim": 2.847,
            "previous_oos_sample": "20 days (2026-03-05 to 2026-04-01)",
            "reframing": (
                "The previous claim was built on 20 days. In the rolling-window "
                "methodology overhaul (13 disjoint 60-day OOS windows, 780 days total), "
                "V1 (always_put) loses to V0 in 12 of 13 windows (aggregate PF 0.888 "
                "vs V0 1.132). The 20-day finding was a regime-specific artifact. "
                "The methodologically-robust champion is V0 + time-stop."
            ),
        },
        "conditional_rule_test": {
            "tested": True,
            "aggregate_pf": cond_agg["conditional_pf"],
            "oracle_upper_bound_pf": cond_agg["oracle_pf"],
            "vs_blanket_v0": cond_agg["conditional_vs_v0"],
            "verdict": cond_data["verdict"],
        },
        "cross_window_stats": {
            "n_windows": 13,
            "total_oos_days": 780,
            "v0_pf_distribution": {
                "mean": float(v0_arr.mean()),
                "median": float(np.median(v0_arr)),
                "std": float(v0_arr.std(ddof=1)),
                "min": float(v0_arr.min()),
                "max": float(v0_arr.max()),
                "p25": float(np.percentile(v0_arr, 25)),
                "p75": float(np.percentile(v0_arr, 75)),
                "n_windows_pf_ge_1_0": int((v0_arr >= 1.0).sum()),
                "n_windows_pf_ge_1_2": int((v0_arr >= 1.2).sum()),
                "n_windows_pf_ge_1_5": int((v0_arr >= 1.5).sum()),
            },
            "v1_pf_distribution": {
                "mean": float(v1_arr.mean()),
                "median": float(np.median(v1_arr)),
                "std": float(v1_arr.std(ddof=1)),
                "min": float(v1_arr.min()),
                "max": float(v1_arr.max()),
            },
        },
        "variant_comparison": {
            "V0": v0,
            "V1": v1,
            "V2": v2,
            "V3": variants.get("V3", {}),
            "V4": variants.get("V4", {}),
        },
        "layer3_future_work": {
            "status": "deferred",
            "reason": (
                "The existing augmented L3 model (Phase 2A A3 config) was trained on "
                "chosen+teacher trades from days in the 5-fold structure's training pool. "
                "Many of those days appear in the rolling OOS windows, creating leakage "
                "if we applied the existing L3 to rolling windows. Proper R6+ L3 "
                "evaluation requires walk-forward L3 retraining per rolling window "
                "(~2-3 hours of pipeline development + CPU time) — out of scope for this "
                "methodology overhaul pass."
            ),
            "expected_lift": (
                "If L3 generalizes across windows with similar dynamics as Phase 2A's "
                "V1+A3 OOS lift (+0.68 PF over V1 baseline), composed V0+L3 could reach "
                "PF ~1.6-1.8 cross-window. This is speculative until proper walk-forward "
                "L3 retraining is done."
            ),
        },
        "what_changed": {
            "before": "V1 (always_put) + A3 L3 @ 0.19 | OOS PF 2.847 on 20 days",
            "after": "V0 (model direction) + time-stop | Agg PF 1.132 on 780 days",
            "net_trade": (
                "Exchanged a high-PF claim built on 20 days for a modest-PF claim built "
                "on 780 days. The previous claim was optically better but methodologically "
                "fragile. The new claim is modestly profitable (PF 1.132) but built on "
                "13x more data and 39x more OOS days. Also reversed the directional rule "
                "(always_put → model's direction)."
            ),
        },
    }

    # === Print ===
    print("=" * 100)
    print("Phase R6 — Methodology Overhaul Summary")
    print("=" * 100)
    print(flush=True)
    print(f"NEW CHAMPION: V0 + time-stop")
    print(f"  Aggregate PF (780 OOS days across 13 windows): {v0['agg_pf']:.3f}")
    print(f"  Per-window PF mean={v0['per_window_pf_mean']:.3f} median={v0['per_window_pf_median']:.3f} "
          f"std={v0['per_window_pf_std']:.3f}")
    print(f"  Per-window PF range: [{v0['per_window_pf_min']:.3f}, {v0['per_window_pf_max']:.3f}]")
    print(f"  Windows with PF >= 1.0: {(v0_arr >= 1.0).sum()}/13")
    print(f"  Windows with PF >= 1.2: {(v0_arr >= 1.2).sum()}/13")
    print(f"  Beats V1 in: {v0['beat_V1_count']}/13 windows")
    print(flush=True)
    print(f"Previous 'champion' (reframed):")
    print(f"  V1 + A3 L3 @ 0.19 had OOS PF 2.847 on 20 days")
    print(f"  In the 13-window methodology overhaul, V1 aggregate PF = {v1['agg_pf']:.3f}")
    print(f"  V1 beats V0 in: {v1['beat_V1_count']}/13 windows (V0 wins in 10/13)")
    print(flush=True)
    print(f"Conditional rule (V0 default, switch to V1 when classifier flags):")
    print(f"  Aggregate PF: {cond_agg['conditional_pf']:.3f}")
    print(f"  vs blanket V0: {cond_agg['conditional_vs_v0']:+.3f}")
    print(f"  Oracle upper bound: {cond_agg['oracle_pf']:.3f}")
    print(f"  Verdict: {cond_data['verdict']}")
    print(flush=True)
    print(f"Layer-3 status: deferred (walk-forward L3 retraining required to avoid leakage)")
    print(flush=True)

    out = os.path.join(args.out_dir, "summary.json")
    save_json(out, summary)
    print(f"Saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
