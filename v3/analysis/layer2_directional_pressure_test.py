"""Stage C2 of the post-OOS research workstream: pressure-test the
Stage C1 best-variant Layer-2 directional rework.

Three pressure tests, all on the best variant's selected trades:

1. **Random-direction ablation**: at each chosen bar, replace direction
   with rng.choice(call/put), 10 seeds. If random PF >= variant PF,
   the variant's directional logic is cosmetic.
2. **Per-fold + OOS breakdown**: reread per-fold from C1 output and
   confirm no fold regresses below 0.80 PF. Especially fold 0.
3. **Slippage stress**: apply $0/$10/$25/$50 RT slip on the variant's
   in-sample + OOS trades. Verdict: variant fragile if PF drops below
   1.0 at $25 OOS.

Verdict gates:
- ROBUST: random-direction PF < variant PF * 0.85 AND no fold < 0.80 AND
  slip-25 OOS PF >= 1.0.
- SOFT: at least one criterion borderline.
- FRAGILE: random-direction PF >= variant PF (signal cosmetic) OR
  any fold < 0.80 OR slip-25 OOS PF < 1.0.

Run:
    python -m v3.analysis.layer2_directional_pressure_test --variant Vk
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.analysis.layer3_oos_validation import (
    _apply_layer2_models,
    _build_oos_export_rows,
    _identify_oos_days,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_labeled_day,
    compute_time_stop_pnl_for_direction,
    load_export_bundle,
    load_json,
    load_pickle,
    replay_metrics_from_pnls,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_C1_DIR = os.path.join("v3", "artifacts", "layer2_directional_variants")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer2_directional_pressure_test")
DEFAULT_SEED = 42
N_RANDOM_DIRECTION_SEEDS = 10
SLIPPAGE_GRID = [0.0, 10.0, 25.0, 50.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pressure-test the best Layer-2 directional variant.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--c1-dir", default=DEFAULT_C1_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--variant", default=None,
                   help="Override variant; defaults to best from C1's directional_variants.json")
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-seeds", type=int, default=N_RANDOM_DIRECTION_SEEDS)
    return p.parse_args()


def _agg(trades_df: pd.DataFrame, equity: float, pnl_col: str = "pnl") -> dict[str, float]:
    if trades_df.empty:
        return {"pf": 0.0, "max_dd_pct": 0.0, "mean_pnl": 0.0, "trades": 0.0}
    sorted_df = trades_df.sort_values(["day", "bar_index"])
    pnls = sorted_df[pnl_col].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    m["trades"] = float(len(pnls))
    m["mean_pnl"] = float(np.mean(pnls))
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # === Determine best variant from C1 ===
    c1_json = os.path.join(args.c1_dir, "directional_variants.json")
    with open(c1_json) as f:
        c1 = json.load(f)
    best_variant = args.variant or c1["best_oos_variant"]
    print(f"Pressure-testing variant: {best_variant}")
    print(f"  C1 in-sample PF: {c1['in_sample_variant_metrics'][best_variant]['pf']:.3f}")
    print(f"  C1 OOS PF: {c1['oos_variant_metrics'][best_variant]['pf']:.3f}")

    # === Load variant trade CSVs ===
    in_sample_csv = os.path.join(args.c1_dir, f"in_sample_trades_{best_variant}.csv")
    oos_csv = os.path.join(args.c1_dir, f"oos_trades_{best_variant}.csv")
    if not (os.path.exists(in_sample_csv) and os.path.exists(oos_csv)):
        print(f"ERROR: missing trade CSVs for variant {best_variant} in {args.c1_dir}")
        return 1
    in_sample_trades = pd.read_csv(in_sample_csv)
    oos_trades = pd.read_csv(oos_csv)
    print(f"  Loaded in-sample trades: {len(in_sample_trades)}, OOS trades: {len(oos_trades)}")

    # === Test 1: Random-direction ablation ===
    print()
    print("=" * 100)
    print("Test 1 -- Random-direction ablation (variant edge from gate or direction?)")
    print("=" * 100)

    print("Loading V2Dataset (needed to recompute PnLs with random directions)...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()

    # Load OOF for in-sample lookups (has time_stop_pnl_call / time_stop_pnl_put per bar)
    oof = load_pickle(os.path.join(args.baseline_run_dir, "oof_predictions.pkl"))
    oof_lookup = {(str(r["day"]), int(r["bar_index"])): r for _, r in oof.iterrows()}

    # For OOS, we don't have OOF — need to look up via build_labeled_day + select_contract
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds_meta = list(bundle["meta"]["folds"])
    fold4 = folds_meta[-1]
    fold4_test = list(fold4["test_days"])
    fold4_dir = os.path.join(args.baseline_run_dir, "folds", str(int(fold4["fold_idx"])))
    fold4_calib = load_json(os.path.join(fold4_dir, "calibration.json"))
    feature_names = list(bundle["meta"]["feature_names"])
    manifest = load_pickle(os.path.join(args.baseline_run_dir, "manifest.pkl"))
    direction_mode = manifest.get("direction_mode", "teacher_if_triggered_else_put")
    score_mode = manifest.get("score_mode", "product")
    side_score_weight = float(manifest.get("side_score_weight", 0.15))

    # Build OOS export to look up call/put PnLs
    oos_days = _identify_oos_days(ds, fold4_test)
    print(f"  Building OOS lookup data for {len(oos_days)} days...")
    df_oos, _ = _build_oos_export_rows(ds, cfg, oos_days, args.equity)
    df_oos_pred = _apply_layer2_models(
        df_oos, fold4_dir, feature_names, direction_mode, score_mode, side_score_weight,
        float(fold4_calib["entry_threshold"]), float(fold4_calib["side_threshold"]),
    )
    oos_lookup = {(str(r["day"]), int(r["bar_index"])): r for _, r in df_oos_pred.iterrows()}

    def _random_dir_pf(trades_df: pd.DataFrame, lookup: dict, n_seeds: int) -> dict:
        pfs = []
        for s in range(args.seed, args.seed + n_seeds):
            rng = np.random.default_rng(s + 70000)
            pnls = []
            for _, t in trades_df.iterrows():
                key = (str(t["day"]), int(t["bar_index"]))
                row = lookup.get(key)
                if row is None:
                    continue
                rdir = "call" if rng.random() < 0.5 else "put"
                col = "time_stop_pnl_call" if rdir == "call" else "time_stop_pnl_put"
                p = row.get(col)
                if p is None or (isinstance(p, float) and not np.isfinite(p)):
                    continue
                pnls.append(float(p))
            if not pnls:
                continue
            m = replay_metrics_from_pnls(pnls, args.equity)
            pfs.append(float(m["pf"]))
        return {
            "n_seeds": int(len(pfs)),
            "pf_mean": float(np.mean(pfs)) if pfs else float("nan"),
            "pf_std": float(np.std(pfs)) if pfs else float("nan"),
            "pf_min": float(np.min(pfs)) if pfs else float("nan"),
            "pf_max": float(np.max(pfs)) if pfs else float("nan"),
            "all_pfs": [float(x) for x in pfs],
        }

    rd_in_sample = _random_dir_pf(in_sample_trades, oof_lookup, args.n_seeds)
    rd_oos = _random_dir_pf(oos_trades, oos_lookup, args.n_seeds)
    in_sample_pf = c1["in_sample_variant_metrics"][best_variant]["pf"]
    oos_pf = c1["oos_variant_metrics"][best_variant]["pf"]

    print(f"  In-sample variant PF: {in_sample_pf:.3f}; random-direction over {args.n_seeds} seeds: "
          f"mean {rd_in_sample['pf_mean']:.3f}, range [{rd_in_sample['pf_min']:.3f}, {rd_in_sample['pf_max']:.3f}]")
    print(f"  OOS variant PF: {oos_pf:.3f}; random-direction over {args.n_seeds} seeds: "
          f"mean {rd_oos['pf_mean']:.3f}, range [{rd_oos['pf_min']:.3f}, {rd_oos['pf_max']:.3f}]")

    # === Test 2: Per-fold + OOS breakdown ===
    print()
    print("=" * 100)
    print("Test 2 -- Per-fold + OOS PF breakdown")
    print("=" * 100)
    per_fold = c1["in_sample_variant_metrics"][best_variant]["per_fold"]
    print(f"{'fold':<6}{'PF':>10}{'DD%':>10}{'mean$':>10}{'trades':>10}")
    min_fold_pf = float("inf")
    min_fold_idx = None
    for fi_str in sorted(per_fold, key=lambda x: int(x)):
        m = per_fold[fi_str]
        fi = int(fi_str)
        print(f"{fi:<6}{m['pf']:>10.3f}{m['max_dd_pct']:>10.1f}{m['mean_pnl']:>10.0f}{int(m['trades']):>10}")
        if m["pf"] < min_fold_pf:
            min_fold_pf = float(m["pf"])
            min_fold_idx = fi
    print(f"OOS:  PF={oos_pf:.3f} DD={c1['oos_variant_metrics'][best_variant]['max_dd_pct']:.1f}% "
          f"trades={int(c1['oos_variant_metrics'][best_variant]['trades'])}")
    print(f"  Worst in-sample fold: {min_fold_idx} (PF {min_fold_pf:.3f})")

    # === Test 3: Slippage stress ===
    print()
    print("=" * 100)
    print(f"Test 3 -- Slippage stress on variant {best_variant}")
    print("=" * 100)
    print(f"{'slip $/RT':>10}{'in_PF':>8}{'in_DD%':>9}{'in_mean$':>10}{'oos_PF':>8}{'oos_DD%':>9}{'oos_mean$':>10}")
    slip_results = {}
    for slip in SLIPPAGE_GRID:
        in_adj = in_sample_trades.copy()
        in_adj["pnl_adj"] = in_adj["pnl"].astype(float) - slip
        in_m = _agg(in_adj.assign(pnl=in_adj["pnl_adj"]).drop(columns=["pnl_adj"]), args.equity)
        oos_adj = oos_trades.copy()
        oos_adj["pnl_adj"] = oos_adj["pnl"].astype(float) - slip
        oos_m = _agg(oos_adj.assign(pnl=oos_adj["pnl_adj"]).drop(columns=["pnl_adj"]), args.equity)
        slip_results[slip] = {"in_sample": in_m, "oos": oos_m}
        print(f"{slip:>10.0f}{in_m['pf']:>8.3f}{in_m['max_dd_pct']:>9.1f}{in_m['mean_pnl']:>10.0f}"
              f"{oos_m['pf']:>8.3f}{oos_m['max_dd_pct']:>9.1f}{oos_m['mean_pnl']:>10.0f}")

    # === Verdict ===
    print()
    print("=" * 100)
    print("Stage C2 Verdict")
    print("=" * 100)
    rd_oos_mean = rd_oos["pf_mean"]
    rd_in_sample_mean = rd_in_sample["pf_mean"]
    slip25_oos_pf = slip_results[25.0]["oos"]["pf"]

    direction_robust = (
        np.isfinite(rd_oos_mean) and rd_oos_mean < oos_pf * 0.85
        and np.isfinite(rd_in_sample_mean) and rd_in_sample_mean < in_sample_pf * 0.85
    )
    fold_robust = min_fold_pf >= 0.80
    slip_robust = slip25_oos_pf >= 1.0

    print(f"  Direction robust (random < variant*0.85): {direction_robust} "
          f"(in_sample {rd_in_sample_mean:.3f} vs {in_sample_pf*0.85:.3f}; OOS {rd_oos_mean:.3f} vs {oos_pf*0.85:.3f})")
    print(f"  Fold robust (min fold PF >= 0.80): {fold_robust} (worst {min_fold_idx}={min_fold_pf:.3f})")
    print(f"  Slip robust (OOS PF at $25 RT >= 1.0): {slip_robust} ({slip25_oos_pf:.3f})")

    if direction_robust and fold_robust and slip_robust:
        verdict = "ROBUST -- variant survives all three pressure tests; production candidate"
    elif not direction_robust:
        verdict = (f"FRAGILE -- random-direction matches/exceeds variant PF; directional logic is cosmetic")
    elif not fold_robust:
        verdict = (f"FRAGILE -- fold {min_fold_idx} regressed to PF {min_fold_pf:.3f} < 0.80")
    elif not slip_robust:
        verdict = (f"FRAGILE -- OOS PF at $25/RT slippage drops to {slip25_oos_pf:.3f} < 1.0")
    else:
        verdict = "SOFT -- partial robustness across the three tests"
    print(f"  VERDICT: {verdict}")

    # === Save ===
    payload = {
        "meta": {
            "best_variant": best_variant,
            "in_sample_pf": float(in_sample_pf),
            "oos_pf": float(oos_pf),
            "n_random_direction_seeds": int(args.n_seeds),
        },
        "random_direction_in_sample": rd_in_sample,
        "random_direction_oos": rd_oos,
        "per_fold_in_sample": per_fold,
        "oos_metrics": c1["oos_variant_metrics"][best_variant],
        "slippage_stress": {
            f"slip_{int(s)}": {"in_sample": slip_results[s]["in_sample"], "oos": slip_results[s]["oos"]}
            for s in SLIPPAGE_GRID
        },
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "directional_pressure_test.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
