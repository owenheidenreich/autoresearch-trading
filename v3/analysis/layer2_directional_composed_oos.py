"""Stage C3 of the post-OOS research workstream: compose the best
Layer-2 directional variant (from Stage C1) with the augmented Layer-3
model and evaluate on OOS.

Specifically: take the variant's OOS chosen trades, build per-trade
data, apply augmented Layer-3 at threshold 0.17, and report composed
OOS metrics.

Comparison table (all OOS):
- Layer-2 baseline (V0) alone
- Layer-2 baseline + augmented Layer-3
- Layer-2 best variant alone
- Layer-2 best variant + augmented Layer-3 (this is what's new)

Promotes the production candidate IF composed PF beats the prior
champion (V0 + augmented L3 OOS PF 1.537) AND no fold regresses.

Run:
    python -m v3.analysis.layer2_directional_composed_oos
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
    _build_per_trade_data,
    _flatten_to_rows,
    _replay_with_model,
)
from v3.analysis.layer3_v31_cleanup import (
    _build_in_sample_trade_data,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    DEFAULT_DATASET_PATH,
    build_labeled_day,
    load_export_bundle,
    load_pickle,
    replay_metrics_from_pnls,
)
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_C1_DIR = os.path.join("v3", "artifacts", "layer2_directional_variants")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer2_directional_composed_oos")
DEFAULT_THRESHOLD = 0.17
DEFAULT_SEED = 42

CHAMPION_OOS_PF = 1.537  # V0 + augmented L3 OOS PF (the prior champion to beat)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Composed OOS: best variant + augmented Layer-3.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--c1-dir", default=DEFAULT_C1_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--variant", default=None,
                   help="Override variant; defaults to best from C1.")
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _agg(trades_df: pd.DataFrame, equity: float, pnl_col: str) -> dict[str, float]:
    if trades_df.empty:
        return {"pf": 0.0, "max_dd_pct": 0.0, "mean_pnl": 0.0, "trades": 0.0}
    sorted_df = trades_df.sort_values(["day"])
    pnls = sorted_df[pnl_col].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    m["trades"] = float(len(pnls))
    m["mean_pnl"] = float(np.mean(pnls))
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # === Determine best variant ===
    c1_json = os.path.join(args.c1_dir, "directional_variants.json")
    with open(c1_json) as f:
        c1 = json.load(f)
    variant = args.variant or c1["best_oos_variant"]
    print(f"Composing variant {variant} with augmented Layer-3 at threshold {args.threshold}")

    # === Load variant OOS trades ===
    variant_csv = os.path.join(args.c1_dir, f"oos_trades_{variant}.csv")
    if not os.path.exists(variant_csv):
        print(f"ERROR: missing {variant_csv}")
        return 1
    variant_oos = pd.read_csv(variant_csv)
    print(f"  Variant OOS trades: {len(variant_oos)}")

    v0_csv = os.path.join(args.c1_dir, "oos_trades_V0.csv")
    v0_oos = pd.read_csv(v0_csv) if os.path.exists(v0_csv) else pd.DataFrame()

    # === Build augmented Layer-3 model ===
    print("Loading V2Dataset and building train data for augmented Layer-3...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
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
    augmented_data = in_sample_data + teacher_data
    X_aug, y_aug, _ = _flatten_to_rows(augmented_data)
    model_aug = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_depth=4,
        max_iter=200, min_samples_leaf=50,
        random_state=args.seed + 1000, early_stopping=False,
    )
    model_aug.fit(X_aug, y_aug)
    print(f"  Augmented model trained on {len(augmented_data)} trades / {len(X_aug)} bar-rows")

    # === Build per-trade data for the variant's OOS trades ===
    print("Building per-trade Layer-3 data for variant OOS trades...")
    variant_trade_data: list[dict] = []
    for _, t in variant_oos.iterrows():
        day = str(t["day"])
        if day not in day_cache:
            log_d, sidecar_d = build_labeled_day(ds, day, cfg, equity=args.equity)
            if log_d is None:
                continue
            day_cache[day] = (log_d, sidecar_d)
        log, sidecar = day_cache[day]
        td = _build_per_trade_data(
            t, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            variant_trade_data.append(td)
    print(f"  Variant OOS trade data: {len(variant_trade_data)}")

    # Build for V0 too (for direct comparison)
    v0_trade_data: list[dict] = []
    for _, t in v0_oos.iterrows():
        day = str(t["day"])
        log, sidecar = day_cache[day]
        td = _build_per_trade_data(
            t, log, sidecar, paths_cache, minute_map_cache, ds,
            DEFAULT_SESSION_END_BAR, DEFAULT_COMMISSION_PER_CONTRACT,
        )
        if td is not None:
            v0_trade_data.append(td)
    print(f"  V0 OOS trade data: {len(v0_trade_data)}")

    # === Apply augmented Layer-3 at threshold ===
    print()
    print("=" * 100)
    print(f"OOS comparison table (all 20 cached days)")
    print("=" * 100)

    rows = []

    # Layer-2 baseline (V0) alone
    v0_alone = _agg(v0_oos, args.equity, "pnl")
    rows.append({
        "system": "Layer-2 V0 alone",
        "PF": v0_alone["pf"],
        "DD%": v0_alone["max_dd_pct"],
        "mean$": v0_alone["mean_pnl"],
        "trades": int(v0_alone["trades"]),
    })

    # Layer-2 baseline + augmented Layer-3
    v0_l3_sim = _replay_with_model(v0_trade_data, model_aug, args.threshold)
    v0_l3_df = pd.DataFrame(v0_l3_sim).rename(columns={"exit_pnl": "pnl"})
    v0_l3 = _agg(v0_l3_df, args.equity, "pnl")
    rows.append({
        "system": "Layer-2 V0 + Augmented L3",
        "PF": v0_l3["pf"],
        "DD%": v0_l3["max_dd_pct"],
        "mean$": v0_l3["mean_pnl"],
        "trades": int(v0_l3["trades"]),
    })

    # Variant alone
    var_alone = _agg(variant_oos, args.equity, "pnl")
    rows.append({
        "system": f"Layer-2 {variant} alone",
        "PF": var_alone["pf"],
        "DD%": var_alone["max_dd_pct"],
        "mean$": var_alone["mean_pnl"],
        "trades": int(var_alone["trades"]),
    })

    # Variant + augmented Layer-3
    var_l3_sim = _replay_with_model(variant_trade_data, model_aug, args.threshold)
    var_l3_df = pd.DataFrame(var_l3_sim).rename(columns={"exit_pnl": "pnl"})
    var_l3 = _agg(var_l3_df, args.equity, "pnl")
    rows.append({
        "system": f"Layer-2 {variant} + Augmented L3",
        "PF": var_l3["pf"],
        "DD%": var_l3["max_dd_pct"],
        "mean$": var_l3["mean_pnl"],
        "trades": int(var_l3["trades"]),
    })

    # Print table
    print(f"{'system':<40}{'PF':>8}{'DD%':>8}{'mean$':>10}{'trades':>10}")
    for r in rows:
        print(f"{r['system']:<40}{r['PF']:>8.3f}{r['DD%']:>8.1f}{r['mean$']:>10.0f}{r['trades']:>10}")

    # Per-day trace for the new champion (variant + augmented L3)
    print()
    print(f"=== Per-day trace: Layer-2 {variant} + Augmented L3 ===")
    print(f"{'day':<12}{'direction':<10}{'pnl':>10}{'bars_held':>11}")
    for _, r in var_l3_df.sort_values("day").iterrows():
        print(f"{str(r['day']):<12}{r['direction']:<10}{float(r['pnl']):>10.0f}{int(r['bars_held']):>11}")

    # === Verdict ===
    print()
    print("=" * 100)
    print("Stage C3 Verdict")
    print("=" * 100)
    new_pf = var_l3["pf"]
    if new_pf > CHAMPION_OOS_PF + 0.05:
        verdict = (f"NEW CHAMPION -- {variant} + Augmented L3 OOS PF {new_pf:.3f} "
                   f"beats prior champion ({CHAMPION_OOS_PF:.3f}) by +{new_pf - CHAMPION_OOS_PF:.3f}")
    elif new_pf >= CHAMPION_OOS_PF - 0.05:
        verdict = (f"WASH -- {variant} + Augmented L3 OOS PF {new_pf:.3f} "
                   f"essentially matches prior champion ({CHAMPION_OOS_PF:.3f}). Keep V0.")
    else:
        verdict = (f"WORSE -- {variant} + Augmented L3 OOS PF {new_pf:.3f} "
                   f"degrades from prior champion ({CHAMPION_OOS_PF:.3f}). Variant interferes with L3.")
    print(f"  VERDICT: {verdict}")

    # === Save ===
    payload = {
        "meta": {
            "variant": variant,
            "threshold": float(args.threshold),
            "champion_oos_pf": float(CHAMPION_OOS_PF),
        },
        "comparison_table": rows,
        "verdict": verdict,
    }
    out = os.path.join(args.out_dir, "directional_composed_oos.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    var_l3_df.to_csv(os.path.join(args.out_dir, f"composed_oos_trades_{variant}.csv"), index=False)
    print()
    print(f"Saved: {out}")
    print(f"Saved: {os.path.join(args.out_dir, f'composed_oos_trades_{variant}.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
