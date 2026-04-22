"""Stage C1 of the post-OOS research workstream: Layer-2 directional
variants applied as inference-time post-processing on the frozen
detach-side model. Tests five variants on in-sample (5 folds) AND
OOS (20 cached days):

| Variant | Description |
|---|---|
| V0 | Baseline (current): teacher_if_triggered_else_put, no overrides |
| V1 | Always_put: override every chosen direction to "put" |
| V2 | Sigma_pos veto on calls: drop trade if direction=="call" AND sigma_pos > 0 |
| V3 | Conviction-asymmetric: if direction=="call" AND side_conf < 1.5*side_threshold, override to "put" |
| V4 | Combined V3 + V2: V3 first, then V2 |

Method:
- In-sample: load oof_predictions.pkl (per-fold predictions). For each
  test fold, apply per_day_choice with baseline mode to pick a bar,
  then apply the variant to compute final direction/accept.
- OOS: re-run inference on 20 cached days (same as layer3_oos_validation),
  apply per_day_choice baseline, apply variant.

For each (variant, universe), report PF/DD/mean/call%/per-fold (in-sample)
and OOS PF/DD/mean/call%.

Verdict gates:
- TERMINAL FAIL: NO variant lifts OOS Layer-2-alone PF >= 1.0.
- V1-WIN: V1 (always_put) wins by huge margin -> call signal worthless.
- V0-WIN: baseline still best -> directional rework doesn't help.
- BEST-VARIANT-WINS: some Vk produces OOS PF >= 1.10 AND in-sample
  doesn't crash. That's the production candidate for C2 pressure-test.

Run:
    python -m v3.analysis.layer2_directional_variants
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
    effective_direction,
    load_export_bundle,
    load_json,
    load_pickle,
    per_day_choice,
    replay_metrics_from_pnls,
    teacher_direction_hint_from_row,
)


DEFAULT_BASELINE_RUN = os.path.join("v3", "artifacts", "layer2_shared_enc_fixedq_detach")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer2_directional_variants")
CALL_CONVICTION_MULTIPLIER = 1.5  # for V3
SIGMA_POS_VETO_THRESHOLD = 0.0    # for V2

VARIANTS = ("V0", "V1", "V2", "V3", "V4")
VARIANT_DESCRIPTIONS = {
    "V0": "baseline (teacher_if_triggered_else_put)",
    "V1": "always_put (override every direction)",
    "V2": "sigma_pos veto on calls (drop if direction==call AND sigma_pos > 0)",
    "V3": f"conviction-asymmetric (call->put if side_conf < {CALL_CONVICTION_MULTIPLIER}*side_threshold)",
    "V4": "combined V3 then V2",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-2 directional variants.")
    p.add_argument("--baseline-run-dir", default=DEFAULT_BASELINE_RUN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    return p.parse_args()


def _apply_variant(
    chosen_row: pd.Series, variant: str, side_threshold: float,
) -> tuple[str | None, bool]:
    """Apply a variant transform to chosen_row's effective_direction.

    Returns:
        (final_direction, accepted) -- final_direction in {'call', 'put', None}.
        If accepted=False, the trade is dropped entirely (no PnL).
    """
    if chosen_row is None:
        return None, False
    direction = str(chosen_row.get("effective_direction", ""))
    if direction not in ("call", "put"):
        return None, False
    sigma_pos = float(chosen_row.get("sigma_pos", 0.0))
    side_conf = float(chosen_row.get("side_conf", 0.0))

    if variant == "V0":
        return direction, True

    if variant == "V1":
        return "put", True

    if variant == "V2":
        if direction == "call" and sigma_pos > SIGMA_POS_VETO_THRESHOLD:
            return None, False
        return direction, True

    if variant == "V3":
        if direction == "call" and side_conf < CALL_CONVICTION_MULTIPLIER * side_threshold:
            return "put", True
        return direction, True

    if variant == "V4":
        # V3 first
        if direction == "call" and side_conf < CALL_CONVICTION_MULTIPLIER * side_threshold:
            direction = "put"
        # V2 second
        if direction == "call" and sigma_pos > SIGMA_POS_VETO_THRESHOLD:
            return None, False
        return direction, True

    raise ValueError(f"Unknown variant {variant}")


def _select_trades_with_variants(
    df: pd.DataFrame, day_cache: dict, entry_threshold: float, side_threshold: float,
    score_mode: str, side_score_weight: float, direction_mode: str,
    fold_id: int | None = None,
) -> dict[str, list[dict]]:
    """For each variant, produce the variant's chosen trades by applying
    per_day_choice + variant transform per day."""
    out: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for day, day_rows in df.groupby("day"):
        chosen = per_day_choice(
            day_rows, entry_threshold, side_threshold,
            score_mode=score_mode, side_score_weight=side_score_weight,
            policy_mode="scalar_side", direction_mode=direction_mode,
        )
        if chosen is None:
            continue
        log, sidecar = day_cache.get(day, (None, None))
        if log is None or sidecar is None:
            continue
        bar_index = int(chosen["bar_index"])
        bar = next((b for b in log.bars if b.bar_index == bar_index), None)
        if bar is None:
            continue
        for variant in VARIANTS:
            final_dir, accepted = _apply_variant(chosen, variant, side_threshold)
            if not accepted or final_dir is None:
                continue
            pnl = compute_time_stop_pnl_for_direction(bar, sidecar, final_dir)
            if pnl is None:
                continue
            row = {
                "day": day, "bar_index": bar_index,
                "direction": final_dir,
                "pnl": float(pnl),
                "fold_idx": fold_id if fold_id is not None else -1,
                "sigma_pos": float(chosen.get("sigma_pos", 0.0)),
                "side_conf": float(chosen.get("side_conf", 0.0)),
                "original_direction": str(chosen["effective_direction"]),
            }
            out[variant].append(row)
    return out


def _agg(trades: list[dict], equity: float) -> dict[str, float]:
    if not trades:
        return {"pf": 0.0, "max_dd_pct": 0.0, "mean_pnl": 0.0, "trades": 0.0, "call_pct": 0.0}
    df = pd.DataFrame(trades).sort_values(["day", "bar_index"])
    pnls = df["pnl"].astype(float).tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    m["trades"] = float(len(pnls))
    m["mean_pnl"] = float(np.mean(pnls))
    m["call_pct"] = float((df["direction"] == "call").mean())
    return m


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # === Load metadata ===
    bundle = load_export_bundle(DEFAULT_DATASET_PATH)
    folds_meta = list(bundle["meta"]["folds"])
    feature_names = list(bundle["meta"]["feature_names"])
    manifest = load_pickle(os.path.join(args.baseline_run_dir, "manifest.pkl"))
    direction_mode = manifest.get("direction_mode", "teacher_if_triggered_else_put")
    score_mode = manifest.get("score_mode", "product")
    side_score_weight = float(manifest.get("side_score_weight", 0.15))

    # Fold-specific calibration (use each fold's own thresholds for in-sample)
    fold_calibs = {}
    for f in folds_meta:
        fi = int(f["fold_idx"])
        fold_calibs[fi] = load_json(
            os.path.join(args.baseline_run_dir, "folds", str(fi), "calibration.json")
        )

    # === Load OOF predictions (in-sample) ===
    print("Loading OOF predictions...")
    oof = load_pickle(os.path.join(args.baseline_run_dir, "oof_predictions.pkl"))
    print(f"  OOF rows: {len(oof)}; folds: {sorted(oof['fold_idx'].unique())}")

    # === Build day_cache for in-sample ===
    print("Loading V2Dataset and building per-day labels...")
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    day_cache: dict = {}

    # We need labeled days for ALL test_days in all folds + OOS
    in_sample_test_days = set()
    for f in folds_meta:
        in_sample_test_days.update(f["test_days"])

    print(f"  In-sample test days: {len(in_sample_test_days)}")
    for i, day in enumerate(sorted(in_sample_test_days)):
        if day not in day_cache:
            log, sidecar = build_labeled_day(ds, day, cfg, equity=args.equity)
            day_cache[day] = (log, sidecar)
        if (i + 1) % 50 == 0:
            print(f"    labeled {i+1}/{len(in_sample_test_days)} in-sample days")

    # === Per-fold in-sample replay with variants ===
    print()
    print("=" * 100)
    print("In-sample variant replay (per fold)")
    print("=" * 100)
    fold_variant_trades: dict[int, dict[str, list[dict]]] = {}
    for fold in folds_meta:
        fi = int(fold["fold_idx"])
        calib = fold_calibs[fi]
        entry_t = float(calib["entry_threshold"])
        side_t = float(calib["side_threshold"])
        # OOF rows for this fold's test_days
        fold_oof = oof[oof["fold_idx"] == fi]
        # Limit to test_days only (the OOF predictions for this fold come from this fold's model
        # but cover other days too via the OOF protocol — we want test_days specifically)
        fold_test = set(fold["test_days"])
        fold_oof = fold_oof[fold_oof["day"].isin(fold_test)]
        # Compute effective_direction (OOF may already have this, but force consistency)
        fold_oof = fold_oof.copy()
        fold_oof["effective_direction"] = fold_oof.apply(
            effective_direction, axis=1,
            direction_mode=direction_mode, policy_mode="scalar_side",
        )
        v_trades = _select_trades_with_variants(
            fold_oof, day_cache, entry_t, side_t, score_mode, side_score_weight,
            direction_mode, fold_id=fi,
        )
        fold_variant_trades[fi] = v_trades
        print(f"  fold {fi}: thresholds entry={entry_t:.4f} side={side_t:.4f} | "
              f"V0={len(v_trades['V0'])} V1={len(v_trades['V1'])} V2={len(v_trades['V2'])} "
              f"V3={len(v_trades['V3'])} V4={len(v_trades['V4'])}")

    # === Aggregate in-sample by variant ===
    in_sample_variant_metrics = {}
    for v in VARIANTS:
        all_trades = []
        for fi in fold_variant_trades:
            all_trades.extend(fold_variant_trades[fi][v])
        in_sample_variant_metrics[v] = _agg(all_trades, args.equity)
        in_sample_variant_metrics[v]["per_fold"] = {
            fi: _agg(fold_variant_trades[fi][v], args.equity)
            for fi in sorted(fold_variant_trades)
        }

    print()
    print("=" * 100)
    print("In-sample aggregate (each variant; chronologically sorted across folds)")
    print("=" * 100)
    print(f"{'variant':<6}{'desc':<45}{'PF':>8}{'DD%':>8}{'mean$':>10}{'trades':>8}{'call%':>8}")
    for v in VARIANTS:
        m = in_sample_variant_metrics[v]
        print(f"{v:<6}{VARIANT_DESCRIPTIONS[v]:<45}{m['pf']:>8.3f}{m['max_dd_pct']:>8.1f}"
              f"{m['mean_pnl']:>10.0f}{int(m['trades']):>8}{m['call_pct']*100:>7.1f}%")

    print()
    print("Per-fold PF (in-sample)")
    print(f"{'fold':<6}", end="")
    for v in VARIANTS:
        print(f"{v:>10}", end="")
    print()
    for fi in sorted(fold_variant_trades):
        print(f"{fi:<6}", end="")
        for v in VARIANTS:
            pf = in_sample_variant_metrics[v]["per_fold"][fi]["pf"]
            print(f"{pf:>10.3f}", end="")
        print()

    # === OOS pipeline ===
    print()
    print("=" * 100)
    print("OOS variant replay (20 cached days, fold-4 model + calibration)")
    print("=" * 100)
    fold4 = folds_meta[-1]
    fold4_test = list(fold4["test_days"])
    fold4_dir = os.path.join(args.baseline_run_dir, "folds", str(int(fold4["fold_idx"])))
    fold4_calib = fold_calibs[int(fold4["fold_idx"])]
    entry_t = float(fold4_calib["entry_threshold"])
    side_t = float(fold4_calib["side_threshold"])

    oos_days = _identify_oos_days(ds, fold4_test)
    print(f"  OOS days: {len(oos_days)}")
    df_oos, oos_day_cache = _build_oos_export_rows(ds, cfg, oos_days, args.equity)
    day_cache.update(oos_day_cache)
    df_pred = _apply_layer2_models(
        df_oos, fold4_dir, feature_names, direction_mode, score_mode, side_score_weight,
        entry_t, side_t,
    )
    oos_variant_trades = _select_trades_with_variants(
        df_pred, oos_day_cache, entry_t, side_t, score_mode, side_score_weight,
        direction_mode, fold_id=-1,
    )
    oos_variant_metrics = {v: _agg(oos_variant_trades[v], args.equity) for v in VARIANTS}

    print(f"{'variant':<6}{'desc':<45}{'PF':>8}{'DD%':>8}{'mean$':>10}{'trades':>8}{'call%':>8}")
    for v in VARIANTS:
        m = oos_variant_metrics[v]
        print(f"{v:<6}{VARIANT_DESCRIPTIONS[v]:<45}{m['pf']:>8.3f}{m['max_dd_pct']:>8.1f}"
              f"{m['mean_pnl']:>10.0f}{int(m['trades']):>8}{m['call_pct']*100:>7.1f}%")

    # === Verdict ===
    print()
    print("=" * 100)
    print("Stage C1 Verdict")
    print("=" * 100)
    in_sample_ranked = sorted(VARIANTS, key=lambda v: -in_sample_variant_metrics[v]["pf"])
    oos_ranked = sorted(VARIANTS, key=lambda v: -oos_variant_metrics[v]["pf"])
    best_oos_variant = oos_ranked[0]
    best_oos_pf = oos_variant_metrics[best_oos_variant]["pf"]
    layer2_alone_baseline_oos = oos_variant_metrics["V0"]["pf"]
    print(f"  In-sample ranking by PF: {in_sample_ranked}")
    print(f"  OOS ranking by PF: {oos_ranked}")
    print(f"  Best OOS variant: {best_oos_variant} (PF={best_oos_pf:.3f}); baseline V0 OOS PF={layer2_alone_baseline_oos:.3f}")

    if best_oos_pf < 1.0:
        verdict = (f"TERMINAL FAIL -- no variant achieves OOS PF >= 1.0. "
                   f"Best is {best_oos_variant} at {best_oos_pf:.3f}. "
                   f"Layer-2 entry has no generalizing edge on this dataset.")
    elif best_oos_variant == "V1" and best_oos_pf - oos_variant_metrics["V0"]["pf"] >= 0.30:
        verdict = (f"V1-WIN -- always_put wins by {best_oos_pf - oos_variant_metrics['V0']['pf']:+.3f} PF. "
                   f"Call signal is worthless OOS; production becomes puts-only SPX 0DTE.")
    elif best_oos_variant == "V0":
        verdict = (f"V0-WIN -- baseline already best ({best_oos_pf:.3f}); directional rework doesn't help. "
                   f"OOS failure mode is something else.")
    else:
        verdict = (f"BEST-VARIANT-WINS -- {best_oos_variant} OOS PF {best_oos_pf:.3f} "
                   f"(vs baseline V0 {layer2_alone_baseline_oos:.3f}, lift +{best_oos_pf-layer2_alone_baseline_oos:.3f}). "
                   f"Production candidate for Stage C2 pressure-test.")
    print(f"  VERDICT: {verdict}")

    # === Save ===
    payload = {
        "meta": {
            "baseline_run_dir": args.baseline_run_dir,
            "n_in_sample_folds": len(folds_meta),
            "n_oos_days": int(len(oos_days)),
            "variants": {v: VARIANT_DESCRIPTIONS[v] for v in VARIANTS},
            "call_conviction_multiplier": CALL_CONVICTION_MULTIPLIER,
            "sigma_pos_veto_threshold": SIGMA_POS_VETO_THRESHOLD,
        },
        "in_sample_variant_metrics": in_sample_variant_metrics,
        "oos_variant_metrics": oos_variant_metrics,
        "verdict": verdict,
        "best_oos_variant": best_oos_variant,
    }
    out = os.path.join(args.out_dir, "directional_variants.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # Save per-variant trade CSVs for downstream stages
    for v in VARIANTS:
        in_trades = []
        for fi in fold_variant_trades:
            in_trades.extend(fold_variant_trades[fi][v])
        if in_trades:
            pd.DataFrame(in_trades).to_csv(
                os.path.join(args.out_dir, f"in_sample_trades_{v}.csv"), index=False,
            )
        if oos_variant_trades[v]:
            pd.DataFrame(oos_variant_trades[v]).to_csv(
                os.path.join(args.out_dir, f"oos_trades_{v}.csv"), index=False,
            )

    print()
    print(f"Saved: {out}")
    print(f"Saved per-variant trade CSVs in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
