"""Phase R4 — Per-window V0 vs V1 evaluation across 13 rolling windows.

For each window:
  1. Load per-window OOS predictions (from R3)
  2. Load per-window calibration thresholds
  3. For each OOS day: per_day_choice → variant transform → PnL
  4. Compute per-window metrics per variant (V0, V1, V2, V3, V4)

Aggregates:
- Per-variant: mean/std/median of per-window PF
- Per-variant: strict win count vs V1 and margin win count vs V1
- Margin win = beats V1 by at least +0.10 PF in a window
- Cross-window: aggregate PF, DD, mean$, total trades
- Regime stratification: tag each window by SPX direction + VIX regime

Output: v3/artifacts/rolling_directional_eval/eval.json + per-window
trade CSVs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from v3.analysis.layer2_directional_variants import (
    VARIANTS, _apply_variant,
)
from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    build_labeled_day,
    compute_time_stop_pnl_for_direction,
    effective_direction,
    load_json,
    load_pickle,
    per_day_choice,
    replay_metrics_from_pnls,
    save_json,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROLLING_DIR = os.path.join("v3", "artifacts", "rolling_l2")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "rolling_directional_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rolling-dir", default=DEFAULT_ROLLING_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    return p.parse_args()


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


def evaluate_window(
    window_idx: int, oos_pred: pd.DataFrame, thresholds: dict,
    day_cache: dict, manifest: dict, equity: float,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """For one rolling window, apply all variants and return per-variant
    (trades list, per-variant metrics)."""
    entry_t = float(thresholds["entry_threshold"])
    side_t = float(thresholds["side_threshold"])
    direction_mode = manifest["direction_mode"]
    score_mode = manifest["score_mode"]
    side_score_weight = float(manifest["side_score_weight"])

    # Force effective_direction on the OOS predictions
    oos_pred = oos_pred.copy()
    oos_pred["effective_direction"] = oos_pred.apply(
        effective_direction, axis=1,
        direction_mode=direction_mode, policy_mode="scalar_side",
    )

    variant_trades: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for day, day_rows in oos_pred.groupby("day"):
        chosen = per_day_choice(
            day_rows, entry_t, side_t,
            score_mode=score_mode, side_score_weight=side_score_weight,
            policy_mode="scalar_side", direction_mode=direction_mode,
        )
        if chosen is None:
            continue
        log, sidecar = day_cache.get(str(day), (None, None))
        if log is None or sidecar is None:
            continue
        bar_index = int(chosen["bar_index"])
        bar = next((b for b in log.bars if b.bar_index == bar_index), None)
        if bar is None:
            continue
        for variant in VARIANTS:
            final_dir, accepted = _apply_variant(chosen, variant, side_t)
            if not accepted or final_dir is None:
                continue
            pnl = compute_time_stop_pnl_for_direction(bar, sidecar, final_dir)
            if pnl is None:
                continue
            variant_trades[variant].append({
                "day": str(day), "bar_index": bar_index,
                "direction": final_dir, "pnl": float(pnl),
                "window_idx": window_idx,
                "sigma_pos": float(chosen.get("sigma_pos", 0.0)),
                "side_conf": float(chosen.get("side_conf", 0.0)),
                "original_direction": str(chosen["effective_direction"]),
            })
    variant_metrics = {
        v: _agg(variant_trades[v], equity) for v in VARIANTS
    }
    return variant_trades, variant_metrics


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = load_json(os.path.join(args.rolling_dir, "manifest.json"))
    windows_info = manifest["windows"]
    print(f"Loaded manifest: {len(windows_info)} windows", flush=True)

    print("Loading V2Dataset + building day_cache for all OOS days...", flush=True)
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    day_cache: dict = {}
    all_oos_days = set()
    for w in windows_info:
        # Per-window OOS dates computed from rolling_windows earlier
        pred_path = os.path.join(args.rolling_dir, f"window_{w['window_idx']:02d}", "oos_predictions.pkl")
        oos_pred = load_pickle(pred_path)
        all_oos_days.update(oos_pred["day"].unique().tolist())
    print(f"  Total OOS days to label: {len(all_oos_days)}", flush=True)

    sorted_days = sorted(all_oos_days)
    for i, day in enumerate(sorted_days):
        if day not in day_cache:
            log, sidecar = build_labeled_day(ds, day, cfg, equity=args.equity)
            day_cache[day] = (log, sidecar)
        if (i + 1) % 100 == 0:
            print(f"    labeled {i+1}/{len(sorted_days)} days", flush=True)
    print(f"  day_cache complete: {len(day_cache)} days", flush=True)

    # Per-window evaluation
    print(flush=True)
    print("=" * 100)
    print("Per-window variant evaluation")
    print("=" * 100, flush=True)
    print(f"{'win':<4}{'OOS':<26}{'var':<5}{'PF':>9}{'DD%':>8}{'mean$':>10}{'n':>6}{'call%':>8}",
          flush=True)

    all_variant_trades: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    per_window_per_variant: list[dict] = []

    for w in windows_info:
        wi = w["window_idx"]
        calib_path = os.path.join(args.rolling_dir, f"window_{wi:02d}", "calibration.json")
        pred_path = os.path.join(args.rolling_dir, f"window_{wi:02d}", "oos_predictions.pkl")
        thresholds = load_json(calib_path)
        oos_pred = load_pickle(pred_path)

        variant_trades, variant_metrics = evaluate_window(
            wi, oos_pred, thresholds, day_cache, manifest, args.equity,
        )

        oos_range = f"{w['oos_start']}..{w['oos_end']}"
        for v in VARIANTS:
            m = variant_metrics[v]
            print(f"{wi:<4}{oos_range:<26}{v:<5}"
                  f"{m['pf']:>9.3f}{m['max_dd_pct']:>8.1f}"
                  f"{m['mean_pnl']:>10.0f}{int(m['trades']):>6}{m['call_pct']*100:>7.1f}%",
                  flush=True)
            per_window_per_variant.append({
                "window_idx": wi,
                "oos_start": w["oos_start"], "oos_end": w["oos_end"],
                "variant": v,
                **m,
            })
            all_variant_trades[v].extend(variant_trades[v])

    # === Cross-window aggregates ===
    print(flush=True)
    print("=" * 100)
    print("Cross-window aggregates per variant")
    print("=" * 100, flush=True)
    print(f"{'var':<5}{'agg_PF':>10}{'agg_DD%':>10}{'mean$':>10}{'trades':>8}"
          f"{'win_pf_mean':>14}{'win_pf_med':>13}{'win_pf_std':>12}"
          f"{'>V1':>8}{'>V1+0.10':>12}", flush=True)

    v1_per_window_pf = {
        r["window_idx"]: r["pf"] for r in per_window_per_variant if r["variant"] == "V1"
    }

    aggregate_per_variant = {}
    for v in VARIANTS:
        trades = all_variant_trades[v]
        agg = _agg(trades, args.equity)
        # Per-window PF values
        per_win_pfs = [
            r["pf"] for r in per_window_per_variant if r["variant"] == v
        ]
        pf_arr = np.array(per_win_pfs, dtype=np.float64)
        strict_win_count = sum(
            1 for r in per_window_per_variant
            if r["variant"] == v
            and r["pf"] > v1_per_window_pf.get(r["window_idx"], -np.inf)
        )
        margin_win_count = sum(
            1 for r in per_window_per_variant
            if r["variant"] == v
            and r["pf"] > v1_per_window_pf.get(r["window_idx"], -np.inf) + 0.10
        )
        print(f"{v:<5}"
              f"{agg['pf']:>10.3f}{agg['max_dd_pct']:>10.1f}{agg['mean_pnl']:>10.0f}"
              f"{int(agg['trades']):>8}"
              f"{pf_arr.mean():>14.3f}{np.median(pf_arr):>13.3f}{pf_arr.std(ddof=1):>12.3f}"
              f"{strict_win_count:>8}/{len(per_win_pfs)}"
              f"{margin_win_count:>12}/{len(per_win_pfs)}",
              flush=True)
        aggregate_per_variant[v] = {
            "agg_pf": agg["pf"], "agg_dd_pct": agg["max_dd_pct"],
            "agg_mean_pnl": agg["mean_pnl"], "agg_trades": int(agg["trades"]),
            "agg_call_pct": agg["call_pct"],
            "per_window_pf_mean": float(pf_arr.mean()),
            "per_window_pf_median": float(np.median(pf_arr)),
            "per_window_pf_std": float(pf_arr.std(ddof=1)),
            "per_window_pf_min": float(pf_arr.min()),
            "per_window_pf_max": float(pf_arr.max()),
            "strict_win_vs_V1_count": strict_win_count,
            "margin_win_vs_V1_count": margin_win_count,
            "total_windows": len(per_win_pfs),
        }

    # === Save ===
    payload = {
        "meta": {
            "rolling_dir": args.rolling_dir,
            "n_windows": len(windows_info),
            "equity": float(args.equity),
        },
        "per_window_per_variant": per_window_per_variant,
        "aggregate_per_variant": aggregate_per_variant,
    }
    out = os.path.join(args.out_dir, "eval.json")
    save_json(out, payload)
    for v in VARIANTS:
        df_v = pd.DataFrame(all_variant_trades[v])
        if not df_v.empty:
            df_v.to_csv(os.path.join(args.out_dir, f"trades_{v}.csv"), index=False)
    print(flush=True)
    print(f"Saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
