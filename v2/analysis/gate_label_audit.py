"""Audit gate-label density before spending GPU on a new screen.

Examples:
    python3 -m v2.analysis.gate_label_audit --mode sparse_high_conviction --screen-mode mini
    python3 -m v2.analysis.gate_label_audit --mode bar_quality --screen-mode mini
    python3 -m v2.analysis.gate_label_audit --mode slice --folds 4
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from v2.core.chain_data import load_sidecar_cached
from v2.core.features import _FEAT_IDX
from v2.core.policy import DEFAULT_POLICY
from v2.core.walkforward import generate_folds, resolve_fold_indices
from v2.train import (
    BAR_QUALITY_PASS_THRESHOLD,
    LOOKBACK,
    _compute_bar_opportunity_quality,
    _compute_consensus_opportunity,
    _compute_sparse_high_conviction_opportunity,
    _compute_strict_opportunity,
)


def _label_for_mode(sidecar: dict, local_bar: int, mode: str) -> bool:
    if mode == "bar_quality":
        return _compute_bar_opportunity_quality(sidecar, local_bar) >= BAR_QUALITY_PASS_THRESHOLD
    if mode == "strict":
        return _compute_strict_opportunity(sidecar, local_bar)
    if mode == "consensus":
        return _compute_consensus_opportunity(sidecar, local_bar)
    if mode in {"sparse", "sparse_high_conviction"}:
        return _compute_sparse_high_conviction_opportunity(sidecar, local_bar)
    if mode == "high_threshold":
        bp = sidecar.get("bar_slice_best_pnl", sidecar.get("bar_best_pnl"))
        return bp is not None and float(bp[local_bar]) > 0.20
    if mode == "old":
        return bool(sidecar["bar_label_trade"][local_bar])
    return bool(sidecar.get("bar_slice_label_trade", sidecar["bar_label_trade"])[local_bar])


def _vix_bucket(v: float) -> str:
    if v < 0:
        return "low"
    if v < 0.5:
        return "mid"
    return "high"


def _summarize_rows(rows: list[dict]) -> dict[str, object]:
    if not rows:
        return {
            "eligible": 0,
            "labelable": 0,
            "positives": 0,
            "pos_rate": 0.0,
            "avg_best_pnl": 0.0,
            "avg_bar_quality": 0.0,
            "vix": {},
        }
    eligible = len(rows)
    labelable = sum(1 for r in rows if r["labelable"])
    positives = sum(1 for r in rows if r["positive"])
    avg_best_pnl = float(np.mean([r["best_pnl"] for r in rows if np.isfinite(r["best_pnl"])])) if rows else 0.0
    avg_bar_quality = float(np.mean([r["bar_quality"] for r in rows])) if rows else 0.0
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["vix_bucket"]].append(row)
    vix_summary = {}
    for name, items in buckets.items():
        pos = sum(1 for r in items if r["positive"])
        lab = sum(1 for r in items if r["labelable"])
        vix_summary[name] = {
            "eligible": len(items),
            "labelable": lab,
            "positives": pos,
            "pos_rate": pos / max(lab, 1),
            "avg_bar_quality": float(np.mean([r["bar_quality"] for r in items])) if items else 0.0,
        }
    return {
        "eligible": eligible,
        "labelable": labelable,
        "positives": positives,
        "pos_rate": positives / max(labelable, 1),
        "avg_best_pnl": avg_best_pnl,
        "avg_bar_quality": avg_bar_quality,
        "vix": vix_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit gate-label sparsity by fold and VIX regime.")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--mode", default="sparse_high_conviction")
    parser.add_argument("--screen-mode", choices=("latest", "mini", "full"), default="mini")
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold indices.")
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    dates = data["dates"]
    bar_of_day = data["bar_of_day"].numpy()
    features = data["X"].numpy()
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    unique_dates = sorted(set(dates))
    vix_idx = _FEAT_IDX.get("vix_regime", 14)

    explicit = None
    if args.folds:
        explicit = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    folds = generate_folds(unique_dates)
    selected = resolve_fold_indices(args.screen_mode, len(folds), explicit)

    print(f"Gate label audit: mode={args.mode} folds={selected}")
    print("-" * 72)

    overall_rows: list[dict] = []
    for fold in folds:
        if fold.fold_idx not in selected:
            continue
        test_days = set(fold.test_days)
        fold_rows: list[dict] = []
        for global_bar in range(LOOKBACK, len(dates)):
            day = dates[global_bar]
            if day not in test_days:
                continue
            local_bar = int(bar_of_day[global_bar])
            if local_bar < DEFAULT_POLICY.no_trade_before_bar or local_bar >= DEFAULT_POLICY.no_trade_after_bar:
                continue
            sidecar = load_sidecar_cached(os.path.join(sidecar_dir, f"{day}.pt"))
            labelable = bool(sidecar.get("bar_slice_labelable", sidecar["bar_labelable"])[local_bar])
            best_pnl_arr = sidecar.get("bar_slice_best_pnl", sidecar.get("bar_best_pnl"))
            best_pnl = float(best_pnl_arr[local_bar]) if best_pnl_arr is not None else float("nan")
            bar_quality = _compute_bar_opportunity_quality(sidecar, local_bar) if labelable else 0.0
            positive = _label_for_mode(sidecar, local_bar, args.mode) if labelable else False
            row = {
                "labelable": labelable,
                "positive": positive,
                "best_pnl": best_pnl,
                "bar_quality": bar_quality,
                "vix_bucket": _vix_bucket(float(features[global_bar, vix_idx])),
            }
            fold_rows.append(row)
            overall_rows.append(row)

        s = _summarize_rows(fold_rows)
        print(
            f"fold {fold.fold_idx}: eligible={s['eligible']:,} labelable={s['labelable']:,}"
            f" positives={s['positives']:,} pos_rate={s['pos_rate']:.2%}"
            f" avg_best_pnl={s['avg_best_pnl']:.4f}"
            f" avg_bar_quality={s['avg_bar_quality']:.3f}"
        )
        for bucket_name in ("low", "mid", "high"):
            bucket = s["vix"].get(bucket_name)
            if not bucket:
                continue
            print(
                f"  vix={bucket_name:<4} eligible={bucket['eligible']:,}"
                f" labelable={bucket['labelable']:,} positives={bucket['positives']:,}"
                f" pos_rate={bucket['pos_rate']:.2%}"
                f" avg_bar_quality={bucket['avg_bar_quality']:.3f}"
            )

    overall = _summarize_rows(overall_rows)
    print("-" * 72)
    print(
        f"overall: eligible={overall['eligible']:,} labelable={overall['labelable']:,}"
        f" positives={overall['positives']:,} pos_rate={overall['pos_rate']:.2%}"
        f" avg_best_pnl={overall['avg_best_pnl']:.4f}"
        f" avg_bar_quality={overall['avg_bar_quality']:.3f}"
    )


if __name__ == "__main__":
    main()
