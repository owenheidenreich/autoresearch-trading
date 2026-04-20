"""V2-pruned regime-awareness audit.

Tests whether V2-pruned's edge concentrates in a coarse subset of opening
regimes. Frozen pipeline (feature set, model, V1B gates, contract
selection, exits, one-trade-per-day, folds) — only diagnostic tooling
added.

Two methodological choices worth naming up front:

1. **Population-mean Control A, not seed-sampled.** For every strategy
   trade, we enumerate ALL V1B-admissible same-day same-side bars and
   realize the trade at each, then take the mean. This is the RNG-free
   limit of "random-bar control" — it eliminates the seed variance that
   earlier runs exposed (+2.56pp vs +0.31pp on the same 266 strategy
   trades depending on which control seed was used).

2. **Paired-delta bootstrap per bucket.** Within a regime bucket, for
   each strategy trade we compute `delta = strategy_net_pct -
   pop_ctrl_A_mean`. We then bootstrap those deltas over day-level
   resampling within the bucket. This isolates the strategy's edge
   relative to random-bar-on-the-same-day, without mixing across days.

Regime axes (4 independent univariate slices, fixed thresholds, no
train-selected tuning — the point is descriptive interpretation):

| axis | buckets | thresholds |
|---|---|---|
| `opening_gap_pct` | down / flat / up | ±30 bps |
| `first15_close_position` | low / mid / high | 0.333, 0.667 |
| `first15_range_pct` | narrow / wide | 20 bps |
| `vix_regime` | calm / low / mid / high | native discrete {-1, -.33, .33, 1} |

Decision rule (promotable bucket): gap > 0, bootstrap `frac>0` ≥ 0.90,
n ≥ 20, ≥ 3 folds contribute.

Plan upstream: [v2/docs/mechanical_baseline_v2_plan.md]. V2-pruned and
thresholding results in `v2/lab_notebook.md` (2026-04-19 / 2026-04-20).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from v2.core.chain_data import load_sidecar_cached, sidecar_path
from v2.analysis.mechanical_baseline_opening_reversion import (
    BAR_LO,
    BAR_HI,
    BaselineTrade,
    SkipRecord,
    SKIP_FIELDS,
    build_folds,
    context_spread_at_bar,
    day_ranges,
    feature_index_map,
    load_data,
    summarize_trades,
    write_csv,
    write_json,
)
from v2.analysis.mechanical_baseline_v1b_opening_reversion import (
    check_gates,
    V1BGateConfig,
)
from v2.analysis import mechanical_baseline_v2_learned_scorer as v2_mod


# ---------------------------------------------------------------------------
# Freeze V2-pruned feature set
# ---------------------------------------------------------------------------

DROPPED_FEATURES = ("vwap_reclaim_state",)


def _activate_pruned_feature_set() -> None:
    active = tuple(n for n in v2_mod.CORE_FEATURE_NAMES if n not in DROPPED_FEATURES)
    v2_mod.CORE_FEATURE_NAMES = active
    v2_mod.N_MODEL_FEATURES = len(active) + 1
    v2_mod.SIDE_INDICATOR_IDX = len(active)


# ---------------------------------------------------------------------------
# Regime axes (fixed thresholds, descriptive)
# ---------------------------------------------------------------------------

@dataclass
class RegimeAxis:
    name: str
    feature_key: str              # key used to look up raw feature value
    bucket_fn_label: str          # human label for docs
    edges: tuple[float, ...] | None   # for numeric binning (exclusive upper except last)
    labels: tuple[str, ...]


REGIME_AXES: tuple[RegimeAxis, ...] = (
    RegimeAxis(
        name="opening_gap_pct",
        feature_key="opening_gap_pct",
        bucket_fn_label="±30 bps",
        edges=(-float("inf"), -0.003, 0.003, float("inf")),
        labels=("down", "flat", "up"),
    ),
    RegimeAxis(
        name="first15_close_position",
        feature_key="first15_close_position",
        bucket_fn_label="0.333, 0.667",
        edges=(-0.01, 0.333, 0.667, 1.01),
        labels=("low", "mid", "high"),
    ),
    RegimeAxis(
        name="first15_range_pct",
        feature_key="first15_range_pct",
        bucket_fn_label="20 bps",
        edges=(-float("inf"), 0.002, float("inf")),
        labels=("narrow", "wide"),
    ),
    RegimeAxis(
        name="vix_regime",
        feature_key="vix_regime",
        bucket_fn_label="native discrete",
        # vix_regime is encoded as {-1, -0.33, 0.33, 1} in X_sim
        edges=(-1.1, -0.5, 0.0, 0.5, 1.1),
        labels=("calm", "low", "mid", "high"),
    ),
)


def bucket_value(axis: RegimeAxis, value: float) -> str | None:
    if not np.isfinite(value):
        return None
    edges = axis.edges
    labels = axis.labels
    if edges is None:
        return None
    for i in range(len(labels)):
        lo = edges[i]
        hi = edges[i + 1]
        if lo < value <= hi or (i == 0 and value == lo):
            return labels[i]
    # Handle float precision at the upper edge
    if value <= edges[-1] + 1e-9:
        return labels[-1]
    return None


# ---------------------------------------------------------------------------
# Extended trade record — carries regime values and pop-ctrl-A stats
# ---------------------------------------------------------------------------

@dataclass
class RegimeTrade(v2_mod.V2Trade):
    # regime values at entry bar (from X_sim)
    opening_gap_pct: float = 0.0
    first15_close_position: float = 0.0
    first15_range_pct: float = 0.0
    # population-mean Control A for this trade (expected random-bar return)
    pop_ctrl_a_mean_net_pct: float = 0.0
    pop_ctrl_a_n_bars: int = 0
    delta_net_pct: float = 0.0   # strategy - pop_ctrl_a


REGIME_TRADE_FIELDS = list(RegimeTrade.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# Population-mean Control A
# ---------------------------------------------------------------------------

def population_control_A_mean(
    *,
    sc: dict[str, Any],
    X_day: np.ndarray,
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config: V1BGateConfig,
    strategy_side: str,
    strategy_bar: int,
) -> tuple[float, int]:
    """Enumerate V1B-admissible same-side bars on this day; realize each;
    return mean net_pct and count. Strategy's own bar is INCLUDED in the
    population (uniform random-bar semantics). Returns (0.0, 0) if no
    candidates.
    """
    cands = v2_mod._enumerate_candidates_on_day(
        sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    if not cands:
        return 0.0, 0
    side_cands = [c for c in cands if c["side"] == strategy_side]
    if not side_cands:
        return 0.0, 0
    net_pcts: list[float] = []
    for cand in side_cands:
        trade, _ = v2_mod._realize_trade(
            day="_pop", fold_idx=-1,
            strategy_label="pop_ctrl_a", paired_trade_id=-1,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map, sc=sc,
            entry_local=cand["local_i"], side=cand["side"],
            picked=cand["picked"], context_spread=cand["context_spread"],
        )
        if trade is not None:
            net_pcts.append(trade.net_pct)
    if not net_pcts:
        return 0.0, 0
    return float(np.mean(net_pcts)), len(net_pcts)


# ---------------------------------------------------------------------------
# Per-fold run: V2-pruned strategy + pop-Control-A per trade + regime values
# ---------------------------------------------------------------------------

def run_fold(
    *,
    fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
) -> dict:
    t0 = time.time()
    fold_idx = fold_spec.fold_idx
    print(f"\n=== Fold {fold_idx} | window={fold_spec.window_id} ===", flush=True)

    # Gate selection
    config, _diag, _n_tr, _n_ent = v2_mod.select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    if config is None:
        print("  NO VIABLE GATE CONFIG", flush=True)
        return {
            "fold_idx": fold_idx, "window_id": fold_spec.window_id,
            "gate_config": {}, "strategy_trades": [],
            "elapsed_sec": time.time() - t0,
        }
    print(f"  gates: iv_max={config.iv_max} vrp_max={config.vrp_max:+.5f} "
          f"({config.vrp_source})", flush=True)

    # Training samples + RF fit
    feats, labels, n_train_days = v2_mod.collect_training_samples(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    model = v2_mod.train_scorer(feats, labels, seed=fold_idx * 10007 + 1)
    print(f"  train: n_samples={feats.shape[0]}", flush=True)

    # Feature indices for regime lookups
    idx_gap = idx_map["opening_gap_pct"]
    idx_f15_close = idx_map["first15_close_position"]
    idx_f15_range = idx_map["first15_range_pct"]
    idx_vix = idx_map["vix_regime"]

    # Test inference + population Control A + regime values
    strategy_trades: list[RegimeTrade] = []
    t1 = time.time()
    for day in fold_spec.test_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        X_day = X[ds:de]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        cands = v2_mod._enumerate_candidates_on_day(
            sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, core_indices=core_indices, config=config,
        )
        if not cands:
            continue
        feat_mat = np.vstack([c["features"] for c in cands]).astype(np.float32)
        preds = model.predict(feat_mat)
        pos = int(np.argmax(preds))
        chosen = cands[pos]

        v2trade, status = v2_mod._realize_trade(
            day=day, fold_idx=fold_idx,
            strategy_label="strategy", paired_trade_id=-1,
            X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, sc=sc,
            entry_local=chosen["local_i"], side=chosen["side"],
            picked=chosen["picked"], context_spread=chosen["context_spread"],
            predicted_net_pct=float(preds[pos]),
            score_rank=1, n_candidates=len(cands),
        )
        if v2trade is None:
            continue

        # Population-mean Control A for this trade's day/side
        pop_mean, pop_n = population_control_A_mean(
            sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, core_indices=core_indices, config=config,
            strategy_side=chosen["side"], strategy_bar=chosen["local_i"],
        )

        # Regime values at the entry bar
        rt = RegimeTrade(
            **{**asdict(v2trade),
               "opening_gap_pct": float(X_sim_day[chosen["local_i"], idx_gap]),
               "first15_close_position": float(X_sim_day[chosen["local_i"], idx_f15_close]),
               "first15_range_pct": float(X_sim_day[chosen["local_i"], idx_f15_range]),
               "pop_ctrl_a_mean_net_pct": float(pop_mean),
               "pop_ctrl_a_n_bars": int(pop_n),
               "delta_net_pct": float(v2trade.net_pct - pop_mean),
               # vix_regime_at_entry already populated by _realize_trade
               },
        )
        strategy_trades.append(rt)

    print(f"  test: n_strategy={len(strategy_trades)}, test+popA elapsed={time.time() - t1:.1f}s",
          flush=True)

    return {
        "fold_idx": fold_idx, "window_id": fold_spec.window_id,
        "gate_config": asdict(config),
        "strategy_trades": strategy_trades,
        "n_train_samples": int(feats.shape[0]),
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Regime bucketing
# ---------------------------------------------------------------------------

def value_for_axis(trade: RegimeTrade, axis: RegimeAxis) -> float:
    if axis.feature_key == "vix_regime":
        return float(trade.vix_regime_at_entry)
    return float(getattr(trade, axis.feature_key))


def bucket_trades(
    trades: list[RegimeTrade],
    axis: RegimeAxis,
) -> dict[str, list[RegimeTrade]]:
    out: dict[str, list[RegimeTrade]] = {label: [] for label in axis.labels}
    for t in trades:
        v = value_for_axis(t, axis)
        bkt = bucket_value(axis, v)
        if bkt is None:
            continue
        out[bkt].append(t)
    return out


# ---------------------------------------------------------------------------
# Paired-delta bootstrap per bucket
# ---------------------------------------------------------------------------

def bootstrap_bucket_delta(
    bucket_trades: list[RegimeTrade],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    if not bucket_trades:
        return {"n": 0, "strategy_mean": 0.0, "pop_ctrl_a_mean": 0.0,
                "gap": 0.0, "gap_ci_low": 0.0, "gap_ci_high": 0.0,
                "gap_frac_gt_0": 0.0, "strategy_dollar_pf": 0.0,
                "target_hit_frac": 0.0, "stop_hit_frac": 0.0,
                "n_folds_contributing": 0}
    # Group by (fold, date) for day-level resample
    by_day: dict[tuple[int, str], list[RegimeTrade]] = {}
    for t in bucket_trades:
        by_day.setdefault((t.fold, t.date), []).append(t)
    day_keys = list(by_day.keys())

    # Summary of observed
    strat_arr = np.array([t.net_pct for t in bucket_trades], dtype=np.float64)
    pop_arr = np.array([t.pop_ctrl_a_mean_net_pct for t in bucket_trades], dtype=np.float64)
    delta_arr = strat_arr - pop_arr

    # Bootstrap mean of delta — day-level block
    rng = np.random.default_rng(seed)
    n_days = len(day_keys)
    boots: list[float] = []
    for _ in range(n_boot):
        sampled_keys = [day_keys[i] for i in rng.integers(0, n_days, size=n_days)]
        sampled_trades = [t for k in sampled_keys for t in by_day[k]]
        if not sampled_trades:
            boots.append(0.0)
            continue
        s = np.array([t.net_pct for t in sampled_trades])
        p = np.array([t.pop_ctrl_a_mean_net_pct for t in sampled_trades])
        boots.append(float((s - p).mean()))
    boot_arr = np.array(boots, dtype=np.float64)

    # PF
    dollars = np.array([t.net_pnl_dollars for t in bucket_trades], dtype=np.float64)
    gp = float(dollars[dollars > 0].sum())
    gl = float(-dollars[dollars < 0].sum())
    pf = gp / gl if gl > 1e-12 else (float("inf") if gp > 0 else 0.0)

    exits = [t.exit_reason for t in bucket_trades]
    target = sum(1 for r in exits if r == "target_first15") / len(exits) if exits else 0.0
    stop = sum(1 for r in exits if r == "stop_vwap") / len(exits) if exits else 0.0

    fold_set = set(t.fold for t in bucket_trades)
    return {
        "n": len(bucket_trades),
        "n_days": len(day_keys),
        "strategy_mean": float(strat_arr.mean()),
        "strategy_std": float(strat_arr.std(ddof=1)) if len(strat_arr) > 1 else 0.0,
        "pop_ctrl_a_mean": float(pop_arr.mean()),
        "gap": float(delta_arr.mean()),
        "gap_ci_low": float(np.quantile(boot_arr, 0.025)),
        "gap_ci_high": float(np.quantile(boot_arr, 0.975)),
        "gap_frac_gt_0": float((boot_arr > 0).mean()),
        "strategy_dollar_pf": float(pf),
        "target_hit_frac": float(target),
        "stop_hit_frac": float(stop),
        "n_folds_contributing": len(fold_set),
        "per_fold_counts": {int(f): sum(1 for t in bucket_trades if t.fold == f)
                             for f in sorted(fold_set)},
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_regime_audit"
N_BOOTSTRAP = 1000


PROMOTABLE_RULES = {
    "min_gap": 0.0,                 # gap > 0
    "min_frac_gt_0": 0.90,          # 90% one-sided bootstrap
    "min_n": 20,
    "min_folds_contributing": 3,
}


def classify_bucket(bucket_stats: dict) -> str:
    """Return one of: 'promotable', 'promising', 'neutral', 'adverse'."""
    if bucket_stats["n"] == 0:
        return "empty"
    gap = bucket_stats["gap"]
    frac = bucket_stats["gap_frac_gt_0"]
    n = bucket_stats["n"]
    folds = bucket_stats["n_folds_contributing"]
    if (gap > PROMOTABLE_RULES["min_gap"]
            and frac >= PROMOTABLE_RULES["min_frac_gt_0"]
            and n >= PROMOTABLE_RULES["min_n"]
            and folds >= PROMOTABLE_RULES["min_folds_contributing"]):
        return "promotable"
    if gap > 0 and frac >= 0.75 and n >= PROMOTABLE_RULES["min_n"]:
        return "promising"
    if gap <= 0 and frac < 0.25:
        return "adverse"
    return "neutral"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    _activate_pruned_feature_set()
    print(f"[regime-audit] Active CORE_FEATURE_NAMES: {list(v2_mod.CORE_FEATURE_NAMES)}")
    for axis in REGIME_AXES:
        print(f"[regime-axis] {axis.name}: {axis.labels} ({axis.bucket_fn_label})")

    t0 = time.time()
    data = load_data(args.data)
    dates = list(data["dates"])
    feature_names = list(data["feature_names"])
    idx_map = feature_index_map(feature_names)
    required = (list(v2_mod.CORE_FEATURE_NAMES)
                + ["vix_regime", "iv_percentile", "vrp",
                   "opening_gap_pct", "first15_close_position", "first15_range_pct"])
    missing = [n for n in required if n not in idx_map]
    if missing:
        print(f"FATAL: feature_names missing: {missing}", file=sys.stderr)
        return 2
    core_idx = v2_mod.core_feature_indices(idx_map)

    X = data["X"].numpy() if isinstance(data["X"], torch.Tensor) else np.asarray(data["X"])
    X_sim = data["X_sim"].numpy() if isinstance(data["X_sim"], torch.Tensor) else np.asarray(data["X_sim"])
    spot_prices = data["spot_prices"].numpy() if isinstance(data["spot_prices"], torch.Tensor) else np.asarray(data["spot_prices"])

    day_to_range, _ = day_ranges(dates)
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dataset_fp = str(data["metadata"].get("fingerprint", "unknown"))

    folds = build_folds(dates)
    os.makedirs(args.out_dir, exist_ok=True)

    # Per-fold strategy + pop-A + regime features
    fold_results: list[dict] = []
    for fs in folds:
        fold_results.append(run_fold(
            fold_spec=fs, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X=X, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, core_indices=core_idx,
        ))

    all_trades: list[RegimeTrade] = []
    for r in fold_results:
        all_trades.extend(r["strategy_trades"])
    print(f"\nTotal strategy trades: {len(all_trades)}")
    # Aggregate baseline (all trades, no regime)
    agg = bootstrap_bucket_delta(all_trades, n_boot=args.n_boot, seed=42)
    print(f"\n=== Aggregate (no regime bucketing) ===")
    print(f"  n={agg['n']}, strategy_mean={agg['strategy_mean']:+.5f}, "
          f"pop_ctrl_a_mean={agg['pop_ctrl_a_mean']:+.5f}")
    print(f"  gap={agg['gap']:+.5f}, 95% CI=[{agg['gap_ci_low']:+.5f}, {agg['gap_ci_high']:+.5f}], "
          f"frac>0={agg['gap_frac_gt_0']:.3f}")
    print(f"  dollar_pf={agg['strategy_dollar_pf']:.3f}, "
          f"target={agg['target_hit_frac']:.3f}, stop={agg['stop_hit_frac']:.3f}")

    # Per-axis bucketing
    print("\n=== Regime-bucket table ===")
    print(f"{'axis':<24} {'bucket':<8} {'n':>4} {'strat_mean':>11} {'popA_mean':>11} "
          f"{'gap':>10} {'ci_low':>10} {'ci_high':>10} {'f>0':>6} {'folds':>6} {'class':<12}")
    axis_tables: dict[str, list[dict]] = {}
    for axis in REGIME_AXES:
        buckets = bucket_trades(all_trades, axis)
        rows: list[dict] = []
        for label in axis.labels:
            b = buckets[label]
            seed = 42 + hash((axis.name, label)) % 10000
            stats = bootstrap_bucket_delta(b, n_boot=args.n_boot, seed=seed)
            stats["axis"] = axis.name
            stats["bucket"] = label
            stats["classification"] = classify_bucket(stats)
            rows.append(stats)
            print(f"{axis.name:<24} {label:<8} {stats['n']:>4d} "
                  f"{stats['strategy_mean']:>+11.5f} {stats['pop_ctrl_a_mean']:>+11.5f} "
                  f"{stats['gap']:>+10.5f} {stats['gap_ci_low']:>+10.5f} "
                  f"{stats['gap_ci_high']:>+10.5f} {stats['gap_frac_gt_0']:>6.3f} "
                  f"{stats['n_folds_contributing']:>6d} {stats['classification']:<12}")
        axis_tables[axis.name] = rows

    # Find promotable / promising buckets
    promotable = [r for rs in axis_tables.values() for r in rs if r["classification"] == "promotable"]
    promising = [r for rs in axis_tables.values() for r in rs if r["classification"] == "promising"]
    print(f"\n=== Verdict ===")
    print(f"  promotable buckets: {len(promotable)}")
    for r in promotable:
        print(f"    {r['axis']}/{r['bucket']}: n={r['n']}, gap={r['gap']:+.5f}, "
              f"frac>0={r['gap_frac_gt_0']:.3f}, folds={r['n_folds_contributing']}")
    print(f"  promising buckets: {len(promising)}")
    for r in promising:
        print(f"    {r['axis']}/{r['bucket']}: n={r['n']}, gap={r['gap']:+.5f}, "
              f"frac>0={r['gap_frac_gt_0']:.3f}, folds={r['n_folds_contributing']}")

    # Write outputs
    write_csv(os.path.join(args.out_dir, "regime_trades.csv"), all_trades, REGIME_TRADE_FIELDS)
    write_json(os.path.join(args.out_dir, "regime_audit_summary.json"), {
        "dataset_fingerprint": dataset_fp,
        "axes": [{"name": a.name, "labels": list(a.labels),
                  "bucket_spec": a.bucket_fn_label,
                  "edges": list(a.edges) if a.edges is not None else None}
                 for a in REGIME_AXES],
        "promotable_rules": PROMOTABLE_RULES,
        "aggregate_no_regime": agg,
        "per_axis": {k: v for k, v in axis_tables.items()},
        "n_bootstrap": args.n_boot,
        "note": "Paired-delta bootstrap per bucket; pop_ctrl_a is population-mean "
                "over admissible same-day same-side bars (RNG eliminated).",
    })
    print(f"\nWrote summary to {args.out_dir}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
