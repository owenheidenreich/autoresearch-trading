"""V2-pruned-gated nested block-CV sizing calibration.

Fixes the train-side calibration failure in the sizing audit (2026-04-20
lab notebook entry): OOB argmax predictions use only ~37% of trees and
make within-day ranking noisier than the deployed full-forest
predictions, causing the train trade stream to have *negative*
expectancy on every fold while the deployed test stream has +2.98%
expectancy. The selection rule "max train CAGR subject to max train
DD <= 20%" then degenerates to "smallest f" in every fold.

This branch replaces the OOB train trade stream with a nested
blocked-CV stream: honest pseudo-out-of-sample predictions from
full-forest models fit on train-minus-block-minus-embargo segments.

Everything else is frozen:
- V2-pruned-gated trade selection (same RF hparams, same feature set,
  same V1B gates, same fixed first15_range_pct >= 20 bps gate)
- contract selection + exit engine + one-trade-per-day
- sizing grid {0.25%, 0.50%, 1.00%, 1.50%, 2.00%}
- selection objective: highest train CAGR subject to max train DD <= 20%
- test-side trade stream unchanged

Nested block-CV parameters (pre-declared):
- n_blocks = 5 (equal contiguous chunks)
- embargo = ±5 trading days around each validation block
- inner RF hparams = outer hparams (n_estimators=200, max_depth=5, min_samples_leaf=20)
- inner model uses FULL FOREST predictions on the held-out block
  (not OOB — that's what broke the OOB run)

Plan: lab_notebook entry 2026-04-20 (sizing audit) and next-branch spec.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from v2.core.chain_data import load_sidecar_cached, sidecar_path
from v2.analysis.mechanical_baseline_opening_reversion import (
    BAR_LO,
    BAR_HI,
    SkipRecord,
    build_folds,
    day_ranges,
    feature_index_map,
    load_data,
    write_csv,
    write_json,
)
from v2.analysis.mechanical_baseline_v1b_opening_reversion import V1BGateConfig
from v2.analysis import mechanical_baseline_v2_learned_scorer as v2_mod
from v2.analysis.mechanical_baseline_v2_pruned_gated import (
    FIRST15_RANGE_GATE,
    FIRST15_RANGE_GATE_BPS,
    GatedTrade,
    GATED_TRADE_FIELDS,
    _activate_pruned_feature_set,
    day_passes_gate,
)
from v2.analysis.mechanical_baseline_v2_pruned_gated_sizing import (
    SIZING_GRID,
    MAX_DD_THRESHOLD_TRAIN,
    TRADING_DAYS_PER_YEAR,
    _train_rf_with_oob,
    aggregate_test_equity_fixed_f,
    aggregate_test_equity_with_per_fold_f,
    build_test_trade_stream,
    build_train_trade_stream_oob,
    collect_train_samples_with_meta,
    equity_metrics,
    kelly_clipped,
    select_fraction_from_train,
    simulate_equity,
)


# ---------------------------------------------------------------------------
# Nested block-CV knobs (pre-declared; no sweep)
# ---------------------------------------------------------------------------

N_INNER_BLOCKS = 5
EMBARGO_DAYS = 5

EXPERIMENT_ID = "mechbase_opening_reversion_v2_pruned_gated_sized_nested"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated_sized_nested"


# ---------------------------------------------------------------------------
# Inner fold spec
# ---------------------------------------------------------------------------

class _InnerFoldSpec:
    def __init__(self, fold_idx: int, window_id: str, train_days: list[str]):
        self.fold_idx = fold_idx
        self.window_id = window_id
        self.train_days = train_days
        # val/test lists are unused by v2_mod.collect_training_samples
        self.val_days: list[str] = []
        self.test_days: list[str] = []


# ---------------------------------------------------------------------------
# Pseudo-OOS train trade stream via nested block-CV
# ---------------------------------------------------------------------------

def build_block_cv_train_trade_stream(
    *,
    outer_fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config: V1BGateConfig,
    idx_first15_range: int,
    n_blocks: int = N_INNER_BLOCKS,
    embargo: int = EMBARGO_DAYS,
    seed_base: int = 0,
) -> tuple[list[GatedTrade], dict]:
    """Partition outer train_days into n_blocks contiguous segments; for
    each segment, fit an inner RF on (outer_train - segment - ±embargo)
    and predict the segment with full-forest scoring. Only gate-passing
    days contribute to the pseudo-OOS train trade stream.

    Returns (trades_sorted_by_date, diagnostics_dict).
    """
    train_days = list(outer_fold_spec.train_days)
    n = len(train_days)
    block_size = max(1, n // n_blocks)

    trades: list[GatedTrade] = []
    block_diag: list[dict] = []

    for k in range(n_blocks):
        start = k * block_size
        end = (k + 1) * block_size if k < n_blocks - 1 else n
        val_indices = set(range(start, end))
        embargo_start = max(0, start - embargo)
        embargo_end = min(n, end + embargo)
        embargo_indices = set(range(embargo_start, embargo_end))
        inner_train_indices = set(range(n)) - embargo_indices
        inner_train_days = [train_days[i] for i in sorted(inner_train_indices)]
        inner_val_days = [train_days[i] for i in sorted(val_indices)]

        inner_spec = _InnerFoldSpec(
            fold_idx=outer_fold_spec.fold_idx,
            window_id=f"{outer_fold_spec.window_id}_inner{k}",
            train_days=inner_train_days,
        )

        # Fit inner RF on inner_train samples
        inner_feats, inner_labels, _ = v2_mod.collect_training_samples(
            fold_spec=inner_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X=X, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, core_indices=core_indices, config=config,
        )
        if inner_feats.shape[0] == 0:
            block_diag.append({"block": k, "n_inner_train": 0, "n_val_trades": 0})
            continue

        inner_model = RandomForestRegressor(
            n_estimators=v2_mod.RF_N_ESTIMATORS,
            max_depth=v2_mod.RF_MAX_DEPTH,
            min_samples_leaf=v2_mod.RF_MIN_SAMPLES_LEAF,
            random_state=seed_base + k,
            n_jobs=-1,
            bootstrap=True,
        )
        inner_model.fit(inner_feats, inner_labels)

        # Score each val day with inner_model (full forest); argmax; realize
        trades_this_block = 0
        days_considered = 0
        days_gate_passed = 0
        for day in inner_val_days:
            if day not in day_to_range:
                continue
            days_considered += 1
            path = sidecar_path(sidecar_dir, day)
            if not os.path.exists(path):
                continue
            ds, de = day_to_range[day]
            X_sim_day = X_sim[ds:de]
            ok, _ = day_passes_gate(X_sim_day, idx_first15_range)
            if not ok:
                continue
            days_gate_passed += 1
            sc = load_sidecar_cached(path)
            X_day = X[ds:de]
            spot_day = spot_prices[ds:de]
            cands = v2_mod._enumerate_candidates_on_day(
                sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
                idx_map=idx_map, core_indices=core_indices, config=config,
            )
            if not cands:
                continue
            feat_mat = np.vstack([c["features"] for c in cands]).astype(np.float32)
            preds = inner_model.predict(feat_mat)
            pos = int(np.argmax(preds))
            chosen = cands[pos]
            v2t, _ = v2_mod._realize_trade(
                day=day, fold_idx=outer_fold_spec.fold_idx,
                strategy_label="train_block_cv", paired_trade_id=-1,
                X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map, sc=sc,
                entry_local=chosen["local_i"], side=chosen["side"],
                picked=chosen["picked"], context_spread=chosen["context_spread"],
                predicted_net_pct=float(preds[pos]),
                score_rank=1, n_candidates=len(cands),
            )
            if v2t is None:
                continue
            first15_val = float(X_sim_day[15, idx_first15_range]) if X_sim_day.shape[0] > 15 else 0.0
            trades.append(GatedTrade(
                **{**asdict(v2t),
                   "first15_range_pct": first15_val,
                   "pop_ctrl_a_mean_net_pct": 0.0, "pop_ctrl_a_n_bars": 0,
                   "pop_ctrl_b_mean_net_pct": 0.0, "pop_ctrl_b_n_days": 0,
                   "delta_vs_pop_A": 0.0, "delta_vs_pop_B": 0.0},
            ))
            trades_this_block += 1
        block_diag.append({
            "block": k,
            "val_start_idx": start, "val_end_idx": end,
            "inner_train_days": len(inner_train_days),
            "inner_train_samples": int(inner_feats.shape[0]),
            "val_days_total": len(inner_val_days),
            "val_days_considered": days_considered,
            "val_days_gate_passed": days_gate_passed,
            "n_val_trades": trades_this_block,
        })

    trades.sort(key=lambda t: t.date)
    return trades, {"blocks": block_diag, "n_blocks": n_blocks, "embargo_days": embargo}


# ---------------------------------------------------------------------------
# Per-fold driver
# ---------------------------------------------------------------------------

@dataclass
class FoldRunNested:
    fold_idx: int
    window_id: str
    # train streams
    train_trades_oob: list[GatedTrade]
    train_trades_blockcv: list[GatedTrade]
    test_trades: list[GatedTrade]
    # grids + selections
    train_grid_oob: dict[float, dict]
    train_grid_blockcv: dict[float, dict]
    test_grid: dict[float, dict]
    selected_f_oob: float
    selected_f_blockcv: float
    rationale_oob: str
    rationale_blockcv: str
    # horizons
    train_days_horizon: int
    test_days_horizon: int
    # block diagnostics
    block_cv_diag: dict


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
) -> FoldRunNested:
    t0 = time.time()
    fold_idx = fold_spec.fold_idx
    print(f"\n=== Fold {fold_idx} | window={fold_spec.window_id} ===", flush=True)

    # V1B gate selection on outer train
    config, _diag, _n_tr, _n_ent = v2_mod.select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    assert config is not None
    idx_first15 = idx_map["first15_range_pct"]

    # --- OOB train stream (for side-by-side) ---
    feats, labels, meta = collect_train_samples_with_meta(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
        first15_range_gate=FIRST15_RANGE_GATE, idx_first15_range=idx_first15,
    )
    outer_model = _train_rf_with_oob(feats, labels, seed=fold_idx * 10007 + 1)
    assert outer_model is not None
    oob_train_trades = build_train_trade_stream_oob(
        fold_idx=fold_idx, metadata=meta, oob_preds=outer_model.oob_prediction_,
        day_to_range=day_to_range, X_sim=X_sim,
        idx_first15_range=idx_first15, first15_range_gate=FIRST15_RANGE_GATE,
    )
    print(f"  outer train samples n={feats.shape[0]}, OOB train trades n={len(oob_train_trades)}",
          flush=True)

    # --- Nested block-CV train stream ---
    t_cv_start = time.time()
    blockcv_train_trades, block_diag = build_block_cv_train_trade_stream(
        outer_fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
        idx_first15_range=idx_first15,
        seed_base=fold_idx * 100003 + 7,
    )
    print(f"  block-CV train trades n={len(blockcv_train_trades)}, "
          f"elapsed={time.time() - t_cv_start:.1f}s", flush=True)

    # --- Test stream (V2-pruned-gated; outer model) ---
    test_trades = build_test_trade_stream(
        fold_spec=fold_spec, model=outer_model, config=config,
        day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, idx_first15_range=idx_first15,
    )
    print(f"  test trades n={len(test_trades)}", flush=True)

    # --- Grid evaluations ---
    train_days_horizon = len(fold_spec.train_days)
    test_days_horizon = len(fold_spec.test_days)

    train_grid_oob: dict[float, dict] = {}
    train_grid_blockcv: dict[float, dict] = {}
    test_grid: dict[float, dict] = {}
    for f in SIZING_GRID:
        train_grid_oob[f] = equity_metrics(
            simulate_equity(oob_train_trades, f), oob_train_trades, train_days_horizon)
        train_grid_blockcv[f] = equity_metrics(
            simulate_equity(blockcv_train_trades, f), blockcv_train_trades, train_days_horizon)
        test_grid[f] = equity_metrics(
            simulate_equity(test_trades, f), test_trades, test_days_horizon)

    f_oob, rat_oob = select_fraction_from_train(train_grid_oob)
    f_bcv, rat_bcv = select_fraction_from_train(train_grid_blockcv)

    print(f"\n  Train stream means:  OOB={np.mean([t.net_pct for t in oob_train_trades]):+.5f}, "
          f"block-CV={np.mean([t.net_pct for t in blockcv_train_trades]):+.5f}", flush=True)
    print(f"  OOB-calibrated train grid:")
    for f, m in train_grid_oob.items():
        mark = " << OOB-selected" if f == f_oob else ""
        print(f"    f={f:.4f}: cagr={m['cagr']:+.4f}, max_dd={m['max_dd']:.4f}, "
              f"calmar={m['calmar']:+.3f}{mark}")
    print(f"  Block-CV-calibrated train grid:")
    for f, m in train_grid_blockcv.items():
        mark = " << block-CV-selected" if f == f_bcv else ""
        print(f"    f={f:.4f}: cagr={m['cagr']:+.4f}, max_dd={m['max_dd']:.4f}, "
              f"calmar={m['calmar']:+.3f}{mark}")
    print(f"  Test grid:")
    for f, m in test_grid.items():
        mark_oob = " <OOB sel>" if f == f_oob else ""
        mark_bcv = " <bcv sel>" if f == f_bcv else ""
        print(f"    f={f:.4f}: cagr={m['cagr']:+.4f}, max_dd={m['max_dd']:.4f}, "
              f"calmar={m['calmar']:+.3f}, worst20={m['worst_20_trade_dd']:.4f}"
              f"{mark_oob}{mark_bcv}")

    return FoldRunNested(
        fold_idx=fold_idx, window_id=fold_spec.window_id,
        train_trades_oob=oob_train_trades,
        train_trades_blockcv=blockcv_train_trades,
        test_trades=test_trades,
        train_grid_oob=train_grid_oob,
        train_grid_blockcv=train_grid_blockcv,
        test_grid=test_grid,
        selected_f_oob=f_oob,
        selected_f_blockcv=f_bcv,
        rationale_oob=rat_oob,
        rationale_blockcv=rat_bcv,
        train_days_horizon=train_days_horizon,
        test_days_horizon=test_days_horizon,
        block_cv_diag=block_diag,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers (reuse sizing module's)
# ---------------------------------------------------------------------------

def aggregate_test_with_per_fold_f(
    fold_results: list[FoldRunNested], variant: str,
) -> dict:
    """variant: 'oob' or 'blockcv' — which per-fold f to apply."""
    trades_with_f: list[tuple[GatedTrade, float]] = []
    for fr in fold_results:
        f = fr.selected_f_blockcv if variant == "blockcv" else fr.selected_f_oob
        for t in fr.test_trades:
            trades_with_f.append((t, f))
    trades_with_f.sort(key=lambda pair: pair[0].date)
    equity = np.empty(len(trades_with_f) + 1, dtype=np.float64)
    equity[0] = 1.0
    flat_trades: list[GatedTrade] = []
    per_fold_f_applied = []
    for i, (t, f) in enumerate(trades_with_f):
        e = equity[i] * (1.0 + f * float(t.net_pct))
        equity[i + 1] = max(e, 0.0)
        flat_trades.append(t)
        per_fold_f_applied.append(f)
    total_cal = sum(fr.test_days_horizon for fr in fold_results)
    m = equity_metrics(equity, flat_trades, total_cal)
    m["per_fold_f_applied_unique"] = sorted(set(per_fold_f_applied))
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    _activate_pruned_feature_set()
    print(f"[nested-cv] Active CORE_FEATURE_NAMES: {list(v2_mod.CORE_FEATURE_NAMES)}")
    print(f"[nested-cv] Gate: first15_range_pct >= {FIRST15_RANGE_GATE:.4f}")
    print(f"[nested-cv] Sizing grid: {[f'{f:.4f}' for f in SIZING_GRID]}")
    print(f"[nested-cv] Inner blocks: {N_INNER_BLOCKS}, embargo: ±{EMBARGO_DAYS} days")

    t0 = time.time()
    data = load_data(args.data)
    dates = list(data["dates"])
    feature_names = list(data["feature_names"])
    idx_map = feature_index_map(feature_names)
    required = (list(v2_mod.CORE_FEATURE_NAMES)
                + ["vix_regime", "iv_percentile", "vrp", "first15_range_pct"])
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

    fold_results: list[FoldRunNested] = []
    for fs in folds:
        fold_results.append(run_fold(
            fold_spec=fs, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X=X, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, core_indices=core_idx,
        ))

    # Aggregates
    agg_oob = aggregate_test_with_per_fold_f(fold_results, variant="oob")
    agg_bcv = aggregate_test_with_per_fold_f(fold_results, variant="blockcv")

    # Full test grid (reuse sizing module pattern)
    agg_grid_test: dict[float, dict] = {}
    for f in SIZING_GRID:
        all_test = [t for fr in fold_results for t in fr.test_trades]
        all_test.sort(key=lambda t: t.date)
        eq = simulate_equity(all_test, f)
        total_cal = sum(fr.test_days_horizon for fr in fold_results)
        agg_grid_test[f] = equity_metrics(eq, all_test, total_cal)

    # Kelly appendix
    all_test = [t for fr in fold_results for t in fr.test_trades]
    all_oob = [t for fr in fold_results for t in fr.train_trades_oob]
    all_bcv = [t for fr in fold_results for t in fr.train_trades_blockcv]
    kelly_test = kelly_clipped(all_test)
    kelly_oob = kelly_clipped(all_oob)
    kelly_bcv = kelly_clipped(all_bcv)

    print("\n=== Side-by-side comparison ===")
    print(f"{'fold':>4} {'f_OOB':>7} {'f_bcv':>7} {'oob_mean':>10} {'bcv_mean':>10}")
    for fr in fold_results:
        om = float(np.mean([t.net_pct for t in fr.train_trades_oob])) if fr.train_trades_oob else 0.0
        bm = float(np.mean([t.net_pct for t in fr.train_trades_blockcv])) if fr.train_trades_blockcv else 0.0
        print(f"{fr.fold_idx:>4d} {fr.selected_f_oob:>7.4f} {fr.selected_f_blockcv:>7.4f} "
              f"{om:>+10.5f} {bm:>+10.5f}")

    print("\n=== Aggregate test curves ===")
    def _row(name, m):
        print(f"  {name:<28}: total={m.get('total_return', 0):+.4f}, "
              f"cagr={m.get('cagr', 0):+.4f}, max_dd={m.get('max_dd', 0):.4f}, "
              f"calmar={m.get('calmar', 0):+.3f}, worst20={m.get('worst_20_trade_dd', 0):.4f}, "
              f"n={m.get('n_trades', 0)}")
    _row("OOB-calibrated (per-fold f)", agg_oob)
    _row("block-CV (per-fold f)", agg_bcv)

    print("\n=== Full test grid (descriptive) ===")
    print(f"{'f':>8} {'n':>5} {'total':>10} {'cagr':>10} {'max_dd':>10} {'calmar':>8} "
          f"{'worst20':>9}")
    for f, m in agg_grid_test.items():
        print(f"{f:>8.4f} {m['n_trades']:>5} {m['total_return']:>+10.4f} "
              f"{m['cagr']:>+10.4f} {m['max_dd']:>10.4f} {m['calmar']:>+8.2f} "
              f"{m['worst_20_trade_dd']:>9.4f}")

    print("\n=== Kelly appendix ===")
    for name, k in [("test", kelly_test), ("OOB-train", kelly_oob), ("block-CV-train", kelly_bcv)]:
        print(f"  {name:<18}: mean={k['mean']:+.5f}, var={k['var']:.5f}, "
              f"raw={k['kelly_raw']:+.4f}, clipped={k['kelly_clipped']:.4f}")

    # Dump
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint": dataset_fp,
        "gate_threshold_bps": FIRST15_RANGE_GATE_BPS,
        "sizing_grid": list(SIZING_GRID),
        "max_dd_threshold_train": MAX_DD_THRESHOLD_TRAIN,
        "nested_cv": {"n_blocks": N_INNER_BLOCKS, "embargo_days": EMBARGO_DAYS},
        "aggregate_test_selected_oob": agg_oob,
        "aggregate_test_selected_blockcv": agg_bcv,
        "aggregate_test_grid": {str(f): m for f, m in agg_grid_test.items()},
        "per_fold": [
            {
                "fold_idx": fr.fold_idx,
                "window_id": fr.window_id,
                "selected_f_oob": fr.selected_f_oob,
                "selected_f_blockcv": fr.selected_f_blockcv,
                "rationale_oob": fr.rationale_oob,
                "rationale_blockcv": fr.rationale_blockcv,
                "train_trades_oob_n": len(fr.train_trades_oob),
                "train_trades_blockcv_n": len(fr.train_trades_blockcv),
                "test_trades_n": len(fr.test_trades),
                "train_grid_oob": {str(f): m for f, m in fr.train_grid_oob.items()},
                "train_grid_blockcv": {str(f): m for f, m in fr.train_grid_blockcv.items()},
                "test_grid": {str(f): m for f, m in fr.test_grid.items()},
                "oob_train_mean_net_pct": float(np.mean([t.net_pct for t in fr.train_trades_oob])) if fr.train_trades_oob else 0.0,
                "blockcv_train_mean_net_pct": float(np.mean([t.net_pct for t in fr.train_trades_blockcv])) if fr.train_trades_blockcv else 0.0,
                "block_cv_diag": fr.block_cv_diag,
            }
            for fr in fold_results
        ],
        "kelly_appendix": {"test": kelly_test, "train_oob": kelly_oob, "train_blockcv": kelly_bcv},
    }
    write_json(os.path.join(args.out_dir, "summary.json"), payload)
    print(f"\nWrote summary to {args.out_dir}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
