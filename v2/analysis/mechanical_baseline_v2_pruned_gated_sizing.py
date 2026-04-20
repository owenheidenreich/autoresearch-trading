"""V2-pruned-gated fixed-fraction sizing audit.

Frozen baseline: trade selection, gate, RF, exits, controls unchanged.
Only added: fixed-fraction equity sizing overlay on the realized net_pct
trade stream.

Sizing rule:
    equity_{t+1} = equity_t * (1 + f * net_pct_t)

Fixed-fraction grid (predeclared, user-locked): f ∈ {0.25%, 0.50%, 1.00%,
1.50%, 2.00%} of equity per trade.

Per-fold selection:
    Choose f from the grid that maximizes train CAGR subject to max
    train DD <= 20%. If no f passes the DD cap, fall back to the smallest
    f in the grid.

Applied to held-out test per fold; aggregate test equity curve walks
across all 5 fold test windows in chronological order, with the
fold-selected f applied to each fold's trades.

Honest train trade stream via RF OOB predictions — for each gate-passing
train day, the OOB-argmax sample's label is the "train trade" outcome.
This avoids the in-sample optimism that plain RF fit-and-predict would
introduce on the training set.

Kelly-style number reported as descriptive appendix only, clipped to
[0, 2%], non-authoritative.

User plan: 2026-04-20 lab_notebook entry (V2-pruned-gated promotion).
"""
from __future__ import annotations

import argparse
import math
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
    SKIP_FIELDS,
    build_folds,
    day_ranges,
    feature_index_map,
    load_data,
    summarize_trades,
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


SIZING_GRID = (0.0025, 0.0050, 0.0100, 0.0150, 0.0200)
MAX_DD_THRESHOLD_TRAIN = 0.20
TRADING_DAYS_PER_YEAR = 252
EXPERIMENT_ID = "mechbase_opening_reversion_v2_pruned_gated_sized"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated_sized"


# ---------------------------------------------------------------------------
# Sizing + equity metrics
# ---------------------------------------------------------------------------

def simulate_equity(trades_sorted: list[GatedTrade], f: float) -> np.ndarray:
    """Apply fixed-fraction sizing in chronological order. Equity is floored
    at 0. Returns an array of length len(trades) + 1 (includes starting 1.0).
    """
    equity = np.empty(len(trades_sorted) + 1, dtype=np.float64)
    equity[0] = 1.0
    for i, t in enumerate(trades_sorted):
        e = equity[i] * (1.0 + f * float(t.net_pct))
        equity[i + 1] = max(e, 0.0)
    return equity


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity)
    return np.where(running_max > 0, (running_max - equity) / running_max, 0.0)


def _worst_n_trade_dd(equity: np.ndarray, window: int = 20) -> float:
    """Maximum peak-to-trough drawdown over any `window` consecutive trades."""
    if len(equity) <= window:
        return float(drawdown_series(equity).max() if len(equity) else 0.0)
    worst = 0.0
    for start in range(0, len(equity) - window):
        w = equity[start:start + window + 1]
        rm = np.maximum.accumulate(w)
        wd = np.where(rm > 0, (rm - w) / rm, 0.0)
        worst = max(worst, float(wd.max()))
    return worst


def equity_metrics(
    equity: np.ndarray,
    trades: list[GatedTrade],
    n_calendar_days: int,
) -> dict:
    if len(equity) < 2:
        return {
            "n_trades": 0, "total_return": 0.0, "cagr": 0.0,
            "max_dd": 0.0, "calmar": 0.0, "ulcer_index": 0.0,
            "worst_20_trade_dd": 0.0, "final_equity": 1.0,
            "n_calendar_days": n_calendar_days,
        }
    final = float(equity[-1])
    total_ret = final - 1.0
    years = max(n_calendar_days / TRADING_DAYS_PER_YEAR, 1e-6)
    cagr = (final ** (1.0 / years)) - 1.0 if final > 0 else -1.0
    dd = drawdown_series(equity)
    max_dd = float(dd.max())
    ulcer = float(np.sqrt(np.mean(dd ** 2))) if len(dd) > 0 else 0.0
    if max_dd > 1e-9:
        calmar = cagr / max_dd
    else:
        calmar = float("inf") if cagr > 0 else 0.0
    worst20 = _worst_n_trade_dd(equity, window=20)
    return {
        "n_trades": len(trades),
        "total_return": total_ret,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "ulcer_index": ulcer,
        "worst_20_trade_dd": worst20,
        "final_equity": final,
        "n_calendar_days": n_calendar_days,
    }


def select_fraction_from_train(train_grid: dict[float, dict]) -> tuple[float, str]:
    """Choose f from SIZING_GRID maximizing train CAGR subject to
    max_dd <= MAX_DD_THRESHOLD_TRAIN. Fall back to smallest f if none
    pass.
    """
    viable = [(f, m) for f, m in train_grid.items()
              if m["max_dd"] <= MAX_DD_THRESHOLD_TRAIN]
    if not viable:
        f_min = min(train_grid.keys())
        return f_min, f"fallback_smallest_f (all exceed {MAX_DD_THRESHOLD_TRAIN:.0%} DD)"
    f_best, _ = max(viable, key=lambda kv: kv[1]["cagr"])
    return f_best, "max_cagr_subject_to_dd_cap"


# ---------------------------------------------------------------------------
# Trade-stream generation per fold
# ---------------------------------------------------------------------------

def _train_rf_with_oob(features: np.ndarray, labels: np.ndarray, seed: int):
    if features.shape[0] == 0:
        return None
    model = RandomForestRegressor(
        n_estimators=v2_mod.RF_N_ESTIMATORS,
        max_depth=v2_mod.RF_MAX_DEPTH,
        min_samples_leaf=v2_mod.RF_MIN_SAMPLES_LEAF,
        random_state=seed, n_jobs=-1,
        oob_score=True, bootstrap=True,
    )
    model.fit(features, labels)
    return model


def collect_train_samples_with_meta(
    *,
    fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config: V1BGateConfig,
    first15_range_gate: float,
    idx_first15_range: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, str, float, float]]]:
    """Collect train candidates with realized trades. Metadata records
    (day, local_i, side, net_pct, net_pnl_dollars) per sample. Only
    gate-passing days contribute — this matches V2-pruned-gated's
    deployment profile on train (though V1B gate selection itself ran
    on the full train_days upstream).

    Wait: the user asked us to NOT retrain on gated train only — that
    branch already failed. But for the sizing audit, we need a train
    TRADE STREAM that mirrors what the deployed strategy would look
    like on train days. So: train RF on all admissible samples (same
    as V2-pruned-gated), then apply the gate at train inference time
    to mirror deployment. We do this below at inference, not here.
    """
    feats, labels, _ = v2_mod.collect_training_samples(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    # We need metadata (day, local_i, side, label, dollars) per sample.
    # v2_mod.collect_training_samples doesn't return metadata. Re-walk the
    # train_days and collect in parallel.
    meta: list[tuple[str, int, str, float, float]] = []
    for day in fold_spec.train_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        X_sim_day = X_sim[ds:de]
        cands = v2_mod._enumerate_candidates_on_day(
            sc=sc, X_day=X[ds:de], X_sim_day=X_sim_day,
            spot_day=spot_prices[ds:de], idx_map=idx_map,
            core_indices=core_indices, config=config,
        )
        for cand in cands:
            trade, _ = v2_mod._realize_trade(
                day=day, fold_idx=fold_spec.fold_idx,
                strategy_label="train_sample", paired_trade_id=-1,
                X_sim_day=X_sim_day, spot_day=spot_prices[ds:de],
                idx_map=idx_map, sc=sc,
                entry_local=cand["local_i"], side=cand["side"],
                picked=cand["picked"], context_spread=cand["context_spread"],
            )
            if trade is None:
                continue
            meta.append((day, int(cand["local_i"]), cand["side"],
                         float(trade.net_pct), float(trade.net_pnl_dollars)))
    assert len(meta) == feats.shape[0], \
        f"meta/feats mismatch: {len(meta)} vs {feats.shape[0]}"
    return feats, labels, meta


def build_train_trade_stream_oob(
    *,
    fold_idx: int,
    metadata: list[tuple[str, int, str, float, float]],
    oob_preds: np.ndarray,
    day_to_range: dict[str, tuple[int, int]],
    X_sim: np.ndarray,
    idx_first15_range: int,
    first15_range_gate: float,
) -> list[GatedTrade]:
    """For each gate-passing train day, find the OOB-argmax sample; its
    label is the realized net_pct.
    """
    # For each day, find best OOB prediction over samples belonging to that day
    by_day_best: dict[str, tuple[float, int]] = {}  # day -> (oob_pred, sample_idx)
    for i, (day, local_i, side, net_pct, dollars) in enumerate(metadata):
        p = float(oob_preds[i])
        if not np.isfinite(p):
            continue
        cur = by_day_best.get(day)
        if cur is None or p > cur[0]:
            by_day_best[day] = (p, i)

    # Filter to gate-passing days
    kept_day_trades: list[tuple[str, int]] = []
    for day, (pred, idx) in by_day_best.items():
        if day not in day_to_range:
            continue
        ds, de = day_to_range[day]
        ok, _ = day_passes_gate(X_sim[ds:de], idx_first15_range)
        if ok:
            kept_day_trades.append((day, idx))

    trades: list[GatedTrade] = []
    for day, idx in kept_day_trades:
        m_day, m_local, m_side, m_net, m_dollars = metadata[idx]
        ds, de = day_to_range[day]
        first15_val = float(X_sim[ds + 15, idx_first15_range]) if de - ds > 15 else 0.0
        t = GatedTrade(
            date=day, fold=fold_idx,
            strategy_label="train_strategy", paired_trade_id=-1,
            bar_entry=int(m_local), side=m_side, strike=0.0, contract_idx=0,
            delta_at_entry=0.0, spread_at_entry=0.0, context_spread_at_entry=0.0,
            entry_mid=0.0, trigger_vwap_dist_min_prior=0.0,
            trigger_vwap_dist_now=0.0, first15_acceptance_at_entry=0.0,
            bar_delta_at_entry=0.0, bar_exit=0, exit_mid=0.0, bars_held=0,
            exit_reason="", gross_pct=0.0, spread_cost_pct=0.0,
            net_pct=float(m_net), net_pnl_dollars=float(m_dollars),
            vix_regime_at_entry=0.0, predicted_net_pct=float(by_day_best[day][0]),
            score_rank_in_day=1, n_candidates_in_day=0,
            first15_range_pct=first15_val,
            pop_ctrl_a_mean_net_pct=0.0, pop_ctrl_a_n_bars=0,
            pop_ctrl_b_mean_net_pct=0.0, pop_ctrl_b_n_days=0,
            delta_vs_pop_A=0.0, delta_vs_pop_B=0.0,
        )
        trades.append(t)
    trades.sort(key=lambda t: t.date)
    return trades


def build_test_trade_stream(
    *,
    fold_spec,
    model,
    config: V1BGateConfig,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    idx_first15_range: int,
) -> list[GatedTrade]:
    trades: list[GatedTrade] = []
    for day in fold_spec.test_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        ds, de = day_to_range[day]
        X_sim_day = X_sim[ds:de]
        ok, _ = day_passes_gate(X_sim_day, idx_first15_range)
        if not ok:
            continue
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
        preds = model.predict(feat_mat)
        pos = int(np.argmax(preds))
        chosen = cands[pos]
        v2trade, _ = v2_mod._realize_trade(
            day=day, fold_idx=fold_spec.fold_idx,
            strategy_label="strategy", paired_trade_id=-1,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map, sc=sc,
            entry_local=chosen["local_i"], side=chosen["side"],
            picked=chosen["picked"], context_spread=chosen["context_spread"],
            predicted_net_pct=float(preds[pos]),
            score_rank=1, n_candidates=len(cands),
        )
        if v2trade is None:
            continue
        first15_val = float(X_sim_day[15, idx_first15_range]) if X_sim_day.shape[0] > 15 else 0.0
        trades.append(GatedTrade(
            **{**asdict(v2trade),
               "first15_range_pct": first15_val,
               "pop_ctrl_a_mean_net_pct": 0.0, "pop_ctrl_a_n_bars": 0,
               "pop_ctrl_b_mean_net_pct": 0.0, "pop_ctrl_b_n_days": 0,
               "delta_vs_pop_A": 0.0, "delta_vs_pop_B": 0.0},
        ))
    trades.sort(key=lambda t: t.date)
    return trades


# ---------------------------------------------------------------------------
# Per-fold driver
# ---------------------------------------------------------------------------

@dataclass
class FoldSizingResult:
    fold_idx: int
    window_id: str
    train_trades: list[GatedTrade]
    test_trades: list[GatedTrade]
    train_grid: dict[float, dict]         # f -> equity metrics
    test_grid: dict[float, dict]
    selected_f: float
    selection_rationale: str
    train_days_horizon: int
    test_days_horizon: int


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
) -> FoldSizingResult:
    t0 = time.time()
    fold_idx = fold_spec.fold_idx
    print(f"\n=== Fold {fold_idx} | window={fold_spec.window_id} ===", flush=True)

    # V1B gate selection (unchanged)
    config, _diag, _n_tr, _n_ent = v2_mod.select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    assert config is not None, "V1B gate selection must succeed on real data"

    idx_first15 = idx_map["first15_range_pct"]

    # Train samples + metadata, RF with OOB
    feats, labels, meta = collect_train_samples_with_meta(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=config,
        first15_range_gate=FIRST15_RANGE_GATE, idx_first15_range=idx_first15,
    )
    model = _train_rf_with_oob(feats, labels, seed=fold_idx * 10007 + 1)
    assert model is not None
    oob_preds = model.oob_prediction_
    print(f"  train samples n={feats.shape[0]}, OOB valid "
          f"{int(np.isfinite(oob_preds).sum())}", flush=True)

    train_trades = build_train_trade_stream_oob(
        fold_idx=fold_idx, metadata=meta, oob_preds=oob_preds,
        day_to_range=day_to_range, X_sim=X_sim,
        idx_first15_range=idx_first15, first15_range_gate=FIRST15_RANGE_GATE,
    )
    test_trades = build_test_trade_stream(
        fold_spec=fold_spec, model=model, config=config,
        day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices,
        idx_first15_range=idx_first15,
    )

    # Calendar-day horizons for CAGR annualization
    train_days_horizon = len(fold_spec.train_days)
    test_days_horizon = len(fold_spec.test_days)

    # Grid evaluation
    train_grid: dict[float, dict] = {}
    test_grid: dict[float, dict] = {}
    for f in SIZING_GRID:
        eq_tr = simulate_equity(train_trades, f)
        eq_te = simulate_equity(test_trades, f)
        train_grid[f] = equity_metrics(eq_tr, train_trades, train_days_horizon)
        test_grid[f] = equity_metrics(eq_te, test_trades, test_days_horizon)

    f_sel, rationale = select_fraction_from_train(train_grid)
    print(f"  train trades n={len(train_trades)}, test trades n={len(test_trades)}",
          flush=True)
    print(f"  train grid:", flush=True)
    for f, m in train_grid.items():
        mark = " << selected" if f == f_sel else ""
        print(f"    f={f:.4f}: total={m['total_return']:+.4f}, "
              f"cagr={m['cagr']:+.4f}, max_dd={m['max_dd']:.4f}, "
              f"calmar={m['calmar']:+.3f}{mark}", flush=True)
    print(f"  test  grid:", flush=True)
    for f, m in test_grid.items():
        mark = " << applied (selected from train)" if f == f_sel else ""
        print(f"    f={f:.4f}: total={m['total_return']:+.4f}, "
              f"cagr={m['cagr']:+.4f}, max_dd={m['max_dd']:.4f}, "
              f"calmar={m['calmar']:+.3f}, "
              f"worst20={m['worst_20_trade_dd']:.4f}{mark}", flush=True)

    return FoldSizingResult(
        fold_idx=fold_idx, window_id=fold_spec.window_id,
        train_trades=train_trades, test_trades=test_trades,
        train_grid=train_grid, test_grid=test_grid,
        selected_f=f_sel, selection_rationale=rationale,
        train_days_horizon=train_days_horizon,
        test_days_horizon=test_days_horizon,
    )


# ---------------------------------------------------------------------------
# Kelly appendix (descriptive, clipped)
# ---------------------------------------------------------------------------

def kelly_clipped(trades: list[GatedTrade], clip_high: float = 0.02) -> dict:
    """Descriptive continuous-Kelly estimate: f* ~= mean / variance.
    Clipped to [0, clip_high]. NOT authoritative — just a reference.
    """
    if not trades:
        return {"kelly_raw": 0.0, "kelly_clipped": 0.0, "mean": 0.0, "var": 0.0}
    x = np.array([t.net_pct for t in trades], dtype=np.float64)
    mean = float(x.mean())
    var = float(x.var(ddof=1)) if len(x) > 1 else 0.0
    kelly_raw = mean / var if var > 1e-9 else 0.0
    kelly_clipped = max(0.0, min(clip_high, kelly_raw))
    return {"kelly_raw": kelly_raw, "kelly_clipped": kelly_clipped,
            "mean": mean, "var": var}


# ---------------------------------------------------------------------------
# Aggregate: cross-fold test equity curve with per-fold selected f
# ---------------------------------------------------------------------------

def aggregate_test_equity_with_per_fold_f(
    fold_results: list[FoldSizingResult],
) -> tuple[np.ndarray, list[GatedTrade], dict]:
    all_trades: list[GatedTrade] = []
    for fr in fold_results:
        for t in fr.test_trades:
            all_trades.append((t, fr.selected_f))
    all_trades.sort(key=lambda pair: pair[0].date)
    equity = np.empty(len(all_trades) + 1, dtype=np.float64)
    equity[0] = 1.0
    per_fold_f_applied = []
    flat_trades: list[GatedTrade] = []
    for i, (t, f) in enumerate(all_trades):
        e = equity[i] * (1.0 + f * float(t.net_pct))
        equity[i + 1] = max(e, 0.0)
        per_fold_f_applied.append(f)
        flat_trades.append(t)
    total_cal_days = sum(fr.test_days_horizon for fr in fold_results)
    metrics = equity_metrics(equity, flat_trades, total_cal_days)
    metrics["per_fold_f_applied_unique"] = sorted(set(per_fold_f_applied))
    return equity, flat_trades, metrics


def aggregate_test_equity_fixed_f(
    fold_results: list[FoldSizingResult], f: float
) -> tuple[np.ndarray, list[GatedTrade], dict]:
    all_trades = [t for fr in fold_results for t in fr.test_trades]
    all_trades.sort(key=lambda t: t.date)
    equity = simulate_equity(all_trades, f)
    total_cal_days = sum(fr.test_days_horizon for fr in fold_results)
    return equity, all_trades, equity_metrics(equity, all_trades, total_cal_days)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    _activate_pruned_feature_set()
    print(f"[sizing] Active CORE_FEATURE_NAMES: {list(v2_mod.CORE_FEATURE_NAMES)}")
    print(f"[sizing] Gate: first15_range_pct >= {FIRST15_RANGE_GATE:.4f} "
          f"({FIRST15_RANGE_GATE_BPS} bps)")
    print(f"[sizing] Grid: f in {[f'{f:.4f}' for f in SIZING_GRID]}")
    print(f"[sizing] Selection: max train CAGR subject to max train DD <= "
          f"{MAX_DD_THRESHOLD_TRAIN:.0%}")

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

    fold_results: list[FoldSizingResult] = []
    for fs in folds:
        fold_results.append(run_fold(
            fold_spec=fs, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X=X, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, core_indices=core_idx,
        ))

    # Aggregate: per-fold f applied
    eq_sel, _tr_sel, agg_sel = aggregate_test_equity_with_per_fold_f(fold_results)

    # Aggregate: full grid fixed f
    agg_grid: dict[float, dict] = {}
    for f in SIZING_GRID:
        _, _, m = aggregate_test_equity_fixed_f(fold_results, f)
        agg_grid[f] = m

    print(f"\n=== Aggregate test equity curve — per-fold selected f ===")
    for k, v in agg_sel.items():
        if isinstance(v, float):
            print(f"  {k}: {v:+.5f}" if abs(v) < 1 else f"  {k}: {v:+.3f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n=== Aggregate test grid (fixed f across folds) ===")
    print(f"{'f':>8} {'n':>5} {'total':>10} {'cagr':>10} {'max_dd':>10} "
          f"{'calmar':>8} {'ulcer':>8} {'worst20':>9}")
    for f, m in agg_grid.items():
        print(f"{f:>8.4f} {m['n_trades']:>5} {m['total_return']:>+10.4f} "
              f"{m['cagr']:>+10.4f} {m['max_dd']:>10.4f} {m['calmar']:>+8.2f} "
              f"{m['ulcer_index']:>8.4f} {m['worst_20_trade_dd']:>9.4f}")

    # Kelly appendix
    all_test_trades = [t for fr in fold_results for t in fr.test_trades]
    kelly_test = kelly_clipped(all_test_trades)
    kelly_train = kelly_clipped(
        [t for fr in fold_results for t in fr.train_trades]
    )
    print(f"\n=== Kelly appendix (descriptive, clipped to [0, 2%]) ===")
    print(f"  test:  mean={kelly_test['mean']:+.5f}, var={kelly_test['var']:.5f}, "
          f"raw={kelly_test['kelly_raw']:+.4f}, clipped={kelly_test['kelly_clipped']:.4f}")
    print(f"  train: mean={kelly_train['mean']:+.5f}, var={kelly_train['var']:.5f}, "
          f"raw={kelly_train['kelly_raw']:+.4f}, clipped={kelly_train['kelly_clipped']:.4f}")

    # Dump
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint": dataset_fp,
        "gate_threshold_bps": FIRST15_RANGE_GATE_BPS,
        "sizing_grid": list(SIZING_GRID),
        "max_dd_threshold_train": MAX_DD_THRESHOLD_TRAIN,
        "aggregate_selected_per_fold": agg_sel,
        "aggregate_grid_test": {str(f): m for f, m in agg_grid.items()},
        "per_fold": [
            {
                "fold_idx": fr.fold_idx,
                "window_id": fr.window_id,
                "selected_f": fr.selected_f,
                "selection_rationale": fr.selection_rationale,
                "train_trade_count": len(fr.train_trades),
                "test_trade_count": len(fr.test_trades),
                "train_grid": {str(f): m for f, m in fr.train_grid.items()},
                "test_grid": {str(f): m for f, m in fr.test_grid.items()},
            }
            for fr in fold_results
        ],
        "kelly_appendix": {"test": kelly_test, "train": kelly_train},
        "note": "Fixed-fraction sizing overlay on frozen V2-pruned-gated trade "
                "stream. OOB predictions for train-side argmax to avoid "
                "in-sample optimism. No adaptive / score / regime sizing.",
    }
    write_json(os.path.join(args.out_dir, "summary.json"), payload)
    # Flat test trade CSV for downstream inspection
    all_test_flat = [t for fr in fold_results for t in fr.test_trades]
    write_csv(os.path.join(args.out_dir, "test_trades.csv"), all_test_flat, GATED_TRADE_FIELDS)

    print(f"\nWrote summary to {args.out_dir}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
