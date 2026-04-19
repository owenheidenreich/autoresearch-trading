"""Mechanical baseline V2 — learned scorer over the 12-core reversion features.

Replaces V1A/V1B's hand-coded reclaim/reject trigger with a
`sklearn.ensemble.RandomForestRegressor` trained per fold on simulated
trade outcomes at every V1B-admissible (bar, side) pair. Everything else
is frozen from V1B (gates, contract selection, exits, one trade per day,
two controls, falsification structure).

Plan: [v2/docs/mechanical_baseline_v2_plan.md].

Inputs at inference and training:
- 12 core features (from `data["X"]`, the rolling-z-score normalized
  tensor) plus a side indicator.

Training data (per fold):
- Walk `fold.train_days`, at each V1B-gate-passing bar enumerate both
  sides, simulate each trade, record (features, side, net_pct).

Inference (per test day):
- Enumerate V1B-gate-passing bars × sides; score each candidate with the
  trained model; take argmax as the day's trade.

Controls use the V2 trade's (bar, side) choice:
- A = same-day, random-bar, same-side, V1B gates applied
- B = same-bar, random other test-day, same-side, V1B gates applied
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from v2.core.chain_data import load_sidecar_cached, sidecar_path
from v2.core.walkforward import CANONICAL_N_FOLDS

from v2.analysis.mechanical_baseline_opening_reversion import (
    BAR_LO,
    BAR_HI,
    BaselineTrade,
    SkipRecord,
    SKIP_FIELDS,
    TRADE_FIELDS,
    build_folds,
    compute_pnl,
    context_spread_at_bar,
    day_ranges,
    falsification_verdict as falsification_verdict_base,
    feature_index_map,
    first15_levels,
    load_data,
    run_exit,
    select_contract,
    summarize_trades,
    write_csv,
    write_json,
)
from v2.analysis.mechanical_baseline_v1b_opening_reversion import (
    V1BGateConfig,
    check_gates,
    evaluate_triggers_on_day,
    pick_best_config,
)


# ---------------------------------------------------------------------------
# V2 constants
# ---------------------------------------------------------------------------

CORE_FEATURE_NAMES = (
    "vwap_dist",
    "vwap_reclaim_state",
    "vwap_slope",
    "opening_gap_pct",
    "session_open_dist",
    "first15_close_position",
    "first15_acceptance",
    "bar_delta",
    "volume_climax_signal",
    "atm_gamma",
    "atm_theta_per_bar",
    "option_spread_pct",
)
SIDE_INDICATOR_IDX = len(CORE_FEATURE_NAMES)  # 12
N_MODEL_FEATURES = len(CORE_FEATURE_NAMES) + 1  # 13

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 20

EXPERIMENT_ID = "mechbase_opening_reversion_v2_learned"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2"


# ---------------------------------------------------------------------------
# Extended trade record (adds V2-specific fields)
# ---------------------------------------------------------------------------

@dataclass
class V2Trade(BaselineTrade):
    predicted_net_pct: float = 0.0
    score_rank_in_day: int = -1      # 1-indexed rank (1 == argmax)
    n_candidates_in_day: int = 0


V2_TRADE_FIELDS = list(V2Trade.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# Core-feature extraction
# ---------------------------------------------------------------------------

def core_feature_indices(idx_map: dict[str, int]) -> list[int]:
    missing = [n for n in CORE_FEATURE_NAMES if n not in idx_map]
    if missing:
        raise ValueError(f"feature_names missing core entries: {missing}")
    return [idx_map[n] for n in CORE_FEATURE_NAMES]


def feature_row(
    X_day: np.ndarray,
    local_i: int,
    core_indices: list[int],
    side: str,
) -> np.ndarray:
    row = np.zeros(N_MODEL_FEATURES, dtype=np.float32)
    row[:len(core_indices)] = X_day[local_i, core_indices]
    row[SIDE_INDICATOR_IDX] = 1.0 if side == "C" else -1.0
    return row


# ---------------------------------------------------------------------------
# Build a single realized trade record given all ingredients
# ---------------------------------------------------------------------------

def _realize_trade(
    *,
    day: str,
    fold_idx: int,
    strategy_label: str,
    paired_trade_id: int,
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
    sc: dict[str, Any],
    entry_local: int,
    side: str,
    picked: dict,
    context_spread: float,
    predicted_net_pct: float = 0.0,
    score_rank: int = -1,
    n_candidates: int = 0,
) -> tuple[V2Trade | None, str]:
    day_n_bars = X_sim_day.shape[0]
    first15_hi, first15_lo = first15_levels(spot_day)
    idx_vwap_dist = idx_map["vwap_dist"]
    idx_vix_regime = idx_map["vix_regime"]
    idx_bar_delta = idx_map["bar_delta"]
    idx_first15_accept = idx_map["first15_acceptance"]

    vix_regime = float(X_sim_day[entry_local, idx_vix_regime])
    mtc_bars = max(1.0, 390.0 - float(entry_local))
    exit_local, exit_reason, exit_mid = run_exit(
        sc=sc,
        contract_idx=picked["contract_idx"],
        entry_local=entry_local,
        side=side,
        X_sim_day=X_sim_day,
        spot_day=spot_day,
        idx_vwap_dist=idx_vwap_dist,
        first15_hi=first15_hi,
        first15_lo=first15_lo,
        day_n_bars=day_n_bars,
    )
    if exit_local is None:
        return None, exit_reason
    pnl = compute_pnl(picked["entry_mid"], exit_mid, mtc_bars, vix_regime)
    trade = V2Trade(
        date=day,
        fold=fold_idx,
        strategy_label=strategy_label,
        paired_trade_id=paired_trade_id,
        bar_entry=int(entry_local),
        side=side,
        strike=picked["strike"],
        contract_idx=picked["contract_idx"],
        delta_at_entry=picked["delta"],
        spread_at_entry=picked["spread_fraction"],
        context_spread_at_entry=float(context_spread),
        entry_mid=picked["entry_mid"],
        trigger_vwap_dist_min_prior=0.0,             # n/a for V2
        trigger_vwap_dist_now=float(X_sim_day[entry_local, idx_vwap_dist]),
        first15_acceptance_at_entry=float(X_sim_day[entry_local, idx_first15_accept]),
        bar_delta_at_entry=float(X_sim_day[entry_local, idx_bar_delta]),
        bar_exit=int(exit_local),
        exit_mid=float(exit_mid),
        bars_held=int(exit_local - entry_local),
        exit_reason=exit_reason,
        gross_pct=pnl["gross_pct"],
        spread_cost_pct=pnl["spread_cost_pct"],
        net_pct=pnl["net_pct"],
        net_pnl_dollars=pnl["net_pnl_dollars"],
        vix_regime_at_entry=float(vix_regime),
        predicted_net_pct=float(predicted_net_pct),
        score_rank_in_day=int(score_rank),
        n_candidates_in_day=int(n_candidates),
    )
    return trade, "ok"


# ---------------------------------------------------------------------------
# Training sample collection
# ---------------------------------------------------------------------------

def _enumerate_candidates_on_day(
    *,
    sc: dict[str, Any],
    X_day: np.ndarray,
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config: V1BGateConfig,
) -> list[dict]:
    """Enumerate every V1B-admissible (bar, side) candidate on a day.

    For each candidate, record:
    - local_i, side
    - feature_row (13-dim)
    - picked (contract dict)
    - context_spread
    - spot_day ref (unchanged per-day)
    """
    day_n_bars = X_sim_day.shape[0]
    idx_iv_pct = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]
    idx_option_spread = idx_map["option_spread_pct"]

    candidates: list[dict] = []
    for local_i in range(BAR_LO, min(BAR_HI + 1, day_n_bars)):
        ok, _reason = check_gates(X_sim_day, local_i, idx_iv_pct, idx_vrp, config)
        if not ok:
            continue
        context_spread = context_spread_at_bar(X_sim_day, local_i, idx_option_spread)
        for side in ("C", "P"):
            picked, reason = select_contract(sc, local_i, side, context_spread)
            if picked is None:
                continue
            candidates.append({
                "local_i": int(local_i),
                "side": side,
                "features": feature_row(X_day, local_i, core_indices, side),
                "picked": picked,
                "context_spread": float(context_spread),
            })
    return candidates


def collect_training_samples(
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
) -> tuple[np.ndarray, np.ndarray, int]:
    """Walk train_days, simulate each admissible (bar, side), return
    (feature_matrix, net_pct_labels, n_days_contributing).
    """
    rows: list[np.ndarray] = []
    labels: list[float] = []
    days_with_samples = 0
    for day in fold_spec.train_days:
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
        cands = _enumerate_candidates_on_day(
            sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, core_indices=core_indices, config=config,
        )
        had_sample = False
        for cand in cands:
            trade, status = _realize_trade(
                day=day, fold_idx=fold_spec.fold_idx,
                strategy_label="train_sample", paired_trade_id=-1,
                X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
                sc=sc, entry_local=cand["local_i"], side=cand["side"],
                picked=cand["picked"], context_spread=cand["context_spread"],
            )
            if trade is None:
                continue
            rows.append(cand["features"])
            labels.append(trade.net_pct)
            had_sample = True
        if had_sample:
            days_with_samples += 1
    if not rows:
        return np.zeros((0, N_MODEL_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.float32), 0
    feat_mat = np.vstack(rows).astype(np.float32)
    label_vec = np.array(labels, dtype=np.float32)
    return feat_mat, label_vec, days_with_samples


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_scorer(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> RandomForestRegressor | None:
    if features.shape[0] == 0 or labels.shape[0] == 0:
        return None
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


# ---------------------------------------------------------------------------
# Test-day inference and strategy-trade realization
# ---------------------------------------------------------------------------

def run_v2_strategy_day(
    *,
    day: str,
    fold_idx: int,
    sc: dict[str, Any],
    X_day: np.ndarray,
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config: V1BGateConfig,
    model: RandomForestRegressor,
) -> tuple[list[V2Trade], list[SkipRecord]]:
    skips: list[SkipRecord] = []
    cands = _enumerate_candidates_on_day(
        sc=sc, X_day=X_day, X_sim_day=X_sim_day, spot_day=spot_day,
        idx_map=idx_map, core_indices=core_indices, config=config,
    )
    if not cands:
        # Record the day as 'no_candidates'. One record per day is enough.
        skips.append(SkipRecord(day, fold_idx, "strategy", -1, "X", "no_v1b_candidates"))
        return [], skips

    feats = np.vstack([c["features"] for c in cands]).astype(np.float32)
    preds = model.predict(feats)
    order = np.argsort(-preds)                      # descending
    chosen_pos = int(order[0])
    chosen = cands[chosen_pos]
    predicted = float(preds[chosen_pos])

    trade, status = _realize_trade(
        day=day, fold_idx=fold_idx,
        strategy_label="strategy", paired_trade_id=-1,
        X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
        sc=sc, entry_local=chosen["local_i"], side=chosen["side"],
        picked=chosen["picked"], context_spread=chosen["context_spread"],
        predicted_net_pct=predicted,
        score_rank=1,
        n_candidates=len(cands),
    )
    if trade is None:
        skips.append(SkipRecord(day, fold_idx, "strategy", chosen["local_i"], chosen["side"], status))
        return [], skips
    return [trade], skips


# ---------------------------------------------------------------------------
# Controls (apply V1B gates and V2's chosen side)
# ---------------------------------------------------------------------------

def run_v2_control_A(
    *,
    strategy_trades: list[V2Trade],
    fold_idx: int,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    config: V1BGateConfig,
    seed: int,
) -> tuple[list[V2Trade], list[SkipRecord]]:
    rng = np.random.default_rng(seed)
    trades: list[V2Trade] = []
    skips: list[SkipRecord] = []
    idx_iv_pct = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]
    idx_option_spread = idx_map["option_spread_pct"]
    for pid, st in enumerate(strategy_trades):
        sc = load_sidecar_cached(sidecar_path(sidecar_dir, st.date))
        ds, de = day_to_range[st.date]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        day_n_bars = X_sim_day.shape[0]
        lo, hi = BAR_LO, min(BAR_HI, day_n_bars - 1)
        if hi < lo:
            skips.append(SkipRecord(st.date, fold_idx, "control_A", -1, st.side, "day_too_short"))
            continue
        entry_local = int(rng.integers(lo, hi + 1))
        ok, reason = check_gates(X_sim_day, entry_local, idx_iv_pct, idx_vrp, config)
        if not ok:
            skips.append(SkipRecord(st.date, fold_idx, "control_A", entry_local, st.side, reason))
            continue
        context_spread = context_spread_at_bar(X_sim_day, entry_local, idx_option_spread)
        picked, reason = select_contract(sc, entry_local, st.side, context_spread)
        if picked is None:
            skips.append(SkipRecord(st.date, fold_idx, "control_A", entry_local, st.side, reason))
            continue
        trade, status = _realize_trade(
            day=st.date, fold_idx=fold_idx,
            strategy_label="control_A", paired_trade_id=pid,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
            sc=sc, entry_local=entry_local, side=st.side,
            picked=picked, context_spread=context_spread,
        )
        if trade is None:
            skips.append(SkipRecord(st.date, fold_idx, "control_A", entry_local, st.side, status))
            continue
        trades.append(trade)
    return trades, skips


def run_v2_control_B(
    *,
    strategy_trades: list[V2Trade],
    fold_idx: int,
    fold_test_days: list[str],
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    config: V1BGateConfig,
    seed: int,
) -> tuple[list[V2Trade], list[SkipRecord]]:
    rng = np.random.default_rng(seed)
    trades: list[V2Trade] = []
    skips: list[SkipRecord] = []
    idx_iv_pct = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]
    idx_option_spread = idx_map["option_spread_pct"]
    available = [d for d in fold_test_days if d in day_to_range]
    if len(available) < 2:
        return trades, skips
    for pid, st in enumerate(strategy_trades):
        pool = [d for d in available if d != st.date]
        if not pool:
            skips.append(SkipRecord(st.date, fold_idx, "control_B", st.bar_entry, st.side, "no_other_day"))
            continue
        pick_day = pool[int(rng.integers(0, len(pool)))]
        sc = load_sidecar_cached(sidecar_path(sidecar_dir, pick_day))
        ds, de = day_to_range[pick_day]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        entry_local = st.bar_entry
        day_n_bars = X_sim_day.shape[0]
        if entry_local >= day_n_bars:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", entry_local, st.side, "out_of_window"))
            continue
        ok, reason = check_gates(X_sim_day, entry_local, idx_iv_pct, idx_vrp, config)
        if not ok:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", entry_local, st.side, reason))
            continue
        context_spread = context_spread_at_bar(X_sim_day, entry_local, idx_option_spread)
        picked, reason = select_contract(sc, entry_local, st.side, context_spread)
        if picked is None:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", entry_local, st.side, reason))
            continue
        trade, status = _realize_trade(
            day=pick_day, fold_idx=fold_idx,
            strategy_label="control_B", paired_trade_id=pid,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
            sc=sc, entry_local=entry_local, side=st.side,
            picked=picked, context_spread=context_spread,
        )
        if trade is None:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", entry_local, st.side, status))
            continue
        trades.append(trade)
    return trades, skips


# ---------------------------------------------------------------------------
# Gate selection — reuse V1B full pipeline
# ---------------------------------------------------------------------------

def select_gates_for_fold(
    *,
    fold_spec,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
) -> tuple[V1BGateConfig | None, list[dict], int, int]:
    per_day_evals: dict[str, list] = {}
    for day in fold_spec.train_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        per_day_evals[day] = evaluate_triggers_on_day(
            day=day, fold_idx=fold_spec.fold_idx, sc=sc,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
        )
    n_triggered = sum(len(v) for v in per_day_evals.values())
    n_entered = sum(1 for v in per_day_evals.values() for e in v if e.entered)
    config, diag = pick_best_config(per_day_evals)
    return config, diag, n_triggered, n_entered


# ---------------------------------------------------------------------------
# Fold driver
# ---------------------------------------------------------------------------

def run_v2_fold(
    *,
    idx_map: dict[str, int],
    core_indices: list[int],
    day_to_range: dict[str, tuple[int, int]],
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    fold_spec,
    sidecar_dir: str,
    out_dir: str,
    run_controls: bool,
) -> dict:
    t0 = time.time()
    fold_idx = fold_spec.fold_idx
    print(f"\n=== Fold {fold_idx} | window={fold_spec.window_id} ===", flush=True)
    print(f"  train_days={len(fold_spec.train_days)} "
          f"({fold_spec.train_days[0]} → {fold_spec.train_days[-1]})")
    print(f"  test_days={len(fold_spec.test_days)} "
          f"({fold_spec.test_days[0]} → {fold_spec.test_days[-1]})")

    # --- Gate selection (reuse V1B full pipeline on train_days) ---
    config, grid_diag, n_trig, n_entered = select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    if config is None:
        print("  NO VIABLE GATE CONFIG — fold skipped", flush=True)
        return {
            "fold_idx": fold_idx, "window_id": fold_spec.window_id,
            "selected_config": None, "grid_diagnostics": grid_diag,
            "strategy_trades": [], "control_A_trades": [], "control_B_trades": [],
            "skips": [],
            "strategy_summary": summarize_trades([]),
            "control_A_summary": summarize_trades([]),
            "control_B_summary": summarize_trades([]),
            "feature_importance": [],
            "elapsed_sec": time.time() - t0,
        }
    print(f"  selected gates: iv_max={config.iv_max} vrp_max={config.vrp_max:+.5f} "
          f"({config.vrp_source}), train_n={config.train_n}, "
          f"train_mean_net_pct={config.train_mean_net_pct:+.5f}", flush=True)

    # --- Training data collection (every admissible (bar, side) on train) ---
    print("  collecting training samples...", flush=True)
    t1 = time.time()
    feats, labels, n_train_days = collect_training_samples(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
        core_indices=core_indices, config=config,
    )
    print(f"  train samples: n={feats.shape[0]}, "
          f"days_contributing={n_train_days}, "
          f"label_mean={float(labels.mean()) if len(labels) else 0.0:+.5f}, "
          f"label_std={float(labels.std()) if len(labels) else 0.0:.5f}, "
          f"elapsed={time.time() - t1:.1f}s", flush=True)

    # --- Train the scorer ---
    seed = fold_idx * 10007 + 1
    model = train_scorer(feats, labels, seed=seed)
    if model is None:
        print("  NO TRAINING SAMPLES — fold skipped", flush=True)
        return {
            "fold_idx": fold_idx, "window_id": fold_spec.window_id,
            "selected_config": asdict(config), "grid_diagnostics": grid_diag,
            "strategy_trades": [], "control_A_trades": [], "control_B_trades": [],
            "skips": [],
            "strategy_summary": summarize_trades([]),
            "control_A_summary": summarize_trades([]),
            "control_B_summary": summarize_trades([]),
            "feature_importance": [],
            "elapsed_sec": time.time() - t0,
        }
    fi = [
        {"feature": (CORE_FEATURE_NAMES + ("side_indicator",))[i],
         "importance": float(model.feature_importances_[i])}
        for i in range(N_MODEL_FEATURES)
    ]

    # --- Test inference ---
    strategy_trades: list[V2Trade] = []
    all_skips: list[SkipRecord] = []
    for day in fold_spec.test_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        day_trades, day_skips = run_v2_strategy_day(
            day=day, fold_idx=fold_idx, sc=sc,
            X_day=X[ds:de], X_sim_day=X_sim[ds:de], spot_day=spot_prices[ds:de],
            idx_map=idx_map, core_indices=core_indices,
            config=config, model=model,
        )
        strategy_trades.extend(day_trades)
        all_skips.extend(day_skips)

    strat_summary = summarize_trades(strategy_trades)
    print(f"  strategy: n={strat_summary['n']}, "
          f"target={strat_summary['target_hit_frac']:.3f}, "
          f"stop={strat_summary['stop_hit_frac']:.3f}, "
          f"mean_net_pct={strat_summary['mean_net_pct']:+.5f}, "
          f"dollar_pf={strat_summary['dollar_pf']:.3f}", flush=True)

    # --- Controls (apply V1B gates + V2's chosen side) ---
    ctrl_A_trades: list[V2Trade] = []
    ctrl_B_trades: list[V2Trade] = []
    if run_controls and strategy_trades:
        ctrl_A_trades, ctrl_A_skips = run_v2_control_A(
            strategy_trades=strategy_trades, fold_idx=fold_idx,
            day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
            config=config, seed=seed,
        )
        ctrl_B_trades, ctrl_B_skips = run_v2_control_B(
            strategy_trades=strategy_trades, fold_idx=fold_idx,
            fold_test_days=fold_spec.test_days, day_to_range=day_to_range,
            sidecar_dir=sidecar_dir, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, config=config, seed=seed + 1,
        )
        all_skips.extend(ctrl_A_skips)
        all_skips.extend(ctrl_B_skips)

    ctrl_A_summary = summarize_trades(ctrl_A_trades)
    ctrl_B_summary = summarize_trades(ctrl_B_trades)
    if run_controls:
        print(f"  control_A: n={ctrl_A_summary['n']}, "
              f"mean_net_pct={ctrl_A_summary['mean_net_pct']:+.5f}, "
              f"dollar_pf={ctrl_A_summary['dollar_pf']:.3f}", flush=True)
        print(f"  control_B: n={ctrl_B_summary['n']}, "
              f"mean_net_pct={ctrl_B_summary['mean_net_pct']:+.5f}, "
              f"dollar_pf={ctrl_B_summary['dollar_pf']:.3f}", flush=True)

    all_trades = strategy_trades + ctrl_A_trades + ctrl_B_trades
    write_csv(os.path.join(out_dir, f"trades_fold{fold_idx}.csv"), all_trades, V2_TRADE_FIELDS)

    write_json(os.path.join(out_dir, f"report_fold{fold_idx}.json"), {
        "experiment_id": EXPERIMENT_ID,
        "fold_idx": fold_idx,
        "window_id": fold_spec.window_id,
        "selected_config": asdict(config),
        "grid_diagnostics": grid_diag,
        "feature_importance": fi,
        "n_train_samples": int(feats.shape[0]),
        "n_train_days_contributing": int(n_train_days),
        "strategy_summary": strat_summary,
        "control_A_summary": ctrl_A_summary,
        "control_B_summary": ctrl_B_summary,
        "strategy_trades": [asdict(t) for t in strategy_trades],
    })

    return {
        "fold_idx": fold_idx,
        "window_id": fold_spec.window_id,
        "selected_config": asdict(config),
        "grid_diagnostics": grid_diag,
        "feature_importance": fi,
        "n_train_samples": int(feats.shape[0]),
        "n_train_days_contributing": int(n_train_days),
        "strategy_summary": strat_summary,
        "control_A_summary": ctrl_A_summary,
        "control_B_summary": ctrl_B_summary,
        "strategy_trades": strategy_trades,
        "control_A_trades": ctrl_A_trades,
        "control_B_trades": ctrl_B_trades,
        "skips": all_skips,
        "elapsed_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--fold", default="all", help="fold index (0..4) or 'all'")
    ap.add_argument("--controls", dest="controls", action="store_true", default=True)
    ap.add_argument("--no-controls", dest="controls", action="store_false")
    args = ap.parse_args()

    t0 = time.time()
    data = load_data(args.data)

    dates = list(data["dates"])
    feature_names = list(data["feature_names"])
    idx_map = feature_index_map(feature_names)
    required = list(CORE_FEATURE_NAMES) + [
        "vix_regime", "iv_percentile", "vrp",
    ]
    missing = [n for n in required if n not in idx_map]
    if missing:
        print(f"FATAL: feature_names missing: {missing}", file=sys.stderr)
        return 2
    core_idx = core_feature_indices(idx_map)

    X = data["X"].numpy() if isinstance(data["X"], torch.Tensor) else np.asarray(data["X"])
    X_sim = data["X_sim"].numpy() if isinstance(data["X_sim"], torch.Tensor) else np.asarray(data["X_sim"])
    spot_prices = data["spot_prices"].numpy() if isinstance(data["spot_prices"], torch.Tensor) else np.asarray(data["spot_prices"])

    day_to_range, _ = day_ranges(dates)
    sidecar_dir = data["metadata"]["chain_sidecar_dir"]
    dataset_fp = str(data["metadata"].get("fingerprint", "unknown"))

    folds = build_folds(dates)
    if args.fold == "all":
        selected = folds
    else:
        try:
            idx = int(args.fold)
        except ValueError:
            print(f"FATAL: --fold must be int 0..{CANONICAL_N_FOLDS - 1} or 'all'", file=sys.stderr)
            return 2
        selected = [f for f in folds if f.fold_idx == idx]
        if not selected:
            print(f"FATAL: fold {idx} not found", file=sys.stderr)
            return 2

    os.makedirs(args.out_dir, exist_ok=True)
    fold_results: list[dict] = []
    for fs in selected:
        fold_results.append(run_v2_fold(
            idx_map=idx_map, core_indices=core_idx,
            day_to_range=day_to_range, X=X, X_sim=X_sim, spot_prices=spot_prices,
            fold_spec=fs, sidecar_dir=sidecar_dir, out_dir=args.out_dir,
            run_controls=args.controls,
        ))

    all_strat: list[V2Trade] = []
    all_A: list[V2Trade] = []
    all_B: list[V2Trade] = []
    all_skips: list[SkipRecord] = []
    for r in fold_results:
        all_strat.extend(r["strategy_trades"])
        all_A.extend(r["control_A_trades"])
        all_B.extend(r["control_B_trades"])
        all_skips.extend(r["skips"])

    write_csv(os.path.join(args.out_dir, "trades.csv"),
              all_strat + all_A + all_B, V2_TRADE_FIELDS)
    write_csv(os.path.join(args.out_dir, "skips.csv"), all_skips, SKIP_FIELDS)

    strat_agg = summarize_trades(all_strat)
    A_agg = summarize_trades(all_A)
    B_agg = summarize_trades(all_B)
    verdict = falsification_verdict_base(strat_agg, A_agg, B_agg)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint": dataset_fp,
        "folds_run": [r["fold_idx"] for r in fold_results],
        "strategy_aggregate": strat_agg,
        "control_A_aggregate": A_agg,
        "control_B_aggregate": B_agg,
        "verdict": verdict,
        "per_fold": [
            {
                "fold_idx": r["fold_idx"],
                "window_id": r["window_id"],
                "selected_config": r["selected_config"],
                "n_train_samples": r.get("n_train_samples", 0),
                "n_train_days_contributing": r.get("n_train_days_contributing", 0),
                "feature_importance": r["feature_importance"],
                "strategy_summary": r["strategy_summary"],
                "control_A_summary": r["control_A_summary"],
                "control_B_summary": r["control_B_summary"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in fold_results
        ],
        "note": "Train-only RF fit on V1B-admissible (bar, side) samples. "
                "Test picks argmax per day. Primary comparator: Control A.",
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    write_json(os.path.join(args.out_dir, "controls.json"),
               {"control_A": A_agg, "control_B": B_agg,
                "per_fold": [{"fold_idx": r["fold_idx"],
                              "control_A": r["control_A_summary"],
                              "control_B": r["control_B_summary"]}
                             for r in fold_results]})

    print(f"\n=== V2 SUMMARY ({EXPERIMENT_ID}) ===")
    print(f"  strategy:  n={strat_agg['n']}, "
          f"target={strat_agg['target_hit_frac']:.3f}, "
          f"stop={strat_agg['stop_hit_frac']:.3f}, "
          f"mean_net_pct={strat_agg['mean_net_pct']:+.5f} "
          f"(±{strat_agg['mean_net_pct_stderr']:.5f}), "
          f"dollar_pf={strat_agg['dollar_pf']:.3f}")
    if all_A:
        print(f"  control_A: n={A_agg['n']}, mean_net_pct={A_agg['mean_net_pct']:+.5f}, "
              f"dollar_pf={A_agg['dollar_pf']:.3f}")
    if all_B:
        print(f"  control_B: n={B_agg['n']}, mean_net_pct={B_agg['mean_net_pct']:+.5f}, "
              f"dollar_pf={B_agg['dollar_pf']:.3f}")
    print(f"  verdict:   {verdict['verdict']}")
    for r in verdict.get("reasons", []):
        print(f"    - {r}")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    print(f"  outputs under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
