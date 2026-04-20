"""V2-pruned coverage-based thresholding.

Narrow extension of the V2-pruned mechanical baseline. Freezes everything
else (feature set, model class/hparams, V1B gates, candidate universe,
contract selection, exit engine, one-trade-per-day rule, folds, controls)
and adds only a score-based abstention layer.

Hypothesis under test:
    The V2-pruned score is rank-ordered enough that restricting trades
    to higher-score bars improves expectancy and PF versus the all-trades
    baseline.

Discipline (all locked):
- Coverage grid: {0.2, 0.4, 0.6, 0.8, 1.0} (fractions of train days admitted).
- Thresholds chosen per fold from `fold.train_days` ONLY, using OOB
  predictions from the RandomForest (honest train-side ranking).
- Fixed-coverage selection (not score-value) to avoid per-fold score drift.
- Per-fold best coverage = highest train mean_net_pct with n >= MIN_N_TRAIN_SELECT.
- Controls inherit strategy's abstention (paired to included days).

Plan: [v2/docs/mechanical_baseline_v2_plan.md] (V2 pipeline, unchanged).
V2-pruned promotion: `lab_notebook.md` entries on 2026-04-19.

Runs in ~1 minute on CPU.
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
from sklearn.ensemble import RandomForestRegressor

from v2.core.chain_data import load_sidecar_cached, sidecar_path
from v2.core.walkforward import CANONICAL_N_FOLDS
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
# Knobs
# ---------------------------------------------------------------------------

DROPPED_FEATURES = ("vwap_reclaim_state",)         # inherits V2-pruned
COVERAGE_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
MIN_N_TRAIN_SELECT = 20                             # min trades for a coverage
                                                    # to be selection-eligible
N_BOOTSTRAP = 1000
BASE_SEED = 42

EXPERIMENT_ID = "mechbase_opening_reversion_v2_pruned_thresholded"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_thresholded"


def _activate_pruned_feature_set() -> None:
    """Patch v2_mod's module globals so its helpers use the 12-feature set."""
    active = tuple(n for n in v2_mod.CORE_FEATURE_NAMES if n not in DROPPED_FEATURES)
    v2_mod.CORE_FEATURE_NAMES = active
    v2_mod.N_MODEL_FEATURES = len(active) + 1
    v2_mod.SIDE_INDICATOR_IDX = len(active)


# ---------------------------------------------------------------------------
# Extended trade record
# ---------------------------------------------------------------------------

@dataclass
class ThresholdedTrade(v2_mod.V2Trade):
    coverage: float = 1.0                          # coverage level this trade was included at
    argmax_pred: float = 0.0                        # predicted score at argmax
    threshold_used: float = -np.inf                 # train-derived threshold


THRESHOLDED_TRADE_FIELDS = list(ThresholdedTrade.__dataclass_fields__.keys())


# ---------------------------------------------------------------------------
# Training sample collection (extended with per-sample metadata)
# ---------------------------------------------------------------------------

def collect_training_samples_with_meta(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int, str]]]:
    """Like v2_mod.collect_training_samples but also records each sample's
    (day, local_i, side) AND the per-sample net_pnl_dollars for dollar-PF.
    """
    rows: list[np.ndarray] = []
    labels: list[float] = []
    dollars: list[float] = []
    meta: list[tuple[str, int, str]] = []
    for day in fold_spec.train_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        cands = v2_mod._enumerate_candidates_on_day(
            sc=sc, X_day=X[ds:de], X_sim_day=X_sim[ds:de],
            spot_day=spot_prices[ds:de], idx_map=idx_map,
            core_indices=core_indices, config=config,
        )
        for cand in cands:
            trade, _ = v2_mod._realize_trade(
                day=day, fold_idx=fold_spec.fold_idx,
                strategy_label="train_sample", paired_trade_id=-1,
                X_sim_day=X_sim[ds:de], spot_day=spot_prices[ds:de],
                idx_map=idx_map, sc=sc,
                entry_local=cand["local_i"], side=cand["side"],
                picked=cand["picked"], context_spread=cand["context_spread"],
            )
            if trade is None:
                continue
            rows.append(cand["features"])
            labels.append(trade.net_pct)
            dollars.append(trade.net_pnl_dollars)
            meta.append((day, int(cand["local_i"]), cand["side"]))
    if not rows:
        return (np.zeros((0, v2_mod.N_MODEL_FEATURES), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                [])
    return (np.vstack(rows).astype(np.float32),
            np.array(labels, dtype=np.float32),
            np.array(dollars, dtype=np.float32),
            meta)


# ---------------------------------------------------------------------------
# OOB-based train-side argmax per day
# ---------------------------------------------------------------------------

def train_scorer_with_oob(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> RandomForestRegressor | None:
    if features.shape[0] == 0 or labels.shape[0] == 0:
        return None
    model = RandomForestRegressor(
        n_estimators=v2_mod.RF_N_ESTIMATORS,
        max_depth=v2_mod.RF_MAX_DEPTH,
        min_samples_leaf=v2_mod.RF_MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    model.fit(features, labels)
    return model


def argmax_by_day_oob(
    metadata: list[tuple[str, int, str]],
    oob_preds: np.ndarray,
    labels: np.ndarray,
    dollars: np.ndarray,
) -> dict[str, dict]:
    """For each train day, find the OOB-argmax (bar, side) and its
    realized net_pct + net_pnl_dollars.
    """
    by_day: dict[str, dict] = {}
    for i, (day, local_i, side) in enumerate(metadata):
        p = float(oob_preds[i])
        if not np.isfinite(p):
            continue
        cur = by_day.get(day)
        if cur is None or p > cur["pred"]:
            by_day[day] = {
                "pred": p,
                "local_i": int(local_i),
                "side": side,
                "net_pct": float(labels[i]),
                "net_pnl_dollars": float(dollars[i]),
                "sample_idx": int(i),
            }
    return by_day


# ---------------------------------------------------------------------------
# Coverage evaluation (train side, using OOB argmax per day)
# ---------------------------------------------------------------------------

def _fake_trade_from_train_argmax(by_day_entry: dict, day: str) -> BaselineTrade:
    """Build a minimal BaselineTrade shell holding just net_pct and
    net_pnl_dollars for summarize_trades. Other fields are placeholder.
    Sufficient for summarize_trades aggregation.
    """
    return BaselineTrade(
        date=day,
        fold=-1,
        strategy_label="train_oob_argmax",
        paired_trade_id=-1,
        bar_entry=int(by_day_entry["local_i"]),
        side=by_day_entry["side"],
        strike=0.0, contract_idx=0, delta_at_entry=0.0,
        spread_at_entry=0.0, context_spread_at_entry=0.0,
        entry_mid=0.0, trigger_vwap_dist_min_prior=0.0,
        trigger_vwap_dist_now=0.0, first15_acceptance_at_entry=0.0,
        bar_delta_at_entry=0.0, bar_exit=0, exit_mid=0.0,
        bars_held=0, exit_reason="target_first15" if by_day_entry["net_pct"] > 0 else "stop_vwap",
        gross_pct=0.0, spread_cost_pct=0.0,
        net_pct=float(by_day_entry["net_pct"]),
        net_pnl_dollars=float(by_day_entry["net_pnl_dollars"]),
        vix_regime_at_entry=0.0,
    )


def evaluate_train_coverage(
    by_day: dict[str, dict],
    coverage: float,
) -> tuple[float, dict]:
    """Threshold train days by OOB argmax >= quantile(1 - coverage).
    Returns (threshold, train_summary_dict).
    """
    if not by_day:
        return float("-inf"), {"n": 0, "mean_net_pct": 0.0, "dollar_pf": 0.0,
                               "target_hit_frac": 0.0, "stop_hit_frac": 0.0}
    preds = np.array([v["pred"] for v in by_day.values()])
    if coverage >= 1.0:
        threshold = float("-inf")
    else:
        threshold = float(np.quantile(preds, 1.0 - coverage))
    included = [v for v in by_day.values() if v["pred"] >= threshold]
    # Guard: quantile may fall exactly on a value; the strict `>=` keeps ties.
    if not included:
        return threshold, {"n": 0, "mean_net_pct": 0.0, "dollar_pf": 0.0,
                           "target_hit_frac": 0.0, "stop_hit_frac": 0.0}
    fake = [_fake_trade_from_train_argmax(v, day) for day, v in by_day.items()
            if v["pred"] >= threshold]
    summary = summarize_trades(fake)
    return threshold, summary


def pick_best_coverage(
    by_day: dict[str, dict],
    coverages: tuple[float, ...] = COVERAGE_GRID,
    min_n: int = MIN_N_TRAIN_SELECT,
) -> tuple[float | None, list[dict]]:
    """Evaluate each coverage on train; pick the one with highest
    mean_net_pct subject to n >= min_n. Returns (chosen_coverage, per_cov_diag).
    """
    diag: list[dict] = []
    best_coverage: float | None = None
    best_mean = -np.inf
    for cov in coverages:
        threshold, s = evaluate_train_coverage(by_day, cov)
        row = {
            "coverage": float(cov),
            "threshold": float(threshold),
            "n": int(s["n"]),
            "mean_net_pct": float(s["mean_net_pct"]),
            "dollar_pf": float(s["dollar_pf"]),
            "target_hit_frac": float(s["target_hit_frac"]),
            "stop_hit_frac": float(s["stop_hit_frac"]),
            "viable": bool(s["n"] >= min_n),
            "selected": False,
        }
        diag.append(row)
        if row["viable"] and row["mean_net_pct"] > best_mean:
            best_mean = row["mean_net_pct"]
            best_coverage = float(cov)
    if best_coverage is not None:
        for row in diag:
            if row["coverage"] == best_coverage:
                row["selected"] = True
                break
    return best_coverage, diag


# ---------------------------------------------------------------------------
# Test-side application — run V2 pipeline but gate entries by score threshold
# ---------------------------------------------------------------------------

def run_test_at_threshold(
    *,
    fold_idx: int,
    fold_test_days: list[str],
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X: np.ndarray,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    core_indices: list[int],
    config: V1BGateConfig,
    model: RandomForestRegressor,
    threshold: float,
    coverage: float,
) -> tuple[list[ThresholdedTrade], list[SkipRecord]]:
    """For each test day: enumerate candidates, score, argmax. If argmax
    score >= threshold, the trade is taken; else abstain.
    """
    trades: list[ThresholdedTrade] = []
    skips: list[SkipRecord] = []
    for day in fold_test_days:
        if day not in day_to_range:
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        cands = v2_mod._enumerate_candidates_on_day(
            sc=sc, X_day=X[ds:de], X_sim_day=X_sim[ds:de],
            spot_day=spot_prices[ds:de], idx_map=idx_map,
            core_indices=core_indices, config=config,
        )
        if not cands:
            skips.append(SkipRecord(day, fold_idx, "strategy", -1, "X", "no_v1b_candidates"))
            continue
        feat_mat = np.vstack([c["features"] for c in cands]).astype(np.float32)
        preds = model.predict(feat_mat)
        argmax_pos = int(np.argmax(preds))
        argmax_pred = float(preds[argmax_pos])
        if argmax_pred < threshold:
            skips.append(SkipRecord(day, fold_idx, "strategy", -1, "X", "abstain_below_threshold"))
            continue
        chosen = cands[argmax_pos]
        v2trade, status = v2_mod._realize_trade(
            day=day, fold_idx=fold_idx,
            strategy_label="strategy", paired_trade_id=-1,
            X_sim_day=X_sim[ds:de], spot_day=spot_prices[ds:de],
            idx_map=idx_map, sc=sc,
            entry_local=chosen["local_i"], side=chosen["side"],
            picked=chosen["picked"], context_spread=chosen["context_spread"],
            predicted_net_pct=argmax_pred,
            score_rank=1,
            n_candidates=len(cands),
        )
        if v2trade is None:
            skips.append(SkipRecord(day, fold_idx, "strategy", chosen["local_i"], chosen["side"], status))
            continue
        trades.append(ThresholdedTrade(
            **{**asdict(v2trade), "coverage": float(coverage),
               "argmax_pred": float(argmax_pred),
               "threshold_used": float(threshold)},
        ))
    return trades, skips


# ---------------------------------------------------------------------------
# Controls (paired to strategy's included days — automatic under thresholding)
# ---------------------------------------------------------------------------

def run_controls_paired(
    strategy_trades: list[ThresholdedTrade],
    fold_idx: int,
    fold_test_days: list[str],
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    config: V1BGateConfig,
    seed: int,
) -> tuple[list[ThresholdedTrade], list[ThresholdedTrade], list[SkipRecord]]:
    """Run control A (random-bar same-day) and control B (random other
    day same-bar) paired to the thresholded strategy trades. Since the
    pairing is per strategy trade, controls automatically inherit the
    strategy's abstention.
    """
    rng_A = np.random.default_rng(seed)
    rng_B = np.random.default_rng(seed + 1)
    ctrl_A: list[ThresholdedTrade] = []
    ctrl_B: list[ThresholdedTrade] = []
    skips: list[SkipRecord] = []
    idx_iv = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]
    idx_spread = idx_map["option_spread_pct"]
    available_days = [d for d in fold_test_days if d in day_to_range]

    for pid, st in enumerate(strategy_trades):
        # Control A — same day, random bar, strategy side
        sc_a = load_sidecar_cached(sidecar_path(sidecar_dir, st.date))
        ds_a, de_a = day_to_range[st.date]
        X_sim_day_a = X_sim[ds_a:de_a]
        spot_day_a = spot_prices[ds_a:de_a]
        lo = BAR_LO
        hi = min(BAR_HI, X_sim_day_a.shape[0] - 1)
        if hi < lo:
            skips.append(SkipRecord(st.date, fold_idx, "control_A", -1, st.side, "day_too_short"))
        else:
            bar_a = int(rng_A.integers(lo, hi + 1))
            ok, reason = check_gates(X_sim_day_a, bar_a, idx_iv, idx_vrp, config)
            if not ok:
                skips.append(SkipRecord(st.date, fold_idx, "control_A", bar_a, st.side, reason))
            else:
                cs = context_spread_at_bar(X_sim_day_a, bar_a, idx_spread)
                picked_a, reason = v2_mod.select_contract(sc_a, bar_a, st.side, cs)
                if picked_a is None:
                    skips.append(SkipRecord(st.date, fold_idx, "control_A", bar_a, st.side, reason))
                else:
                    v2t, status = v2_mod._realize_trade(
                        day=st.date, fold_idx=fold_idx,
                        strategy_label="control_A", paired_trade_id=pid,
                        X_sim_day=X_sim_day_a, spot_day=spot_day_a,
                        idx_map=idx_map, sc=sc_a,
                        entry_local=bar_a, side=st.side,
                        picked=picked_a, context_spread=cs,
                    )
                    if v2t is not None:
                        ctrl_A.append(ThresholdedTrade(
                            **{**asdict(v2t), "coverage": st.coverage,
                               "argmax_pred": 0.0, "threshold_used": st.threshold_used},
                        ))
                    else:
                        skips.append(SkipRecord(st.date, fold_idx, "control_A", bar_a, st.side, status))

        # Control B — random other day, same bar as strategy, same side
        pool = [d for d in available_days if d != st.date]
        if not pool:
            skips.append(SkipRecord(st.date, fold_idx, "control_B", st.bar_entry, st.side, "no_other_day"))
            continue
        pick_day = pool[int(rng_B.integers(0, len(pool)))]
        sc_b = load_sidecar_cached(sidecar_path(sidecar_dir, pick_day))
        ds_b, de_b = day_to_range[pick_day]
        X_sim_day_b = X_sim[ds_b:de_b]
        spot_day_b = spot_prices[ds_b:de_b]
        bar_b = st.bar_entry
        if bar_b >= X_sim_day_b.shape[0]:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", bar_b, st.side, "out_of_window"))
            continue
        ok, reason = check_gates(X_sim_day_b, bar_b, idx_iv, idx_vrp, config)
        if not ok:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", bar_b, st.side, reason))
            continue
        cs = context_spread_at_bar(X_sim_day_b, bar_b, idx_spread)
        picked_b, reason = v2_mod.select_contract(sc_b, bar_b, st.side, cs)
        if picked_b is None:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", bar_b, st.side, reason))
            continue
        v2t, status = v2_mod._realize_trade(
            day=pick_day, fold_idx=fold_idx,
            strategy_label="control_B", paired_trade_id=pid,
            X_sim_day=X_sim_day_b, spot_day=spot_day_b,
            idx_map=idx_map, sc=sc_b,
            entry_local=bar_b, side=st.side,
            picked=picked_b, context_spread=cs,
        )
        if v2t is None:
            skips.append(SkipRecord(pick_day, fold_idx, "control_B", bar_b, st.side, status))
            continue
        ctrl_B.append(ThresholdedTrade(
            **{**asdict(v2t), "coverage": st.coverage,
               "argmax_pred": 0.0, "threshold_used": st.threshold_used},
        ))
    return ctrl_A, ctrl_B, skips


# ---------------------------------------------------------------------------
# Bootstrap (day-level, by fold, using test_days)
# ---------------------------------------------------------------------------

def bootstrap_day_level(
    strategy_trades: list[ThresholdedTrade],
    control_A_trades: list[ThresholdedTrade],
    control_B_trades: list[ThresholdedTrade],
    caches_test_days: list[tuple[int, list[str]]],
    n_boot: int,
    seed: int,
) -> dict:
    """Resample test_days per fold with replacement; recompute aggregates."""
    rng = np.random.default_rng(seed)
    # Group by (fold, date)
    def _group(trades):
        grp: dict[tuple[int, str], list] = {}
        for t in trades:
            grp.setdefault((t.fold, t.date), []).append(t)
        return grp
    strat_grp = _group(strategy_trades)
    # Control A's date = strategy date (per run_controls_paired). Fine.
    ctrl_A_grp = _group(control_A_trades)
    # Control B's date = the picked random day. We re-key it to the paired
    # strategy's date for apples-to-apples resampling.
    ctrl_B_by_pair: dict[tuple[int, str], list] = {}
    for t in control_B_trades:
        paired_date = strategy_trades[t.paired_trade_id].date if 0 <= t.paired_trade_id < len(strategy_trades) else t.date
        ctrl_B_by_pair.setdefault((t.fold, paired_date), []).append(t)

    metrics: dict[str, list[float]] = {
        "strategy_mean_net_pct": [], "strategy_dollar_pf": [],
        "control_A_mean_net_pct": [], "gap_vs_A": [],
        "control_B_mean_net_pct": [], "gap_vs_B": [],
    }
    for _rep in range(n_boot):
        all_s, all_a, all_b = [], [], []
        for fold_idx, test_days in caches_test_days:
            sampled = rng.choice(test_days, size=len(test_days), replace=True)
            for d in sampled:
                all_s.extend(strat_grp.get((fold_idx, d), []))
                all_a.extend(ctrl_A_grp.get((fold_idx, d), []))
                all_b.extend(ctrl_B_by_pair.get((fold_idx, d), []))
        s = summarize_trades(all_s)
        a = summarize_trades(all_a)
        b = summarize_trades(all_b)
        metrics["strategy_mean_net_pct"].append(s["mean_net_pct"])
        metrics["strategy_dollar_pf"].append(s["dollar_pf"])
        metrics["control_A_mean_net_pct"].append(a["mean_net_pct"])
        metrics["gap_vs_A"].append(s["mean_net_pct"] - a["mean_net_pct"])
        metrics["control_B_mean_net_pct"].append(b["mean_net_pct"])
        metrics["gap_vs_B"].append(s["mean_net_pct"] - b["mean_net_pct"])

    out: dict[str, dict] = {}
    for k, v in metrics.items():
        arr = np.array(v, dtype=np.float64)
        out[k] = {
            "p2_5": float(np.quantile(arr, 0.025)),
            "p50": float(np.quantile(arr, 0.50)),
            "p97_5": float(np.quantile(arr, 0.975)),
            "mean": float(arr.mean()),
            "frac_gt_0": float((arr > 0).mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Per-fold driver
# ---------------------------------------------------------------------------

@dataclass
class FoldRun:
    fold_idx: int
    window_id: str
    test_days: list[str]
    gate_config: dict
    selected_coverage: float | None
    selected_threshold: float
    train_coverage_diag: list[dict]
    per_coverage_test: dict[float, dict]            # cov -> summary dict
    per_coverage_strategy_trades: dict[float, list[ThresholdedTrade]]
    per_coverage_A: dict[float, list[ThresholdedTrade]]
    per_coverage_B: dict[float, list[ThresholdedTrade]]
    skips: list[SkipRecord] = field(default_factory=list)
    elapsed_sec: float = 0.0


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
) -> FoldRun:
    t0 = time.time()
    fold_idx = fold_spec.fold_idx
    print(f"\n=== Fold {fold_idx} | window={fold_spec.window_id} ===", flush=True)
    print(f"  train_days={len(fold_spec.train_days)} "
          f"({fold_spec.train_days[0]} → {fold_spec.train_days[-1]})")
    print(f"  test_days={len(fold_spec.test_days)} "
          f"({fold_spec.test_days[0]} → {fold_spec.test_days[-1]})")

    # Gate selection (reuse V1B pipeline via v2_mod.select_gates_for_fold)
    gate_config, _diag, _n_tr, _n_ent = v2_mod.select_gates_for_fold(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
    )
    if gate_config is None:
        print("  NO VIABLE GATE CONFIG — skipping fold", flush=True)
        return FoldRun(
            fold_idx=fold_idx, window_id=fold_spec.window_id,
            test_days=list(fold_spec.test_days), gate_config={},
            selected_coverage=None, selected_threshold=float("-inf"),
            train_coverage_diag=[], per_coverage_test={},
            per_coverage_strategy_trades={}, per_coverage_A={}, per_coverage_B={},
            elapsed_sec=time.time() - t0,
        )
    print(f"  gates: iv_max={gate_config.iv_max} vrp_max={gate_config.vrp_max:+.5f} "
          f"({gate_config.vrp_source})", flush=True)

    # Training samples + metadata
    feats, labels, dollars, meta = collect_training_samples_with_meta(
        fold_spec=fold_spec, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
        X=X, X_sim=X_sim, spot_prices=spot_prices,
        idx_map=idx_map, core_indices=core_indices, config=gate_config,
    )
    print(f"  train: n_samples={feats.shape[0]}, "
          f"n_days_represented={len(set(m[0] for m in meta))}", flush=True)

    # RF with OOB
    seed = fold_idx * 10007 + 1
    model = train_scorer_with_oob(feats, labels, seed=seed)
    if model is None:
        print("  NO TRAINING SAMPLES — skipping fold", flush=True)
        return FoldRun(
            fold_idx=fold_idx, window_id=fold_spec.window_id,
            test_days=list(fold_spec.test_days), gate_config=asdict(gate_config),
            selected_coverage=None, selected_threshold=float("-inf"),
            train_coverage_diag=[], per_coverage_test={},
            per_coverage_strategy_trades={}, per_coverage_A={}, per_coverage_B={},
            elapsed_sec=time.time() - t0,
        )
    oob_preds = model.oob_prediction_
    n_oob_valid = int(np.isfinite(oob_preds).sum())
    print(f"  OOB: {n_oob_valid}/{len(oob_preds)} valid", flush=True)

    # Per-day argmax on train
    train_day_argmax = argmax_by_day_oob(meta, oob_preds, labels, dollars)
    print(f"  train: {len(train_day_argmax)} days with OOB argmax", flush=True)

    # Coverage grid on train
    best_cov, cov_diag = pick_best_coverage(train_day_argmax)
    for row in cov_diag:
        marker = "  <<" if row["selected"] else ""
        print(f"    train coverage {row['coverage']:.1f}: "
              f"threshold={row['threshold']:+.5f}, n={row['n']:3d}, "
              f"mean_net_pct={row['mean_net_pct']:+.5f}, "
              f"dollar_pf={row['dollar_pf']:.3f}{marker}", flush=True)
    print(f"  selected coverage: {best_cov}", flush=True)

    # Evaluate EVERY coverage on test (for the full curve) — minimal overhead.
    per_cov_test: dict[float, dict] = {}
    per_cov_strat: dict[float, list[ThresholdedTrade]] = {}
    per_cov_A: dict[float, list[ThresholdedTrade]] = {}
    per_cov_B: dict[float, list[ThresholdedTrade]] = {}
    all_skips: list[SkipRecord] = []
    for cov in COVERAGE_GRID:
        preds = np.array([v["pred"] for v in train_day_argmax.values()])
        thr = float("-inf") if cov >= 1.0 else float(np.quantile(preds, 1.0 - cov))
        strat_trades, strat_skips = run_test_at_threshold(
            fold_idx=fold_idx, fold_test_days=fold_spec.test_days,
            day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X=X, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, core_indices=core_indices, config=gate_config,
            model=model, threshold=thr, coverage=cov,
        )
        ctrl_A_trades, ctrl_B_trades, ctrl_skips = run_controls_paired(
            strategy_trades=strat_trades, fold_idx=fold_idx,
            fold_test_days=fold_spec.test_days, day_to_range=day_to_range,
            sidecar_dir=sidecar_dir, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, config=gate_config, seed=seed + int(cov * 1000),
        )
        per_cov_strat[cov] = strat_trades
        per_cov_A[cov] = ctrl_A_trades
        per_cov_B[cov] = ctrl_B_trades
        s = summarize_trades(strat_trades)
        a = summarize_trades(ctrl_A_trades)
        b = summarize_trades(ctrl_B_trades)
        per_cov_test[cov] = {
            "threshold": float(thr),
            "strategy": s, "control_A": a, "control_B": b,
            "gap_vs_A": s["mean_net_pct"] - a["mean_net_pct"],
            "gap_vs_B": s["mean_net_pct"] - b["mean_net_pct"],
        }
        all_skips.extend(strat_skips)
        all_skips.extend(ctrl_skips)
        marker = "  <<" if (best_cov is not None and abs(cov - best_cov) < 1e-9) else ""
        print(f"    test  coverage {cov:.1f}: n_strat={s['n']:3d} "
              f"mean_net_pct={s['mean_net_pct']:+.5f} pf={s['dollar_pf']:.3f} "
              f"gap_vs_A={s['mean_net_pct'] - a['mean_net_pct']:+.5f}{marker}", flush=True)

    selected_threshold = (
        per_cov_test[best_cov]["threshold"] if best_cov is not None else float("-inf")
    )

    return FoldRun(
        fold_idx=fold_idx, window_id=fold_spec.window_id,
        test_days=list(fold_spec.test_days),
        gate_config=asdict(gate_config),
        selected_coverage=best_cov, selected_threshold=selected_threshold,
        train_coverage_diag=cov_diag, per_coverage_test=per_cov_test,
        per_coverage_strategy_trades=per_cov_strat,
        per_coverage_A=per_cov_A, per_coverage_B=per_cov_B,
        skips=all_skips, elapsed_sec=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------

def aggregate_selected(fold_runs: list[FoldRun]) -> dict:
    """Per-fold: use each fold's selected coverage. Aggregate across folds."""
    all_s: list[ThresholdedTrade] = []
    all_a: list[ThresholdedTrade] = []
    all_b: list[ThresholdedTrade] = []
    per_fold: list[dict] = []
    for fr in fold_runs:
        if fr.selected_coverage is None:
            per_fold.append({"fold_idx": fr.fold_idx, "selected_coverage": None})
            continue
        cov = fr.selected_coverage
        s = fr.per_coverage_strategy_trades[cov]
        a = fr.per_coverage_A[cov]
        b = fr.per_coverage_B[cov]
        all_s.extend(s)
        all_a.extend(a)
        all_b.extend(b)
        ss = summarize_trades(s)
        per_fold.append({
            "fold_idx": fr.fold_idx,
            "selected_coverage": cov,
            "selected_threshold": fr.selected_threshold,
            "strategy": ss,
            "control_A": summarize_trades(a),
            "control_B": summarize_trades(b),
            "gap_vs_A": ss["mean_net_pct"] - summarize_trades(a)["mean_net_pct"],
        })
    s_agg = summarize_trades(all_s)
    a_agg = summarize_trades(all_a)
    b_agg = summarize_trades(all_b)
    return {
        "strategy": s_agg,
        "control_A": a_agg,
        "control_B": b_agg,
        "gap_vs_A": s_agg["mean_net_pct"] - a_agg["mean_net_pct"],
        "gap_vs_B": s_agg["mean_net_pct"] - b_agg["mean_net_pct"],
        "per_fold": per_fold,
        "trades": {"strategy": all_s, "control_A": all_a, "control_B": all_b},
    }


def aggregate_fixed_coverage(fold_runs: list[FoldRun], coverage: float) -> dict:
    """Apply the SAME coverage across folds. For the coverage curve."""
    all_s: list[ThresholdedTrade] = []
    all_a: list[ThresholdedTrade] = []
    all_b: list[ThresholdedTrade] = []
    for fr in fold_runs:
        if coverage in fr.per_coverage_strategy_trades:
            all_s.extend(fr.per_coverage_strategy_trades[coverage])
            all_a.extend(fr.per_coverage_A[coverage])
            all_b.extend(fr.per_coverage_B[coverage])
    s = summarize_trades(all_s)
    a = summarize_trades(all_a)
    b = summarize_trades(all_b)
    return {
        "coverage": coverage,
        "strategy": s, "control_A": a, "control_B": b,
        "gap_vs_A": s["mean_net_pct"] - a["mean_net_pct"],
        "gap_vs_B": s["mean_net_pct"] - b["mean_net_pct"],
        "trades": {"strategy": all_s, "control_A": all_a, "control_B": all_b},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v2/data.pt")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()

    _activate_pruned_feature_set()
    print(f"[thresholding] Active CORE_FEATURE_NAMES: {list(v2_mod.CORE_FEATURE_NAMES)}")
    print(f"[thresholding] Coverage grid: {list(COVERAGE_GRID)}")

    t0 = time.time()
    data = load_data(args.data)
    dates = list(data["dates"])
    feature_names = list(data["feature_names"])
    idx_map = feature_index_map(feature_names)
    required = list(v2_mod.CORE_FEATURE_NAMES) + ["vix_regime", "iv_percentile", "vrp"]
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

    fold_runs: list[FoldRun] = []
    for fs in folds:
        fold_runs.append(run_fold(
            fold_spec=fs, day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X=X, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, core_indices=core_idx,
        ))

    # Selected-per-fold aggregate (primary result)
    print("\n=== Selected-per-fold aggregate (PRIMARY) ===")
    selected = aggregate_selected(fold_runs)
    s = selected["strategy"]
    a = selected["control_A"]
    b = selected["control_B"]
    print(f"  strategy:   n={s['n']}, target={s['target_hit_frac']:.3f}, "
          f"stop={s['stop_hit_frac']:.3f}, mean_net_pct={s['mean_net_pct']:+.5f} "
          f"(±{s['mean_net_pct_stderr']:.5f}), dollar_pf={s['dollar_pf']:.3f}")
    print(f"  control_A:  n={a['n']}, mean_net_pct={a['mean_net_pct']:+.5f}, "
          f"dollar_pf={a['dollar_pf']:.3f}")
    print(f"  control_B:  n={b['n']}, mean_net_pct={b['mean_net_pct']:+.5f}, "
          f"dollar_pf={b['dollar_pf']:.3f}")
    print(f"  gap_vs_A={selected['gap_vs_A']:+.5f}, gap_vs_B={selected['gap_vs_B']:+.5f}")

    # Per-fixed-coverage aggregate (coverage curve)
    print("\n=== Coverage curve (same coverage applied across all folds) ===")
    curve: list[dict] = []
    for cov in COVERAGE_GRID:
        r = aggregate_fixed_coverage(fold_runs, cov)
        curve.append(r)
        print(f"  coverage {cov:.1f}: n={r['strategy']['n']:3d}, "
              f"mean_net_pct={r['strategy']['mean_net_pct']:+.5f}, "
              f"dollar_pf={r['strategy']['dollar_pf']:.3f}, "
              f"gap_vs_A={r['gap_vs_A']:+.5f}, "
              f"control_A mean={r['control_A']['mean_net_pct']:+.5f}")

    # Bootstrap at each coverage
    print("\n=== Bootstrap CIs ===")
    caches_test_days = [(fr.fold_idx, fr.test_days) for fr in fold_runs]
    boot_by_cov: dict[float, dict] = {}
    for cov in COVERAGE_GRID:
        r = curve[COVERAGE_GRID.index(cov)]
        boot = bootstrap_day_level(
            strategy_trades=r["trades"]["strategy"],
            control_A_trades=r["trades"]["control_A"],
            control_B_trades=r["trades"]["control_B"],
            caches_test_days=caches_test_days,
            n_boot=args.n_boot, seed=BASE_SEED + int(cov * 1000),
        )
        boot_by_cov[cov] = boot
        print(f"  coverage {cov:.1f}: mean_net CI=[{boot['strategy_mean_net_pct']['p2_5']:+.5f},"
              f"{boot['strategy_mean_net_pct']['p97_5']:+.5f}] "
              f"gap_vs_A CI=[{boot['gap_vs_A']['p2_5']:+.5f},"
              f"{boot['gap_vs_A']['p97_5']:+.5f}] frac>0={boot['gap_vs_A']['frac_gt_0']:.3f}")

    # Bootstrap on selected-per-fold aggregate
    print("\n=== Bootstrap on selected-per-fold aggregate ===")
    boot_selected = bootstrap_day_level(
        strategy_trades=selected["trades"]["strategy"],
        control_A_trades=selected["trades"]["control_A"],
        control_B_trades=selected["trades"]["control_B"],
        caches_test_days=caches_test_days,
        n_boot=args.n_boot, seed=BASE_SEED + 777,
    )
    print(f"  strategy_mean_net_pct CI=[{boot_selected['strategy_mean_net_pct']['p2_5']:+.5f},"
          f"{boot_selected['strategy_mean_net_pct']['p97_5']:+.5f}] "
          f"frac>0={boot_selected['strategy_mean_net_pct']['frac_gt_0']:.3f}")
    print(f"  gap_vs_A CI=[{boot_selected['gap_vs_A']['p2_5']:+.5f},"
          f"{boot_selected['gap_vs_A']['p97_5']:+.5f}] "
          f"frac>0={boot_selected['gap_vs_A']['frac_gt_0']:.3f}")

    # Dump trades + summary
    all_strat_trades = selected["trades"]["strategy"]
    all_A_trades = selected["trades"]["control_A"]
    all_B_trades = selected["trades"]["control_B"]
    write_csv(os.path.join(args.out_dir, "trades_selected.csv"),
              all_strat_trades + all_A_trades + all_B_trades,
              THRESHOLDED_TRADE_FIELDS)
    for cov in COVERAGE_GRID:
        r = curve[COVERAGE_GRID.index(cov)]
        all_trades_cov = r["trades"]["strategy"] + r["trades"]["control_A"] + r["trades"]["control_B"]
        write_csv(os.path.join(args.out_dir, f"trades_coverage_{int(cov * 100)}.csv"),
                  all_trades_cov, THRESHOLDED_TRADE_FIELDS)

    # Skip summary across all coverages
    all_skips = [s for fr in fold_runs for s in fr.skips]
    write_csv(os.path.join(args.out_dir, "skips.csv"), all_skips, SKIP_FIELDS)

    def _strip_trades(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "trades"}

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint": dataset_fp,
        "coverage_grid": list(COVERAGE_GRID),
        "min_n_train_select": MIN_N_TRAIN_SELECT,
        "selected_per_fold": _strip_trades(selected),
        "coverage_curve": [_strip_trades(r) for r in curve],
        "bootstrap_by_coverage": boot_by_cov,
        "bootstrap_selected": boot_selected,
        "per_fold": [
            {
                "fold_idx": fr.fold_idx,
                "window_id": fr.window_id,
                "selected_coverage": fr.selected_coverage,
                "selected_threshold": fr.selected_threshold,
                "gate_config": fr.gate_config,
                "train_coverage_diag": fr.train_coverage_diag,
                "per_coverage_test_summary": {
                    str(cov): {
                        "threshold": v["threshold"],
                        "strategy": v["strategy"],
                        "control_A": v["control_A"],
                        "control_B": v["control_B"],
                        "gap_vs_A": v["gap_vs_A"],
                        "gap_vs_B": v["gap_vs_B"],
                    }
                    for cov, v in fr.per_coverage_test.items()
                },
                "elapsed_sec": fr.elapsed_sec,
            }
            for fr in fold_runs
        ],
        "note": "Fixed-coverage thresholding over RF OOB-ranked train argmax. "
                "Primary comparator: 1.0-coverage baseline (unthresholded V2-pruned).",
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print(f"\nWrote summary + trades to {args.out_dir}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
