"""Mechanical baseline V1B — V1A + premium-sanity gates (iv_percentile, vrp).

Hypothesis under test: opening-structure reversion is a valid day/side thesis
(V1A beat Control B by 2.15pp on aggregate), but the reclaim/reject trigger
does not meaningfully pick better bars within those days (V1A lost narrowly
to Control A). V1B adds two no-trade premium-sanity gates to the same V1A
trigger, selected per fold from train-only data, then evaluated on held-out
test folds. Primary comparator: Control A.

Discipline:
- Trigger and contract selection are unchanged from V1A.
- Two gates: `iv_percentile <= iv_max` and `vrp <= vrp_max`. Both must pass.
- Gate thresholds chosen per fold from fold.train_days ONLY, from a
  predeclared 3x3 grid:
    iv_max in {0.6, 0.7, 0.8}
    vrp_max in {train_median, train_p75, 0 if feasible}
- Selection criterion: highest train mean_net_pct with n >= MIN_N_TRAIN.
- Test evaluated on fold.test_days with the selected config.
- Controls apply the same gates (apples-to-apples isolation of trigger
  marginal value over random-bar-on-same-day with gates held fixed).

This module imports V1A helpers rather than duplicating them; V1B only adds
gate selection, gate application, and a V1B-specific driver + outputs.
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
    detect_trigger,
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


# ---------------------------------------------------------------------------
# V1B constants
# ---------------------------------------------------------------------------

GATE_IV_CANDIDATES = (0.6, 0.7, 0.8)
VRP_ZERO_MIN_FRACTION = 0.10          # vrp<=0 candidate included only if this
                                      # fraction of entered train trades pass
MIN_N_TRAIN = 30                      # minimum trades a config must produce
                                      # on train before it can be selected
EXPERIMENT_ID = "mechbase_opening_reversion_v1b"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v1b"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class V1BGateConfig:
    iv_max: float
    vrp_max: float
    vrp_source: str                   # "median" | "p75" | "zero"
    train_n: int
    train_mean_net_pct: float
    train_dollar_pf: float


@dataclass
class TriggerEval:
    """All the state we need to know about one triggered bar on train.

    'entered' means the V1A structure+contract pipeline would have opened a
    trade at this bar (with no gates applied). If entered, `trade` holds the
    simulated trade outcome.
    """
    day: str
    local_i: int
    side: str
    iv_percentile: float
    vrp: float
    entered: bool
    skip_reason: str
    trade: BaselineTrade | None = None


# ---------------------------------------------------------------------------
# Gate check (applied at every entry attempt in V1B)
# ---------------------------------------------------------------------------

def check_gates(
    X_sim_day: np.ndarray,
    local_i: int,
    idx_iv_pct: int,
    idx_vrp: int,
    config: V1BGateConfig,
) -> tuple[bool, str]:
    iv = float(X_sim_day[local_i, idx_iv_pct])
    if not np.isfinite(iv) or iv > config.iv_max:
        return False, "gated_iv_percentile"
    vrp = float(X_sim_day[local_i, idx_vrp])
    if not np.isfinite(vrp) or vrp > config.vrp_max:
        return False, "gated_vrp"
    return True, "ok"


# ---------------------------------------------------------------------------
# Per-bar trigger evaluation (train collection only — no traded_today)
# ---------------------------------------------------------------------------

def evaluate_triggers_on_day(
    *,
    day: str,
    fold_idx: int,
    sc: dict[str, Any],
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
) -> list[TriggerEval]:
    """Walk bars [BAR_LO, BAR_HI], at each triggering bar run V1A's contract
    selection + exit + PnL. No 'one trade per day' constraint — we need every
    triggered bar so the V1B gate simulation can pick the first one that
    passes gates.
    """
    out: list[TriggerEval] = []
    day_n_bars = X_sim_day.shape[0]
    first15_hi, first15_lo = first15_levels(spot_day)

    idx_vwap_dist = idx_map["vwap_dist"]
    idx_bar_delta = idx_map["bar_delta"]
    idx_first15_accept = idx_map["first15_acceptance"]
    idx_option_spread = idx_map["option_spread_pct"]
    idx_vix_regime = idx_map["vix_regime"]
    idx_iv_pct = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]

    for local_i in range(BAR_LO, min(BAR_HI + 1, day_n_bars)):
        side, info = detect_trigger(
            X_sim_day, local_i, idx_vwap_dist, idx_bar_delta, idx_first15_accept,
        )
        if side is None or side == "ambiguous":
            continue

        iv_pct_val = float(X_sim_day[local_i, idx_iv_pct])
        vrp_val = float(X_sim_day[local_i, idx_vrp])

        context_spread = context_spread_at_bar(X_sim_day, local_i, idx_option_spread)
        picked, reason = select_contract(sc, local_i, side, context_spread)
        if picked is None:
            out.append(TriggerEval(
                day=day, local_i=int(local_i), side=side,
                iv_percentile=iv_pct_val, vrp=vrp_val,
                entered=False, skip_reason=reason, trade=None,
            ))
            continue

        vix_regime = float(X_sim_day[local_i, idx_vix_regime])
        mtc_bars = max(1.0, 390.0 - float(local_i))
        exit_local, exit_reason, exit_mid = run_exit(
            sc=sc,
            contract_idx=picked["contract_idx"],
            entry_local=local_i,
            side=side,
            X_sim_day=X_sim_day,
            spot_day=spot_day,
            idx_vwap_dist=idx_vwap_dist,
            first15_hi=first15_hi,
            first15_lo=first15_lo,
            day_n_bars=day_n_bars,
        )
        if exit_local is None:
            out.append(TriggerEval(
                day=day, local_i=int(local_i), side=side,
                iv_percentile=iv_pct_val, vrp=vrp_val,
                entered=False, skip_reason=exit_reason, trade=None,
            ))
            continue

        pnl = compute_pnl(picked["entry_mid"], exit_mid, mtc_bars, vix_regime)
        trade = BaselineTrade(
            date=day,
            fold=fold_idx,
            strategy_label="train_eval",          # overwritten downstream
            paired_trade_id=-1,
            bar_entry=int(local_i),
            side=side,
            strike=picked["strike"],
            contract_idx=picked["contract_idx"],
            delta_at_entry=picked["delta"],
            spread_at_entry=picked["spread_fraction"],
            context_spread_at_entry=float(context_spread),
            entry_mid=picked["entry_mid"],
            trigger_vwap_dist_min_prior=(
                info["vwap_dist_min_prior"] if side == "C" else info["vwap_dist_max_prior"]
            ),
            trigger_vwap_dist_now=info["vwap_dist_now"],
            first15_acceptance_at_entry=info["first15_acceptance_now"],
            bar_delta_at_entry=info["bar_delta_now"],
            bar_exit=int(exit_local),
            exit_mid=float(exit_mid),
            bars_held=int(exit_local - local_i),
            exit_reason=exit_reason,
            gross_pct=pnl["gross_pct"],
            spread_cost_pct=pnl["spread_cost_pct"],
            net_pct=pnl["net_pct"],
            net_pnl_dollars=pnl["net_pnl_dollars"],
            vix_regime_at_entry=float(vix_regime),
        )
        out.append(TriggerEval(
            day=day, local_i=int(local_i), side=side,
            iv_percentile=iv_pct_val, vrp=vrp_val,
            entered=True, skip_reason="", trade=trade,
        ))
    return out


# ---------------------------------------------------------------------------
# Train-based gate config selection
# ---------------------------------------------------------------------------

def _simulate_gate_config_on_trainevals(
    per_day_evals: dict[str, list[TriggerEval]],
    iv_max: float,
    vrp_max: float,
) -> list[BaselineTrade]:
    """For each day, pick the first entered TriggerEval (in bar order) whose
    gate values pass; that trade is the V1B trade for that day under this
    config. Skip the day if no eval passes.
    """
    selected: list[BaselineTrade] = []
    for _day, evals in per_day_evals.items():
        for ev in evals:
            if not ev.entered:
                continue
            if not np.isfinite(ev.iv_percentile) or ev.iv_percentile > iv_max:
                continue
            if not np.isfinite(ev.vrp) or ev.vrp > vrp_max:
                continue
            assert ev.trade is not None
            selected.append(ev.trade)
            break
    return selected


def pick_best_config(
    per_day_evals: dict[str, list[TriggerEval]],
) -> tuple[V1BGateConfig | None, list[dict]]:
    """Evaluate the 3x3 grid on train and pick the highest mean_net_pct
    config. Returns (config or None, grid_diagnostic).
    """
    all_entered = [ev for evs in per_day_evals.values() for ev in evs if ev.entered]
    if not all_entered:
        return None, []

    vrps = np.array([ev.vrp for ev in all_entered if np.isfinite(ev.vrp)])
    if vrps.size == 0:
        return None, []

    vrp_median = float(np.median(vrps))
    vrp_p75 = float(np.quantile(vrps, 0.75))
    vrp_zero_feasible = float((vrps <= 0.0).mean()) >= VRP_ZERO_MIN_FRACTION

    vrp_candidates = [("median", vrp_median), ("p75", vrp_p75)]
    if vrp_zero_feasible:
        vrp_candidates.append(("zero", 0.0))

    diagnostics: list[dict] = []
    best: V1BGateConfig | None = None

    for iv_max in GATE_IV_CANDIDATES:
        for vrp_src, vrp_max in vrp_candidates:
            trades = _simulate_gate_config_on_trainevals(per_day_evals, iv_max, vrp_max)
            n = len(trades)
            if n == 0:
                diagnostics.append({
                    "iv_max": iv_max, "vrp_max": vrp_max, "vrp_source": vrp_src,
                    "n": 0, "mean_net_pct": 0.0, "dollar_pf": 0.0, "selected": False,
                    "viable": False, "reason": "no_trades",
                })
                continue
            summary = summarize_trades(trades)
            viable = n >= MIN_N_TRAIN
            diag = {
                "iv_max": iv_max, "vrp_max": vrp_max, "vrp_source": vrp_src,
                "n": n,
                "mean_net_pct": summary["mean_net_pct"],
                "dollar_pf": summary["dollar_pf"],
                "target_hit_frac": summary["target_hit_frac"],
                "stop_hit_frac": summary["stop_hit_frac"],
                "viable": viable,
                "selected": False,
            }
            diagnostics.append(diag)
            if not viable:
                continue
            if best is None or summary["mean_net_pct"] > best.train_mean_net_pct:
                best = V1BGateConfig(
                    iv_max=float(iv_max),
                    vrp_max=float(vrp_max),
                    vrp_source=vrp_src,
                    train_n=n,
                    train_mean_net_pct=float(summary["mean_net_pct"]),
                    train_dollar_pf=float(summary["dollar_pf"]),
                )

    if best is not None:
        for d in diagnostics:
            if (d["iv_max"] == best.iv_max and d["vrp_max"] == best.vrp_max
                    and d["vrp_source"] == best.vrp_source):
                d["selected"] = True
    return best, diagnostics


# ---------------------------------------------------------------------------
# V1B entry helpers (strategy + controls)
# ---------------------------------------------------------------------------

def _build_trade(
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
    info: dict[str, float],
) -> BaselineTrade | tuple[None, str]:
    day_n_bars = X_sim_day.shape[0]
    first15_hi, first15_lo = first15_levels(spot_day)
    idx_vwap_dist = idx_map["vwap_dist"]
    idx_vix_regime = idx_map["vix_regime"]

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
    return BaselineTrade(
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
        trigger_vwap_dist_min_prior=info.get("min_prior", 0.0),
        trigger_vwap_dist_now=info.get("vwap_dist_now", 0.0),
        first15_acceptance_at_entry=info.get("first15_acceptance_now", 0.0),
        bar_delta_at_entry=info.get("bar_delta_now", 0.0),
        bar_exit=int(exit_local),
        exit_mid=float(exit_mid),
        bars_held=int(exit_local - entry_local),
        exit_reason=exit_reason,
        gross_pct=pnl["gross_pct"],
        spread_cost_pct=pnl["spread_cost_pct"],
        net_pct=pnl["net_pct"],
        net_pnl_dollars=pnl["net_pnl_dollars"],
        vix_regime_at_entry=float(vix_regime),
    )


def run_v1b_strategy_day(
    *,
    day: str,
    fold_idx: int,
    sc: dict[str, Any],
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
    config: V1BGateConfig,
) -> tuple[list[BaselineTrade], list[SkipRecord]]:
    trades: list[BaselineTrade] = []
    skips: list[SkipRecord] = []
    day_n_bars = X_sim_day.shape[0]

    idx_vwap_dist = idx_map["vwap_dist"]
    idx_bar_delta = idx_map["bar_delta"]
    idx_first15_accept = idx_map["first15_acceptance"]
    idx_option_spread = idx_map["option_spread_pct"]
    idx_iv_pct = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]

    for local_i in range(BAR_LO, min(BAR_HI + 1, day_n_bars)):
        side, info = detect_trigger(
            X_sim_day, local_i, idx_vwap_dist, idx_bar_delta, idx_first15_accept,
        )
        if side is None:
            continue
        if side == "ambiguous":
            skips.append(SkipRecord(day, fold_idx, "strategy", local_i, "X", "ambiguous_trigger"))
            continue

        # V1B gate check — before any contract work
        ok, gate_reason = check_gates(X_sim_day, local_i, idx_iv_pct, idx_vrp, config)
        if not ok:
            skips.append(SkipRecord(day, fold_idx, "strategy", local_i, side, gate_reason))
            continue

        context_spread = context_spread_at_bar(X_sim_day, local_i, idx_option_spread)
        picked, reason = select_contract(sc, local_i, side, context_spread)
        if picked is None:
            skips.append(SkipRecord(day, fold_idx, "strategy", local_i, side, reason))
            continue

        info_full = {
            "min_prior": (info["vwap_dist_min_prior"] if side == "C" else info["vwap_dist_max_prior"]),
            "vwap_dist_now": info["vwap_dist_now"],
            "first15_acceptance_now": info["first15_acceptance_now"],
            "bar_delta_now": info["bar_delta_now"],
        }
        built = _build_trade(
            day=day, fold_idx=fold_idx, strategy_label="strategy",
            paired_trade_id=-1, X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, sc=sc, entry_local=local_i, side=side,
            picked=picked, context_spread=context_spread, info=info_full,
        )
        if isinstance(built, tuple):
            skips.append(SkipRecord(day, fold_idx, "strategy", local_i, side, built[1]))
            continue
        trades.append(built)
        break                                     # one trade per day

    return trades, skips


def _enter_control_at_bar(
    *,
    day: str,
    fold_idx: int,
    label: str,
    paired_trade_id: int,
    sc: dict[str, Any],
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
    entry_local: int,
    side: str,
    config: V1BGateConfig,
) -> tuple[BaselineTrade | None, SkipRecord | None]:
    idx_iv_pct = idx_map["iv_percentile"]
    idx_vrp = idx_map["vrp"]
    idx_option_spread = idx_map["option_spread_pct"]
    idx_vwap_dist = idx_map["vwap_dist"]
    idx_bar_delta = idx_map["bar_delta"]
    idx_first15_accept = idx_map["first15_acceptance"]
    day_n_bars = X_sim_day.shape[0]

    if entry_local < BAR_LO or entry_local > BAR_HI or entry_local >= day_n_bars:
        return None, SkipRecord(day, fold_idx, label, entry_local, side, "out_of_window")

    ok, gate_reason = check_gates(X_sim_day, entry_local, idx_iv_pct, idx_vrp, config)
    if not ok:
        return None, SkipRecord(day, fold_idx, label, entry_local, side, gate_reason)

    context_spread = context_spread_at_bar(X_sim_day, entry_local, idx_option_spread)
    picked, reason = select_contract(sc, entry_local, side, context_spread)
    if picked is None:
        return None, SkipRecord(day, fold_idx, label, entry_local, side, reason)

    info = {
        "min_prior": 0.0,
        "vwap_dist_now": float(X_sim_day[entry_local, idx_vwap_dist]),
        "first15_acceptance_now": float(X_sim_day[entry_local, idx_first15_accept]),
        "bar_delta_now": float(X_sim_day[entry_local, idx_bar_delta]),
    }
    built = _build_trade(
        day=day, fold_idx=fold_idx, strategy_label=label,
        paired_trade_id=paired_trade_id, X_sim_day=X_sim_day, spot_day=spot_day,
        idx_map=idx_map, sc=sc, entry_local=entry_local, side=side,
        picked=picked, context_spread=context_spread, info=info,
    )
    if isinstance(built, tuple):
        return None, SkipRecord(day, fold_idx, label, entry_local, side, built[1])
    return built, None


def run_v1b_control_A(
    *,
    strategy_trades: list[BaselineTrade],
    fold_idx: int,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    config: V1BGateConfig,
    seed: int,
) -> tuple[list[BaselineTrade], list[SkipRecord]]:
    rng = np.random.default_rng(seed)
    trades: list[BaselineTrade] = []
    skips: list[SkipRecord] = []
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
        trade, skip = _enter_control_at_bar(
            day=st.date, fold_idx=fold_idx, label="control_A", paired_trade_id=pid,
            sc=sc, X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
            entry_local=entry_local, side=st.side, config=config,
        )
        if trade is not None:
            trades.append(trade)
        if skip is not None:
            skips.append(skip)
    return trades, skips


def run_v1b_control_B(
    *,
    strategy_trades: list[BaselineTrade],
    fold_idx: int,
    fold_test_days: list[str],
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    config: V1BGateConfig,
    seed: int,
) -> tuple[list[BaselineTrade], list[SkipRecord]]:
    rng = np.random.default_rng(seed)
    trades: list[BaselineTrade] = []
    skips: list[SkipRecord] = []
    available = [d for d in fold_test_days if d in day_to_range]
    if len(available) < 2:
        return trades, skips
    for pid, st in enumerate(strategy_trades):
        pool = [d for d in available if d != st.date]
        if not pool:
            skips.append(SkipRecord(st.date, fold_idx, "control_B", st.bar_entry, st.side, "no_other_day"))
            continue
        pick = pool[int(rng.integers(0, len(pool)))]
        sc = load_sidecar_cached(sidecar_path(sidecar_dir, pick))
        ds, de = day_to_range[pick]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        trade, skip = _enter_control_at_bar(
            day=pick, fold_idx=fold_idx, label="control_B", paired_trade_id=pid,
            sc=sc, X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
            entry_local=st.bar_entry, side=st.side, config=config,
        )
        if trade is not None:
            trades.append(trade)
        if skip is not None:
            skips.append(skip)
    return trades, skips


# ---------------------------------------------------------------------------
# Fold driver
# ---------------------------------------------------------------------------

def run_v1b_fold(
    *,
    idx_map: dict[str, int],
    day_to_range: dict[str, tuple[int, int]],
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
    print(f"  train_days={len(fold_spec.train_days)} ({fold_spec.train_days[0]} → {fold_spec.train_days[-1]})")
    print(f"  test_days={len(fold_spec.test_days)} ({fold_spec.test_days[0]} → {fold_spec.test_days[-1]})")

    # --- Train collection ---
    train_missing = 0
    train_evals: dict[str, list[TriggerEval]] = {}
    for day in fold_spec.train_days:
        if day not in day_to_range:
            train_missing += 1
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            train_missing += 1
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        train_evals[day] = evaluate_triggers_on_day(
            day=day, fold_idx=fold_idx, sc=sc,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
        )
    n_train_trigger_bars = sum(len(v) for v in train_evals.values())
    n_train_entered = sum(1 for v in train_evals.values() for e in v if e.entered)
    print(f"  train: triggered_bars={n_train_trigger_bars}, entered={n_train_entered}, "
          f"missing_sidecars={train_missing}", flush=True)

    # --- Config selection ---
    config, grid_diag = pick_best_config(train_evals)
    if config is None:
        print("  NO VIABLE CONFIG on train — fold skipped", flush=True)
        return {
            "fold_idx": fold_idx,
            "window_id": fold_spec.window_id,
            "n_train_triggered_bars": n_train_trigger_bars,
            "n_train_entered": n_train_entered,
            "selected_config": None,
            "grid_diagnostics": grid_diag,
            "strategy_trades": [],
            "control_A_trades": [],
            "control_B_trades": [],
            "skips": [],
            "strategy_summary": summarize_trades([]),
            "control_A_summary": summarize_trades([]),
            "control_B_summary": summarize_trades([]),
            "elapsed_sec": time.time() - t0,
        }

    print(f"  selected: iv_max={config.iv_max} vrp_max={config.vrp_max:+.5f} "
          f"({config.vrp_source}), train_n={config.train_n}, "
          f"train_mean_net_pct={config.train_mean_net_pct:+.5f}, "
          f"train_dollar_pf={config.train_dollar_pf:.3f}", flush=True)

    # Free the train-eval memory before test runs (sidecars remain cached)
    train_evals.clear()

    # --- Test run ---
    strategy_trades: list[BaselineTrade] = []
    all_skips: list[SkipRecord] = []
    test_missing = 0
    for day in fold_spec.test_days:
        if day not in day_to_range:
            test_missing += 1
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            test_missing += 1
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        day_trades, day_skips = run_v1b_strategy_day(
            day=day, fold_idx=fold_idx, sc=sc,
            X_sim_day=X_sim_day, spot_day=spot_day,
            idx_map=idx_map, config=config,
        )
        strategy_trades.extend(day_trades)
        all_skips.extend(day_skips)

    strat_summary = summarize_trades(strategy_trades)
    print(f"  strategy: n={strat_summary['n']}, "
          f"target={strat_summary['target_hit_frac']:.3f}, "
          f"stop={strat_summary['stop_hit_frac']:.3f}, "
          f"mean_net_pct={strat_summary['mean_net_pct']:+.5f}, "
          f"dollar_pf={strat_summary['dollar_pf']:.3f}", flush=True)

    ctrl_A_trades: list[BaselineTrade] = []
    ctrl_B_trades: list[BaselineTrade] = []
    if run_controls and strategy_trades:
        seed = fold_idx * 10007 + 1
        ctrl_A_trades, ctrl_A_skips = run_v1b_control_A(
            strategy_trades=strategy_trades, fold_idx=fold_idx,
            day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map,
            config=config, seed=seed,
        )
        ctrl_B_trades, ctrl_B_skips = run_v1b_control_B(
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
    write_csv(os.path.join(out_dir, f"trades_fold{fold_idx}.csv"), all_trades, TRADE_FIELDS)

    write_json(os.path.join(out_dir, f"report_fold{fold_idx}.json"), {
        "experiment_id": EXPERIMENT_ID,
        "fold_idx": fold_idx,
        "window_id": fold_spec.window_id,
        "selected_config": asdict(config),
        "grid_diagnostics": grid_diag,
        "strategy_summary": strat_summary,
        "control_A_summary": ctrl_A_summary,
        "control_B_summary": ctrl_B_summary,
        "strategy_trades": [asdict(t) for t in strategy_trades],
        "n_train_triggered_bars": n_train_trigger_bars,
        "n_train_entered": n_train_entered,
        "test_missing_sidecars": test_missing,
    })

    return {
        "fold_idx": fold_idx,
        "window_id": fold_spec.window_id,
        "n_train_triggered_bars": n_train_trigger_bars,
        "n_train_entered": n_train_entered,
        "selected_config": asdict(config),
        "grid_diagnostics": grid_diag,
        "strategy_summary": strat_summary,
        "control_A_summary": ctrl_A_summary,
        "control_B_summary": ctrl_B_summary,
        "strategy_trades": strategy_trades,
        "control_A_trades": ctrl_A_trades,
        "control_B_trades": ctrl_B_trades,
        "skips": all_skips,
        "test_missing_sidecars": test_missing,
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
    required = ["vwap_dist", "bar_delta", "first15_acceptance", "option_spread_pct",
                "vix_regime", "iv_percentile", "vrp"]
    missing = [n for n in required if n not in idx_map]
    if missing:
        print(f"FATAL: feature_names missing: {missing}", file=sys.stderr)
        return 2

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
        fold_results.append(run_v1b_fold(
            idx_map=idx_map, day_to_range=day_to_range,
            X_sim=X_sim, spot_prices=spot_prices, fold_spec=fs,
            sidecar_dir=sidecar_dir, out_dir=args.out_dir,
            run_controls=args.controls,
        ))

    all_strat: list[BaselineTrade] = []
    all_A: list[BaselineTrade] = []
    all_B: list[BaselineTrade] = []
    all_skips: list[SkipRecord] = []
    for r in fold_results:
        all_strat.extend(r["strategy_trades"])
        all_A.extend(r["control_A_trades"])
        all_B.extend(r["control_B_trades"])
        all_skips.extend(r["skips"])

    write_csv(os.path.join(args.out_dir, "trades.csv"),
              all_strat + all_A + all_B, TRADE_FIELDS)
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
                "n_train_triggered_bars": r["n_train_triggered_bars"],
                "n_train_entered": r["n_train_entered"],
                "test_missing_sidecars": r["test_missing_sidecars"],
                "strategy_summary": r["strategy_summary"],
                "control_A_summary": r["control_A_summary"],
                "control_B_summary": r["control_B_summary"],
                "grid_diagnostics": r["grid_diagnostics"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in fold_results
        ],
        "note": "Train-only threshold selection; test_days held out. Primary "
                "comparator: Control A. Secondary: Control B.",
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    write_json(os.path.join(args.out_dir, "controls.json"),
               {"control_A": A_agg, "control_B": B_agg,
                "per_fold": [{"fold_idx": r["fold_idx"],
                              "control_A": r["control_A_summary"],
                              "control_B": r["control_B_summary"]}
                             for r in fold_results]})

    print(f"\n=== V1B SUMMARY ({EXPERIMENT_ID}) ===")
    print(f"  strategy:  n={strat_agg['n']}, "
          f"target={strat_agg['target_hit_frac']:.3f}, "
          f"stop={strat_agg['stop_hit_frac']:.3f}, "
          f"mean_net_pct={strat_agg['mean_net_pct']:+.5f} (±{strat_agg['mean_net_pct_stderr']:.5f}), "
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
