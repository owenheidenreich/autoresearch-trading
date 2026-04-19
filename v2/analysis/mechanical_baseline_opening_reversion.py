"""Mechanical baseline — opening-structure reversion (V1A).

Implements the plan locked in
[v2/docs/mechanical_baseline_plan_opening_reversion.md] against the live
repo/data contract. No ML, no simulator integration — spot-driven entries
and exits with direct contract-mid PnL accounting. See also:

- [v2/docs/strategy_card_opening_reversion.md] (thesis, exact triggers)
- [v2/docs/feature_shortlist_opening_reversion.md] (feature policy)
- [v2/docs/feature_schema.md] (79-feature audit; feature indices)

Reads:
- v2/data.pt (unnormalized X_sim, spot_prices, dates, bar_of_day, metadata)
- v2/data_sidecars/{day}.pt (per-day contract matrices + executable rows)

Writes (under --out-dir, default v2/artifacts/mechanical_baseline_opening_reversion/):
- trades.csv, trades_fold{N}.csv
- skips.csv
- controls.json
- report_fold{N}.json (EvalReport)
- summary.json (falsification verdict)

Run:
    python3 -m v2.analysis.mechanical_baseline_opening_reversion --fold 0
    python3 -m v2.analysis.mechanical_baseline_opening_reversion --fold all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from v2.core.chain_data import load_sidecar_cached, sidecar_path
from v2.core.features import compute_adaptive_spread_bps
from v2.core.walkforward import CANONICAL_N_FOLDS, generate_folds


# ---------------------------------------------------------------------------
# Locked parameters (plan: mechanical_baseline_plan_opening_reversion.md)
# ---------------------------------------------------------------------------

VWAP_BPS = 0.0010        # 10 bps
LOOKBACK = 10            # bars of prior VWAP distance history
BAR_LO, BAR_HI = 15, 120 # session entry window
TIME_STOP_MIN = 30       # bars held
DELTA_LO, DELTA_HI = 0.45, 0.55
SPREAD_CAP_CONTEXT = 0.20
SPREAD_CAP_CONTRACT = 0.20
QUALITY_VALID = 2

OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion"
EXPERIMENT_ID = "mechbase_opening_reversion_v1a"


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class BaselineTrade:
    date: str
    fold: int
    strategy_label: str          # "strategy", "control_A", "control_B"
    paired_trade_id: int = -1    # for controls, index into strategy trades
    # entry
    bar_entry: int = 0           # bar_of_day at entry
    side: str = ""               # "C" or "P"
    strike: float = 0.0
    contract_idx: int = 0
    delta_at_entry: float = 0.0
    spread_at_entry: float = 0.0
    context_spread_at_entry: float = 0.0
    entry_mid: float = 0.0
    # hypothesis conditions (for attribution on failure)
    trigger_vwap_dist_min_prior: float = 0.0   # bull: min; bear: max (prior 10 bars)
    trigger_vwap_dist_now: float = 0.0
    first15_acceptance_at_entry: float = 0.0
    bar_delta_at_entry: float = 0.0
    # exit
    bar_exit: int = 0
    exit_mid: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    # pnl
    gross_pct: float = 0.0
    spread_cost_pct: float = 0.0
    net_pct: float = 0.0
    net_pnl_dollars: float = 0.0
    vix_regime_at_entry: float = 0.0


@dataclass
class SkipRecord:
    date: str
    fold: int
    strategy_label: str
    bar: int
    side_intended: str
    reason: str


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> dict:
    print(f"Loading {path}...", flush=True)
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def day_ranges(dates: list[str]) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Map day string → (global_start, global_end). Also return ordered days."""
    day_to_range: dict[str, tuple[int, int]] = {}
    ordered: list[str] = []
    if not dates:
        return day_to_range, ordered
    start = 0
    cur = dates[0]
    for i in range(1, len(dates)):
        if dates[i] != cur:
            day_to_range[cur] = (start, i)
            ordered.append(cur)
            start = i
            cur = dates[i]
    day_to_range[cur] = (start, len(dates))
    ordered.append(cur)
    return day_to_range, ordered


def feature_index_map(feature_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def build_folds(dates: list[str]) -> list:
    unique_dates = sorted(set(dates))
    return generate_folds(unique_dates, n_folds=CANONICAL_N_FOLDS)


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------

def detect_trigger(
    X_sim_day: np.ndarray,
    local_i: int,
    idx_vwap_dist: int,
    idx_bar_delta: int,
    idx_first15_accept: int,
) -> tuple[str | None, dict[str, float]]:
    """Return ('C', info) for bull, ('P', info) for bear, or (None, info).

    `info` holds the four hypothesis-attribution values regardless of
    whether a trigger fired, for diagnostic logging.
    """
    if local_i < BAR_LO or local_i > BAR_HI:
        return None, {}
    prior = X_sim_day[max(0, local_i - LOOKBACK):local_i, idx_vwap_dist]
    if prior.size == 0:
        return None, {}

    vd_now = float(X_sim_day[local_i, idx_vwap_dist])
    bd_now = float(X_sim_day[local_i, idx_bar_delta])
    acc_now = float(X_sim_day[local_i, idx_first15_accept])
    vd_min_prior = float(np.nanmin(prior))
    vd_max_prior = float(np.nanmax(prior))

    bull_ok = (
        vd_min_prior <= -VWAP_BPS
        and vd_now > 0.0
        and acc_now >= 0.0
        and bd_now > 0.0
    )
    bear_ok = (
        vd_max_prior >= VWAP_BPS
        and vd_now < 0.0
        and acc_now <= 0.0
        and bd_now < 0.0
    )

    info = {
        "vwap_dist_now": vd_now,
        "bar_delta_now": bd_now,
        "first15_acceptance_now": acc_now,
        "vwap_dist_min_prior": vd_min_prior,
        "vwap_dist_max_prior": vd_max_prior,
    }

    if bull_ok and bear_ok:
        return "ambiguous", info
    if bull_ok:
        return "C", info
    if bear_ok:
        return "P", info
    return None, info


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------

def context_spread_at_bar(X_sim_day: np.ndarray, local_i: int, idx_spread: int) -> float:
    val = float(X_sim_day[local_i, idx_spread])
    if not np.isfinite(val):
        return float("inf")
    return val


def select_contract(
    sc: dict[str, Any],
    local_i: int,
    side: str,
    context_spread: float,
) -> tuple[dict | None, str]:
    """Return (selected_row_dict, reason).

    Gate order: context spread → delta-band pool → per-contract spread →
    tiebreak. First failing gate determines the skip reason.
    """
    if context_spread > SPREAD_CAP_CONTEXT:
        return None, "skipped_context_spread"

    ptrs = sc["bar_ptrs"]
    start = int(ptrs[local_i])
    end = int(ptrs[local_i + 1])
    if end <= start:
        return None, "skipped_no_contracts_at_bar"
    feats = np.asarray(sc["row_features"][start:end])
    contract_ids = np.asarray(sc["row_contract_idx"][start:end])
    if feats.shape[0] == 0:
        return None, "skipped_no_contracts_at_bar"

    valid = feats[:, 0] > 0.5
    quality = feats[:, 14] >= QUALITY_VALID
    mid_ok = feats[:, 3] > 0.0

    is_put = feats[:, 2] > 0.5
    if side == "C":
        side_mask = ~is_put
        delta_band = (feats[:, 8] >= DELTA_LO) & (feats[:, 8] <= DELTA_HI)
    else:
        side_mask = is_put
        delta_band = (feats[:, 8] >= -DELTA_HI) & (feats[:, 8] <= -DELTA_LO)

    pool = valid & quality & mid_ok & side_mask & delta_band
    if not pool.any():
        return None, "skipped_no_delta_contract"

    spread_ok = feats[:, 4] <= SPREAD_CAP_CONTRACT
    pool2 = pool & spread_ok
    if not pool2.any():
        return None, "skipped_contract_spread"

    abs_delta = np.abs(feats[:, 8])
    rank_d = np.abs(abs_delta - 0.50)
    idxs = np.where(pool2)[0]
    sort_key = list(zip(
        rank_d[idxs],
        feats[idxs, 4],
        np.abs(feats[idxs, 11]),
        idxs.tolist(),
    ))
    sort_key.sort()
    chosen_local = sort_key[0][3]

    return (
        {
            "local_row": int(chosen_local),
            "contract_idx": int(contract_ids[chosen_local]),
            "strike": float(feats[chosen_local, 1]),
            "delta": float(feats[chosen_local, 8]),
            "spread_fraction": float(feats[chosen_local, 4]),
            "entry_mid": float(feats[chosen_local, 3]),
            "moneyness_pct": float(feats[chosen_local, 11]),
        },
        "ok",
    )


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------

def first15_levels(spot_day: np.ndarray) -> tuple[float, float]:
    n = min(15, len(spot_day))
    if n == 0:
        return float("nan"), float("nan")
    window = spot_day[:n]
    finite = np.isfinite(window) & (window > 0)
    if not finite.any():
        return float("nan"), float("nan")
    return float(np.nanmax(window[finite])), float(np.nanmin(window[finite]))


def run_exit(
    sc: dict[str, Any],
    contract_idx: int,
    entry_local: int,
    side: str,
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_vwap_dist: int,
    first15_hi: float,
    first15_lo: float,
    day_n_bars: int,
) -> tuple[int | None, str, float]:
    """Run the spot-driven exit loop.

    Returns (exit_local, reason, exit_mid). On failure, exit_local is None
    and reason encodes the failure.
    """
    latest_exit = min(entry_local + TIME_STOP_MIN, BAR_HI, day_n_bars - 1)
    if latest_exit <= entry_local:
        return None, "zero_hold", 0.0

    mid_series = np.asarray(sc["contract_mid"][contract_idx])
    quality_series = np.asarray(sc["contract_quality"][contract_idx])

    def priced_mid(tau: int) -> float:
        m = float(mid_series[tau])
        q = int(quality_series[tau])
        if np.isfinite(m) and m > 0 and q >= QUALITY_VALID:
            return m
        return float("nan")

    for tau in range(entry_local + 1, latest_exit + 1):
        vd = float(X_sim_day[tau, idx_vwap_dist])
        px = float(spot_day[tau])

        # Priority 1: hard stop (VWAP re-cross) — checked first so same-bar
        # stop+target resolves to stop.
        if side == "C" and np.isfinite(vd) and vd <= 0.0:
            m = priced_mid(tau)
            return tau, "stop_vwap", (m if np.isfinite(m) else 0.0)
        if side == "P" and np.isfinite(vd) and vd >= 0.0:
            m = priced_mid(tau)
            return tau, "stop_vwap", (m if np.isfinite(m) else 0.0)

        # Priority 2: profit target (first-15 boundary touch, close-based)
        if side == "C" and np.isfinite(px) and np.isfinite(first15_hi) and px >= first15_hi:
            m = priced_mid(tau)
            return tau, "target_first15", (m if np.isfinite(m) else 0.0)
        if side == "P" and np.isfinite(px) and np.isfinite(first15_lo) and px <= first15_lo:
            m = priced_mid(tau)
            return tau, "target_first15", (m if np.isfinite(m) else 0.0)

    m = priced_mid(latest_exit)
    if not np.isfinite(m):
        # Walk back for the last priced mid in the window
        for tau in range(latest_exit - 1, entry_local, -1):
            m = priced_mid(tau)
            if np.isfinite(m):
                return tau, "time_stop", m
        return None, "exit_unpriced", 0.0
    return latest_exit, "time_stop", m


# ---------------------------------------------------------------------------
# PnL
# ---------------------------------------------------------------------------

def compute_pnl(entry_mid: float, exit_mid: float, mtc_bars: float, vix_regime: float) -> dict[str, float]:
    if entry_mid <= 0 or not np.isfinite(entry_mid):
        return {"gross_pct": 0.0, "spread_cost_pct": 0.0, "net_pct": 0.0, "net_pnl_dollars": 0.0}
    gross = (exit_mid - entry_mid) / entry_mid
    spread_bps = compute_adaptive_spread_bps(float(mtc_bars), float(vix_regime), is_otm=False)
    spread_cost = 2.0 * spread_bps / 10000.0
    net = gross - spread_cost
    net_dollars = net * entry_mid * 100.0  # SPX multiplier; 1-contract notional
    return {
        "gross_pct": float(gross),
        "spread_cost_pct": float(spread_cost),
        "net_pct": float(net),
        "net_pnl_dollars": float(net_dollars),
    }


# ---------------------------------------------------------------------------
# Per-day strategy run
# ---------------------------------------------------------------------------

def run_strategy_day(
    *,
    day: str,
    fold_idx: int,
    sc: dict[str, Any],
    X_sim_day: np.ndarray,
    spot_day: np.ndarray,
    idx_map: dict[str, int],
) -> tuple[list[BaselineTrade], list[SkipRecord]]:
    trades: list[BaselineTrade] = []
    skips: list[SkipRecord] = []

    first15_hi, first15_lo = first15_levels(spot_day)
    day_n_bars = X_sim_day.shape[0]

    idx_vwap_dist = idx_map["vwap_dist"]
    idx_bar_delta = idx_map["bar_delta"]
    idx_first15_accept = idx_map["first15_acceptance"]
    idx_option_spread = idx_map["option_spread_pct"]
    idx_vix_regime = idx_map["vix_regime"]

    traded_today = False
    for local_i in range(BAR_LO, min(BAR_HI + 1, day_n_bars)):
        if traded_today:
            break
        side, info = detect_trigger(
            X_sim_day, local_i, idx_vwap_dist, idx_bar_delta, idx_first15_accept,
        )
        if side is None:
            continue
        if side == "ambiguous":
            skips.append(SkipRecord(day, fold_idx, "strategy", local_i, "X", "ambiguous_trigger"))
            continue

        context_spread = context_spread_at_bar(X_sim_day, local_i, idx_option_spread)
        picked, reason = select_contract(sc, local_i, side, context_spread)
        if picked is None:
            skips.append(SkipRecord(day, fold_idx, "strategy", local_i, side, reason))
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
            skips.append(SkipRecord(day, fold_idx, "strategy", local_i, side, exit_reason))
            continue

        pnl = compute_pnl(picked["entry_mid"], exit_mid, mtc_bars, vix_regime)

        trades.append(BaselineTrade(
            date=day,
            fold=fold_idx,
            strategy_label="strategy",
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
        ))
        traded_today = True

    return trades, skips


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def _enter_at_bar(
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
) -> tuple[BaselineTrade | None, SkipRecord | None]:
    """Shared entry/exit for controls. No trigger check."""
    idx_option_spread = idx_map["option_spread_pct"]
    idx_vwap_dist = idx_map["vwap_dist"]
    idx_vix_regime = idx_map["vix_regime"]
    idx_bar_delta = idx_map["bar_delta"]
    idx_first15_accept = idx_map["first15_acceptance"]
    day_n_bars = X_sim_day.shape[0]
    if entry_local < BAR_LO or entry_local > BAR_HI or entry_local >= day_n_bars:
        return None, SkipRecord(day, fold_idx, label, entry_local, side, "out_of_window")

    context_spread = context_spread_at_bar(X_sim_day, entry_local, idx_option_spread)
    picked, reason = select_contract(sc, entry_local, side, context_spread)
    if picked is None:
        return None, SkipRecord(day, fold_idx, label, entry_local, side, reason)

    first15_hi, first15_lo = first15_levels(spot_day)
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
        return None, SkipRecord(day, fold_idx, label, entry_local, side, exit_reason)

    pnl = compute_pnl(picked["entry_mid"], exit_mid, mtc_bars, vix_regime)
    return BaselineTrade(
        date=day,
        fold=fold_idx,
        strategy_label=label,
        paired_trade_id=paired_trade_id,
        bar_entry=int(entry_local),
        side=side,
        strike=picked["strike"],
        contract_idx=picked["contract_idx"],
        delta_at_entry=picked["delta"],
        spread_at_entry=picked["spread_fraction"],
        context_spread_at_entry=float(context_spread),
        entry_mid=picked["entry_mid"],
        trigger_vwap_dist_min_prior=0.0,
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
    ), None


def run_control_A(
    *,
    strategy_trades: list[BaselineTrade],
    fold_idx: int,
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    seed: int,
) -> tuple[list[BaselineTrade], list[SkipRecord]]:
    """Same day, random bar, strategy's side."""
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
        trade, skip = _enter_at_bar(
            day=st.date, fold_idx=fold_idx, label="control_A", paired_trade_id=pid,
            sc=sc, X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
            entry_local=entry_local, side=st.side,
        )
        if trade is not None:
            trades.append(trade)
        if skip is not None:
            skips.append(skip)
    return trades, skips


def run_control_B(
    *,
    strategy_trades: list[BaselineTrade],
    fold_idx: int,
    fold_test_days: list[str],
    day_to_range: dict[str, tuple[int, int]],
    sidecar_dir: str,
    X_sim: np.ndarray,
    spot_prices: np.ndarray,
    idx_map: dict[str, int],
    seed: int,
) -> tuple[list[BaselineTrade], list[SkipRecord]]:
    """Same bar-of-day, random other day in the fold, strategy's side."""
    rng = np.random.default_rng(seed)
    trades: list[BaselineTrade] = []
    skips: list[SkipRecord] = []
    available_days = [d for d in fold_test_days if d in day_to_range]
    if len(available_days) < 2:
        return trades, skips
    for pid, st in enumerate(strategy_trades):
        candidates = [d for d in available_days if d != st.date]
        if not candidates:
            skips.append(SkipRecord(st.date, fold_idx, "control_B", st.bar_entry, st.side, "no_other_day"))
            continue
        pick = candidates[int(rng.integers(0, len(candidates)))]
        sc = load_sidecar_cached(sidecar_path(sidecar_dir, pick))
        ds, de = day_to_range[pick]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        trade, skip = _enter_at_bar(
            day=pick, fold_idx=fold_idx, label="control_B", paired_trade_id=pid,
            sc=sc, X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
            entry_local=st.bar_entry, side=st.side,
        )
        if trade is not None:
            trades.append(trade)
        if skip is not None:
            skips.append(skip)
    return trades, skips


# ---------------------------------------------------------------------------
# Aggregation + verdict
# ---------------------------------------------------------------------------

def summarize_trades(trades: list[BaselineTrade]) -> dict[str, float]:
    if not trades:
        return {
            "n": 0,
            "target_hit_frac": 0.0,
            "stop_hit_frac": 0.0,
            "time_stop_frac": 0.0,
            "mean_net_pct": 0.0,
            "std_net_pct": 0.0,
            "mean_net_pct_stderr": 0.0,
            "dollar_gross_profit": 0.0,
            "dollar_gross_loss": 0.0,
            "dollar_net_pnl": 0.0,
            "dollar_pf": 0.0,
        }
    net = np.array([t.net_pct for t in trades], dtype=np.float64)
    dollars = np.array([t.net_pnl_dollars for t in trades], dtype=np.float64)
    exits = [t.exit_reason for t in trades]
    n = len(trades)
    n_target = sum(1 for r in exits if r == "target_first15")
    n_stop = sum(1 for r in exits if r == "stop_vwap")
    n_time = sum(1 for r in exits if r == "time_stop")
    gp = float(dollars[dollars > 0].sum())
    gl = float(-dollars[dollars < 0].sum())
    pf = (gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else 0.0)
    std = float(net.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    return {
        "n": n,
        "target_hit_frac": n_target / n,
        "stop_hit_frac": n_stop / n,
        "time_stop_frac": n_time / n,
        "mean_net_pct": float(net.mean()),
        "std_net_pct": std,
        "mean_net_pct_stderr": float(se),
        "dollar_gross_profit": gp,
        "dollar_gross_loss": gl,
        "dollar_net_pnl": float(dollars.sum()),
        "dollar_pf": pf,
    }


def falsification_verdict(
    strat: dict[str, float],
    ctrl_A: dict[str, float],
    ctrl_B: dict[str, float],
) -> dict[str, Any]:
    reasons: list[str] = []
    n = int(strat["n"])
    mean = strat["mean_net_pct"]
    se = strat["mean_net_pct_stderr"]
    margin = 0.5 * se

    if n < 20:
        return {
            "verdict": "inconclusive",
            "reasons": [f"N={n} < 20"],
            "near_miss": False,
            "mean_net_pct_minus_margin": mean - margin,
            "margin": margin,
        }

    if strat["target_hit_frac"] <= strat["stop_hit_frac"]:
        reasons.append(f"target_hit_frac={strat['target_hit_frac']:.3f} ≤ stop_hit_frac={strat['stop_hit_frac']:.3f}")
    if mean <= 0:
        reasons.append(f"mean_net_pct={mean:.5f} ≤ 0")
    if ctrl_A["n"] > 0 and ctrl_A["mean_net_pct"] >= mean - margin:
        reasons.append(f"control_A mean_net_pct={ctrl_A['mean_net_pct']:.5f} ≥ strategy−margin={mean - margin:.5f}")
    if ctrl_B["n"] > 0 and ctrl_B["mean_net_pct"] >= mean - margin:
        reasons.append(f"control_B mean_net_pct={ctrl_B['mean_net_pct']:.5f} ≥ strategy−margin={mean - margin:.5f}")

    if not reasons:
        return {"verdict": "passed", "reasons": [], "near_miss": False}

    target_gap = strat["target_hit_frac"] - strat["stop_hit_frac"]
    near_miss = (mean > 0) and (
        any("control_A" in r for r in reasons) or any("control_B" in r for r in reasons)
        or (-0.03 <= target_gap <= 0.0)
    )
    clear_fail = mean <= -0.01
    verdict = "failed_clear" if clear_fail else ("failed_near_miss" if near_miss else "failed")
    return {"verdict": verdict, "reasons": reasons, "near_miss": near_miss}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

TRADE_FIELDS = list(BaselineTrade.__dataclass_fields__.keys())
SKIP_FIELDS = list(SkipRecord.__dataclass_fields__.keys())


def write_csv(path: str, rows: list, fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def write_fold_report(out_dir: str, fold_idx: int, strat_summary: dict, trades: list[BaselineTrade]) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "fold_idx": fold_idx,
        "summary": strat_summary,
        "trades": [asdict(t) for t in trades],
    }
    write_json(os.path.join(out_dir, f"report_fold{fold_idx}.json"), payload)


# ---------------------------------------------------------------------------
# Fold driver
# ---------------------------------------------------------------------------

def run_fold(
    *,
    data: dict,
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
    print(f"\n=== Fold {fold_idx} | window={fold_spec.window_id} | "
          f"test_days={len(fold_spec.test_days)} "
          f"({fold_spec.test_days[0]} → {fold_spec.test_days[-1]}) ===", flush=True)

    strat_trades: list[BaselineTrade] = []
    strat_skips: list[SkipRecord] = []
    missing = 0
    for day in fold_spec.test_days:
        if day not in day_to_range:
            missing += 1
            continue
        path = sidecar_path(sidecar_dir, day)
        if not os.path.exists(path):
            missing += 1
            continue
        sc = load_sidecar_cached(path)
        ds, de = day_to_range[day]
        X_sim_day = X_sim[ds:de]
        spot_day = spot_prices[ds:de]
        day_trades, day_skips = run_strategy_day(
            day=day, fold_idx=fold_idx, sc=sc,
            X_sim_day=X_sim_day, spot_day=spot_day, idx_map=idx_map,
        )
        strat_trades.extend(day_trades)
        strat_skips.extend(day_skips)

    strat_summary = summarize_trades(strat_trades)
    print(f"  strategy: n={strat_summary['n']}, "
          f"target={strat_summary['target_hit_frac']:.3f}, "
          f"stop={strat_summary['stop_hit_frac']:.3f}, "
          f"mean_net_pct={strat_summary['mean_net_pct']:.5f}, "
          f"dollar_pf={strat_summary['dollar_pf']:.3f}", flush=True)

    ctrl_A_trades: list[BaselineTrade] = []
    ctrl_B_trades: list[BaselineTrade] = []
    ctrl_A_skips: list[SkipRecord] = []
    ctrl_B_skips: list[SkipRecord] = []
    if run_controls and strat_trades:
        seed = fold_idx * 10007 + 1
        ctrl_A_trades, ctrl_A_skips = run_control_A(
            strategy_trades=strat_trades, fold_idx=fold_idx,
            day_to_range=day_to_range, sidecar_dir=sidecar_dir,
            X_sim=X_sim, spot_prices=spot_prices, idx_map=idx_map, seed=seed,
        )
        ctrl_B_trades, ctrl_B_skips = run_control_B(
            strategy_trades=strat_trades, fold_idx=fold_idx,
            fold_test_days=fold_spec.test_days, day_to_range=day_to_range,
            sidecar_dir=sidecar_dir, X_sim=X_sim, spot_prices=spot_prices,
            idx_map=idx_map, seed=seed + 1,
        )

    ctrl_A_summary = summarize_trades(ctrl_A_trades)
    ctrl_B_summary = summarize_trades(ctrl_B_trades)
    if run_controls:
        print(f"  control_A: n={ctrl_A_summary['n']}, "
              f"mean_net_pct={ctrl_A_summary['mean_net_pct']:.5f}, "
              f"dollar_pf={ctrl_A_summary['dollar_pf']:.3f}", flush=True)
        print(f"  control_B: n={ctrl_B_summary['n']}, "
              f"mean_net_pct={ctrl_B_summary['mean_net_pct']:.5f}, "
              f"dollar_pf={ctrl_B_summary['dollar_pf']:.3f}", flush=True)

    all_trades = strat_trades + ctrl_A_trades + ctrl_B_trades
    all_skips = strat_skips + ctrl_A_skips + ctrl_B_skips

    write_csv(os.path.join(out_dir, f"trades_fold{fold_idx}.csv"), all_trades, TRADE_FIELDS)
    write_fold_report(out_dir, fold_idx, strat_summary, strat_trades)

    elapsed = time.time() - t0
    print(f"  missing_sidecars={missing}, elapsed={elapsed:.1f}s", flush=True)

    return {
        "fold_idx": fold_idx,
        "window_id": fold_spec.window_id,
        "n_test_days": len(fold_spec.test_days),
        "missing_sidecars": missing,
        "strategy_summary": strat_summary,
        "control_A_summary": ctrl_A_summary,
        "control_B_summary": ctrl_B_summary,
        "strategy_trades": strat_trades,
        "control_A_trades": ctrl_A_trades,
        "control_B_trades": ctrl_B_trades,
        "skips": all_skips,
        "elapsed_sec": elapsed,
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
                "vix_regime"]
    missing = [n for n in required if n not in idx_map]
    if missing:
        print(f"FATAL: feature_names missing required entries: {missing}", file=sys.stderr)
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
            print(f"FATAL: fold {idx} not found; valid={[f.fold_idx for f in folds]}", file=sys.stderr)
            return 2

    os.makedirs(args.out_dir, exist_ok=True)

    fold_results: list[dict] = []
    for fs in selected:
        fold_results.append(run_fold(
            data=data, idx_map=idx_map, day_to_range=day_to_range,
            X_sim=X_sim, spot_prices=spot_prices, fold_spec=fs,
            sidecar_dir=sidecar_dir, out_dir=args.out_dir,
            run_controls=args.controls,
        ))

    all_strat_trades: list[BaselineTrade] = []
    all_ctrl_A: list[BaselineTrade] = []
    all_ctrl_B: list[BaselineTrade] = []
    all_skips: list[SkipRecord] = []
    for r in fold_results:
        all_strat_trades.extend(r["strategy_trades"])
        all_ctrl_A.extend(r["control_A_trades"])
        all_ctrl_B.extend(r["control_B_trades"])
        all_skips.extend(r["skips"])

    write_csv(os.path.join(args.out_dir, "trades.csv"),
              all_strat_trades + all_ctrl_A + all_ctrl_B, TRADE_FIELDS)
    write_csv(os.path.join(args.out_dir, "skips.csv"), all_skips, SKIP_FIELDS)

    strat_agg = summarize_trades(all_strat_trades)
    ctrl_A_agg = summarize_trades(all_ctrl_A)
    ctrl_B_agg = summarize_trades(all_ctrl_B)
    verdict = falsification_verdict(strat_agg, ctrl_A_agg, ctrl_B_agg)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_fingerprint": dataset_fp,
        "folds_run": [r["fold_idx"] for r in fold_results],
        "strategy_aggregate": strat_agg,
        "control_A_aggregate": ctrl_A_agg,
        "control_B_aggregate": ctrl_B_agg,
        "verdict": verdict,
        "per_fold": [
            {
                "fold_idx": r["fold_idx"],
                "window_id": r["window_id"],
                "n_test_days": r["n_test_days"],
                "missing_sidecars": r["missing_sidecars"],
                "strategy_summary": r["strategy_summary"],
                "control_A_summary": r["control_A_summary"],
                "control_B_summary": r["control_B_summary"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in fold_results
        ],
        "note": "A weak result only falsifies this opening-reversion expression, "
                "not the broader 0DTE long-premium hypothesis (see strategy card).",
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    write_json(os.path.join(args.out_dir, "controls.json"),
               {"control_A": ctrl_A_agg, "control_B": ctrl_B_agg,
                "per_fold": [{"fold_idx": r["fold_idx"],
                              "control_A": r["control_A_summary"],
                              "control_B": r["control_B_summary"]}
                             for r in fold_results]})

    print(f"\n=== SUMMARY ({EXPERIMENT_ID}) ===")
    print(f"  strategy: n={strat_agg['n']}, "
          f"target={strat_agg['target_hit_frac']:.3f}, "
          f"stop={strat_agg['stop_hit_frac']:.3f}, "
          f"mean_net_pct={strat_agg['mean_net_pct']:.5f} (±{strat_agg['mean_net_pct_stderr']:.5f}), "
          f"dollar_pf={strat_agg['dollar_pf']:.3f}")
    if all_ctrl_A:
        print(f"  control_A: n={ctrl_A_agg['n']}, "
              f"mean_net_pct={ctrl_A_agg['mean_net_pct']:.5f}, "
              f"dollar_pf={ctrl_A_agg['dollar_pf']:.3f}")
    if all_ctrl_B:
        print(f"  control_B: n={ctrl_B_agg['n']}, "
              f"mean_net_pct={ctrl_B_agg['mean_net_pct']:.5f}, "
              f"dollar_pf={ctrl_B_agg['dollar_pf']:.3f}")
    print(f"  verdict: {verdict['verdict']}")
    for r in verdict.get("reasons", []):
        print(f"    - {r}")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    print(f"  outputs under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
