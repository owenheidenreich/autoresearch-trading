"""Transferable entry-timing pattern audit for SPXW 0DTE.

This is the follow-up to the broad edge-existence audit. It does not ask
"which regime was profitable?" It asks whether specific, causal entry timing
patterns can survive different market environments without being selected on
March or frozen Q4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.environment_diagnostics import time_bucket
from v4.model.hypothesis_protocol import MarketStructureCache
from v4.model.supervised_pilot import session_from_path
from v4.scripts.run_edge_existence_audit import (
    _add_rule_bins,
    _build_random_lookup,
    _candidate_summary,
    _load_1s_audit,
    _normalized_audit,
    _random_same_times,
    _safe_float,
    _stress_metrics,
    _trade_metrics,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


_NY = ZoneInfo("America/New_York")

PATTERN_COLUMNS: tuple[str, ...] = (
    "sigma_trend_continuation",
    "vwap_reclaim",
    "vwap_hold_continuation",
    "vwap_pullback_resume",
    "last10_breakout",
    "compression_breakout",
    "momentum_ignition",
    "pullback_resume",
    "omar_mid_reclaim",
    "omar_retest_bounce",
    "first15_acceptance_break",
    "first15_inside_reversal",
)

RULE_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("side", "entry_pattern"),
    ("side", "entry_pattern", "moneyness_bucket"),
    ("side", "entry_pattern", "time_bucket"),
    ("side", "entry_pattern", "premium_bucket"),
    ("side", "entry_pattern", "spread_quality"),
    ("side", "entry_pattern", "moneyness_bucket", "premium_bucket"),
    ("side", "entry_pattern", "moneyness_bucket", "spread_quality"),
    ("side", "entry_pattern", "time_bucket", "moneyness_bucket"),
)


@dataclass(frozen=True)
class PatternRule:
    policy_index: int
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @property
    def name(self) -> str:
        parts = [f"policy{self.policy_index}"]
        parts.extend(f"{c}={v}" for c, v in zip(self.columns, self.values))
        return "|".join(parts)

    @property
    def rule_id(self) -> str:
        return hashlib.sha1(self.name.encode("utf-8")).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized"))
    parser.add_argument("--cbbo-1s-audit", type=Path, default=Path("v4/audit/cbbo_1m_vs_1s_audit_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/entry_timing_patterns"))
    parser.add_argument("--policy-indexes", nargs="*", type=int, default=sorted(POLICY_META), choices=sorted(POLICY_META))
    parser.add_argument("--min-discovery-candidates", type=int, default=80)
    parser.add_argument("--min-selection-candidates", type=int, default=20)
    parser.add_argument("--max-rules-per-policy", type=int, default=160)
    parser.add_argument("--max-simulated-rules", type=int, default=360)
    parser.add_argument("--max-trades-per-day", type=int, default=2)
    parser.add_argument("--random-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser.parse_args()


def _split_for_session(session: str) -> str:
    year = int(session[:4])
    month = int(session[5:7])
    day = int(session[8:10])
    if year == 2025:
        return "q4"
    if month == 1:
        return "train"
    if month == 2 and day <= 13:
        return "calibration"
    if month == 2:
        return "selection"
    if month == 3:
        return "march"
    return "unknown"


def _paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.pkl"))


def _side_value(value: float, side: str) -> float:
    direction = 1.0 if side == "C" else -1.0
    return direction * value if math.isfinite(value) else 0.0


def _market_sequence_features(market_window: np.ndarray, structure: np.ndarray, side: str) -> dict[str, bool | float]:
    close = market_window[:, 0].astype(float)
    vwap = market_window[:, 2].astype(float)
    omar = market_window[:, 3].astype(float)
    if len(close) < 2:
        return {name: False for name in PATTERN_COLUMNS}

    current = float(close[-1])
    prev = float(close[-2])
    gap = close - vwap
    current_gap = float(gap[-1])
    prev_gap = float(gap[-2]) if len(gap) >= 2 else 0.0
    side_gap = _side_value(current_gap, side)
    side_prev_gap = _side_value(prev_gap, side)
    prior5_gap = gap[-6:-1] if len(gap) >= 6 else gap[:-1]
    prior5_close = close[-6:-1] if len(close) >= 6 else close[:-1]
    prior15_close = close[-16:-1] if len(close) >= 16 else close[:-1]

    move1 = current - prev
    move3 = current - float(close[-4]) if len(close) >= 4 else move1
    move5 = current - float(close[-6]) if len(close) >= 6 else move3
    move15 = current - float(close[-16]) if len(close) >= 16 else move5
    side_move1 = _side_value(move1, side)
    side_move3 = _side_value(move3, side)
    side_move5 = _side_value(move5, side)
    side_move15 = _side_value(move15, side)

    last5_range = float(np.nanmax(close[-5:]) - np.nanmin(close[-5:])) if len(close) >= 5 else 0.0
    last15_range = float(np.nanmax(close[-15:]) - np.nanmin(close[-15:])) if len(close) >= 15 else max(last5_range, 0.01)
    compression_ratio = last5_range / max(last15_range, 0.01)

    sigma_pos = _safe_float(structure[2], 0.0)
    omar_mid = _safe_float(structure[9], current)
    omar_retest_dist = _safe_float(structure[13], 9.0)
    first15_acceptance = _safe_float(structure[17], 0.0)
    inside_first15 = _safe_float(structure[18], 0.0)
    last10_break_state = _safe_float(structure[21], 0.0)
    atr15_points = max(_safe_float(structure[22], 0.0) * max(abs(current), 1.0), 0.5)

    side_sigma = _side_value(sigma_pos, side)
    side_last10_break = _side_value(last10_break_state, side)
    side_first15 = _side_value(first15_acceptance, side)
    side_omar_now = _side_value(current - omar_mid, side)
    side_omar_prev = _side_value(prev - omar_mid, side)

    prior_counter_or_near_vwap = bool(len(prior5_gap) and np.nanmin([_side_value(x, side) for x in prior5_gap]) <= 0.75)
    prior_aligned_vwap = bool(len(prior5_gap) and np.nanmin([_side_value(x, side) for x in prior5_gap]) > 0.75)
    finite_prior5 = prior5_close[np.isfinite(prior5_close)]
    if len(finite_prior5):
        pullback_reference = float(np.nanmax(finite_prior5) if side == "P" else np.nanmin(finite_prior5))
        prior_counter_move = _side_value(current - pullback_reference, side) > 0
    else:
        prior_counter_move = False
    prior15_compressed = compression_ratio <= 0.45 or last5_range <= 0.85 * atr15_points

    return {
        "side_gap": side_gap,
        "side_sigma": side_sigma,
        "side_move1": side_move1,
        "side_move3": side_move3,
        "side_move5": side_move5,
        "side_move15": side_move15,
        "compression_ratio": compression_ratio,
        "sigma_trend_continuation": bool(side_sigma > 0.25 and side_move5 > 1.0 and side_move15 > 2.0),
        "vwap_reclaim": bool(side_gap > 0.75 and side_prev_gap <= 0.0 and side_move1 > 0.5),
        "vwap_hold_continuation": bool(side_sigma > 0.25 and prior_aligned_vwap and side_move5 > 0.5),
        "vwap_pullback_resume": bool(side_sigma > 0.25 and side_gap > 0.75 and prior_counter_or_near_vwap and side_move1 > 0.5),
        "last10_breakout": bool(side_last10_break > 0.5 and side_move1 > 0.5),
        "compression_breakout": bool(side_last10_break > 0.5 and prior15_compressed and side_move1 > 0.5),
        "momentum_ignition": bool(side_move1 > 0.75 and side_move3 > 1.5 and side_move5 > 2.0),
        "pullback_resume": bool(side_sigma > 0.25 and prior_counter_move and side_move1 > 0.5),
        "omar_mid_reclaim": bool(side_omar_now > 0.0 and side_omar_prev <= 0.0 and side_move1 > 0.5),
        "omar_retest_bounce": bool(side_sigma > 0.25 and omar_retest_dist <= 0.35 and side_move1 > 0.5),
        "first15_acceptance_break": bool(side_first15 > 0.65 and side_move1 > 0.5),
        "first15_inside_reversal": bool(inside_first15 >= 0.5 and side_first15 > 0.15 and side_move1 > 0.5),
    }


def _moneyness_steps(offset: float, side: str) -> float:
    return offset / 5.0 if side == "C" else -offset / 5.0


def _quality_score(row: dict) -> float:
    spread_frac = _safe_float(row["spread_frac"], 1.0)
    spread = _safe_float(row["spread"], 5.0)
    min_size = max(min(_safe_float(row["bid_size"], 0.0), _safe_float(row["ask_size"], 0.0)), 0.0)
    volume = max(_safe_float(row["option_ohlcv_volume"], 0.0), 0.0)
    ask = max(_safe_float(row["ask"], 0.0), 0.0)
    pattern_count = max(_safe_float(row.get("pattern_count"), 0.0), 0.0)
    return (
        0.20 * pattern_count
        - 8.0 * spread_frac
        - 0.4 * spread
        - 0.04 * abs(_safe_float(row["moneyness_steps"], 0.0))
        + 0.04 * math.log1p(min_size)
        + 0.015 * math.log1p(volume)
        - 0.01 * max(ask - 10.0, 0.0)
    )


def _load_pattern_candidates(
    paths: Sequence[Path],
    *,
    policies: Sequence[int],
    market_cache: MarketStructureCache,
) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        session = session_from_path(path)
        split = _split_for_session(session)
        with path.open("rb") as handle:
            day_rows = pickle.load(handle)
        for decision in day_rows:
            decision_time = decision["decision_time"]
            local = decision_time.astimezone(_NY)
            structure = market_cache.features_for(decision_time)
            market_window = np.asarray(decision["market_window"], dtype=np.float32)
            market_last = market_window[-1]
            feature_names = list(decision["feature_names"])
            index = {name: idx for idx, name in enumerate(feature_names)}
            mask = np.asarray(decision["candidate_mask"], dtype=bool)
            ladder = np.asarray(decision["option_ladder"], dtype=float)
            labels_net = np.asarray(decision["labels_net_pnl"], dtype=float)
            labels_mid = np.asarray(decision["labels_mid_pnl"], dtype=float)
            offsets = np.asarray(decision["strike_offsets"], dtype=float)
            rights = tuple(decision["rights"])
            contract_ids = np.asarray(decision["contract_ids"], dtype=object)
            for strike_idx, right_idx in zip(*np.where(mask)):
                side = str(rights[right_idx])
                option = ladder[strike_idx, right_idx]
                offset = float(offsets[strike_idx])
                patterns = _market_sequence_features(market_window, structure, side)
                pattern_names = [name for name in PATTERN_COLUMNS if patterns.get(name)]
                if not pattern_names:
                    continue
                base = {
                    "session": session,
                    "split": split,
                    "decision_time": decision_time.isoformat(),
                    "local_time": f"{local.hour:02d}:{local.minute:02d}",
                    "time_bucket": time_bucket(decision_time),
                    "side": side,
                    "offset": offset,
                    "moneyness_steps": _moneyness_steps(offset, side),
                    "contract_id": str(contract_ids[strike_idx, right_idx]),
                    "bid": _safe_float(option[index["bid"]]),
                    "ask": _safe_float(option[index["ask"]]),
                    "mid": _safe_float(option[index["mid"]]),
                    "spread": _safe_float(option[index["spread"]]),
                    "spread_frac": _safe_float(option[index["spread_frac"]]),
                    "bid_size": _safe_float(option[index["bid_size"]]),
                    "ask_size": _safe_float(option[index["ask_size"]]),
                    "option_ohlcv_volume": _safe_float(option[index["option_ohlcv_volume"]], 0.0),
                    "stat_open_interest": _safe_float(option[index["stat_open_interest"]], 0.0),
                    "iv": _safe_float(option[index["iv"]]),
                    "delta": _safe_float(option[index["delta"]]),
                    "gamma": _safe_float(option[index["gamma"]]),
                    "theta": _safe_float(option[index["theta"]]),
                    "breakeven_distance": _safe_float(option[index["breakeven_distance"]]),
                    "spx_close": _safe_float(market_last[0]),
                    "vix_close": _safe_float(market_last[1]),
                    "spx_vwap": _safe_float(market_last[2]),
                    "omar": _safe_float(market_last[3]),
                    "session_range": _safe_float(market_last[4]),
                    "momentum_5m": _safe_float(market_last[5]),
                    "momentum_15m": _safe_float(market_last[6]),
                    "sigma_pos": _safe_float(structure[2]),
                    "vwap_dist_pct": _safe_float(structure[5]),
                    "first15_acceptance": _safe_float(structure[17]),
                    "last10_break_state": _safe_float(structure[21]),
                    "atr15_pct": _safe_float(structure[22]),
                    "pattern_count": len(pattern_names),
                    **{name: bool(patterns.get(name, False)) for name in PATTERN_COLUMNS},
                }
                base["quality_score"] = _quality_score(base)
                for entry_pattern in pattern_names:
                    for policy_idx in policies:
                        pnl = _safe_float(labels_net[strike_idx, right_idx, policy_idx])
                        mid_pnl = _safe_float(labels_mid[strike_idx, right_idx, policy_idx])
                        if not math.isfinite(pnl):
                            continue
                        rows.append(
                            {
                                **base,
                                "entry_pattern": entry_pattern,
                                "policy_index": int(policy_idx),
                                "policy_name": POLICY_META[int(policy_idx)][0],
                                "pnl": pnl,
                                "mid_pnl": mid_pnl,
                                "mid_minus_net": mid_pnl - pnl if math.isfinite(mid_pnl) else math.nan,
                            }
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("pattern candidate frame is empty")
    return _add_rule_bins(frame)


def _group_metrics(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame[list(columns) + ["pnl"]].copy()
    work["gross_profit"] = work["pnl"].clip(lower=0)
    work["gross_loss"] = -work["pnl"].clip(upper=0)
    work["wins"] = (work["pnl"] > 0).astype(float)
    grouped = work.groupby(list(columns), observed=True, dropna=False).agg(
        count=("pnl", "size"),
        total=("pnl", "sum"),
        avg=("pnl", "mean"),
        median=("pnl", "median"),
        gross_profit=("gross_profit", "sum"),
        gross_loss=("gross_loss", "sum"),
        win_rate=("wins", "mean"),
    )
    grouped["pf"] = grouped["gross_profit"] / grouped["gross_loss"].replace(0.0, np.nan)
    grouped["pf"] = grouped["pf"].replace([np.inf, -np.inf], np.nan).fillna(999.0)
    return grouped.reset_index()


def _discover_rules(
    frame: pd.DataFrame,
    *,
    min_discovery_candidates: int,
    min_selection_candidates: int,
    max_per_policy: int,
) -> list[dict]:
    discovery = frame[frame["split"].isin(["train", "calibration"])]
    selection = frame[frame["split"] == "selection"]
    rows: list[dict] = []
    for policy_index in sorted(frame["policy_index"].unique()):
        policy_rows: list[dict] = []
        d_policy = discovery[discovery["policy_index"] == policy_index]
        s_policy = selection[selection["policy_index"] == policy_index]
        for columns in RULE_TEMPLATES:
            d_metrics = _group_metrics(d_policy, columns)
            s_metrics = _group_metrics(s_policy, columns)
            if d_metrics.empty or s_metrics.empty:
                continue
            merged = d_metrics.merge(s_metrics, on=list(columns), suffixes=("_discovery", "_selection"))
            merged = merged[
                (merged["count_discovery"] >= min_discovery_candidates)
                & (merged["count_selection"] >= min_selection_candidates)
                & (merged["avg_discovery"] > 0)
                & (merged["avg_selection"] > 0)
                & (merged["pf_discovery"] >= 1.05)
                & (merged["pf_selection"] >= 1.05)
            ]
            for record in merged.to_dict("records"):
                values = tuple(str(record[col]) for col in columns)
                rule = PatternRule(int(policy_index), tuple(columns), values)
                policy_rows.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule": rule.name,
                        "policy_index": int(policy_index),
                        "columns": list(columns),
                        "values": list(values),
                        "discovery_candidates": int(record["count_discovery"]),
                        "selection_candidates": int(record["count_selection"]),
                        "discovery_avg_candidate_pnl": float(record["avg_discovery"]),
                        "selection_avg_candidate_pnl": float(record["avg_selection"]),
                        "discovery_pf_candidate": float(record["pf_discovery"]),
                        "selection_pf_candidate": float(record["pf_selection"]),
                        "selection_total_candidate_pnl": float(record["total_selection"]),
                    }
                )
        deduped = {}
        for row in sorted(
            policy_rows,
            key=lambda r: (
                r["selection_avg_candidate_pnl"],
                r["selection_pf_candidate"],
                r["selection_candidates"],
                r["discovery_avg_candidate_pnl"],
            ),
            reverse=True,
        ):
            deduped.setdefault(row["rule_id"], row)
        rows.extend(list(deduped.values())[:max_per_policy])
    return rows


def _rule_mask(frame: pd.DataFrame, rule: PatternRule) -> pd.Series:
    mask = frame["policy_index"].eq(rule.policy_index)
    for column, value in zip(rule.columns, rule.values):
        mask &= frame[column].astype(str).eq(str(value))
    return mask


def _simulate_rule(
    frame: pd.DataFrame,
    rule: PatternRule,
    *,
    split: str,
    cooldown_minutes: int,
    max_trades_per_day: int,
) -> list[dict]:
    candidates = frame[(frame["split"] == split) & _rule_mask(frame, rule)].copy()
    if candidates.empty:
        return []
    candidates = candidates.sort_values(
        ["session", "decision_time", "pattern_count", "quality_score", "spread_frac"],
        ascending=[True, True, False, False, True],
    )
    chosen_by_minute = candidates.groupby(["session", "decision_time"], as_index=False, sort=False).head(1)
    trades: list[dict] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    for record in chosen_by_minute.sort_values(["session", "decision_time"]).to_dict("records"):
        session = str(record["session"])
        if trades_by_session.get(session, 0) >= max_trades_per_day:
            continue
        decision_time = datetime.fromisoformat(record["decision_time"])
        next_time = next_time_by_session.get(session)
        if next_time is not None and decision_time < next_time:
            continue
        trades.append(
            {
                "rule_id": rule.rule_id,
                "session": session,
                "decision_time": record["decision_time"],
                "pnl": float(record["pnl"]),
                "side": str(record["side"]),
                "offset": float(record["offset"]),
                "entry_pattern": str(record["entry_pattern"]),
                "time_bucket": str(record["time_bucket"]),
                "vix_bucket": str(record.get("vix_bucket", "unknown")),
                "range_bucket": str(record.get("range_bucket", "unknown")),
                "contract_id": str(record["contract_id"]),
            }
        )
        trades_by_session[session] = trades_by_session.get(session, 0) + 1
        next_time_by_session[session] = decision_time + timedelta(minutes=cooldown_minutes)
    return trades


def _trade_metrics_by_bucket(trades: Sequence[dict], column: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for trade in trades:
        buckets.setdefault(str(trade.get(column, "unknown")), []).append(trade)
    return {key: _trade_metrics(value) for key, value in sorted(buckets.items())}


def _transfer_score(row: dict) -> tuple:
    selection = row["metrics_by_split"]["selection"]
    march = row["metrics_by_split"]["march"]
    q4 = row["metrics_by_split"]["q4"]
    stress = row["slippage_stress_by_split"]
    return (
        row["transfer_gate_pass"],
        row["cross_regime_positive_bucket_fraction"],
        march["total_pnl"] > 0 and q4["total_pnl"] > 0,
        selection["total_pnl"],
        march["total_pnl"] + q4["total_pnl"],
        stress["march"]["25"]["total_pnl"] + stress["q4"]["25"]["total_pnl"],
    )


def _simulate_rules(
    frame: pd.DataFrame,
    discovered: Sequence[dict],
    *,
    max_rules: int,
    max_trades_per_day: int,
    random_runs: int,
    seed: int,
) -> list[dict]:
    random_lookup = _build_random_lookup(frame)
    selected = sorted(
        discovered,
        key=lambda r: (
            r["selection_avg_candidate_pnl"],
            r["selection_pf_candidate"],
            r["selection_candidates"],
        ),
        reverse=True,
    )[:max_rules]
    rows: list[dict] = []
    for row in selected:
        rule = PatternRule(
            policy_index=int(row["policy_index"]),
            columns=tuple(row["columns"]),
            values=tuple(row["values"]),
        )
        cooldown = POLICY_META[rule.policy_index][1]
        metrics_by_split = {}
        random_by_split = {}
        stress_by_split = {}
        regime_by_split = {}
        all_bucket_metrics = []
        for split in ("selection", "march", "q4"):
            trades = _simulate_rule(
                frame,
                rule,
                split=split,
                cooldown_minutes=cooldown,
                max_trades_per_day=max_trades_per_day,
            )
            metrics_by_split[split] = _trade_metrics(trades)
            random_by_split[split] = _random_same_times(
                frame,
                trades,
                policy_index=rule.policy_index,
                split=split,
                lookup=random_lookup,
                runs=random_runs,
                seed=seed + int(rule.rule_id[:6], 16),
            )
            regime_by_split[split] = {
                "time_bucket": _trade_metrics_by_bucket(trades, "time_bucket"),
                "vix_bucket": _trade_metrics_by_bucket(trades, "vix_bucket"),
                "range_bucket": _trade_metrics_by_bucket(trades, "range_bucket"),
            }
            for bucket_table in regime_by_split[split].values():
                all_bucket_metrics.extend(m for m in bucket_table.values() if m["trades"] >= 5)
            if split in {"march", "q4"}:
                stress_by_split[split] = _stress_metrics(trades)
        selection = metrics_by_split["selection"]
        march = metrics_by_split["march"]
        q4 = metrics_by_split["q4"]
        positive_bucket_fraction = (
            float(np.mean([m["total_pnl"] > 0 for m in all_bucket_metrics])) if all_bucket_metrics else 0.0
        )
        transfer_pass = (
            selection["trades"] >= 8
            and selection["total_pnl"] > 0
            and selection["profit_factor"] >= 1.10
            and march["trades"] >= 10
            and march["total_pnl"] > 0
            and march["profit_factor"] >= 1.05
            and q4["trades"] >= 20
            and q4["total_pnl"] > 0
            and q4["profit_factor"] >= 1.05
            and positive_bucket_fraction >= 0.55
            and stress_by_split["march"]["25"]["total_pnl"] > 0
            and stress_by_split["q4"]["25"]["total_pnl"] > 0
            and march["total_pnl"] > random_by_split["march"]["total_pnl_median"]
            and q4["total_pnl"] > random_by_split["q4"]["total_pnl_median"]
        )
        rows.append(
            {
                **row,
                "transfer_gate_pass": bool(transfer_pass),
                "cross_regime_positive_bucket_fraction": positive_bucket_fraction,
                "metrics_by_split": metrics_by_split,
                "random_same_time_by_split": random_by_split,
                "slippage_stress_by_split": stress_by_split,
                "regime_metrics_by_split": regime_by_split,
            }
        )
    rows.sort(key=_transfer_score, reverse=True)
    return rows


def _pattern_summary(frame: pd.DataFrame) -> dict:
    rows = []
    for pattern, group in frame.groupby("entry_pattern", observed=True):
        rows.append(
            {
                "entry_pattern": str(pattern),
                "rows": int(len(group)),
                "sessions": int(group["session"].nunique()),
                "selection_rows": int((group["split"] == "selection").sum()),
                "march_rows": int((group["split"] == "march").sum()),
                "q4_rows": int((group["split"] == "q4").sum()),
            }
        )
    rows.sort(key=lambda r: r["rows"], reverse=True)
    return {"patterns": rows, "pattern_count": len(rows)}


def _write_markdown(path: Path, payload: dict) -> None:
    champion = payload["champion"]
    lines = [
        "# Entry Timing Pattern Audit",
        "",
        "Non-neural audit for transferable, causal entry timing primitives.",
        "",
        f"Transfer gate pass count: `{payload['transfer_gate_pass_count']}`",
        f"Discovered pattern rules: `{payload['discovered_rule_count']}`",
        f"Simulated pattern rules: `{payload['simulated_rule_count']}`",
        "",
        "## Champion",
        "",
    ]
    if champion:
        lines.extend(
            [
                f"Best pattern rule: `{champion['rule']}`",
                f"Cross-regime positive bucket fraction: `{champion['cross_regime_positive_bucket_fraction']:.2f}`",
                "",
                "| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in ("selection", "march", "q4"):
            metrics = champion["metrics_by_split"][split]
            random = champion["random_same_time_by_split"][split]
            lines.append(
                f"| {split} | {metrics['trades']} | {metrics['total_pnl']:.0f} | "
                f"{metrics['profit_factor']:.3f} | {metrics['max_drawdown']:.0f} | "
                f"{metrics['positive_day_fraction']:.2f} | {random['total_pnl_median']:.0f} |"
            )
    lines.extend(
        [
            "",
            "## Top Pattern Rules",
            "",
            "| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(payload["top_simulated_rules"][:25], start=1):
        sel = row["metrics_by_split"]["selection"]
        march = row["metrics_by_split"]["march"]
        q4 = row["metrics_by_split"]["q4"]
        lines.append(
            f"| {idx} | `{row['rule']}` | {sel['total_pnl']:.0f}/{sel['profit_factor']:.3f} | "
            f"{march['total_pnl']:.0f}/{march['profit_factor']:.3f} | "
            f"{q4['total_pnl']:.0f}/{q4['profit_factor']:.3f} | "
            f"{row['cross_regime_positive_bucket_fraction']:.2f} | {row['transfer_gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Data Notes",
            "",
            f"- Pattern candidate rows: `{payload['data_summary']['candidate_frame']['rows']:,}`.",
            f"- CBBO-1m minute-boundary audit acceptable: `{payload['data_summary']['cbbo_1m_vs_1s'].get('acceptable_for_minute_boundary_prototype')}`.",
            f"- Median p95 intraminute mid range: `${payload['data_summary']['cbbo_1m_vs_1s'].get('median_p95_intraminute_mid_range', math.nan):.2f}`.",
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    market_cache = MarketStructureCache()
    all_paths = _paths(args.data_dir) + _paths(args.q4_data_dir)
    sessions = [session_from_path(path) for path in all_paths]
    frame = _load_pattern_candidates(all_paths, policies=args.policy_indexes, market_cache=market_cache)
    discovered = _discover_rules(
        frame,
        min_discovery_candidates=args.min_discovery_candidates,
        min_selection_candidates=args.min_selection_candidates,
        max_per_policy=args.max_rules_per_policy,
    )
    simulated = _simulate_rules(
        frame,
        discovered,
        max_rules=args.max_simulated_rules,
        max_trades_per_day=args.max_trades_per_day,
        random_runs=args.random_runs,
        seed=args.seed,
    )
    pass_count = sum(1 for row in simulated if row["transfer_gate_pass"])
    champion = simulated[0] if simulated else None
    payload = {
        "audit_id": "v4_entry_timing_pattern_audit_001",
        "framing": "Test transferable, causal entry timing primitives before changing the neural model.",
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "policy_indexes": args.policy_indexes,
            "min_discovery_candidates": args.min_discovery_candidates,
            "min_selection_candidates": args.min_selection_candidates,
            "max_trades_per_day": args.max_trades_per_day,
            "random_runs": args.random_runs,
        },
        "pattern_definitions": list(PATTERN_COLUMNS),
        "pattern_summary": _pattern_summary(frame),
        "data_summary": {
            "candidate_frame": _candidate_summary(frame),
            "normalized": _normalized_audit(args.normalized_dir, sessions),
            "cbbo_1m_vs_1s": _load_1s_audit(args.cbbo_1s_audit),
        },
        "discovered_rule_count": len(discovered),
        "simulated_rule_count": len(simulated),
        "transfer_gate_pass_count": pass_count,
        "champion": champion,
        "top_simulated_rules": simulated[:60],
        "top_discovered_rules": discovered[:60],
        "interpretation": (
            "Transferable timing leads exist if transfer_gate_pass_count is positive. "
            "The next neural target should learn these pattern primitives directly, but still abstain unless "
            "the broader causal context says the pattern is worth paying the spread. These results are research leads, "
            "not live-trading permission."
            if pass_count
            else "No timing primitive survived the transfer gate. The next step would be improving data resolution or "
            "changing the trade structure rather than adding neural complexity."
        ),
    }
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(args.out_dir / "report.md", payload)
    print(json.dumps({
        "discovered_rule_count": len(discovered),
        "simulated_rule_count": len(simulated),
        "transfer_gate_pass_count": pass_count,
        "champion": champion["rule"] if champion else None,
        "champion_selection_pnl": champion["metrics_by_split"]["selection"]["total_pnl"] if champion else None,
        "champion_march_pnl": champion["metrics_by_split"]["march"]["total_pnl"] if champion else None,
        "champion_q4_pnl": champion["metrics_by_split"]["q4"]["total_pnl"] if champion else None,
    }, indent=2))
    print(f"WROTE {args.out_dir / 'report.json'}")
    print(f"WROTE {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
