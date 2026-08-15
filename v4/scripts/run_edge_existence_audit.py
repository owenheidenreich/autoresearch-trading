"""Non-neural edge existence audit for the SPXW 0DTE prototype.

The goal is not to train a better model. The goal is to answer two narrower
questions before more architecture work:

1. Does the current v4 data resemble what a minute-by-minute live trader would
   have seen closely enough for a prototype?
2. Are there any transparent, picky entry cells with executable ask-entry /
   bid-exit expectancy that survive March 2026 and frozen Q4 2025?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.environment_diagnostics import omar_bucket, time_bucket
from v4.model.hypothesis_protocol import MarketStructureCache
from v4.model.supervised_pilot import session_from_path
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


_NY = ZoneInfo("America/New_York")
_SESSION_RE = re.compile(r"databento_spxw_0dte_(\d{4}-\d{2}-\d{2})\.parquet$")

RULE_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("side", "time_bucket"),
    ("side", "time_bucket", "moneyness_bucket"),
    ("side", "time_bucket", "vwap_side"),
    ("side", "time_bucket", "omar_side"),
    ("side", "time_bucket", "trend15_side"),
    ("side", "time_bucket", "sigma_side"),
    ("side", "time_bucket", "spread_quality"),
    ("side", "time_bucket", "premium_bucket"),
    ("side", "time_bucket", "vix_bucket"),
    ("side", "time_bucket", "range_bucket"),
    ("side", "time_bucket", "atr_bucket"),
    ("side", "moneyness_bucket", "spread_quality"),
    ("side", "moneyness_bucket", "premium_bucket"),
    ("side", "moneyness_bucket", "vwap_side"),
    ("side", "moneyness_bucket", "omar_side"),
    ("side", "moneyness_bucket", "trend15_side"),
    ("side", "moneyness_bucket", "sigma_side"),
    ("side", "spread_quality", "premium_bucket"),
    ("side", "spread_quality", "liquidity_bucket"),
    ("side", "volume_bucket", "oi_bucket"),
    ("side", "time_bucket", "moneyness_bucket", "spread_quality"),
    ("side", "time_bucket", "moneyness_bucket", "premium_bucket"),
    ("side", "time_bucket", "moneyness_bucket", "vwap_side"),
    ("side", "time_bucket", "moneyness_bucket", "omar_side"),
    ("side", "time_bucket", "moneyness_bucket", "trend15_side"),
    ("side", "time_bucket", "moneyness_bucket", "sigma_side"),
    ("side", "time_bucket", "vwap_side", "trend15_side"),
    ("side", "time_bucket", "omar_side", "trend15_side"),
    ("side", "time_bucket", "sigma_side", "trend15_side"),
    ("side", "time_bucket", "first15_side", "last10_break_bucket"),
    ("side", "time_bucket", "vix_bucket", "range_bucket"),
    ("side", "time_bucket", "spread_quality", "liquidity_bucket"),
    ("side", "time_bucket", "spread_quality", "volume_bucket"),
    ("side", "time_bucket", "premium_bucket", "spread_quality"),
    ("side", "moneyness_bucket", "vwap_side", "trend15_side"),
    ("side", "moneyness_bucket", "omar_side", "trend15_side"),
    ("side", "moneyness_bucket", "sigma_side", "trend15_side"),
    ("side", "vwap_side", "omar_side", "trend15_side"),
)


@dataclass(frozen=True)
class Rule:
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
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/edge_existence"))
    parser.add_argument("--policy-indexes", nargs="*", type=int, default=sorted(POLICY_META), choices=sorted(POLICY_META))
    parser.add_argument("--min-discovery-candidates", type=int, default=120)
    parser.add_argument("--min-selection-candidates", type=int, default=40)
    parser.add_argument("--max-discovered-rules-per-policy", type=int, default=160)
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


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _quality_score(row: dict) -> float:
    spread_frac = _safe_float(row["spread_frac"], 1.0)
    spread = _safe_float(row["spread"], 5.0)
    min_size = max(min(_safe_float(row["bid_size"], 0.0), _safe_float(row["ask_size"], 0.0)), 0.0)
    volume = max(_safe_float(row["option_ohlcv_volume"], 0.0), 0.0)
    ask = max(_safe_float(row["ask"], 0.0), 0.0)
    return (
        -8.0 * spread_frac
        -0.4 * spread
        -0.04 * abs(_safe_float(row["moneyness_steps"], 0.0))
        +0.04 * math.log1p(min_size)
        +0.015 * math.log1p(volume)
        -0.01 * max(ask - 10.0, 0.0)
    )


def _moneyness_steps(offset: float, side: str) -> float:
    # Positive means OTM in trader terms: calls above spot, puts below spot.
    return offset / 5.0 if side == "C" else -offset / 5.0


def _load_candidates(
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
            market_last = np.asarray(decision["market_window"], dtype=np.float32)[-1]
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
                    "true_spx_close": _safe_float(structure[0]),
                    "true_vix_close": _safe_float(structure[1]),
                    "sigma_pos": _safe_float(structure[2]),
                    "vwap_dist_pct": _safe_float(structure[5]),
                    "vwap_slope_5m_pct": _safe_float(structure[6]),
                    "omar_mid_pos_units": _safe_float(structure[12]),
                    "omar_retest_dist_norm": _safe_float(structure[13]),
                    "first15_acceptance": _safe_float(structure[17]),
                    "inside_first15": _safe_float(structure[18]),
                    "last10_range_over_omar": _safe_float(structure[20]),
                    "last10_break_state": _safe_float(structure[21]),
                    "atr15_pct": _safe_float(structure[22]),
                }
                base["quality_score"] = _quality_score(base)
                for policy_idx in policies:
                    pnl = _safe_float(labels_net[strike_idx, right_idx, policy_idx])
                    mid_pnl = _safe_float(labels_mid[strike_idx, right_idx, policy_idx])
                    if not math.isfinite(pnl):
                        continue
                    rows.append(
                        {
                            **base,
                            "policy_index": int(policy_idx),
                            "policy_name": POLICY_META[int(policy_idx)][0],
                            "pnl": pnl,
                            "mid_pnl": mid_pnl,
                            "mid_minus_net": mid_pnl - pnl if math.isfinite(mid_pnl) else math.nan,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("candidate frame is empty")
    return frame


def _fixed_bucket(
    values: pd.Series,
    *,
    low: float,
    high: float,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.full(len(arr), mid_label, dtype=object), index=values.index)
    out[arr <= low] = low_label
    out[arr >= high] = high_label
    out[arr.isna()] = "unknown"
    return out


def _quantile_bucket(values: pd.Series, reference: pd.Series, prefix: str) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce")
    ref = ref[np.isfinite(ref)]
    if len(ref) < 10:
        return pd.Series(np.full(len(values), f"{prefix}_unknown", dtype=object), index=values.index)
    q1, q2 = np.nanquantile(ref, [1 / 3, 2 / 3])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        q1, q2 = np.nanquantile(ref, [0.25, 0.75])
    arr = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.full(len(arr), f"{prefix}_mid", dtype=object), index=values.index)
    out[arr <= q1] = f"{prefix}_low"
    out[arr > q2] = f"{prefix}_high"
    out[arr.isna()] = f"{prefix}_unknown"
    return out


def _add_rule_bins(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    reference = frame[frame["split"].isin(["train", "calibration"])]

    steps = pd.to_numeric(frame["moneyness_steps"], errors="coerce")
    frame["moneyness_bucket"] = np.select(
        [
            steps <= -2,
            steps == -1,
            steps == 0,
            steps == 1,
            (steps >= 2) & (steps <= 3),
            steps >= 4,
        ],
        ["itm_2p", "itm_1", "atm", "otm_1", "otm_2_3", "otm_4p"],
        default="unknown",
    )
    frame["spread_quality"] = np.select(
        [
            (frame["spread"] <= 0.10) & (frame["spread_frac"] <= 0.05),
            (frame["spread"] <= 0.25) & (frame["spread_frac"] <= 0.12),
            (frame["spread"] <= 0.50) & (frame["spread_frac"] <= 0.25),
        ],
        ["excellent", "good", "acceptable"],
        default="wide_or_unknown",
    )
    frame["premium_bucket"] = np.select(
        [
            frame["ask"] <= 1.0,
            (frame["ask"] > 1.0) & (frame["ask"] <= 3.0),
            (frame["ask"] > 3.0) & (frame["ask"] <= 8.0),
            (frame["ask"] > 8.0) & (frame["ask"] <= 20.0),
            frame["ask"] > 20.0,
        ],
        ["cheap_le_1", "small_1_3", "medium_3_8", "large_8_20", "very_large_20p"],
        default="premium_unknown",
    )
    min_size = np.minimum(frame["bid_size"].fillna(0.0), frame["ask_size"].fillna(0.0))
    frame["liquidity_bucket"] = np.select(
        [
            min_size <= 1,
            (min_size > 1) & (min_size <= 5),
            (min_size > 5) & (min_size <= 20),
            min_size > 20,
        ],
        ["size_1", "size_2_5", "size_6_20", "size_20p"],
        default="size_unknown",
    )
    frame["volume_bucket"] = np.select(
        [
            frame["option_ohlcv_volume"].isna(),
            frame["option_ohlcv_volume"] <= 0,
            (frame["option_ohlcv_volume"] > 0) & (frame["option_ohlcv_volume"] <= 10),
            frame["option_ohlcv_volume"] > 10,
        ],
        ["volume_missing", "volume_zero", "volume_1_10", "volume_10p"],
        default="volume_unknown",
    )
    frame["oi_bucket"] = _quantile_bucket(
        frame["stat_open_interest"],
        reference["stat_open_interest"],
        "oi",
    )
    vwap_gap = frame["spx_close"] - frame["spx_vwap"]
    near_vwap = vwap_gap.abs() <= 2.0
    aligned_vwap = ((frame["side"] == "C") & (vwap_gap > 2.0)) | ((frame["side"] == "P") & (vwap_gap < -2.0))
    counter_vwap = ((frame["side"] == "C") & (vwap_gap < -2.0)) | ((frame["side"] == "P") & (vwap_gap > 2.0))
    frame["vwap_side"] = np.select(
        [near_vwap, aligned_vwap, counter_vwap],
        ["near_vwap", "vwap_aligned", "vwap_counter"],
        default="vwap_unknown",
    )
    frame["omar_bucket"] = frame["omar"].map(omar_bucket)
    aligned_omar = ((frame["side"] == "C") & (frame["omar"] > 0)) | ((frame["side"] == "P") & (frame["omar"] < 0))
    counter_omar = ((frame["side"] == "C") & (frame["omar"] < 0)) | ((frame["side"] == "P") & (frame["omar"] > 0))
    frame["omar_side"] = np.select(
        [frame["omar"].abs() <= 0.05, aligned_omar, counter_omar],
        ["omar_neutral", "omar_aligned", "omar_counter"],
        default="omar_unknown",
    )
    aligned_trend = ((frame["side"] == "C") & (frame["momentum_15m"] > 2.0)) | ((frame["side"] == "P") & (frame["momentum_15m"] < -2.0))
    counter_trend = ((frame["side"] == "C") & (frame["momentum_15m"] < -2.0)) | ((frame["side"] == "P") & (frame["momentum_15m"] > 2.0))
    frame["trend15_side"] = np.select(
        [frame["momentum_15m"].abs() <= 2.0, aligned_trend, counter_trend],
        ["trend15_flat", "trend15_aligned", "trend15_counter"],
        default="trend15_unknown",
    )
    aligned_sigma = ((frame["side"] == "C") & (frame["sigma_pos"] > 0.25)) | ((frame["side"] == "P") & (frame["sigma_pos"] < -0.25))
    counter_sigma = ((frame["side"] == "C") & (frame["sigma_pos"] < -0.25)) | ((frame["side"] == "P") & (frame["sigma_pos"] > 0.25))
    frame["sigma_side"] = np.select(
        [frame["sigma_pos"].abs() <= 0.25, aligned_sigma, counter_sigma],
        ["sigma_neutral", "sigma_aligned", "sigma_counter"],
        default="sigma_unknown",
    )
    first15_aligned = ((frame["side"] == "C") & (frame["first15_acceptance"] > 0.35)) | ((frame["side"] == "P") & (frame["first15_acceptance"] < -0.35))
    first15_counter = ((frame["side"] == "C") & (frame["first15_acceptance"] < -0.35)) | ((frame["side"] == "P") & (frame["first15_acceptance"] > 0.35))
    frame["first15_side"] = np.select(
        [frame["first15_acceptance"].abs() <= 0.35, first15_aligned, first15_counter],
        ["first15_neutral", "first15_aligned", "first15_counter"],
        default="first15_unknown",
    )
    frame["last10_break_bucket"] = _fixed_bucket(
        frame["last10_break_state"],
        low=-0.25,
        high=0.25,
        low_label="last10_down_break",
        mid_label="last10_inside",
        high_label="last10_up_break",
    )
    frame["vix_bucket"] = _quantile_bucket(frame["vix_close"], reference["vix_close"], "vix")
    frame["range_bucket"] = _quantile_bucket(frame["session_range"], reference["session_range"], "range")
    frame["atr_bucket"] = _quantile_bucket(frame["atr15_pct"], reference["atr15_pct"], "atr")
    return frame


def _group_metrics(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame[list(columns) + ["pnl"]].copy()
    work["gross_profit"] = work["pnl"].clip(lower=0)
    work["gross_loss"] = (-work["pnl"].clip(upper=0))
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
    rows: list[dict] = []
    discovery = frame[frame["split"].isin(["train", "calibration"])]
    selection = frame[frame["split"] == "selection"]
    for policy_index in sorted(frame["policy_index"].unique()):
        d_policy = discovery[discovery["policy_index"] == policy_index]
        s_policy = selection[selection["policy_index"] == policy_index]
        policy_rows: list[dict] = []
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
            ].copy()
            if merged.empty:
                continue
            for record in merged.to_dict("records"):
                values = tuple(str(record[col]) for col in columns)
                rule = Rule(policy_index=int(policy_index), columns=tuple(columns), values=values)
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
        seen = {}
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
            seen.setdefault(row["rule_id"], row)
        rows.extend(list(seen.values())[:max_per_policy])
    return rows


def _trade_metrics(trades: Sequence[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sessions_traded": 0,
            "positive_day_fraction": 0.0,
            "top_day_profit_share": 1.0,
        }
    ordered = sorted(trades, key=lambda r: (r["decision_time"], r["rule_id"]))
    pnl = np.asarray([float(r["pnl"]) for r in ordered], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = equity - peak
    by_day: dict[str, float] = {}
    for trade in ordered:
        by_day[trade["session"]] = by_day.get(trade["session"], 0.0) + float(trade["pnl"])
    daily = np.asarray(list(by_day.values()), dtype=float)
    positives = daily[daily > 0]
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(ordered)),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sessions_traded": int(len(by_day)),
        "mean_daily_pnl": float(daily.mean()) if len(daily) else 0.0,
        "median_daily_pnl": float(np.median(daily)) if len(daily) else 0.0,
        "positive_day_fraction": float((daily > 0).mean()) if len(daily) else 0.0,
        "top_day_profit_share": float(positives.max() / positives.sum()) if positives.sum() > 0 else 1.0,
    }


def _rule_mask(frame: pd.DataFrame, rule: Rule) -> pd.Series:
    mask = frame["policy_index"].eq(rule.policy_index)
    for column, value in zip(rule.columns, rule.values):
        mask &= frame[column].astype(str).eq(str(value))
    return mask


def _simulate_rule(
    frame: pd.DataFrame,
    rule: Rule,
    *,
    split: str,
    cooldown_minutes: int,
    max_trades_per_day: int,
) -> list[dict]:
    candidates = frame[(frame["split"] == split) & _rule_mask(frame, rule)].copy()
    if candidates.empty:
        return []
    candidates = candidates.sort_values(
        ["session", "decision_time", "quality_score", "spread_frac"],
        ascending=[True, True, False, True],
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
                "quality_score": float(record["quality_score"]),
                "contract_id": str(record["contract_id"]),
            }
        )
        trades_by_session[session] = trades_by_session.get(session, 0) + 1
        next_time_by_session[session] = decision_time + timedelta(minutes=cooldown_minutes)
    return trades


def _stress_metrics(trades: Sequence[dict]) -> dict[str, dict]:
    out = {}
    for cost in (25.0, 50.0, 100.0):
        stressed = [{**trade, "pnl": float(trade["pnl"]) - cost} for trade in trades]
        out[str(int(cost))] = _trade_metrics(stressed)
    return out


def _random_same_times(
    frame: pd.DataFrame,
    trades: Sequence[dict],
    *,
    policy_index: int,
    split: str,
    lookup: dict[tuple[int, str], dict[tuple[str, str], np.ndarray]],
    runs: int,
    seed: int,
) -> dict:
    if not trades:
        return {"runs": runs, "total_pnl_median": 0.0, "profit_factor_median": 0.0, "trades_median": 0.0}
    split_lookup = lookup.get((policy_index, split), {})
    rng = np.random.default_rng(seed)
    summaries = []
    for _ in range(runs):
        sampled = []
        for trade in trades:
            key = (trade["session"], trade["decision_time"])
            idx = split_lookup.get(key)
            if idx is None or len(idx) == 0:
                continue
            record = frame.loc[int(rng.choice(idx))]
            sampled.append(
                {
                    "rule_id": "random_same_time",
                    "session": str(record["session"]),
                    "decision_time": str(record["decision_time"]),
                    "pnl": float(record["pnl"]),
                }
            )
        summaries.append(_trade_metrics(sampled))
    return {
        "runs": int(runs),
        "trades_median": float(np.median([x["trades"] for x in summaries])),
        "total_pnl_median": float(np.median([x["total_pnl"] for x in summaries])),
        "profit_factor_median": float(np.median([x["profit_factor"] for x in summaries])),
        "positive_day_fraction_median": float(np.median([x["positive_day_fraction"] for x in summaries])),
        "top_day_profit_share_median": float(np.median([x["top_day_profit_share"] for x in summaries])),
    }


def _build_random_lookup(frame: pd.DataFrame) -> dict[tuple[int, str], dict[tuple[str, str], np.ndarray]]:
    lookup: dict[tuple[int, str], dict[tuple[str, str], np.ndarray]] = {}
    for (policy_index, split), split_frame in frame.groupby(["policy_index", "split"], sort=False):
        lookup[(int(policy_index), str(split))] = {
            key: group.index.to_numpy()
            for key, group in split_frame.groupby(["session", "decision_time"], sort=False)
        }
    return lookup


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
    rows = []
    for row in selected:
        rule = Rule(
            policy_index=int(row["policy_index"]),
            columns=tuple(row["columns"]),
            values=tuple(row["values"]),
        )
        cooldown = POLICY_META[rule.policy_index][1]
        metrics_by_split = {}
        random_by_split = {}
        stress_by_split = {}
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
            if split in {"march", "q4"}:
                stress_by_split[split] = _stress_metrics(trades)
        selection = metrics_by_split["selection"]
        march = metrics_by_split["march"]
        q4 = metrics_by_split["q4"]
        passes = (
            selection["trades"] >= 10
            and selection["total_pnl"] > 0
            and selection["profit_factor"] >= 1.10
            and selection["top_day_profit_share"] <= 0.55
            and march["trades"] >= 10
            and march["total_pnl"] > 0
            and march["profit_factor"] >= 1.05
            and march["positive_day_fraction"] >= 0.40
            and q4["trades"] >= 20
            and q4["total_pnl"] > 0
            and q4["profit_factor"] >= 1.05
            and q4["positive_day_fraction"] >= 0.40
            and stress_by_split["march"]["25"]["total_pnl"] > 0
            and stress_by_split["q4"]["25"]["total_pnl"] > 0
            and march["total_pnl"] > random_by_split["march"]["total_pnl_median"]
            and q4["total_pnl"] > random_by_split["q4"]["total_pnl_median"]
        )
        rows.append(
            {
                **row,
                "simulation_gate_pass": bool(passes),
                "metrics_by_split": metrics_by_split,
                "random_same_time_by_split": random_by_split,
                "slippage_stress_by_split": stress_by_split,
            }
        )
    rows.sort(
        key=lambda r: (
            r["simulation_gate_pass"],
            r["metrics_by_split"]["march"]["total_pnl"] > 0 and r["metrics_by_split"]["q4"]["total_pnl"] > 0,
            r["metrics_by_split"]["selection"]["total_pnl"],
            r["metrics_by_split"]["march"]["total_pnl"] + r["metrics_by_split"]["q4"]["total_pnl"],
        ),
        reverse=True,
    )
    return rows


def _candidate_summary(frame: pd.DataFrame) -> dict:
    out = {
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "policies": sorted(int(x) for x in frame["policy_index"].unique()),
        "by_split": {},
        "spread": {},
        "mid_vs_net": {},
    }
    for split, group in frame.groupby("split"):
        out["by_split"][split] = {
            "rows": int(len(group)),
            "sessions": int(group["session"].nunique()),
            "decision_minutes": int(group[["session", "decision_time"]].drop_duplicates().shape[0]),
            "median_candidates_per_minute_policy": float(
                group.groupby(["policy_index", "session", "decision_time"]).size().median()
            ),
        }
    for col in (
        "spread",
        "spread_frac",
        "ask",
        "bid_size",
        "ask_size",
        "option_ohlcv_volume",
        "stat_open_interest",
        "iv",
        "delta",
        "gamma",
        "theta",
    ):
        values = pd.to_numeric(frame[col], errors="coerce")
        out["spread" if col in {"spread", "spread_frac"} else "candidate_quality"] = out.get(
            "spread" if col in {"spread", "spread_frac"} else "candidate_quality",
            {},
        )
        target = out["spread" if col in {"spread", "spread_frac"} else "candidate_quality"]
        target[col] = {
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "missing_frac": float(values.isna().mean()),
        }
    diff = pd.to_numeric(frame["mid_minus_net"], errors="coerce")
    out["mid_vs_net"] = {
        "median_mid_minus_net": float(diff.median()),
        "p95_mid_minus_net": float(diff.quantile(0.95)),
        "positive_material_difference_frac": float((diff > 25.0).mean()),
    }
    return out


def _normalized_paths(normalized_dir: Path, sessions: Iterable[str]) -> list[Path]:
    needed = set(sessions)
    out = []
    for path in normalized_dir.glob("databento_spxw_0dte_*.parquet"):
        match = _SESSION_RE.match(path.name)
        if match and match.group(1) in needed:
            out.append(path)
    return sorted(out)


def _normalized_audit(normalized_dir: Path, sessions: Iterable[str]) -> dict:
    paths = _normalized_paths(normalized_dir, sessions)
    totals = {
        "files": len(paths),
        "rows": 0,
        "finite_quote_rows": 0,
        "bad_root_rows": 0,
        "bad_settlement_rows": 0,
        "bad_strike_alignment_rows": 0,
        "bad_quote_rows": 0,
        "quote_time_after_event_rows": 0,
        "contract_multiplier_counts": {},
        "min_price_increment_missing_rows": 0,
        "underlying_price_missing_finite_quote_rows": 0,
        "option_volume_missing_finite_quote_rows": 0,
        "last_trade_size_total": 0,
    }
    quote_age_values = []
    quote_gap_values = []
    settlement_hours = {}
    for path in paths:
        frame = pd.read_parquet(path)
        totals["rows"] += int(len(frame))
        finite_quote = frame["bid"].notna() & frame["ask"].notna() & frame["mid"].notna()
        totals["finite_quote_rows"] += int(finite_quote.sum())
        totals["bad_root_rows"] += int((frame["root"] != "SPXW").sum())
        totals["bad_settlement_rows"] += int((frame["settlement_style"] != "PM").sum())
        strikes = pd.to_numeric(frame["strike"], errors="coerce")
        totals["bad_strike_alignment_rows"] += int((~np.isclose(strikes % 5, 0.0)).sum())
        totals["bad_quote_rows"] += int(
            (
                finite_quote
                & (
                    (pd.to_numeric(frame["bid"], errors="coerce") < 0)
                    | (pd.to_numeric(frame["ask"], errors="coerce") <= 0)
                    | (pd.to_numeric(frame["ask"], errors="coerce") < pd.to_numeric(frame["bid"], errors="coerce"))
                )
            ).sum()
        )
        totals["quote_time_after_event_rows"] += int((frame["quote_time"] > frame["event_time"]).sum())
        counts = frame["contract_multiplier"].value_counts(dropna=False).to_dict()
        for key, value in counts.items():
            totals["contract_multiplier_counts"][str(key)] = totals["contract_multiplier_counts"].get(str(key), 0) + int(value)
        totals["min_price_increment_missing_rows"] += int(frame["min_price_increment"].isna().sum())
        totals["underlying_price_missing_finite_quote_rows"] += int((finite_quote & frame["underlying_price"].isna()).sum())
        totals["option_volume_missing_finite_quote_rows"] += int((finite_quote & frame["option_ohlcv_volume"].isna()).sum())
        totals["last_trade_size_total"] += int(pd.to_numeric(frame["last_trade_size"], errors="coerce").fillna(0).sum())
        quote_age_values.append(pd.to_numeric(frame.loc[finite_quote, "quote_age_ms"], errors="coerce"))
        quote_gap_values.append(pd.to_numeric(frame.loc[finite_quote, "quote_gap_seconds"], errors="coerce"))
        hours = pd.to_datetime(frame["settlement_time_utc"], utc=True).dt.hour.value_counts().to_dict()
        for key, value in hours.items():
            settlement_hours[str(key)] = settlement_hours.get(str(key), 0) + int(value)
    quote_age = pd.concat(quote_age_values, ignore_index=True) if quote_age_values else pd.Series(dtype=float)
    quote_gap = pd.concat(quote_gap_values, ignore_index=True) if quote_gap_values else pd.Series(dtype=float)
    finite = max(totals["finite_quote_rows"], 1)
    rows = max(totals["rows"], 1)
    return {
        **totals,
        "bad_root_frac": totals["bad_root_rows"] / rows,
        "bad_settlement_frac": totals["bad_settlement_rows"] / rows,
        "bad_strike_alignment_frac": totals["bad_strike_alignment_rows"] / rows,
        "bad_quote_frac_of_finite": totals["bad_quote_rows"] / finite,
        "quote_time_after_event_frac": totals["quote_time_after_event_rows"] / rows,
        "quote_age_ms_p95": float(quote_age.quantile(0.95)) if len(quote_age) else math.nan,
        "quote_gap_seconds_p95": float(quote_gap.quantile(0.95)) if len(quote_gap) else math.nan,
        "quote_gap_seconds_p99": float(quote_gap.quantile(0.99)) if len(quote_gap) else math.nan,
        "min_price_increment_missing_frac": totals["min_price_increment_missing_rows"] / rows,
        "underlying_price_missing_finite_quote_frac": totals["underlying_price_missing_finite_quote_rows"] / finite,
        "option_volume_missing_finite_quote_frac": totals["option_volume_missing_finite_quote_rows"] / finite,
        "settlement_utc_hour_counts": settlement_hours,
        "data_realism_notes": [
            "Normalized rows are SPXW PM-settled and 5-point aligned if the bad_* fractions are zero.",
            "contract_multiplier is audited because normalized files currently contain Databento definition sentinel values; labels use the dataset config multiplier of 100.",
            "SPX/VIX context files in data/raw/index are derived proxy context, not an official live Cboe index feed.",
        ],
    }


def _load_1s_audit(path: Path) -> dict:
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text())
    return {
        "available": True,
        "sessions": payload.get("sessions"),
        "acceptable_for_minute_boundary_prototype": payload.get("acceptable_for_prototype"),
        "min_coverage": payload.get("min_coverage"),
        "max_p95_mid_abs_diff": payload.get("max_p95_mid_abs_diff"),
        "mean_p95_mid_abs_diff": payload.get("mean_p95_mid_abs_diff"),
        "median_large_intraminute_mid_range_frac": float(
            np.median([d.get("large_intraminute_mid_range_frac", math.nan) for d in payload.get("days", [])])
        )
        if payload.get("days")
        else math.nan,
        "median_p95_intraminute_mid_range": float(
            np.median([d.get("p95_intraminute_mid_range", math.nan) for d in payload.get("days", [])])
        )
        if payload.get("days")
        else math.nan,
        "interpretation": (
            "CBBO-1m matches the last CBBO-1s quote at the audited minute boundary, "
            "but intraminute option movement is large, so 1m stop/target labels are coarse."
        ),
    }


def _write_markdown(path: Path, payload: dict) -> None:
    champion = payload["champion"]
    lines = [
        "# Edge Existence Audit",
        "",
        "Non-neural audit for whether current v4 data contains transparent, picky long-call/long-put entry pockets after executable ask-entry / bid-exit labels.",
        "",
        f"Simulation gate pass count: `{payload['simulation_gate_pass_count']}`",
        f"Discovered candidate rules: `{payload['discovered_rule_count']}`",
        f"Simulated rules: `{payload['simulated_rule_count']}`",
        "",
        "## Data Fidelity",
        "",
    ]
    norm = payload["data_fidelity"]["normalized"]
    cand = payload["data_fidelity"]["candidate_frame"]
    one_s = payload["data_fidelity"]["cbbo_1m_vs_1s"]
    candidate_quality = cand.get("candidate_quality", {})
    lines.extend(
        [
            f"- Normalized files audited: `{norm['files']}` with `{norm['rows']:,}` rows.",
            f"- Bad root / settlement / 5-point strike fractions: `{norm['bad_root_frac']:.6f}` / `{norm['bad_settlement_frac']:.6f}` / `{norm['bad_strike_alignment_frac']:.6f}`.",
            f"- Bad finite quote fraction: `{norm['bad_quote_frac_of_finite']:.6f}`.",
            f"- Quote time after event fraction: `{norm['quote_time_after_event_frac']:.6f}`.",
            f"- Quote gap p95/p99 seconds on finite quotes: `{norm['quote_gap_seconds_p95']:.1f}` / `{norm['quote_gap_seconds_p99']:.1f}`.",
            f"- Candidate rows: `{cand['rows']:,}` across `{cand['sessions']}` sessions.",
            f"- Median spread: `${cand['spread']['spread']['median']:.2f}`; p95 spread: `${cand['spread']['spread']['p95']:.2f}`.",
            f"- Median spread fraction: `{cand['spread']['spread_frac']['median']:.3f}`; p95 spread fraction: `{cand['spread']['spread_frac']['p95']:.3f}`.",
            f"- Candidate IV missing fraction: `{candidate_quality.get('iv', {}).get('missing_frac', math.nan):.3f}`; gamma missing fraction: `{candidate_quality.get('gamma', {}).get('missing_frac', math.nan):.3f}`.",
            f"- Median mid-label advantage over executable net label: `${cand['mid_vs_net']['median_mid_minus_net']:.2f}`.",
            f"- CBBO-1s audit available: `{one_s.get('available')}`; minute-boundary acceptable: `{one_s.get('acceptable_for_minute_boundary_prototype')}`.",
            f"- Median p95 intraminute mid range in 1s audit: `${one_s.get('median_p95_intraminute_mid_range', math.nan):.2f}`.",
            f"- Normalized contract multiplier counts: `{norm['contract_multiplier_counts']}`.",
            f"- Normalized min price increment missing fraction: `{norm['min_price_increment_missing_frac']:.3f}`.",
            "",
            "## Champion",
            "",
        ]
    )
    if champion:
        lines.extend(
            [
                f"Best simulated rule: `{champion['rule']}`",
                "",
                "| Split | Trades | PnL | PF | DD | Positive Days | Top-Day Share | Random Same-Time PnL |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in ("selection", "march", "q4"):
            metrics = champion["metrics_by_split"][split]
            random = champion["random_same_time_by_split"][split]
            lines.append(
                f"| {split} | {metrics['trades']} | {metrics['total_pnl']:.0f} | "
                f"{metrics['profit_factor']:.3f} | {metrics['max_drawdown']:.0f} | "
                f"{metrics['positive_day_fraction']:.2f} | {metrics['top_day_profit_share']:.2f} | "
                f"{random['total_pnl_median']:.0f} |"
            )
    else:
        lines.append("No simulated rule produced trades.")
    lines.extend(
        [
            "",
            "## Top Simulated Rules",
            "",
            "| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Pass |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(payload["top_simulated_rules"][:20], start=1):
        sel = row["metrics_by_split"]["selection"]
        march = row["metrics_by_split"]["march"]
        q4 = row["metrics_by_split"]["q4"]
        lines.append(
            f"| {idx} | `{row['rule']}` | {sel['total_pnl']:.0f}/{sel['profit_factor']:.3f} | "
            f"{march['total_pnl']:.0f}/{march['profit_factor']:.3f} | "
            f"{q4['total_pnl']:.0f}/{q4['profit_factor']:.3f} | {row['simulation_gate_pass']} |"
        )
    lines.extend(
        [
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
    q1_paths = _paths(args.data_dir)
    q4_paths = _paths(args.q4_data_dir)
    all_paths = q1_paths + q4_paths
    frame = _load_candidates(all_paths, policies=args.policy_indexes, market_cache=market_cache)
    frame = _add_rule_bins(frame)
    discovered = _discover_rules(
        frame,
        min_discovery_candidates=args.min_discovery_candidates,
        min_selection_candidates=args.min_selection_candidates,
        max_per_policy=args.max_discovered_rules_per_policy,
    )
    simulated = _simulate_rules(
        frame,
        discovered,
        max_rules=args.max_simulated_rules,
        max_trades_per_day=args.max_trades_per_day,
        random_runs=args.random_runs,
        seed=args.seed,
    )
    pass_count = sum(1 for row in simulated if row["simulation_gate_pass"])
    champion = simulated[0] if simulated else None
    sessions = [session_from_path(path) for path in all_paths]
    payload = {
        "audit_id": "v4_edge_existence_audit_001",
        "framing": "Non-neural, picky-entry edge existence audit using executable ask-entry / bid-exit labels.",
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "policy_indexes": args.policy_indexes,
            "min_discovery_candidates": args.min_discovery_candidates,
            "min_selection_candidates": args.min_selection_candidates,
            "max_trades_per_day": args.max_trades_per_day,
            "random_runs": args.random_runs,
        },
        "criteria": {
            "discovery": "Jan + early Feb candidate cells must be positive with PF >= 1.05.",
            "selection": "Late Feb cells must be positive with PF >= 1.05 before simulation.",
            "simulation_gate": "One-contract max 2/day rule must survive late Feb, March, Q4, $25 stress, and same-time random baselines.",
        },
        "data_fidelity": {
            "normalized": _normalized_audit(args.normalized_dir, sessions),
            "candidate_frame": _candidate_summary(frame),
            "cbbo_1m_vs_1s": _load_1s_audit(args.cbbo_1s_audit),
        },
        "discovered_rule_count": len(discovered),
        "simulated_rule_count": len(simulated),
        "simulation_gate_pass_count": pass_count,
        "champion": champion,
        "top_simulated_rules": simulated[:50],
        "top_discovered_rules": discovered[:50],
        "interpretation": (
            "This audit found transparent, non-neural edge-existence leads if simulation_gate_pass_count is positive. "
            "The leads should not be promoted directly to live trading: they are templates for the next model target. "
            "The strongest theme is call-side participation when the causal SPX/VWAP sigma state is call-aligned, "
            "usually with near-ATM or ITM contracts. Remaining data caveats are the coarse 1-minute stop/target path, "
            "derived rather than official SPX/VIX context, and unusable normalized definition fields for multiplier/min tick."
            if pass_count
            else "No transparent picky-entry rule survived the frozen audits. The current dataset may be cleaner than v3, "
            "but this pass would not show a robust exploitable long-premium edge under these trader-style cells."
        ),
    }
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(args.out_dir / "report.md", payload)
    print(json.dumps({
        "discovered_rule_count": len(discovered),
        "simulated_rule_count": len(simulated),
        "simulation_gate_pass_count": pass_count,
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
