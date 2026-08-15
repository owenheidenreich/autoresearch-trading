"""Environment-aware diagnostics for SPXW 0DTE action labels.

These helpers do not try to prove an edge. They organize the existing pilot
data into causal environment descriptions so we can see which patterns survive
train/validation/test splits before making any more data purchases.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model.action_pilot import ActionDecision
from v4.model.supervised_pilot import Trade, metrics_for_trades


_NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EnvironmentBinner:
    """Train-fitted bins for continuous environment variables."""

    vix_edges: tuple[float, float]
    range_edges: tuple[float, float]

    @classmethod
    def fit(cls, decisions: Sequence[ActionDecision]) -> "EnvironmentBinner":
        market = np.vstack([d.market_last for d in decisions]).astype(float)
        vix = market[:, 1]
        session_range = market[:, 4]
        vix_edges = tuple(np.nanquantile(vix[np.isfinite(vix)], [1 / 3, 2 / 3]).tolist())
        range_edges = tuple(
            np.nanquantile(session_range[np.isfinite(session_range)], [1 / 3, 2 / 3]).tolist()
        )
        return cls(vix_edges=(float(vix_edges[0]), float(vix_edges[1])), range_edges=(float(range_edges[0]), float(range_edges[1])))

    def bucket(self, value: float, edges: tuple[float, float]) -> str:
        if not np.isfinite(value):
            return "unknown"
        if value <= edges[0]:
            return "low"
        if value <= edges[1]:
            return "mid"
        return "high"

    def vix_bucket(self, value: float) -> str:
        return self.bucket(value, self.vix_edges)

    def range_bucket(self, value: float) -> str:
        return self.bucket(value, self.range_edges)

    def to_dict(self) -> dict:
        return {"vix_edges": list(self.vix_edges), "range_edges": list(self.range_edges)}


def time_bucket(decision_time: datetime) -> str:
    """Refined time bucket matching the user's observed structure."""
    local = decision_time.astimezone(_NY)
    minutes = local.hour * 60 + local.minute
    if minutes < 10 * 60:
        return "first_30"
    if minutes < 11 * 60 + 30:
        return "post_open_morning"
    if minutes < 13 * 60 + 30:
        return "midday"
    return "late_afternoon"


def sign_bucket(value: float, *, eps: float = 0.0) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "flat"


def omar_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= -0.5:
        return "deep_negative"
    if value < 0:
        return "negative"
    if value < 0.5:
        return "positive"
    return "deep_positive"


def row_for_action(
    decision: ActionDecision,
    *,
    side: str,
    binner: EnvironmentBinner,
) -> dict:
    action = 1 if side == "C" else 2
    market = decision.market_last.astype(float)
    spx_close = float(market[0])
    vix_proxy = float(market[1])
    spx_vwap = float(market[2])
    omar = float(market[3])
    session_range = float(market[4])
    momentum_5m = float(market[5])
    momentum_15m = float(market[6])
    above_vwap = spx_close > spx_vwap if np.isfinite(spx_close) and np.isfinite(spx_vwap) else False
    trend_aligned = (side == "C" and momentum_15m > 0) or (side == "P" and momentum_15m < 0)
    mean_reversion_side = (side == "C" and not above_vwap) or (side == "P" and above_vwap)
    return {
        "session": decision.session,
        "decision_time": decision.decision_time.isoformat(),
        "time_bucket": time_bucket(decision.decision_time),
        "side": side,
        "pnl": float(decision.labels[action]),
        "offset": float(decision.offsets[action]),
        "spx_close": spx_close,
        "vix_proxy": vix_proxy,
        "spx_vwap": spx_vwap,
        "omar": omar,
        "session_range": session_range,
        "momentum_5m": momentum_5m,
        "momentum_15m": momentum_15m,
        "above_vwap": bool(above_vwap),
        "below_vwap": bool(not above_vwap),
        "omar_sign": sign_bucket(omar),
        "omar_bucket": omar_bucket(omar),
        "momentum_5m_sign": sign_bucket(momentum_5m, eps=1.0),
        "momentum_15m_sign": sign_bucket(momentum_15m, eps=2.0),
        "vix_bucket": binner.vix_bucket(vix_proxy),
        "range_bucket": binner.range_bucket(session_range),
        "trend_aligned": bool(trend_aligned),
        "mean_reversion_side": bool(mean_reversion_side),
    }


def environment_table(
    decisions_by_split: dict[str, Sequence[ActionDecision]],
    *,
    binner: EnvironmentBinner,
) -> pd.DataFrame:
    rows: list[dict] = []
    for split, decisions in decisions_by_split.items():
        for decision in decisions:
            for side in ("C", "P"):
                row = row_for_action(decision, side=side, binner=binner)
                row["split"] = split
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_group(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Summarize PnL by environment group."""
    rows = []
    for key, group in frame.groupby(list(group_cols), dropna=False):
        pnl = group["pnl"].astype(float).to_numpy()
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_loss = abs(float(losses.sum()))
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "trades": int(len(pnl)),
                "total_pnl": float(pnl.sum()),
                "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
                "median_pnl": float(np.median(pnl)) if len(pnl) else 0.0,
                "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
                "profit_factor": float(wins.sum() / gross_loss) if gross_loss else float("inf"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["total_pnl", "trades"], ascending=[False, False])


@dataclass(frozen=True)
class EnvironmentRule:
    """One transparent environment-gated side rule."""

    side: str
    conditions: tuple[tuple[str, object], ...]

    @property
    def name(self) -> str:
        parts = [self.side]
        parts.extend(f"{k}={v}" for k, v in self.conditions)
        return "|".join(parts)

    def matches(self, row: dict) -> bool:
        if row["side"] != self.side:
            return False
        return all(row.get(key) == value for key, value in self.conditions)


def generate_rules(frame: pd.DataFrame) -> list[EnvironmentRule]:
    """Generate low-dimensional rules from observed validation environments."""
    condition_sets: set[tuple[tuple[str, object], ...]] = set()
    columns = [
        "time_bucket",
        "above_vwap",
        "omar_sign",
        "momentum_15m_sign",
        "vix_bucket",
        "range_bucket",
        "trend_aligned",
        "mean_reversion_side",
    ]
    for col in columns:
        for value in sorted(frame[col].dropna().unique().tolist()):
            condition_sets.add(((col, value),))

    for _, row in frame[["time_bucket", "above_vwap", "momentum_15m_sign"]].drop_duplicates().iterrows():
        condition_sets.add(
            (
                ("time_bucket", row["time_bucket"]),
                ("above_vwap", bool(row["above_vwap"])),
            )
        )
        condition_sets.add(
            (
                ("time_bucket", row["time_bucket"]),
                ("momentum_15m_sign", row["momentum_15m_sign"]),
            )
        )
        condition_sets.add(
            (
                ("time_bucket", row["time_bucket"]),
                ("above_vwap", bool(row["above_vwap"])),
                ("momentum_15m_sign", row["momentum_15m_sign"]),
            )
        )

    # Include the user's recurring hypothesis as a discoverable rule candidate,
    # selected only if validation supports it.
    condition_sets.add((("time_bucket", "post_open_morning"),))
    condition_sets.add((("time_bucket", "late_afternoon"),))

    return [
        EnvironmentRule(side=side, conditions=tuple(sorted(conditions)))
        for side in ("C", "P")
        for conditions in condition_sets
    ]


def simulate_rule(
    frame: pd.DataFrame,
    rule: EnvironmentRule,
    *,
    cooldown_minutes: int,
) -> list[Trade]:
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    for record in frame.sort_values(["decision_time", "side"]).to_dict("records"):
        if not rule.matches(record):
            continue
        decision_time = pd.Timestamp(record["decision_time"]).to_pydatetime()
        next_time = next_time_by_session.get(record["session"])
        if next_time is not None and decision_time < next_time:
            continue
        trades.append(
            Trade(
                session=record["session"],
                decision_time=record["decision_time"],
                pnl=float(record["pnl"]),
                score=None,
                right=str(record["side"]),
                offset=float(record["offset"]),
                strategy=rule.name,
            )
        )
        next_time_by_session[record["session"]] = decision_time + timedelta(
            minutes=cooldown_minutes
        )
    return trades


def rule_search(
    frame: pd.DataFrame,
    *,
    cooldown_minutes: int,
    min_validation_trades: int = 20,
) -> list[dict]:
    """Select rules on validation only and score them on test."""
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    rules = generate_rules(validation)
    rows = []
    for rule in rules:
        validation_metrics = metrics_for_trades(
            simulate_rule(validation, rule, cooldown_minutes=cooldown_minutes)
        )
        if validation_metrics["trades"] < min_validation_trades:
            continue
        if validation_metrics["total_pnl"] <= 0 or validation_metrics["profit_factor"] <= 1.0:
            continue
        test_metrics = metrics_for_trades(simulate_rule(test, rule, cooldown_minutes=cooldown_minutes))
        rows.append(
            {
                "rule": rule.name,
                "side": rule.side,
                "conditions": [{"column": k, "value": v} for k, v in rule.conditions],
                "validation": validation_metrics,
                "test": test_metrics,
            }
        )
    rows.sort(
        key=lambda r: (
            r["validation"]["total_pnl"],
            r["validation"]["profit_factor"],
            r["validation"]["trades"],
        ),
        reverse=True,
    )
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
