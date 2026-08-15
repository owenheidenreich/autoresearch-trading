"""Balanced Profitability Score V2 for Protocol101 hill-climb gating."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


SCHEMA_VERSION = "BalancedProfitabilityScoreV2"


@dataclass(frozen=True)
class BalancedProfitabilityScoreConfigV2:
    min_trade_count: int = 100
    target_trade_count: int = 500
    max_churn_trade_count: int = 2_500
    max_drawdown_fraction: float = 0.25
    min_net_pnl_under_required_slippage: float = 0.0
    max_top_1_day_pnl_fraction: float = 0.25
    max_top_5_day_pnl_fraction: float = 0.50
    max_top_10_trade_pnl_fraction: float = 0.35
    target_robust_net_pnl_per_day: float = 250.0
    target_return_on_premium: float = 0.30
    target_drawdown_efficiency: float = 4.0
    target_win_rate: float = 0.70
    target_expectancy_per_trade: float = 75.0
    target_regime_stability: float = 0.75
    slippage_fragility_penalty_weight: float = 10.0
    outlier_concentration_penalty_weight: float = 10.0
    excessive_churn_penalty_weight: float = 5.0
    tail_loss_penalty_weight: float = 5.0
    live_historical_drift_penalty_weight: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WEIGHTS = {
    "robust_net_pnl_per_day_ratio": 30.0,
    "return_on_premium_ratio": 20.0,
    "drawdown_efficiency_ratio": 15.0,
    "win_rate_ratio": 10.0,
    "expectancy_per_trade_ratio": 10.0,
    "trade_frequency_sufficiency_score": 10.0,
    "regime_stability_score": 5.0,
}


def score_balanced_profitability_v2(
    metrics: dict[str, Any],
    *,
    config: BalancedProfitabilityScoreConfigV2 = BalancedProfitabilityScoreConfigV2(),
) -> dict[str, Any]:
    hard_gate_failures = _hard_gate_failures(metrics, config)
    components = _components(metrics, config)
    penalties = _penalties(metrics, config, components)
    raw_score = sum(WEIGHTS[name] * value for name, value in components.items()) - sum(penalties.values())
    score = max(0.0, round(raw_score, 6))
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "eligible_for_hill_climb_ranking" if not hard_gate_failures else "blocked_by_hard_gates",
        "score": score if not hard_gate_failures else 0.0,
        "raw_score_before_hard_gate": score,
        "hard_gate_failures": hard_gate_failures,
        "components": components,
        "weights": WEIGHTS,
        "penalties": penalties,
        "config": config.to_dict(),
        "notes": [
            "Trade count is capped as a sufficiency metric, not rewarded endlessly.",
            "ROC and ending equity are intentionally not double-counted.",
            "Execution evidence and decision evidence must be reviewed separately before promotion claims.",
        ],
    }


def _hard_gate_failures(metrics: dict[str, Any], config: BalancedProfitabilityScoreConfigV2) -> list[str]:
    failures: list[str] = []
    required_true = (
        "no_leakage",
        "decision_parity_passed",
        "protected_holdout_untouched",
        "all_flat_by_close",
    )
    for key in required_true:
        if metrics.get(key) is not True:
            failures.append(key)
    if float(metrics.get("net_pnl_under_required_slippage", -math.inf)) <= config.min_net_pnl_under_required_slippage:
        failures.append("profitable_under_required_slippage")
    if int(metrics.get("trade_count", 0)) < config.min_trade_count:
        failures.append("min_trade_count")
    if _drawdown_fraction(metrics) > config.max_drawdown_fraction:
        failures.append("max_drawdown_fraction")
    if int(metrics.get("unaffordable_trade_count", 0)) != 0:
        failures.append("unaffordable_trades")
    if int(metrics.get("overlapping_headline_trade_count", 0)) != 0:
        failures.append("overlapping_headline_trades")
    if float(metrics.get("top_1_day_pnl_fraction", 0.0)) > config.max_top_1_day_pnl_fraction:
        failures.append("top_1_day_profit_concentration")
    if float(metrics.get("top_5_day_pnl_fraction", 0.0)) > config.max_top_5_day_pnl_fraction:
        failures.append("top_5_day_profit_concentration")
    if float(metrics.get("top_10_trade_pnl_fraction", 0.0)) > config.max_top_10_trade_pnl_fraction:
        failures.append("top_10_trade_profit_concentration")
    if int(metrics.get("protected_holdout_tuning_count", 0)) != 0:
        failures.append("protected_holdout_tuning")
    return sorted(set(failures))


def _components(metrics: dict[str, Any], config: BalancedProfitabilityScoreConfigV2) -> dict[str, float]:
    robust_net_pnl_per_day = _robust_net_pnl_per_day(metrics)
    return_on_premium = float(metrics.get("return_on_premium", 0.0))
    drawdown_efficiency = float(metrics.get("drawdown_efficiency", _drawdown_efficiency(metrics)))
    win_rate = float(metrics.get("win_rate", 0.0))
    expectancy = float(metrics.get("expectancy_per_trade", _expectancy_per_trade(metrics)))
    trade_count = int(metrics.get("trade_count", 0))
    regime_stability = float(metrics.get("regime_stability_score", 0.0))
    return {
        "robust_net_pnl_per_day_ratio": _ratio(robust_net_pnl_per_day, config.target_robust_net_pnl_per_day),
        "return_on_premium_ratio": _ratio(return_on_premium, config.target_return_on_premium),
        "drawdown_efficiency_ratio": _ratio(drawdown_efficiency, config.target_drawdown_efficiency),
        "win_rate_ratio": _ratio(win_rate, config.target_win_rate),
        "expectancy_per_trade_ratio": _ratio(expectancy, config.target_expectancy_per_trade),
        "trade_frequency_sufficiency_score": _trade_frequency_score(trade_count, config),
        "regime_stability_score": _ratio(regime_stability, config.target_regime_stability),
    }


def _penalties(
    metrics: dict[str, Any],
    config: BalancedProfitabilityScoreConfigV2,
    components: dict[str, float],
) -> dict[str, float]:
    churn_penalty = 1.0 - components["trade_frequency_sufficiency_score"] if int(metrics.get("trade_count", 0)) > config.max_churn_trade_count else 0.0
    return {
        "slippage_fragility": config.slippage_fragility_penalty_weight * _bounded(metrics.get("slippage_fragility", 0.0)),
        "outlier_concentration": config.outlier_concentration_penalty_weight * _bounded(metrics.get("outlier_concentration", 0.0)),
        "excessive_churn": config.excessive_churn_penalty_weight * churn_penalty,
        "tail_loss": config.tail_loss_penalty_weight * _bounded(metrics.get("tail_loss_severity", 0.0)),
        "live_historical_drift": config.live_historical_drift_penalty_weight * _bounded(metrics.get("live_historical_drift", 0.0)),
    }


def _robust_net_pnl_per_day(metrics: dict[str, Any]) -> float:
    if "robust_net_pnl_per_day" in metrics:
        return float(metrics["robust_net_pnl_per_day"])
    if "median_daily_pnl" in metrics:
        return float(metrics["median_daily_pnl"])
    days = max(1.0, float(metrics.get("trading_days", 1.0)))
    return float(metrics.get("net_pnl", 0.0)) / days


def _drawdown_efficiency(metrics: dict[str, Any]) -> float:
    drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    if drawdown <= 0.0:
        return 0.0
    return max(0.0, float(metrics.get("net_pnl", 0.0))) / drawdown


def _drawdown_fraction(metrics: dict[str, Any]) -> float:
    if "max_drawdown_fraction" in metrics:
        return abs(float(metrics["max_drawdown_fraction"]))
    start_cash = float(metrics.get("starting_cash", 0.0))
    if start_cash <= 0.0:
        return math.inf
    return abs(float(metrics.get("max_drawdown", math.inf))) / start_cash


def _expectancy_per_trade(metrics: dict[str, Any]) -> float:
    trades = int(metrics.get("trade_count", 0))
    if trades <= 0:
        return 0.0
    return float(metrics.get("net_pnl", 0.0)) / trades


def _trade_frequency_score(trade_count: int, config: BalancedProfitabilityScoreConfigV2) -> float:
    if trade_count <= 0:
        return 0.0
    sufficiency = min(1.0, trade_count / max(1.0, float(config.target_trade_count)))
    if trade_count <= config.max_churn_trade_count:
        return sufficiency
    excess = trade_count - config.max_churn_trade_count
    decay = max(0.0, 1.0 - excess / max(1.0, float(config.max_churn_trade_count)))
    return min(sufficiency, decay)


def _ratio(value: float, target: float) -> float:
    if target <= 0.0:
        return 0.0
    return _bounded(value / target)


def _bounded(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))
