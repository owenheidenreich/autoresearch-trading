"""Aggregate evaluation metrics for v3 replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from v3.core.schema import EpisodeSummary, TradeRecord


@dataclass
class EvalConfig:
    """Primary evaluation contract for v3 RL."""

    max_drawdown_limit: float = 0.25
    min_trades: int = 10
    min_days: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplayMetrics:
    total_days: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0
    turnover_contracts: int = 0
    exposure_time_pct: float = 0.0
    total_return_pct: float = 0.0
    avg_daily_return_pct: float = 0.0
    daily_vol_pct: float = 0.0
    daily_sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_action_confidence: float = 0.0
    action_reward_corr: float = 0.0
    score: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def beats_all_baselines(policy_metrics: ReplayMetrics, baseline_metrics: dict[str, ReplayMetrics]) -> bool:
    """Require the policy score to exceed every configured baseline score."""

    if not baseline_metrics:
        return False
    return all(policy_metrics.score > base.score for base in baseline_metrics.values())


def _compute_sortino(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=0) if len(downside) else 0.0
    if downside_std <= 1e-9:
        return float(np.sign(daily_returns.mean()) * 5.0) if daily_returns.mean() != 0 else 0.0
    return float(daily_returns.mean() / downside_std)


def compute_score(metrics: ReplayMetrics, config: EvalConfig) -> float:
    if metrics.total_days < config.min_days or metrics.total_trades < config.min_trades:
        return -1.0
    if metrics.total_return_pct <= 0:
        return -0.5
    if metrics.max_drawdown_pct > config.max_drawdown_limit:
        return -0.25
    dd_mult = max(0.0, 1.0 - metrics.max_drawdown_pct / max(config.max_drawdown_limit, 1e-6))
    return float(min(metrics.daily_sortino, 5.0) * metrics.total_return_pct * dd_mult)


def compute_metrics(
    episodes: list[EpisodeSummary],
    trades: list[TradeRecord],
    *,
    action_confidences: list[float] | None = None,
    action_rewards: list[float] | None = None,
    config: EvalConfig | None = None,
) -> ReplayMetrics:
    config = config or EvalConfig()
    metrics = ReplayMetrics()
    metrics.total_days = len(episodes)
    metrics.total_trades = len(trades)
    if trades:
        pnls = np.asarray([t.pnl_dollars for t in trades], dtype=np.float64)
        metrics.win_rate = float((pnls > 0).mean())
        metrics.expectancy = float(pnls.mean())
    if episodes:
        returns = np.asarray(
            [
                (ep.ending_equity - ep.starting_equity) / max(ep.starting_equity, 1e-6)
                for ep in episodes
            ],
            dtype=np.float64,
        )
        metrics.total_return_pct = float(
            (episodes[-1].ending_equity - episodes[0].starting_equity) / max(episodes[0].starting_equity, 1e-6)
        )
        metrics.avg_daily_return_pct = float(returns.mean())
        metrics.daily_vol_pct = float(returns.std(ddof=0))
        metrics.daily_sortino = _compute_sortino(returns)
        metrics.max_drawdown_pct = float(max((ep.max_drawdown for ep in episodes), default=0.0))
        metrics.turnover_contracts = int(sum(ep.turnover_contracts for ep in episodes))
        metrics.exposure_time_pct = float(
            sum(ep.exposure_bars for ep in episodes) / max(len(episodes) * 390, 1)
        )
        metrics.avg_action_confidence = float(
            np.mean([ep.avg_action_confidence for ep in episodes]) if episodes else 0.0
        )
    if action_confidences and action_rewards and len(action_confidences) == len(action_rewards) and len(action_confidences) > 1:
        conf = np.asarray(action_confidences, dtype=np.float64)
        rew = np.asarray(action_rewards, dtype=np.float64)
        if conf.std(ddof=0) > 1e-9 and rew.std(ddof=0) > 1e-9:
            metrics.action_reward_corr = float(np.corrcoef(conf, rew)[0, 1])
            if not np.isfinite(metrics.action_reward_corr):
                metrics.action_reward_corr = 0.0
    metrics.score = compute_score(metrics, config)
    metrics.passed = metrics.score > 0
    return metrics
