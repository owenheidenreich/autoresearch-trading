"""Replay scoring: computes performance metrics from simulated trades.

See docs/v2/evaluator.md for the promotion score formula.

Metrics: profit factor (primary), win rate, Sharpe, max drawdown,
trades per day, direction balance. Score formula produces a single scalar
for keep/revert decisions.

v1 origin: training/replay.py (metrics aggregation), training/train.py
(_score_config, score computation)
"""
# TODO: ReplayMetrics dataclass (profit_factor, win_rate, sharpe, max_drawdown,
#   trades_per_day, call_pct, put_pct)
# TODO: compute_metrics(trades: list[SimulatedTrade]) -> ReplayMetrics
# TODO: compute_score(metrics: ReplayMetrics) -> float (exact formula from evaluator.md)
# TODO: Score config fingerprinting (SHA-256, reset on change)
# TODO: Baseline comparison (random, ATM-always, simple-rules) per baselines.md
