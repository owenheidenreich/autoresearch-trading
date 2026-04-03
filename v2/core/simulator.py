"""Trade simulation engine: executes TradeIntents against historical data.

See docs/v2/evaluator.md for the complete simulation rules.

Given a sequence of TradeIntents and historical price/option data,
simulates execution with realistic costs, stops, targets, and time limits.
Single source of truth for trade P&L computation.

v1 origin: training/replay.py (trade simulation loop, compute_adaptive_spread_bps,
position management, exit logic)
"""
# TODO: SimulatedTrade dataclass (entry, exit, P&L, metadata)
# TODO: TradeSimulator class
# TODO: simulate_day(model, features, option_prices) -> list[SimulatedTrade]
# TODO: simulate_dataset(model, data) -> list[SimulatedTrade]
# TODO: compute_adaptive_spread_bps(minutes_remaining, vix_regime, is_otm) -> float
# TODO: Determinism: fixed seeds, no stochastic fills
