"""Dynamic candidate generation from option chain data.

v1 used a fixed 6-class direction head (call/put x ATM/OTM5/OTM10).
v2 generates candidates dynamically from the visible option chain,
allowing the model to select from real available strikes.

Candidate universe: ATM +/- 30 points in 5-point increments, both calls
and puts. Filtered by liquidity, spread width, and minimum price.
See docs/v2/evaluator.md for the exact candidate spec.

v1 origin: training/trading_rules.py (select_strike, ACTION_* constants)
"""
# TODO: generate_candidates(spot_price, chain_snapshot) -> list[CandidateContract]
# TODO: CandidateContract dataclass (strike, right, bid, ask, mid, delta, gamma, theta)
# TODO: Filter by liquidity, spread width, minimum price
