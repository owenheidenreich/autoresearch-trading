"""Oracle labeler: computes training labels from historical option data.

See docs/v2/labeling.md for the full specification.

At each decision bar, searches the candidate universe x risk parameter grid
to find the best executable trade under the evaluator's rules. Labels are
derived from what actually would have worked, not from proxies.

v1 origin: training/prepare.py (compute_v18_labels, compute_prediction_labels,
compute_dynamic_pnl). v1 used proxy labels (MFE/MAE entry gate, ATR risk targets).
v2 replaces these with oracle trade outcomes.
"""
# TODO: compute_oracle_labels(features, prices, options, evaluator) -> OracleLabels
# TODO: OracleLabels dataclass (trade, direction, strike_offset, stop_pct, target_pct, pnl)
# TODO: Tier 1 (fast): ATM call + put, fixed risk. 2 sims/bar.
# TODO: Tier 2 (medium): 6 candidates x 3 configs. 18 sims/bar.
# TODO: Tier 3 (full): 26 candidates x 150 configs. 3,900 sims/bar.
# TODO: Quality checks (no-trade rate, direction balance, avg hold, win rate)
