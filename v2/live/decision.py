"""Model inference -> TradeIntent.

Takes model forward() output, generates candidates from current chain,
scores candidates, and constructs a TradeIntent (or no-trade).

Key difference from v1: emits TradeIntent (not DecisionIntent), uses
dynamic candidate scoring (not fixed 6-class), and applies the same
policy as replay (no policy divergence).

v1 origin: training/live/decision.py (ModelDecisionEngine, InferenceResult)
"""
# TODO: DecisionEngine class
# TODO: model_output_to_intent(model_output, candidates, bar_context) -> TradeIntent
# TODO: apply_filters(intent, bar_index, last_exit) -> TradeIntent or no-trade
# TODO: Same policy as simulator (evaluator.md rules)
