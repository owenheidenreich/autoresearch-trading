"""Rule-based strategy modules (non-ML).

Each strategy here is a deterministic decision function consuming point-in-time
features + sidecar context and emitting a ``TradeIntent`` (or ``None``). The
package deliberately avoids v2's ML training and promotion infrastructure;
strategies land here when we want to test a stated rule on historical data
without letting a learned model silently decide anything.

See ``v2/docs/pickles_digest.md`` for the extracted rule set that drives
current strategies (Fork A1 tests Row 1).
"""
