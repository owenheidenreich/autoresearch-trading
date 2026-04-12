# v2 Core Contract: Exact-Chain Contract Selection

## Current Truth

Replay selects one exact contract row from the model's `contract_scores` output. There is:

- no direction head (call/put is implicit in which contract wins)
- no coarse strike offset
- no learned risk head

The model scores `NO_TRADE` alongside every executable contract on the current bar. The highest-scoring option wins. Risk parameters (stop, target, hold, exit policy) come from `v2/core/policy.py`.

## Schema

See `v2/core/schema.py` for the exact current dataclasses:

- `TradeIntent`
- `RiskAdjustment`
- `SimulatedTrade`

## Current Runtime Semantics

In replay today:

- `NO_TRADE` wins means skip the bar
- the winning contract determines right (call/put) and strike
- `stop_price`, `take_profit_price`, and `max_hold_bars` come from the fixed policy
- `underlying_price` is used when mapping the intent back to the stored price arrays

## Sidecar Labels

Each day sidecar provides:

- `row_labels`: per-contract forward P&L under the fixed policy
- `bar_best_contract_idx`: index of the best contract on each bar
- `bar_best_pnl`: P&L of the best contract
- `bar_label_trade`: whether the best P&L exceeds the gate threshold
- `bar_labelable`: whether the bar has valid forward labels at all

## Important Limitation

The contract scoring space is large (~38 contracts per bar median). Hard one-hot selection CE across this many classes provides weak gradient. The live reset baseline uses gate BCE plus soft KL selection; more complex side-decomposition ideas are deferred until the baseline is re-established.
