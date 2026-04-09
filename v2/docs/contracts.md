# v2 Core Contract: TradeIntent

`TradeIntent` is the runtime trade object used by replay and intended future live execution.

## Current Truth

- Replay constructs `TradeIntent` objects from model outputs in `v2/replay.py`.
- `v2/core/simulator.py` consumes those intents.
- Live execution is not implemented yet, but the schema already exists in `v2/core/schema.py`.
- Training does not currently store or supervise full serialized `TradeIntent` labels. It supervises component tensors such as directional P&L and risk labels.

## Schema

See `v2/core/schema.py` for the exact current dataclasses:

- `TradeIntent`
- `RiskAdjustment`
- `SimulatedTrade`

Important current fields on `TradeIntent`:

- `trade`
- `expiry`
- `strike`
- `right`
- `qty`
- `entry_ref_price`
- `order_style`
- `tif`
- `stop_price`
- `take_profit_price`
- `max_hold_bars`
- `exit_policy`
- `confidence`
- `reason_codes`
- `bar_index`
- `timestamp`
- `intent_id`
- `underlying_price`
- `policy_version`

## Current Runtime Semantics

In replay today:

- `trade=False` means skip the bar
- `right` is chosen from the model direction head
- `strike` is derived from rounded ATM plus the selected coarse offset
- `stop_price`, `take_profit_price`, and `max_hold_bars` come from the risk head plus policy ranges
- `underlying_price` is used when mapping the intent back to the stored price arrays

## Important Limitation

The schema supports dynamic strike choice, but the current training objective does not strongly supervise it. So the contract is richer than the current learned behavior.
