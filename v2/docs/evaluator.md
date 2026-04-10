# v2 Evaluator Contract

Replay authority lives in:

- `v2/replay.py`
- `v2/core/simulator.py`
- `v2/core/metrics.py`

## What Replay Evaluates

1. load model or artifact
2. load the selected mask from `v2/data.pt`
3. load the matching day sidecars
4. score the executable contracts visible on each decision bar
5. emit an exact-contract `TradeIntent`
6. simulate that contract’s exact intraday price path
7. aggregate trades into account-level metrics

## Exact-Contract Rules

- The selected contract is one exact `(expiry, strike, right)`.
- Replay uses that contract’s own sidecar price series through exit.
- No ATM/OTM ladder remapping is allowed in the main path.
- If a contract is not executable for that bar, it does not appear in the snapshot the model sees.
- If a contract is visible on the bar but lacks an honest forward path, replay refuses the trade instead of substituting a nearby contract.

## Entry And Exit Timing

- decision happens on bar `N`
- entry fill uses bar `N+1`
- if bar `N+1` has no valid option price, the trade is skipped
- stop loss is checked before take profit
- trailing stop tiers apply when `exit_policy == "TRAILING"`
- max hold and EOD flatten are enforced

## Cost Model

Net P&L includes:

- adaptive round-trip spread
- minimum-tick spread floor
- `$1.30` round-trip commission

## Baselines

Replay compares against four baselines on the same exact-chain harness:

1. Random
2. ATM-Always
3. Simple-Rules
4. ATM-Trailing

## Score

```python
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

Hard gates:

- fewer than 30 trades -> `-1.0`
- fewer than 15 traded days -> `-0.5`
- direction balance below `0.15` -> `-0.3`
- account drawdown above `20%` -> `-0.2`
