# v2 Evaluator Contract

This document defines how replay works right now.

The authority is the live code in:

- `v2/replay.py`
- `v2/core/simulator.py`
- `v2/core/metrics.py`

## What Replay Evaluates

Replay evaluates a model against historical option price tensors stored in `v2/data.pt`.

The current flow is:

1. load model or artifact
2. run model inference on the selected mask
3. convert outputs to `TradeIntent`
4. map each intent to one of the stored option price arrays
5. simulate the trade
6. aggregate trades into account-level metrics
7. compute the promotion score

## Model Loading Rules

Default replay now follows this order:

1. promoted artifact bundles only
2. artifact dataset fingerprint must match the current dataset fingerprint
3. if no compatible promoted artifact exists, fall back to the raw checkpoint on disk

That fallback exists so replay still works before the first repaired-era promoted model exists.

## Evaluation Masks

Replay supports:

- `val_mask`
- `promote_mask`
- `shadow_mask`
- temporary walk-forward test masks injected by `v2/core/walkforward.py`

The canonical local evaluation mask is `promote_mask`.

## TradeIntent Construction

`model_to_intent()` currently does this:

- `gate = sigmoid(outputs["gate"])`
- no-trade if gate is below `policy.gate_threshold`
- direction from `softmax(outputs["direction"])`
- strike offset from `softmax(outputs["strike"])`
- strike = rounded ATM plus the selected coarse offset
- stop / target / hold decoded from the risk head and policy ranges

Current important nuance:

- strike choice is represented in replay
- current training does not directly supervise strike selection
- current models remain effectively ATM-biased
- `label_max_hold` may be `390`, but replay decodes model-emitted hold from `policy.max_hold_range`, currently `10-250`

## Replay Price Universe

The evaluator does not load arbitrary contracts from raw option bars at replay time.
It uses the stored coarse price arrays in `v2/data.pt`:

- `atm_*`
- `otm5_*`
- `otm10_*`
- `otm15_*`
- `otm20_*`
- `otm25_*`
- `otm30_*`

These arrays are now keyed to the dynamic nearest ATM per bar.

## Entry And Exit Timing

Simulation rules from `v2/core/simulator.py`:

- decision happens on bar `N`
- entry fill uses bar `N+1`
- if bar `N+1` has no valid option price, the trade is skipped
- options below `$0.50` are skipped
- stop loss is checked before take profit
- trailing stop tiers apply only when `exit_policy == "TRAILING"`
- max hold and EOD flatten are enforced

## Cost Model

Net P&L includes:

- adaptive round-trip spread
- minimum-tick floor on spread fractions
- `$1.30` round-trip commission

Spread depends on:

- minutes to close
- VIX regime
- whether the contract is OTM

## Entry Gating During Replay

`replay_validation()` also enforces:

- no new entries before `policy.no_trade_before_bar`
- no new entries at or after `policy.no_trade_after_bar`
- cooldown after stop losses
- maximum one open position
- no new entries after the daily loss cap is hit
- no new entries if cumulative equity is already wiped out

Current default policy values:

- no-trade before bar `30`
- no-trade at or after bar `270`
- cooldown `5`
- daily loss cap `5%`

## Score

The current score in `v2/core/metrics.py` is:

```python
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

Hard gates:

- fewer than 30 trades -> `-1.0`
- fewer than 15 traded days -> `-0.5`
- direction balance below `0.15` -> `-0.3`
- account drawdown above `20%` -> `-0.2`

If the hard gates pass:

- `dd_mult = 1.0` at drawdown `<= 8%`
- `dd_mult` then decays linearly to `0.0` at `20%`

## Baselines

Replay computes and compares the model against four baselines:

1. Random
2. ATM-Always
3. Simple-Rules
4. ATM-Trailing

Baseline results are cached in `v2/.baseline_cache.json` and keyed to:

- dataset fingerprint
- mask
- policy fingerprint
- max-days override

## Current Limits Of The Evaluator

- It evaluates only the coarse stored ATM / OTM ladders, not arbitrary raw strikes.
- The model architecture still does not learn strike selection in a strong supervised way.
- Live execution parity is not implemented yet.
