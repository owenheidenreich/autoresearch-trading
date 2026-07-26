# Protocol101 Canonical Serial Simulator V2

## Purpose

`protocol101_serial_simulator_v2` defines the account-state replay semantics for future Protocol101 fair-contract folds, selected-candidate replay gates, and paper-readiness evidence. It exists because multiple older replay paths shared most rules but differed in subtle ways around stress, cash, and daily-loss accounting.

Archived March 2026 diagnostic packets remain frozen under their original v1 artifacts. Do not rerun March under v2 to create new evidence. v2 applies forward to backfilled folds and future candidates.

## Version

```text
protocol101_serial_simulator_v2
```

Implementation:

```text
v4/model/protocol101_serial_simulator.py
```

## State Semantics

Daily-loss basis:

```text
raw_realized_net_pnl
```

Cash and affordability basis:

```text
raw_realized_net_pnl
```

Stress application:

```text
metrics_only
```

In plain English:

- The daily-loss guard is driven by realized net PnL as the live system would know it.
- Account cash used for affordability is moved by realized net PnL.
- The stress haircut is a backtest/reporting metric and never drives in-simulation state.
- A trade may be worse in stressed metrics without causing a live-style daily-loss stop.
- A trade may be worse in stressed metrics without reducing live-style buying power.

## Serial Rules

- One account per split/fold.
- One open position per split/session.
- Pending positions are realized at their synthetic lifecycle exit time before considering a new entry at the same or later timestamp.
- New entries are blocked while a same-session position is still open.
- `max_trades_per_session` is enforced after pending exits are realized.
- `max_daily_loss` is enforced from raw realized net PnL after pending exits are realized.
- Affordability checks `entry_ask * contract_multiplier <= current_cash`.
- Current cash is starting cash plus raw realized net PnL, not stressed PnL.
- Stress is recorded on each trade as `stressed_pnl = raw_label_pnl - stress_per_trade`.

## Why V2 Exists

The Round 5 March reconciliation found an ambiguity:

- `simulate_model_policy` used raw labels for the daily-loss guard and used `cash_pnl_adjustment` only for cash/metrics stress.
- The selected-candidate strict replay gate used stressed PnL for realized daily loss and cash.

That was acceptable only as an archived diagnostic artifact because March is now frozen. Going forward, fair-contract folds need one live-reproducible simulator. v2 chooses raw realized net PnL for state because stress haircuts are not live-observable broker/account facts.

## Artifact Requirements

Any future strict replay, fold, model-search, null, or paper-readiness artifact that uses v2 must write:

```text
simulator_version
daily_loss_basis
cash_basis
stress_application
```

Expected values:

```json
{
  "simulator_version": "protocol101_serial_simulator_v2",
  "daily_loss_basis": "raw_realized_net_pnl",
  "cash_basis": "raw_realized_net_pnl",
  "stress_application": "metrics_only"
}
```

## March Migration Rule

March 2026 packets are archived as v1 falsification/diagnostic evidence. They may be used for historical explanation and regression checks against archived hashes, but not rerun under new semantics to mine new conclusions.

Future fold evidence must use v2 unless a later documented simulator version supersedes it.

