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

Exit-time semantics:

```text
synthetic_exit_at_entry_plus_cooldown
```

Cooldown anchor:

```text
entry
```

No new entries after:

```text
15:30 ET
```

The boundary is inclusive: an entry exactly at `15:30:00 ET` is allowed; an entry after that is blocked.

Forced flat before:

```text
15:55 ET
```

Fee model:

```text
none_in_state
```

Account continuity:

```text
cash_compounds_across_sessions_within_split
```

In plain English:

- The daily-loss guard is driven by realized net PnL as the live system would know it.
- Account cash used for affordability is moved by realized net PnL.
- The stress haircut is a backtest/reporting metric and never drives in-simulation state.
- A trade may be worse in stressed metrics without causing a live-style daily-loss stop.
- A trade may be worse in stressed metrics without reducing live-style buying power.
- The current v2 simulator does not debit option premium while a position is open. This is acceptable only because the simulator enforces one open position per split/session. If concurrent positions are ever allowed, premium escrow becomes mandatory.
- `cash_after` on a trade record is projected post-exit cash under the label outcome, not cash immediately after order submission.
- Real fees are different from stress. Real commissions/exchange fees are live-observable and should enter state once the fee model is defined. Synthetic stress remains metrics-only.
- All decision timestamps must be timezone-aware. A timezone-naive timestamp fails closed with `tz_naive_decision_time`.

## Serial Rules

- One account per split/fold.
- One open position per split/session.
- Pending positions are realized at their synthetic lifecycle exit time before considering a new entry at the same or later timestamp.
- New entries are blocked while a same-session position is still open.
- New entries after `15:30 ET` are blocked by the simulator, even if an upstream builder or sampler accidentally emits them.
- Synthetic exit time is capped at `15:55 ET`; a later synthetic hold is not emitted as a live-reproducible exit time.
- `synthetic_exit_time` is emitted as a UTC ISO timestamp, including when the forced-flat cap is applied.
- `max_trades_per_session` is enforced after pending exits are realized.
- `max_daily_loss` is enforced from raw realized net PnL after pending exits are realized.
- Affordability checks `entry_ask * contract_multiplier <= current_cash`.
- Current cash is starting cash plus raw realized net PnL, not stressed PnL.
- Stress is recorded on each trade as `stressed_pnl = raw_label_pnl - stress_per_trade`.
- v2 sets `synthetic_exit_time = decision_time + cooldown_minutes`. This is equivalent to the current frozen policies because their cooldown equals their max hold. Future lifecycle grids where cooldown differs from max hold require a v2.1 simulator where `SerialCandidate` carries an explicit label-derived exit time.
- If a selected-candidate stream provides `max_hold_minutes`, v2 fails closed when `cooldown_minutes != max_hold_minutes`.

## Why V2 Exists

The Round 5 March reconciliation found an ambiguity:

- `simulate_model_policy` used raw labels for the daily-loss guard and used `cash_pnl_adjustment` only for cash/metrics stress.
- The selected-candidate strict replay gate used stressed PnL for realized daily loss and cash.

The March gate-only, null, uplift, and sampler diagnostics were built on `simulate_model_policy`, so those falsification conclusions already used raw daily-loss semantics. Only the selected-candidate strict replay artifacts used the stressed-basis v1 replay gate. Treat those candidate replays as archived diagnostic references with residual incomparability against raw-basis nulls.

Going forward, fair-contract folds need one live-reproducible simulator. v2 chooses raw realized net PnL for state because stress haircuts are not live-observable broker/account facts.

## Artifact Requirements

Any future strict replay, fold, model-search, null, or paper-readiness artifact that uses v2 must write:

```text
simulator_version
daily_loss_basis
cash_basis
stress_application
exit_time_semantics
cooldown_anchor
no_new_entries_after
forced_flat_before
fee_model
account_continuity
stress_per_trade_dollars
simulator_config_hash
candidate_stream_hash
candidate_payload_hash
```

Expected values:

```json
{
  "simulator_version": "protocol101_serial_simulator_v2",
  "daily_loss_basis": "raw_realized_net_pnl",
  "cash_basis": "raw_realized_net_pnl",
  "stress_application": "metrics_only",
  "exit_time_semantics": "synthetic_exit_at_entry_plus_cooldown",
  "cooldown_anchor": "entry",
  "no_new_entries_after": "15:30",
  "forced_flat_before": "15:55",
  "fee_model": "none_in_state",
  "account_continuity": "cash_compounds_across_sessions_within_split"
}
```

Hash purposes:

- `simulator_config_hash`: hash of simulator configuration and state semantics only. It answers "same replay rules?"
- `candidate_stream_hash`: hash of ordered candidate identity keys only: split, session, decision time, contract id, right, and offset. It answers "same decision identities?" and is intentionally vendor-independent.
- `candidate_payload_hash`: hash of ordered candidate identity plus `entry_ask`, `raw_label_pnl`, `cooldown_minutes`, and `max_hold_minutes`. It answers "same decision data?" `max_hold_minutes` is included because the fail-closed guard consumes it; two streams that differ only in max hold can produce different accept/reject behavior under the same config.

If `require_cooldown_equals_max_hold_when_present` is disabled, the effective `simulator_version` is suffixed with:

```text
_cooldown_hold_guard_disabled
```

This prevents a relaxed guard from masquerading as plain v2.

## March Migration Rule

March 2026 packets are archived as falsification/diagnostic evidence. They may be used for historical explanation and regression checks against archived hashes, but not rerun under new semantics to mine new conclusions.

A v1-compat path may exist only for bit-identical hash regression against archived March artifacts. It must be flagged `non_evidence` and must never be used for new configurations.

Future fold evidence must use v2 unless a later documented simulator version supersedes it.

## Changelog

- Round 6-8 implementation-review packets briefly emitted `simulator_semantics_hash` with shifting semantics. Because no fold evidence artifacts existed yet, the field is retired from v2 outputs before first fold use. Use `simulator_config_hash` for config-only replay-rule comparisons.

## V2.1 Prerequisite

Before lifecycle-grid experiments where cooldown differs from max hold, create a successor simulator version where each candidate carries an explicit label-derived exit time. Until then, v2 is valid only for policies where cooldown and max hold are intentionally equal.

Also verify that the label pipeline itself caps late-entry lifecycle labels at `15:55 ET`, especially for policy 2. A simulator cap on emitted exit time is not enough if `labels_net_pnl` was computed from a hold path that live could not take.
