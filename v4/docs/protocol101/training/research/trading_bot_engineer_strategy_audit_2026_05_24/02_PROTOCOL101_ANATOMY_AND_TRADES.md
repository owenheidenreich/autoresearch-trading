# Protocol101 Anatomy And Trade Behavior

Protocol101 is best described as a conservative event-selection strategy over a curated candidate stream.

It is not a full autonomous trading policy. It does not learn a fresh unified entry/hold/exit lifecycle. It selects among candidate entries that already carry frozen downstream exit metadata.

## Encoded Trader Beliefs

Protocol101 encodes these beliefs:

- 0DTE SPXW long calls/puts are the instrument universe.
- Execution is ask-entry and bid-exit.
- One contract, one account, one open position is the operational scope.
- Waiting is the default action.
- The best candidate must beat the wait action by a validation-selected threshold.
- The useful windows are `post_open_morning` and `late_afternoon`.
- Candidate quality should include surface edge, spread, premium, Greeks, SPX context, and short causal event history.
- Recent event context matters: prior candidate count, prior max/mean edge, gamma, theta burden, spread, and call/put edge imbalance.
- Exits are inherited from frozen lifecycle/candidate-exit artifacts, not learned by Protocol101 itself.

## Entry Mechanics

The live entry path in `v4/live/protocol101_entry.py` builds a candidate set from the A+/surface-edge stream. It keeps allowed time buckets, applies a minimum edge threshold, adds causal history features, and feeds an event-set policy with one wait logit plus candidate logits.

The model enters when:

`best_candidate_logit - wait_logit >= threshold`

This matters because Protocol101 is not asking "is this option profitable in isolation?" It is asking "is this current event worth spending the only live slot?"

## Exit Mechanics

The selected candidate carries frozen exit metadata. In replay, once the bot enters, it skips later entries until the frozen exit time. In the seed-1 paper replay:

| Exit reason | Trades | PnL | Win rate | Median duration |
|---|---:|---:|---:|---:|
| `sequence_residual_override` | 691 | $242,950 | 84.1% | 7 min |
| `protocol054_fallback` | 297 | $23,020 | 57.9% | 21 min |
| `target` | 29 | $67,550 | 100.0% | 12 min |
| `hard_stop` | 11 | -$14,470 | 0.0% | 16 min |

This is a key weak point: Protocol101's entry policy is learned, but exit ownership is inherited.

## Seed-1 Paper Replay Snapshot

Source: `v4/audit/autoresearch/v4_aplus_hypothesis_113_protocol101_trade_charts/trades.csv`

| Metric | Value |
|---|---:|
| Trades | 1,028 |
| Start cash | $10,000 |
| Ending equity | $329,050 |
| Total PnL | $319,050 |
| Max closed-trade drawdown | -$2,790 |
| Worst day | -$2,080 |
| Worst intratrade MAE | -$1,840 |
| Median duration | 10 min |
| 90th percentile duration | 24.3 min |
| Max duration | 25 min |

## Side Profile

| Side | Trades | PnL | Win rate | Avg PnL | Median premium |
|---|---:|---:|---:|---:|---:|
| Calls | 670 | $193,140 | 74.8% | $288 | $2,890 |
| Puts | 358 | $125,910 | 78.5% | $352 | $2,440 |

Protocol101 is call-heavy by count, but puts have higher average PnL in this seed-1 paper replay.

## Time And Split Profile

| Block | Trades | PnL | Win rate | Median premium |
|---|---:|---:|---:|---:|
| Q4 2024 external | 211 | $58,280 | 69.2% | $2,540 |
| Q3 2025 | 274 | $68,740 | 85.0% | $2,600 |
| Q4 2025 | 294 | $97,020 | 76.2% | $2,905 |
| Q1 2026 | 249 | $95,010 | 71.9% | $2,890 |

The paper replay has no first-30-minute or midday entries. It is mostly post-open morning with a smaller late-afternoon component.

## Premium And Moneyness Implication

Protocol101 is premium-rich and near-ATM/slightly ITM:

- Median premium in the paper replay: about `$2,750`.
- Calls usually have negative offset, median around `-25`.
- Puts usually have positive offset, median around `+20`.
- Protocol248 showed Protocol101 was almost entirely ITM on Q1/Q3/Q4/recent diagnostics.

This is not an OTM lottery-ticket strategy. It is closer to a disciplined high-premium directional capture/scalp strategy.

## MFE Capture

Median MFE capture in the seed-1 replay was high:

- Overall median MFE capture: about `0.81`.
- `sequence_residual_override` median capture: about `0.97`.
- `protocol054_fallback` median capture: about `0.38`.
- `hard_stop` rows had positive MFE but ended as losses.

Interpretation: Protocol101 often captures much of the move on its main winning exit path, but fallback and hard-stop cases deserve a separate runner/giveback/loss audit.

## Best And Worst Days

Worst days in the seed-1 paper replay:

| Day | Trades | PnL | Win rate |
|---|---:|---:|---:|
| 2024-10-31 | 4 | -$2,080 | 0.0% |
| 2026-03-04 | 5 | -$1,980 | 40.0% |
| 2025-12-11 | 2 | -$1,540 | 50.0% |
| 2025-11-26 | 2 | -$1,420 | 0.0% |
| 2024-11-22 | 5 | -$1,300 | 40.0% |

Best days:

| Day | Trades | PnL | Win rate |
|---|---:|---:|---:|
| 2026-03-03 | 6 | $9,270 | 100.0% |
| 2025-11-21 | 11 | $7,400 | 90.9% |
| 2025-10-29 | 6 | $5,850 | 100.0% |
| 2025-11-14 | 9 | $5,780 | 88.9% |
| 2024-11-21 | 4 | $5,710 | 100.0% |

The paper replay is not top-trade concentrated. The next useful concentration audit should focus on regimes and day types, not only top individual trades.

