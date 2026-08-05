# Protocol 081 Promotion-Readiness Packet

> **LEGACY (2026-08-05).** Describes the legacy Protocol101 paper spine, not the current Path-D path. Current status is [`STATUS.md`](../../STATUS.md).


## Candidate

- Name: Protocol 081 Q4-start residual sequence-lifecycle challenger
- Frozen baseline: Protocol 054
- Frozen entry protocol: Protocol 051
- Freeze manifest: `v4/promotion/PROTOCOL_081_FREEZE.json`
- Deployment artifact manifest: `v4/promotion/PROTOCOL_081_DEPLOYMENT_ARTIFACTS.json`
- Current status: frozen research challenger, not paper/live approved
- Data cost for promotion-readiness replay: `$0`

## Evidence Now Present

- Freeze hashes pass for the Protocol 081 report, selected trades, sequence dataset report, artifact reproduction summary, and deployment artifact manifest.
- Persisted Protocol 081 model artifacts exist for all 4 chronological walk-forward folds and 10 seeds per fold: `40` model/scaler/threshold/manifest bundles, `160` files total.
- Artifact reproduction passed across all `40` saved Protocol 081 artifacts and all `42,170` frozen selected rows with `0` mismatches.
- Paper replay consumes `42,170` frozen selected rows and matches all selected exit steps.
- Replayed bid/ask path PnL matches selected candidate PnL with p99 absolute difference `0`.
- Order-state accounting samples `5,000` records through `candidate_seen -> ... -> exit_filled`.
- One-contract behavior is enforced in replay.
- PM-settled `SPXW` contract IDs are enforced.
- No mid fills are used: entry is ask, exit is bid.
- Conservative extra slippage of `$0.25` per side remains positive in all scored splits.
- Available CBBO-1s lifecycle audit covered `4,160` trades with `0` sign flips; planned lifecycle exits matched 1m path PnL exactly where 1s coverage exists.
- Offline no-order shadow rehearsal passed on a March slice: `1,250` observations, `0` failed rows, official VIX coverage `1.000`, and hold/exit/stop/forced-flat actions represented.
- Fallback ablation rejected removing Protocol 054 fallback because no-fallback damaged Q3/Q4 and lowered PF in Q2/Q3/Q4/Q1/March.
- The live inference helper now matches research fallback precedence: hard stop/target/time-flat first, residual override before the Protocol 054 exit step, and ordinary Protocol 054 fallback exit at the fallback step.
- Protocol 088 no-order router smoke now loads the Protocol 051/054 stack plus Protocol 081 override, emits `1,250` shadow rows, and passes parity with `0` failures. A market-hours IBKR run connected on `127.0.0.1:4002` but blocked because SPX/VIX live market data was not subscribed. A delayed-data plumbing check emitted `24` SPXW rows and passed schema/no-order parity, but remained blocked as delayed data.
- Protocol 089 no-order shadow-paper ledger now consumes the Protocol 088 router JSONL and reconstructs one-contract lifecycle accounting with `0` broker order fields, `0` parity failures, `50` closed trades, and bid/ask-only PnL accounting. It found `19` offline rehearsal trades with rows after a terminal exit/stop action and max observed concurrency of `8`, which is acceptable for selected-trade path rehearsal but must be rejected in a real live stream.
- Protocol 090 strict shadow lifecycle replay now transforms the same offline router JSONL into live-like serial semantics: post-terminal rows are removed, overlapping candidates are skipped, and strict `--require-all-closed --require-terminal-final --enforce-global-one-position` behavior passes. The strict stream kept `13` serial one-contract trades, removed `137` post-terminal rows, skipped `37` overlapping candidate trades, and passed with max concurrency `1`.
- Protocol 091 broader strict shadow lifecycle replay extended the same no-order live-semantics check to `417` March 2026 offline router path rehearsals. The strict serial stream kept `87` one-contract trades, skipped `330` overlaps, removed `323` post-terminal rows, passed strict parity with `0` failed rows, and closed all trades with max concurrency `1`. This broader check used a `60s` offline context-age allowance because historical VIX bars are one-minute bars; the eventual live gate remains strict.

## Current Stress Result

| Scenario | Split | Median Seed PnL | Positive Seeds | Median PF |
|---|---:|---:|---:|---:|
| extra `$0.25` each side | Q2 2025 | `151,835` | `10/10` | `1.741` |
| extra `$0.25` each side | Q3 2025 | `171,365` | `10/10` | `2.999` |
| extra `$0.25` each side | Q4 2025 | `287,480` | `10/10` | `3.633` |
| extra `$0.25` each side | Q1 2026 | `343,515` | `10/10` | `3.270` |
| extra `$0.25` each side | March 2026 | `185,215` | `10/10` | `4.568` |

## Blockers Before Paper Trading

- Live shadow-feed parity: requires non-empty fresh live SPXW rows. The current market-hours attempt produced `0` live rows because IBKR returned unsubscribed/delayed-only SPX/VIX market data.
- Live market-data entitlements: enable live Cboe SPX/VIX index quotes and OPRA/SPX options before rerunning `ibkr-live-capture` without delayed data.
- IBKR account equity: Client Portal currently blocks market-data activation because the account does not meet IBKR's minimum equity requirement for market-data subscriptions.
- Entry quote timestamp retention: the live path must retain the actual entry quote timestamp, not only executable entry price.
- Quote gap/stale quote checks: historical replay lacks usable quote-gap coverage, so live capture must compute and enforce stale-quote rejection.
- High-resolution audit coverage: local 1s coverage supports the path labels where available, but Q3 2025 still has no local 1s coverage.

## Decision

Protocol 081 may remain frozen as the best validated sequence-lifecycle challenger.

Do not approve broker-connected paper trading yet. While waiting for IBKR market-data eligibility, use already-collected data for offline promotion-readiness and research only. Once live market data is enabled, rerun no-order `ibkr-live-capture` for the full modular stack and then run the strict Protocol 090 lifecycle gate before any broker-connected paper trading.
