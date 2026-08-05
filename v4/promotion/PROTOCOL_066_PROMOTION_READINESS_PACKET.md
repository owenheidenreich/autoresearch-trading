# Protocol 066 Promotion-Readiness Packet

> **LEGACY (2026-08-05).** Describes the legacy Protocol101 paper spine, not the current Path-D path. Current status is [`STATUS.md`](../../STATUS.md).


## Candidate

- Name: Protocol 066 residual recovery-penalty sequence lifecycle challenger
- Frozen baseline: Protocol 054
- Frozen entry protocol: Protocol 051
- Freeze manifest: `v4/promotion/PROTOCOL_066_FREEZE.json`
- Protocol 051/054 stack artifact manifest: `v4/promotion/PROTOCOL_054_051_STACK_ARTIFACTS.json`
- Deployment artifact manifest: `v4/promotion/PROTOCOL_066_DEPLOYMENT_ARTIFACTS.json`
- Current status: frozen research challenger, not paper/live approved
- Data cost for promotion-readiness replay: `$0`

## Evidence Now Present

- Freeze hashes pass for the Protocol 066 report, selected trades, override diagnostic, and 1s path audit.
- Paper replay consumes `32,490` frozen selected rows and matches all selected exit steps.
- Replayed bid/ask path PnL matches selected candidate PnL with p99 absolute difference `0`.
- Order-state accounting samples `5,000` records through `candidate_seen -> ... -> exit_filled`.
- One-contract behavior is enforced in replay.
- PM-settled `SPXW` contract IDs are enforced.
- No mid fills are used: entry is ask, exit is bid.
- Conservative extra slippage of `$0.25` per side remains positive in all scored splits.
- Persisted Protocol 066 model artifacts now exist for all 3 walk-forward folds and 10 seeds per fold: `30` model/scaler/threshold/manifest bundles, `120` files total.
- A no-order shadow-feed parity harness now exists and is intentionally blocked until a live JSONL capture is supplied.
- Offline shadow rehearsal passed on a March slice: `1,250` no-order observations, `0` failed rows, official VIX coverage `1.000`, and hold/exit/stop/forced-flat actions represented.
- Artifact reproduction passed across all `30` saved Protocol 066 artifacts and all `32,490` frozen selected rows with `0` mismatches.
- Fallback ablation rejected removing Protocol 054: no-fallback stayed profitable, but damaged Q3, Q4, and March and lowered profit factor.
- Protocol 051 entry plus Protocol 054 fallback artifacts now exist as a frozen-config rerun: `40` fold/seed bundles and `200` files total. This is a reproducible saved-stack candidate, not byte-identical to the original Protocol 054 metrics.

## Current Stress Result

| Scenario | Split | Median Seed PnL | Positive Seeds | Median PF |
|---|---:|---:|---:|---:|
| extra `$0.25` each side | Q3 2025 | `87,492.5` | `10/10` | `1.689` |
| extra `$0.25` each side | Q4 2025 | `211,160` | `10/10` | `2.721` |
| extra `$0.25` each side | Q1 2026 | `315,115` | `10/10` | `2.614` |
| extra `$0.25` each side | March 2026 | `168,990` | `10/10` | `3.557` |

## Blockers Before Paper Trading

- Live shadow-feed parity: compare live decision-time features, SPX/VIX context, option NBBO, quote timestamps, stale-quote status, contract identity, and derived Greeks against the research feature path without placing orders.
- Live inference ensemble/router: persisted fold/seed artifacts exist, but the paper/live path still needs an explicit approved inference selection policy.
- Protocol 054 fallback engine: the frozen-config stack is persisted, but it still needs a live router and parity audit with Protocol 066's residual override.
- Entry quote timestamp retention: Protocol 060 retained the executable entry ask but not the original entry quote timestamp. The live path must retain this.
- Quote gap fields: historical replay currently lacks usable quote-gap coverage, so the live path must compute and enforce stale-quote rejection.
- High-resolution audit coverage: current 1s coverage supports the path labels where available, but Q3 has no local 1s coverage.

## Decision

Protocol 066 may remain frozen as the best validated sequence-lifecycle challenger.

Do not approve broker-connected paper trading yet. The next approved workstream is to make the full modular inference stack explicit: entry protocol, Protocol 054 fallback, and Protocol 066 residual override. Then capture a no-order live shadow feed before any order placement.
