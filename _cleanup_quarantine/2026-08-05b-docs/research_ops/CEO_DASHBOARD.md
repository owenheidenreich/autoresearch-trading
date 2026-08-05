# CEO Dashboard

> **STALE 2026-08-05.** Last updated 2026-05-24, over two months ago, and never tracked in git. Current status is [`STATUS.md`](../STATUS.md).


Last updated: 2026-05-24

## Executive State

| Area | Status | Notes |
|---|---|---|
| Current control | `PAPER_DEFAULT_PROTOCOL101` | No default change authorized. |
| Stage | Stage 0 transition preparation | Governance layer being created around v4. |
| Trading-code mutation | Blocked | No strategy/runtime/model changes in Stage 0 scaffold. |
| Model hill climbing | Blocked | Execution, parity, and untouched-data gates remain unresolved. |
| Broker/paper-submit work | Blocked by default | Requires explicit user authorization and Section 4 controls. |
| Paid data work | Blocked by default | Requires explicit data request and approval. |
| Protected holdout scoring | Blocked | Future untouched block pending collection. |

## Current Control Truth

| Item | Current truth |
|---|---|
| Instrument | SPXW 0DTE PM-settled long calls/puts/no trade. |
| Operational default | `PAPER_DEFAULT_PROTOCOL101`. |
| Current state | Guarded IBKR paper runtime exists; recent inspected paper logs show no submitted broker orders/fills. |
| Replacement status | Research challengers remain research-only. |
| Main unresolved blocker | Execution/fill realism and replay/live parity. |

## Active Blockers

| Priority | Blocker | Why it matters | Owning future artifact |
|---:|---|---|---|
| 1 | Fill/cancel/slippage evidence absent | Replay edge may not be executable. | Verifier report / execution realism packet |
| 2 | Replay/live feature parity not proven | Live candidate scores may not match training semantics. | Verifier report / parity harness RFC |
| 3 | Lifecycle live sequence mismatch | Live exit model may not reproduce replay/training behavior. | Verifier report |
| 4 | Exposed validation splits heavily reused | Current metrics are research evidence, not final proof. | Decision memo / validation governance |
| 5 | Stale-doc contradictions | Agents may follow obsolete operational truth. | Cartography report / stale-doc queue |

## Next Recommended Decisions

1. Approve or revise the AI-agent artifact taxonomy.
2. Decide whether to add `.github` issue and pull request templates in Stage 0
   or defer them to Stage 1.
3. Decide whether `research_ops/ASSUMPTION_REGISTRY.md` becomes the binding
   cross-agent assumption tracker.
4. Decide which blocker gets the first verifier report.

## Dashboard Rule

This dashboard is not evidence. It is an executive routing surface. Evidence
lives in the cited v4 docs, audit artifacts, logs, and verifier reports.

