# CEO Dashboard

Last updated: 2026-05-24

## Executive State

| Area | Status | Notes |
|---|---|---|
| Current control | `PAPER_DEFAULT_PROTOCOL101` | No default change authorized. |
| Stage | Stage 2 research_ops scaffold | Governance layer exists around v4. |
| Trading-code mutation | Blocked | No strategy/runtime/model changes in the scaffold. |
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

1. Approve the Stage 2 scaffold as the binding operating law.
2. Decide which blocker gets the first verifier report.
3. Decide whether `.github` issue and pull request templates should enforce the
   artifact taxonomy.
4. Decide whether the first iteration should target execution realism,
   replay/live parity, lifecycle parity, or stale-doc cartography.

<!-- research_ops:update_dashboard:start -->
## Generated Current State

- Stage: `stage_2_research_ops_scaffold`
- Control tag: `v4-protocol101-control-2026-05-24`
- Operational default: `PAPER_DEFAULT_PROTOCOL101`
- Next recommended prompt: `Create the first verifier RFC for Protocol101 execution realism and replay/live fill parity.`
<!-- research_ops:update_dashboard:end -->

## Dashboard Rule

This dashboard is not evidence. It is an executive routing surface. Evidence
lives in the cited v4 docs, audit artifacts, logs, and verifier reports.
