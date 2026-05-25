# Stale Docs Index

Created: 2026-05-24

Purpose: identify documents whose claims can conflict with the current
Protocol101 operational truth. This is an index only. Stage 9 banner headers
should be added after human review; no old documentation is deleted or
rewritten here.

## Binding Truth Sources

When documents disagree, use these sources before acting:

- `research_ops/CURRENT_STATE.yaml`
- `research_ops/CEO_DASHBOARD.md`
- `research_ops/bootstrap/V4_BASELINE_INVENTORY.md`
- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md`
- `research_ops/iterations/ITER-001_quote_age_truth/05_decision_memo.md`
- `research_ops/iterations/ITER-002_decision_reconstruction/05_decision_memo.md`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/05_decision_memo.md`

Current operational truth, in one sentence: v4 Protocol101 is the guarded IBKR
paper control, not real-money live trading; paper-submit infrastructure exists,
but quote-age truth, decision reconstruction, replay/live feature parity, and
fill realism remain unresolved blockers.

## Editing Policy

- `can edit now`: safe candidate for a short banner after this index is reviewed.
- `separate docs-refresh required`: current truth docs or source-code docstrings
  need a scoped documentation refresh rather than a broad stale banner.
- `must remain archived`: local/generated/older packet should stay as historical
  evidence unless explicitly reintroduced.

## Stale Or Conflicting Documents

| Path | Stale or conflicting claim | Current truth | Source of current truth | Status | Edit/archive policy |
|---|---|---|---|---|---|
| `README.md` | Lines 7, 12, 19, and 29 describe IBKR paper trading as eventual/deferred and `v2/` as the active canonical system. Lines 35-42 provide v2 quickstart commands. | v4 Protocol101 is the current guarded paper control. The README remains directionally true for real-money live trading, but stale for paper-ops and canonical-system status. | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:10-24`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:45-47`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:790-792`, `research_ops/bootstrap/V4_BASELINE_INVENTORY.md:382-385` | partially stale | can edit now with a banner and a pointer to current truth; do not rewrite history in this pass |
| `v4/README.md` | Lines 3 and 41 say v4 is clean-slate and has no shared imports with v2 or v3; line 13 frames v4 as Phase 0 only, with no model training or live trading. | Stage 1 baseline records the actual operational source workspace importing `v2.core.market_structure` from `v4/model/hypothesis_protocol.py:33`. v4 also advanced beyond Phase 0 into Protocol101 guarded paper control. | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:47`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:793`, `research_ops/bootstrap/V4_BASELINE_INVENTORY.md:387-391`; source workspace evidence: `/Users/gduby/Documents/autoresearch-trading/v4/model/hypothesis_protocol.py:33` | partially stale | can edit now with a banner; note that the cited import exists in the frozen source-workspace evidence even though that file is not present in this clean transition worktree |
| `/Users/gduby/Documents/autoresearch-trading/v4/promotion/PROTOCOL_101_PROMOTION_READINESS_PACKET.md` | Lines 6 and 38 say Protocol101 is a research promotion candidate and not broker-connected paper approved. | Guarded Protocol160/158 paper-submit infrastructure, launchd scheduling, and runtime enablement exist. This does not prove tradable edge, but it supersedes the older "not paper approved" operational-status claim. | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:45`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:790-791`, `research_ops/bootstrap/V4_BASELINE_INVENTORY.md:393-396` | superseded | must remain archived; this file is source-workspace/local evidence and is not currently tracked in the clean transition worktree. If reintroduced, add a historical banner first |
| `v4/promotion/PROTOCOL_101_TUESDAY_PAPER_SESSION_RUNBOOK.md` | Lines 3 and 57 say the default morning job starts in `no-order-shadow` and must later be switched to `paper` only after the gate passes. | Current ops truth says the scheduled Protocol101 session is a guarded `paper-submit` runtime. However, Stage 8 still blocks paper-submit trust because quote-age truth, reconstruction, parity, and fill evidence are insufficient. | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:18-21`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:90`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:790-792`, `research_ops/bootstrap/V4_BASELINE_INVENTORY.md:36-39` | partially stale | can edit now with a banner; keep the runbook as historical gate context |
| `v4/promotion/PROTOCOL_101_TUESDAY_LIVE_SHADOW_CHECKLIST.md` | Lines 3 and 33-38 say no paper orders should be placed until no-order live-data parity passes and explicit approval is given. | Guarded paper-submit infrastructure now exists, but the checklist remains useful as older pre-paper control history. Current research_ops decisions still block trust in paper-submit evidence until observability and parity are proven. | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:24`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:790-792`, `research_ops/iterations/ITER-001_quote_age_truth/05_decision_memo.md`, `research_ops/iterations/ITER-002_decision_reconstruction/05_decision_memo.md`, `research_ops/iterations/ITER-003_replay_live_feature_parity/05_decision_memo.md` | historical context | can edit now with a banner; do not change checklist procedures without a separate ops decision |
| `v4/sim/simulator.py` docstrings | Lines 1-16, 67-75, 96-104, and 115-123 describe future fill models and concrete simulators. | Current replay/live research does not have a calibrated fill model. `NullSimulator` rejects fills, replay remains deterministic ask-entry/bid-exit style, and latest inspected paper evidence had no broker endpoint/fill rows. | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:53`, `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md:783-794`, `research_ops/bootstrap/V4_BASELINE_INVENTORY.md:171`, `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md:43-50` | historical context | separate docs-refresh required; this is source-code documentation, not a standalone doc |
| `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` quote-age line | Line 54 says the live bridge sets `quote_age_ms=0`; the same document later notes quote freshness is weak and unresolved. | The correct current claim is narrower: Protocol158/160 code paths have quote-age handling, but Iteration 1 showed existing logs do not prove true quote-age measurement. Quote-age truth remains unknown and blocks paper-submit trust. | `research_ops/bootstrap/V4_BASELINE_INVENTORY.md:398-401`, `research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_truth_report.md`, `research_ops/iterations/ITER-001_quote_age_truth/05_decision_memo.md` | partially stale | separate docs-refresh required; do not banner the source-of-truth doc broadly, update the specific quote-age statement in a reviewed truth-doc refresh |

## Watchlist, Not Yet Bannered

These documents are not automatically stale, but they should be read carefully
against current research_ops decisions.

| Path | Watch item | Current handling |
|---|---|---|
| `v4/docs/PROMOTION_SEQUENCE.md` | Promotion evidence sections can be misread as evidence already captured in current logs. Stage 8 diagnostics show current logs are insufficient for quote-age truth, decision reconstruction, and replay/live feature parity. | Do not banner yet. Clarify only if a future reviewer treats the promotion requirements as satisfied evidence. |
| `v4/docs/PHASE_0_5_VENDOR_VERIFICATION.md` | Describes older vendor-verification phase work. | Treat as historical data/vendor context unless a current RFC explicitly uses it. |
| `docs/TUESDAY_NO_ORDER_EVIDENCE_COLLECTION_PLAN.md` | Mentions no-order evidence collection and bounded paper-submit work. | Keep as historical context; current default and blocked actions live in research_ops. |

## Banner Candidates After Review

Recommended first banner pass:

1. `README.md`
2. `v4/README.md`
3. `v4/promotion/PROTOCOL_101_TUESDAY_PAPER_SESSION_RUNBOOK.md`
4. `v4/promotion/PROTOCOL_101_TUESDAY_LIVE_SHADOW_CHECKLIST.md`

Do not banner `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md`; update its
specific quote-age wording in a scoped docs-refresh iteration. Do not edit
`v4/sim/simulator.py` as part of stale-doc cleanup unless the change is approved
as a source-code documentation patch.

## Human Review Questions

- Should root `README.md` be updated to identify v4 Protocol101 as the current
  paper-control system while preserving the "not real-money live" warning?
- Should source-workspace-only promotion packets be imported as archived evidence,
  or kept out of the clean transition branch?
- Should v4 clean-slate language be replaced with "mostly v4-owned, with known
  legacy v2 dependency in frozen source-workspace evidence"?
- Should the quote-age statement in the source-of-truth be amended now that
  Iteration 1 produced an `unknown` verdict rather than a direct placeholder pass/fail?
