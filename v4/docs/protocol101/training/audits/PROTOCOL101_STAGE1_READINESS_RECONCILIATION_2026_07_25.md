# Protocol101 Stage-1 Readiness Reconciliation

> **SUPERSEDED 2026-08-05.** Dated 2026-07-25; Stage-1 has since closed. Current status is [`STATUS.md`](../../../../../STATUS.md).


Prepared: 2026-07-25

This document is the current status map. It distinguishes evidence that is
complete from work that merely has an old artifact or an optimistic label.

## Authoritative Status

| Item | Status | Meaning |
|---|---|---|
| G1-G9 objective | COMPLETE | Owner-approved 2026-07-07; G8 wording clarified without changing the payoff target. |
| G4 v2 and holdout revision | COMPLETE | Signed 2026-07-19: pooled Calmar >= 1.0 and every fold stays at or above $5,000. |
| Five-fold governed data scaffold | COMPLETE | 301 accepted registry sessions; 271 unique fold-eligible sessions after the 30 protected-holdout sessions are excluded; five chronological expanding folds and one-session embargo. |
| Global all-feature synchronization | NOT PASSED / NOT REQUIRED | Raw feeds and quarantined feature families are not declared identical. |
| Scoped Stage-1 synchronization | COMPLETE | Owner-signed 2026-07-25; exactly 17 features are eligible for initial offline alpha. |
| Trader charter | COMPLETE | Owner-signed 2026-07-25. |
| Canonical Stage-1 training design | COMPLETE | Owner-signed 2026-07-25; exact selection/noise rules are binding. |
| Legacy masked-v2 runner | RETIRED FOR NEW TRAINING | It exposes the full market/option arrays and the obsolete 25% G4 rule. It is historical evidence only. |
| Exact-contract Stage-1 input adapter | IMPLEMENTED | Builds only the authorized 17 features and enforces the feature/quarantine contract. |
| Exact-contract readiness preflight | IMPLEMENTED | Verifies corpus/folds, hashes, guards, noise inputs, signatures, and a feature-construction smoke. |
| Serial simulator accounting | REPAIRED + TESTED | v4 carries cash across sessions and reserves the fee during affordability; stale v2/v3 reference packets are inadmissible. |
| Exact-contract null + G3 heuristic | COMPLETE | Rebuilt on simulator v4. G3 fixed baseline is policy 5 at +$4,592 pooled; it is not itself gate-eligible. |
| Exact-contract HGB runner | COMPLETE + FROZEN | Canonical runner implemented; disposable H3 smoke and independent validation passed with final code hashes. |
| Deterministic G1-G8 aggregator | COMPLETE | Enforces the preregistered seed/fold gates and fails closed on stale or missing artifacts. |
| Exact-contract hill-climb execution | READY FOR SEPARATE OWNER-APPROVED H0 | No real H0 training has run yet. |
| Candidate-specific transfer | FUTURE PER-CANDIDATE GATE | Required after a candidate is frozen and before paper validation. |
| Paper readiness | NOT READY | Requires candidate hardening and no-order shadow evidence. |

## What The Synchronization Pass Means

The project has enough paired evidence to train on a restricted feature set
without waiting for more perfect recorder days. It does not mean the historical
and IBKR feeds match tick for tick. The training model is protected by:

- the exact 17-feature allowlist;
- measured divergence-noise injection;
- boundary-stable intersection guards;
- pessimistic fills and strict serial replay;
- candidate-specific no-order shadow transfer before paper validation.

## Prior Campaign Guidance

The April campaign is a method and hypothesis guide, not a curve to inherit.
Its useful, methodology-correct findings were:

- shallow tree models under chronological tests found a modest opening-structure
  entry edge;
- the best surviving entry scorer improved when a weak VWAP-reclaim premise was
  removed;
- a wide first-15-minute opening condition strengthened the signal;
- entry and exit learning must remain separate;
- learned lifecycle models can materially improve a genuine entry edge;
- overlap, narrow samples, cached paths, and pre-parity results can manufacture
  impressive but untrustworthy curves.

Stage-1 therefore starts with bounded HGB, chronological folds, payoff targets,
strict serial replay, and one narrow hypothesis batch at a time. It does not
revive any pre-parity model or copy its reported PnL.

## Owner Documents

All three current owner documents are signed. The G4 revision and Stage-1
objective/gates remain in force and do not need to be signed again.

## Accounting Repair

The readiness work discovered that serial simulator v2 deferred each session's
last pending trade until the end of the fold. That made the next session appear
to start with fresh buying power. A second edge allowed a contract whose
premium consumed all available cash even though the $3 fee was already present
in realized PnL.

Simulator v4 fixes both defects. The exact null and heuristic packets were
rebuilt after the repair. Any current candidate packet must identify
`protocol101_serial_simulator_v4_account_continuity_fee_reserve`.

## Launch Order

1. Exact-contract readiness preflight. **Complete.**
2. Canonical 17-feature HGB executor. **Complete.**
3. Tiny `plumbing_smoke_only` batch. **Complete; not edge evidence.**
4. Independent verification and runner freeze. **Complete.**
5. Preregister and run H0 manually after separate owner authorization.
6. Verify H0 before starting the next hypothesis.
7. Continue one preregistered hypothesis at a time.

No old masked-v2 result decides whether the new 17-feature contract has edge.
