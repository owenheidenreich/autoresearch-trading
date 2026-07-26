# Protocol101 Current Status

Last reconciled: 2026-07-25

This is the single source of truth for the current phase. Older sealed-day,
masked-v2, feature-recovery, and five-family training plans are provenance,
not current launch authority.

## Plain-English Status

1. **Scoped synchronization passed.** We have enough paired IBKR and
   Databento/ThetaData evidence to begin offline research on exactly 17
   parity-stable model inputs.
2. **Full-feed synchronization did not pass and is not claimed.** Direct
   per-slot price paths, internal IV expansion, VIX changes, and raw quote
   microstructure remain excluded from initial model alpha.
3. **The old masked-v2 hill climb does not answer the new question.** Its
   failures remain historical evidence; they neither prove nor disprove edge
   in the new 17-feature contract.
4. **The training objective and G4 are already approved.** G4 v2 requires
   pooled PnL/max-drawdown >= 1.0 and at least $5,000 equity in every fold.
5. **A material serial-replay bug was repaired.** Simulator v4 now carries
   cash across sessions and reserves the $3 fee during affordability.
6. **Exact null and heuristic references are complete under simulator v4.**
   The fixed G3 heuristic is policy 5 at +$4,592 pooled. It is a positive
   benchmark, not an acceptable strategy: only 3/5 folds are profitable and
   weak-fold equity violates G4.
7. **The Stage-1 machinery is ready, but H0 has not started.** All three owner
   documents are signed; the canonical runner smoke and independent validation
   passed; the frozen H0 path remains separately owner-locked.

## Exact Initial Alpha Contract

Contract ID: `protocol101-scoped-canonical-stage1-v1`

Allowed:

- 12 certified non-VIX context features;
- three near-ATM D-family composites;
- internally computed delta and gamma.

Excluded from initial alpha:

- Family C direct per-slot option-price paths;
- internal IV and IV expansion/compression;
- VIX 5m/15m changes;
- bid, ask, spread, size, quote age, volume, OI, and vendor Greeks.

Excluded raw fields remain available for tradability guards, fills, labels,
PnL, and audit.

## Evidence Boundary

The scoped pass uses 864 eligible minutes from July 15/17/20/21/22 for the
newly admitted D/E families, plus the prior three-full-day certification for
the 12 non-VIX context features.

This permits offline hill climbing without waiting for perfect full-day
recorder captures. It does not permit a paper-readiness claim. Any selected
candidate must later pass candidate-specific no-order decision-shadow transfer
before paper validation.

## Completed

| Item | Status |
|---|---|
| G1-G9 Stage-1 objective | Approved 2026-07-07 |
| G4 v2 + holdout revision | Owner-signed 2026-07-19 |
| 301-session accepted registry | Complete |
| 271-session five-fold expanding CV scope | Complete; 30 protected holdout sessions excluded |
| Scoped synchronization evidence | Passed; owner-signed 2026-07-25 |
| Trader charter | Owner-signed 2026-07-25 |
| Canonical Stage-1 training design | Owner-signed 2026-07-25 |
| Exact 17-feature adapter | Implemented and tested |
| Boundary-stable guard + divergence evidence | Complete |
| Serial simulator v4 accounting repair | Implemented and tested |
| Exact random nulls, seven policies | Complete under simulator v4 |
| Exact fixed G3 heuristic | Complete under simulator v4 |
| Fail-closed readiness preflight | Passed; blockers empty |
| Exact bounded-HGB runner | Implemented; disposable H3 smoke passed |
| Independent smoke validation + runner freeze | Passed; code hashes frozen |
| Deterministic G1-G8 aggregator | Implemented and tested |
| Final machinery readiness | `ready_for_separate_owner_approved_H0` |
| Goal-sized gated training system | Active; RUN -> GATE -> AUDIT per hypothesis |

## Open Gate

The only immediate gate is a separate explicit owner authorization to
preregister and execute the real offline H0 batch. The completed plumbing smoke
is not edge evidence, and it did not authorize research training.

The next goal ID is `S1-H0-RUN`. It must stop after the 105 frozen unit
artifacts are complete and before G1-G8 aggregation.

## Hill-Climb Sequence

1. Preregister and run the 105-unit H0 batch, then stop.
2. Mechanically aggregate H0 G1-G8 in a separate goal, then stop.
3. Independently audit and freeze the H0 verdict, then stop.
4. Repeat the same RUN -> GATE -> AUDIT chain for H1, H2, and H3.
5. Use the April campaign only as a methodology and hypothesis guide:
   shallow bounded trees, chronological testing, opening-structure priors,
   entry/exit separation, and strict protection against overlap or cached-path
   optimism.

If real entry signal clears G2/G5 but fixed exits cause the failure, route to a
separately approved learned-exit Stage-2. If all four hypotheses show no real
signal, do not loosen gates; review a separately parity-tested opening-feature
amendment, a same-vendor data plane, or another strategy class.

## Path To Paper

Offline candidate -> G9 confirmation seed -> immutable final fit -> one-shot
protected holdout -> research-candidate freeze -> candidate-specific
recorder/shadow transfer -> live no-order shadow -> separate owner approval for
guarded IBKR paper trading.

Current highest allowed claim:

> Scoped synchronization and Stage-1 machinery readiness passed; Protocol101
> is ready for separately owner-approved offline H0 hill climbing.

Protocol101 is not paper-ready and no real-money path is authorized.

## Authoritative References

- `PROTOCOL101_STAGE1_READINESS_RECONCILIATION_2026_07_25.md`
- `PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `PROTOCOL101_TRADER_CHARTER.md`
- `PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `PROTOCOL101_STAGE1_TO_LIVE_EXECUTION_PLAN.md`
- `PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md`
- `PROTOCOL101_SERIAL_SIMULATOR_ACCOUNTING_REPAIR_2026_07_25.md`
- `PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`
- `PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md`
