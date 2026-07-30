# Protocol101 D1 Negative-Control And Incremental-Edge Amendment

Status: **OWNER-AUTHORIZED AND BINDING**

Effective date: 2026-07-28

Authorization source: the owner-authorized Codex Goal that required the
official D1 replacement and model-free reaggregation of the frozen Stage-1
campaign.

This amendment supersedes the D1 selection law in the signed Stage-1 regimen
repair amendment. Historical D1 artifacts remain unchanged and continue to
describe the rule that was in force when they were created. They are no longer
selection authority.

## Why D1 Changed

P5 can make substantial historical profit even when entries and contracts are
selected without model features. Therefore, positive absolute PnL from a
shuffled model does not prove that the training pipeline manufactured alpha.

The relevant negative-control question is:

> Does a strongly shuffled model add statistically credible fee-adjusted
> profit beyond a feature-independent random selector playing the same P5
> exposure?

The relevant real-model question is:

> Does the real model add statistically credible fee-adjusted profit beyond
> the same kind of matched-random P5 exposure?

## D1 V2 Negative-Control Law

The negative-control model must use a global within-fit-role target
permutation that destroys temporal target alignment and associations with
contract, right, ladder slot, moneyness, and premium. It must be compared with
a feature-independent random selector.

The comparison must be paired by governed fold and session and use simulator
v5 fee-adjusted PnL. Absolute PnL is diagnostic only. In particular:

```text
shuffled_model_absolute_PnL <= 0
```

is not a gate.

The shuffled-model control passes D1 V2 only when:

1. Every shuffled seed has an acceptable exposure-matched random control.
2. The shuffled-model increment over that control does not have both:
   - a multiplicity-adjusted one-sided `p <= 0.05`; and
   - a 95% paired lower confidence bound strictly greater than zero.
3. Every input, model, control, session, and replay identity is complete and
   hash-valid.

If exposure matching is incomplete, the result is
`INSUFFICIENT_MATCHED_CONTROL`. It blocks selection but is not evidence of a
positive artifact.

## Strict Exposure Matching

Matching is outcome-blind. PnL may not be used to choose a random-control draw.
The comparison must report and pass all of these dimensions:

| Dimension | Frozen rule |
|---|---|
| Opportunity set | Same governed fold/session risk set and exact intent budget; random control timing must be feature-independent |
| Entry-intent count | Exact |
| Executed trade count | Maximum 5% relative drift |
| Calls versus puts | Total-variation distance at most 5% |
| Moneyness | ATM/near/wing total-variation distance at most 6% |
| Executed premium | Maximum 10% relative drift in mean premium at risk |
| Holding time | Maximum 10% relative drift in mean realized holding minutes |
| Occupancy | Maximum 10% relative drift in total realized holding minutes |

Moneyness buckets use canonical ladder distance from ATM slot 10:

- `ATM`: distance at most one slot;
- `NEAR`: distance two through five slots;
- `WING`: distance greater than five slots.

Matching must be evaluated after serial replay as well as from the
pre-execution randomization receipt. A post-replay mismatch fails closed.

Controls that preserve model-selected entry times may diagnose contract or
slot-selection value, but they are not valid hard evidence of a model's total
incremental entry value. The hard D1 and real-candidate comparisons must
randomize both timing and contract choice without using model features.

## Three-Dollar Equivalence Diagnostic

The former `$3 per executed trade` equivalence target remains a useful
precision diagnostic. It is not a hard gate and cannot change D1 V2,
candidate eligibility, ranking, or selection.

This prevents a lack of statistical precision from being mislabeled as
evidence that a shuffled model found positive alpha.

## Real Candidate Incremental-Edge Gate

A real Stage-1 row is selection-eligible only when all previously signed hard
requirements pass and its fee-adjusted increment over a candidate-specific
exposure-matched random P5 control satisfies all of the following:

1. The exposure match passes every dimension above.
2. The paired 95% lower confidence bound is strictly greater than zero.
3. The multiplicity-adjusted one-sided `p <= 0.05`.
4. The comparison includes all costs and uses simulator v5.
5. The evidence carries all 28 preregistered row identities and the canonical
   SHA-256 of that ordered multiplicity family; a count or Boolean alone is
   not a complete receipt.
6. The evidence is complete and independently reproducible.

A lower confidence bound equal to zero fails. Missing candidate-specific
incremental evidence fails closed. The multiplicity family is all 28 attempted
H0-H3/policy rows, not merely the rows that looked promising after training.

## Current Campaign Reaggregation

The frozen 420 models may be rescored and replayed, but they must not be
refitted for this amendment. Reaggregation must:

1. preserve and verify all 420 model hashes;
2. replace the old D1 result with D1 V2;
3. retain G1-G8, D5, D6, and maxT evidence without recomputation unless
   required for the paired contrast;
4. apply the candidate incremental-edge gate to every row;
5. select at most one row;
6. keep G9, seed 45, the protected holdout, and learned exits closed unless a
   row earns the corrected selection rule.

Every control-summary and session-level chunk consumed by reaggregation must
be individually hash-addressed in a preregistered manifest. Independent
verification must write to a separate immutable packet, rehash those source
chunks, and recompute matching receipts and any available paired inference
from source data rather than trusting producer-derived metrics.

## Authorization Boundary

This amendment authorizes the corrected D1 implementation, focused tests,
independent verification, and model-free reaggregation of frozen campaign
evidence. It does not authorize model refitting, G9, protected-holdout access,
learned-exit training, broker connectivity, paper orders, promotion, runtime
changes, launchd changes, paid downloads, or real-money work.
