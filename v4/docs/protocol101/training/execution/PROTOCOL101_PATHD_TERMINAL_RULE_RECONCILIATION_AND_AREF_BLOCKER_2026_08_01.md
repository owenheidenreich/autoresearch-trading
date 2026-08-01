# Protocol101 Path-D Terminal-Rule Reconciliation and `A_ref` Blocker

Date: 2026-08-01

Status: **post-investigation reconciliation; no owner decision adopted; no seal or fit authorized**.

This note closes the interrupted Claude verification of:

- `PROTOCOL101_PATHD_COMPOSITE_CALIBRATION_TERMINAL_RULE_FINDINGS_2026_08_01.md`; and
- `v4/audit/autoresearch/protocol101_pathd_composite_calibration_terminal_rule_analysis/READ_ONLY_EVIDENCE_ANALYSIS_2026_08_01.md`.

It is outside every frozen foundation. It does not mutate corrected-v2, adopt a terminal rule,
authorize model fitting, open any corpus/fold/holdout evidence, or seal machinery or stability.

## Reconciled conclusions

### 1. Exit-power discrepancy resolved

The outer entry history lengths `54/75/97/119/142` are not the learned-exit fit denominator.
The binding learned-exit denominator is the aggregate of calibration-valid nested-entry-OOF
validation sessions assigned to exit weights:

```text
[0, 0, 19, 48, 56]
```

The frozen minimum is 60. Every learned-exit outer fold therefore abstains before exit weights.
This run cannot instantiate learned-exit Boxes C/D or support a combined entry+exit claim.

### 2. Terminal-rule recommendation reconciled

Claude's preliminary Candidate A (fold-scoped abstention with survivor continuation) should not be
adopted for this frozen run. It conflicts with the literal all-five power formulas, exact five-row
reconstruction, and pooled five-fold population, and it creates regime-dependent survivor-selection
risk.

Candidate C (drop an offending target and continue) is rejected because it changes the estimand and
composite weights after seeing calibration behavior.

The evidence-supported recommendation is **B-R: status-preserving whole-run hard stop**:

- Every required outer/final calibration node must be `VALID` before outer economic evidence opens.
- A required evidence-count miss stops the run as `insufficient_evidence`.
- `INVALID_TARGET_COVERAGE` remains distinct and should stop out of band as `invalid_result` through
  a failure-only receipt, never a scientific result or success receipt.
- Lawful preregistered nested skip blocks remain skips; they do not become whole-run terminals.
- No failed fold, row, horizon, action, or required target may be deleted, substituted,
  zero-imputed, reweighted, or pooled only over survivors.
- The existing pooled and at-least-four-of-five economic gates run only after all five required
  outer/final calibration bundles are valid and every frozen minimum-power gate passes.

B-R remains a recommendation. It is not adopted until the owner approves a superseding correction.

## New blocking contradiction: `A_ref` target roles

The corrected-v2 source and frozen preregistration contain a categorical cross-contract conflict:

1. The per-second exit composer uses `A_ref_mean` and `A_ref_q10` only.
2. `exit.diagnostic_target_isolation.action_model.action_consumers` lists only
   `A_ref_mean` and `A_ref_q10`.
3. `A_ref_q50` and `A_ref_q90` are separate diagnostic models.
4. Diagnostic `forbidden_effects` include both `A_ref calibration` and `HOLD/EXIT action`;
   diagnostic failure is declared unable to change the primary verdict.
5. The global monotone calibration contract jointly calibrates each target family's
   q10/q50/q90 outputs and uses q50 as the location supporting calibrated q10 and q90.
6. The binding HOLD reliability gate consumes calibrated `A_ref_q10`.
7. The binding EXIT reliability gate consumes negative calibrated `A_ref_q90`, because
   the lower quantile of `-A_ref` is `-q90(A_ref)`.
8. Combined acceptance requires HOLD and EXIT calibration gates to pass.

Consequently, `A_ref_q90` is simultaneously verdict-required and verdict-isolated. `A_ref_q50` is
simultaneously required by the joint monotone calibration formula and forbidden from influencing
`A_ref` calibration. No current clause establishes precedence.

The registered suite passes despite this conflict:

```text
143 passed in 28.08s
```

That pass is a coverage gap, not exculpatory evidence. The fixed-science isolation test asserts the
mean/q10-versus-q50/q90 split but never cross-checks the binding EXIT dependency on q90.

## Smallest coherent correction — recommended, not adopted

The smallest scientific change is to preserve the existing composer and reclassify the complete
`A_ref` quantile triplet as decision-critical action-calibration support:

- `A_ref_mean`: per-second HOLD/EXIT composer and action predicted mean.
- `A_ref_q10`: per-second downside term and binding HOLD reliability lower bound.
- `A_ref_q50`: joint monotone calibration/noncrossing location support.
- `A_ref_q90`: binding EXIT reliability lower bound through `q10(-A_ref)=-q90(A_ref)`.
- `downside_300`, `recovery_300`, `giveback_300`, and `remaining_tail_300`: the only
  diagnostic-only quantile families; their incompleteness remains nonbinding.

This interpretation preserves the runtime utility:

```text
U_hold(t) = LCB90(mean[A_ref | s_t]) + 0.25 * min(q10[A_ref | s_t], 0)
```

It distinguishes **policy-action inputs** (mean and q10) from **decision-critical acceptance and
calibration support** (q10/q50/q90). It is smaller than deleting q90 from the EXIT gate, which would
require designing a replacement one-sided lower bound for `-A_ref` or weakening a binding gate.

## Owner decision still required

The recommended reclassification still changes which targets may veto a verdict and requires an
exact model/loss/gradient topology that is not currently frozen. The owner must approve both:

1. `AREF_ACTION_CALIBRATION_ROLE`: preserve the composer; make the A_ref q10/q50/q90 triplet
   decision-critical calibration support; keep only the four local-path triplets diagnostic-only.
2. `COMPOSITE_CALIBRATION_TERMINAL_RULE`: adopt B-R with the status precedence
   `invalid_result > insufficient_evidence > owner_decision_required > no_genuine_signal > PASS`.

After approval, implementation must use a distinct corrected-v3/superseding foundation. It must:

- preserve corrected-v2 byte-for-byte;
- bind the exact A_ref model/loss/calibration topology;
- enumerate decision-critical versus diagnostic-only calibration node IDs;
- add an explicit exit minimum-power stop reason;
- add negative tests for diagnostic leakage, binding-gate dependency isolation, target/fold
  deletion, survivor-only pooling, zero imputation, and status relabeling;
- re-freeze before any machinery/stability seal or model fit.

Until then: no machinery seal, no foundation-stability seal, no model fit, no corpus/fold evidence
opening, no protected-holdout access, and no live/broker action.

STOP_FOR_OWNER_DECISION
