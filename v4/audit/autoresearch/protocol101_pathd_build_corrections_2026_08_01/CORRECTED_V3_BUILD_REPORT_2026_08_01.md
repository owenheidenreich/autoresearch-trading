# Protocol101 Path-D Corrected-v3 Foundation Build Report

Date: 2026-08-01

Status: **CORRECTED_V3_FROZEN_AT_BUILD-ONLY PREFIT PAUSE**

Authority:

- `v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_BUILD_CORRECTION_PROPOSAL_2026_08_01.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_TERMINAL_RULE_RECONCILIATION_AND_AREF_BLOCKER_2026_08_01.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_CORRECTED_V3_OWNER_AUTHORIZATION_2026_08_01.json`

Corrected generation:

`v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_v3_2026_08_01`

## Executive disposition

The owner-approved `AREF_ACTION_CALIBRATION_ROLE` and refined B-R
`COMPOSITE_CALIBRATION_TERMINAL_RULE` decisions are implemented in a distinct corrected-v3
preregistration and restoration foundation. Corrected-v2 remains byte-identical and is bound as
the immediate immutable predecessor.

The correction remains a build-only pre-fit pause. The two scientific decisions are resolved,
but machinery sealing, foundation-stability sealing, fitting, corpus decoding, nested or outer
evidence opening, protected-holdout opening, and live or broker action remain unauthorized.

## `A_ref` correction

The per-second utility is unchanged:

```text
U_hold = LCB90(mean[A_ref|s_t]) + 0.25*min(calibrated_q10_A_ref,0)
HOLD iff U_hold > 0; otherwise request EXIT
```

The corrected role closure is:

- `A_ref_mean`: direct composer input and HOLD/EXIT predicted mean.
- `A_ref_q10`: direct downside input and binding HOLD lower-coverage support.
- `A_ref_q50`: decision-critical joint-calibration location and transitive dependency of calibrated q10.
- `A_ref_q90`: decision-critical EXIT upper-coverage support only; never a direct utility input.
- `downside_300`, `recovery_300`, `giveback_300`, and `remaining_tail_300`: the only diagnostic-only families.

The action bundle is fit and frozen first with exactly
`0.5*MSE(A_ref_mean)+0.5*pinball_0.10(A_ref_q10)`. The q50 support bundle is separate and cannot
backpropagate into the action bundle. The q90 support bundle is separate and cannot backpropagate
into the action or q50 bundles. All three use byte-identical frozen A_ref feature/scaler/mask/row/
weight/partition payloads without sharing a mutable scaler object. Joint monotone calibration is
fit only after all raw outputs are frozen.

The raw neural topology is:

```text
q10 = frozen action-core output
q50 = q10 + softplus(gap50), with q10 detached during q50 fitting
q90 = q50 + softplus(gap90), with q10/q50 detached during q90 fitting
```

The HGB topology likewise freezes the q10 estimator before independent q50/q90 support estimators
and uses the same frozen-anchor noncrossing assembly. Its exact constructor arguments, 48-fit
count, loss/quantile assignments, seed order, scaler identity, aggregation, and serialization
order are preregistered. The neural contract freezes seven independent modules per seed and their
graph, losses, detach boundaries, optimizers, update limit, and serialization order.

The binding EXIT reliability identity is coverage-based and tie-safe:

```text
A_ref <= calibrated_q90_upper  iff  -A_ref >= -calibrated_q90_upper
```

Corrected-v3 does not claim the literal empirical-quantile identity
`q10(-A_ref)=-q90(A_ref)` without a tie-compatible convention.

## Refined B-R terminal rule

Corrected-v3 freezes `B_R_STATUS_PRESERVING_WHOLE_RUN_HARD_STOP` with terminal precedence:

```text
invalid_result
  > insufficient_evidence
  > owner_decision_required
  > no_genuine_signal
  > PASS
```

Every calibration-valid nested or outer scope must validate its complete ordered required-node
manifest and seal a write-once `VALID` scope-gate receipt before evidence may open. Outer folds are
strictly sequential `1 -> 2 -> 3 -> 4 -> 5`; each earlier result must be immutable before later-fold
work. Pooled and at-least-four-of-five economics require exactly five chronological `VALID` folds.
The `>=4/5` rule means four positive deltas among five valid folds, never survivor-only pooling or
four folds relabeled as five.

Only the eleven session-assignment blocks already frozen with `calibration_valid=false` may emit a
zero-access structural skip:

```text
outer 1: inner 1,2,3,4
outer 2: inner 1,2,3
outer 3: inner 1,2
outer 4: inner 1
outer 5: inner 1
```

A computed failure in a calibration-valid block cannot be relabeled as a skip.
`INVALID_TARGET_COVERAGE` maps out of band to `invalid_result` with failure-only receipts and no
scientific success receipt. A required count miss maps to `insufficient_evidence`.

The known exit-weight session counts remain `[0,0,19,48,56]`, all below the frozen minimum of 60.
If entry passes, the immutable entry-only verdict is preserved by hash, then the combined campaign
stops `insufficient_evidence / insufficient_exit_evidence` before exit weights. It does not create
exit fits, Boxes C/D, a full-exit artifact, a pre-holdout packet, or a holdout authorization.

Forbidden rescue includes fold/node/target/quantile/horizon/action/control deletion, post-prediction
row deletion, null or zero imputation, substitution, composite reweighting, status relabeling,
survivor-only pooling, four-as-five accounting, alternate calibrators, and same-generation retry,
refit, reseed, or threshold change after an invalid result.

## Preserved Path-D build corrections

The P1/P2/P3 dispositions remain unchanged:

- P1 drift classification is `BENIGN_TEST_CONTAMINATION`. The synthetic fixed transaction wrote a
  zero-access historical fold-1 burn before the original freeze; no corpus/evidence row was opened.
  The burn remains immutable history and folds `[1,2,3,4,5]` are pristine in corrected-v3.
- The foundation-stability gate remains implemented and fail-loud, but its receipt is deliberately
  unsealed at this pause.
- `last_causal_open_interest` remains dropped because it is EOD-only and lacks an intraday live twin.
- `last_causal_minute_volume` remains dropped until an exact shared completed-minute historical/live
  adapter and receipt prove sparse-minute, exact-contract, same-session, and 90-second carry parity.
- Entry remains exactly signed-17; exit remains the corrected ordered 47-feature set.
- `size_imbalance`, optionally with `bid_size` and `ask_size`, remains the pre-vetted forward-only
  `WIDEN-ENTRY` lead. It is not entry alpha in this run.

## Frozen corrected-v3 artifacts

The corrected-v3 root contains exactly six regular files:

| File | Bytes | Raw SHA-256 |
|---|---:|---|
| `feature_lineage.json` | 156,333 | `4521b3e62982f672daaba73d98ce5a9ce72df170b083cd40df7dc979983d1ef0` |
| `foundation_restoration_receipt.json` | 8,215 | `cbe9dd557fd97a27e0da7712205644be240edc8c5ad9621fff942fc3c0dd198a` |
| `preregistration.json` | 458,413 | `8ed2b729e1b6757fa5c876f3584645dd4dc590e5a3c8f8675993732e6c2b1d07` |
| `preregistration.sha256` | 87 | `c2e9c98e92416ed5786b121d625caa9d2ef44116248235eabc4f27cb9e34405b` |
| `preregistration_freeze_receipt.json` | 3,552 | `e18ad1d9162fc17ee67c154f0b2288a74b570fcbe62151044d4c202c02db4f94` |
| `session_assignments.json` | 103,609 | `431cd14879ad6a14b278cb2683e4f3b860eef960e6b74a4f0edb3e620cf82475` |

Foundation generation SHA-256:

`da65e572c7deb6b842d9c7c103c1047f8045f20dc2a1ddd3b12373f5e0ac79f8`

Restoration receipt semantic self-hash:

`a695189e4265d1472264c1b200900dd39a98dbc5108b95436475af46f82bf906`

The corrected-v2 generation SHA-256 remains:

`c1a0b31383524197a383281f9fb43526d4a20174b6557def2794b2eb7994b79a`

All six corrected-v2 raw file sizes and hashes were rechecked after corrected-v3 creation and remain
exactly equal to their frozen bindings.

## Verification evidence and prohibited-artifact audit

The complete registered synthetic pre-fit suite passed:

```text
148 passed in 51.41s
```

The added tests cover A_ref role closure, HGB/neural topology, gradient/detach boundaries, tie-heavy
EXIT coverage identity, dependency tampering, exact corrected-v2 byte preservation, B-R node/status
closure, lawful structural skips, fold/node deletion, zero imputation, survivor-only pooling,
four-as-five accounting, and entry-verdict preservation at the known exit-power stop.

`assert_preregistration_frozen()` and `assert_foundation_restoration_frozen()` both pass. All 57
`source_hashes_at_freeze` entries reproduce current bytes. The corrected-v3 root has no fold or
holdout directory and no machinery, lineage-implementation, corpus-integrity, foundation-stability,
JUnit, pooled, model, calibration-evidence, fitted-weight, access, dataset, result, paper, or live
artifact. Both frozen receipts report `model_fit_executed=false` and `holdout_open_count=0`.

No corpus file was decoded, no nested/outer evidence was opened, no protected holdout was accessed,
and no live, broker, paper, registry, runtime, or promotion action occurred.

STOP_FOR_CLAUDE_VERIFICATION
