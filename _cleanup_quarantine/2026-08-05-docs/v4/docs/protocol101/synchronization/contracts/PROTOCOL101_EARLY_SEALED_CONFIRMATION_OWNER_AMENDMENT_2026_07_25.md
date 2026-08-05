# Protocol101 Early Sealed Confirmation — Owner Amendment

Effective: 2026-07-25, before any sealed July 15-24 market data was opened for
the cross-vendor confirmation comparison.

Owner direction:

> Open the sealed July 15-24 IBKR evidence, acquire the matching
> Databento/ThetaData evidence, and run the originally planned synchronization
> comparison without waiting for FOMC evidence. Determine whether the existing
> evidence is sufficient to begin model training without collecting more IBKR
> recorder days.

This document amends only the evidence-count, regime-coverage, and development
rehearsal prerequisites in the frozen
`protocol101_canonical_v1_sealed_confirmation_preregistration`. It does not
alter any canonical feature, selection rule, epsilon, action/slot agreement
threshold, L0 threshold, L2 threshold, model probe, or materiality definition.

## Early confirmation population

Open and classify every sealed recorder session from 2026-07-15 through
2026-07-24. A session enters confirmation scoring only if objective input
readiness holds before parity results are inspected:

- 360 replay trace rows, including the 09:31 ET context row;
- 15,120 slot rows (360 x 42);
- opening context sufficient for full-session VWAP and momentum features;
- monotonic, live rather than delayed capture evidence;
- no broker order endpoint call and no paper submit;
- matching Databento and ThetaData source-aligned historical replay.

Incomplete sessions are reported and burned by this authorized opening, but
they do not become parity failures. The selection rule is mechanical; no day
may be included or excluded because its parity result looks favorable or
unfavorable.

At least three usable sealed sessions are required. This supports leave-one-day
out source-discriminator evaluation and supplies a fresh confirmation set in
addition to the already-passed July 10/13/14 development rehearsal. If fewer
than three sessions are usable, route to `insufficient_artifacts`.

## Amended prerequisites

- The original requirement for at least nine sealed sessions is replaced by
  the mechanical minimum of three usable sessions defined above.
- The FOMC/event-session requirement is waived for this early confirmation.
  The owner expects to avoid real-money trading on FOMC decision days. This
  waiver does not authorize live or paper orders and does not remove later
  live-shadow regime monitoring.
- The required development rehearsal is satisfied by the frozen battery pass
  on complete sessions 2026-07-10, 2026-07-13, and 2026-07-14, including CPI.
  The partial 2026-07-20 capture is classified `insufficient_artifacts` because
  it lacks opening context; it is not a failed parity session.

## Unchanged confirmation gates

Run the frozen canonical v1.4 L0/L2/L1/L3 battery exactly as preregistered:

- every non-random L1 probe and L3 subset/model action agreement >= 0.98;
- mutually confident selected-slot agreement >= 0.99 when denominator >= 30;
- L0 coverage/drift/bias gates unchanged;
- L2 HGB and logistic AUC <= 0.55, null control in [0.45, 0.55], positive
  control >= 0.80;
- material true-score reordering allowance scaled exactly by the original
  `2 per 10 sessions` rule;
- mandatory per-day reporting;
- no threshold, epsilon, feature, transform, or selection-contract change.

## Result interpretation

Passing routes to:

`canonical_v1_4_early_sealed_confirmed_for_offline_training`

This is sufficient to begin governed offline Stage-1 training and hill
climbing. It is not paper readiness.

Before paper trading, any selected candidate still requires:

- G1-G9 and protected-holdout success;
- candidate-specific historical-versus-IBKR recorder replay;
- no-order live shadow validation;
- separate owner authorization for paper submit.

Failing routes to the original component-specific failure class. All opened
July 15-24 sessions become burned regardless of result. No second confirmation
run may tune against their outcomes.

## Safety scope

Authorized now: opening sealed July 15-24 evidence, matching paid historical
downloads, replay construction, and the frozen offline comparison.

Not authorized: model training, threshold tuning, broker calls, paper submit,
promotion/default changes, launchd/runtime changes, or real-money paths.
