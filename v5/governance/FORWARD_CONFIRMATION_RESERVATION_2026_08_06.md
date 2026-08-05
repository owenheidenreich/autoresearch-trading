# Forward Confirmation Reservation — declaration

**Status: DRAFT — awaiting owner signature. Drafted 2026-08-05 on the owner's decision of the same
day. The reservation binds from the moment the owner signs, retroactive to the start date below.**

## What this declares

The protected historical holdout is spent and no clean historical range exists, so forward sessions
are the only confirmation evidence this project has left. This declaration starts that clock.

> **Every ES and SPXW trading session from 2026-08-06 (inclusive) onward is reserved for
> confirmation only.** No research analysis, screen, model, tuning decision, chart, or summary may
> compute or inspect any strategy outcome, return, or policy economics on reserved sessions until a
> pre-registered confirmation protocol (G8, or an owner-approved equivalent) opens them — once.

## Rules

1. **What is reserved:** strategy and policy *economics* on reserved sessions — returns, P&L, hit
   rates, signal-conditioned outcomes, and anything derived from them.
2. **What is not reserved:** infrastructure evidence that evaluates no policy — arrival-latency
   distributions, feed parity checks, capture completeness, clock receipts. The already-authorized
   Track-A capture on 2026-08-06 and 2026-08-07 analyzes exactly this and is explicitly permitted.
   The line is: *timing and identity of data, yes; what a strategy would have earned, no.*
3. **Development data ends 2026-08-05.** The owned corpus through 2026-08-05 is development data.
   G1 and all research run there.
4. **Opening the reserve** requires a pre-registered protocol frozen before any reserved outcome is
   seen: candidate hash, session list, alternative, bar, N, significance, and stopping rules — the
   G8 pre-registration in the gate-chain audit §10, or a pre-registered sequential (always-valid)
   test. One opening. If the candidate is then changed, the opened range becomes development data
   and the reservation restarts from the change date.
5. **Violations are recorded, not hidden.** An accidental computation on reserved sessions is
   reported in STATUS immediately; the affected sessions move to development data and the
   reservation restarts from the violation date.

## Why the start date is 2026-08-06

Every day between now and a future G8 that is not under reservation is confirmation calendar
permanently lost. The measurement review
([finding](../research/findings/MEASUREMENT_REVIEW_2026_08_05.md)) puts the forward calendar at
~1.1 years for a 2-point edge and longer for smaller ones; the reservation must start before the
edge is found, not after.

## Signature

- Owner decision to reserve: **recorded 2026-08-05** (plan approval conversation).
- Owner signature on this declaration text: **PENDING** — sign by replacing this line with
  `Signed: <name>, <date>`.

*Drafted by Claude Fable 5, 2026-08-05, on owner instruction. This draft contacts nothing and
changes no runtime state.*
