# Protocol101 Canonical Minute-Game Test Design Brief For Fable

Generated: 2026-07-09

## Purpose

This document is for Fable to design concrete tests and execution prompts for the next Protocol101 phase.

The project is no longer trying to prove raw Databento/ThetaData quotes equal raw IBKR recorder quotes. That route produced useful evidence but became too brittle. The new direction is to test whether both data planes can map into a shared **canonical minute-level SPXW 0DTE decision game** that is stable enough to train on.

The end goal remains:

```text
Create a profitable SPXW 0DTE model that is trained on a game equivalent to what it will see live, then validate it enough to enter guarded IBKR paper trading.
```

Fable should help turn this into concrete, testable experiments and Codex `Goal` prompts.

## Current Decision

The prior Group 1 non-VIX uplift run has been skipped/abandoned as a decision-making path. Treat any existing artifacts under:

```text
/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_nonvix_uplift_attempt001/
```

as non-load-bearing unless a later owner-approved review explicitly reclassifies them. The project should not use that run to decide the canonical-game path.

The current focus is:

```text
Design and test Protocol101 canonical minute-game v1.
```

## Why We Are Changing Direction

The old synchronization effort discovered:

- Exact raw cross-vendor quote parity is not realistic.
- Static-ladder universe parity can be achieved.
- SPX/VIX context and ladder/ATM drift were repairable.
- Remaining candidate differences were option quote/tradability boundary effects.
- The masked-v2 baseline did not produce a paper-ready candidate.
- Group 2 geometry/moneyness passed parity under the static-ladder policy but failed uplift and was rejected as no real signal.

Important artifact paths:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_option_quote_source_policy_resolution/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_static_ladder_boundary_stable_policy_audit/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group2_geometry_uplift_attempt002/routing_decision.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_live_v2_group1_index_context_parity_resolution/report.md`

The key unresolved problem:

```text
We need to use the large Databento/ThetaData historical corpus without training on a fake game that IBKR live/paper cannot reproduce.
```

## New Core Hypothesis

```text
A profitable SPXW 0DTE model can be trained on Databento/ThetaData if both Databento history and IBKR recorder data are transformed into the same canonical minute-level decision game.
```

The canonical game should allow stable, coarsened minute-level information as alpha, while keeping feed-sensitive raw quote behavior out of model alpha.

This is a two-part claim:

1. **Transferability claim:** canonical features derived from Databento/ThetaData and IBKR recorder data represent the same abstract game closely enough that a model does not learn source artifacts.
2. **Profitability claim:** that canonical game contains enough edge to train a profitable 0DTE model.

The next phase tests claim 1 first. Do not conflate L0/L2 transferability with profitability.

## Candidate Canonical Alpha Families

Fable should refine this list into a concrete v1 feature specification.

Potential model-facing canonical features:

- SPX minute levels and returns.
- VIX minute levels and returns where causal warm-up exists.
- Time of day and day structure.
- Range position, VWAP distance, opening move context, gap context.
- Static 42-slot SPXW 0DTE ladder identity.
- Moneyness and normalized distance from spot.
- Tick-quantized per-slot option minute-close mid.
- Normalized option mid changes over 1m, 5m, and 15m.
- Option path structure from minute mids.
- Near-ATM cross-strike relationships, initially limited to a fixed band such as ±5 slots.
- Put/call mid ratios within the near-ATM band.
- Internally recomputed IV/Greeks from canonical mid + spot + time, if deterministic and L0/L2 safe.

Fields that should remain guard/fill/audit-only unless separately proven:

- Raw bid.
- Raw ask.
- Raw spread.
- Quote age and update timing.
- Bid/ask sizes and depth.
- Anything sub-minute.
- Volume/OI until live-proven.
- Vendor-computed Greeks.
- Distance-to-guard-boundary indicators.
- Near-spread-cap, near-mid-cap, near-staleness, or similar raw boundary features.

## Existing Evidence To Use

Existing paired recorder days:

```text
2026-06-30
2026-07-01
2026-07-02
```

These days are **burned/design days**. They may be used to design canonical transform v1 and run design-phase L0/L2 audits, but they must not be used for final certification.

Relevant existing evidence:

- Static-ladder universe parity passed on 1077 rows and 45234 static slot pairs.
- Residual raw quote/tradability divergence was 62 side-only candidates across 44 decision rows.
- Divergence concentrated at guard boundaries: mid cap, spread cap, stale quote, invalid bid/ask timing, no quote before decision boundary.
- There were zero action flips in the prior static-ladder parity work.

Fable should treat this as support for the idea that divergence is a boundary phenomenon, but should not assume minute-close mid values transfer until measured.

## Tests Fable Should Design

### L0: Canonical Field Divergence And Bias Test

Purpose:

```text
Determine whether each canonical feature transfers across Databento/ThetaData and IBKR recorder data without dangerous bias.
```

Required design questions:

- How should canonical rows be paired?
- Which fields are compared directly?
- Which fields require normalization or quantization first?
- What moneyness bands should be used?
- What time-of-day bands should be used?
- What volatility/opportunity regimes should be used?
- How should forward-return or label correlation be computed without leakage?
- What correlation/bias thresholds are acceptable?
- What makes a feature `admit`, `repair`, `reject`, or `insufficient evidence`?

Minimum outputs Fable should specify:

- `canonical_feature_definition.json`
- `field_divergence.csv`
- `bias_tests.json`
- per-feature admit/repair/reject table
- measured divergence distributions for later noise-injection design

### L2: Source Discriminator

Purpose:

```text
Determine whether canonical feature vectors still leak whether they came from Databento/ThetaData or IBKR.
```

Required design questions:

- What classifier family should be used first?
- What inputs are allowed?
- How do we avoid a same-day/date artifact?
- Should CV be by day, by timestamp block, or leave-one-day-out?
- What same-source shuffled/day-control baseline is required?
- What AUC threshold should trigger admit/repair/reject?

Minimum outputs Fable should specify:

- `source_discriminator.json`
- AUC and confidence interval.
- Comparison to same-source/day-shuffled baseline.
- Feature importance or source-leak explanation.
- Routing decision.

### L1: Fixed Policy Probe Battery

Purpose:

```text
Test whether simple fixed policies behave similarly on canonical Databento rows and canonical IBKR rows.
```

This should likely be the second Codex goal after L0/L2, not bundled into the first implementation unless Fable strongly disagrees.

Probe ideas:

- Momentum.
- Mean reversion.
- VWAP-side.
- Static side/ATM controls.
- Random-with-guards.
- Simple threshold rules per canonical feature family.
- Crossed with the existing 7 menu-v2 trade shapes.

Required metrics:

- Action agreement rate.
- Trade-set Jaccard overlap.
- Entry timestamp agreement.
- Selected slot/contract agreement.
- Day-PnL correlation.
- Per-trade PnL delta.
- Disagreement concentration by volatility/time/moneyness.

### L3: Transfer Probe Models

Purpose:

```text
Test whether simple models trained on transformed vendor data make equivalent decisions on paired transformed IBKR data.
```

This should happen only after L0/L2 and probably L1 pass.

Probe ideas:

- Small HGB.
- Logistic/linear baseline.
- Random forest or shallow tree baseline.
- Small feature-family-specific models.

Key restriction:

```text
These are transfer probes, not final strategy candidates.
```

## Stress Tests Fable Should Include

Design stress tests for later training/uplift phases:

- Measured source-divergence noise injection.
- Guard-boundary flips.
- Fill ladder: mid, mid plus fraction of spread, touch.
- Timing jitter: previous-minute state, delayed state, and boundary-shift tests.
- Pessimistic edge band reporting.

Fable should define which of these belong in the first scaffold and which should wait.

## Critical Design Rules

Fable should preserve these:

- Existing three recorder days are burned/design-only.
- Future sealed days must be assigned by deterministic rule before inspection.
- Transform code hash should be frozen before sealed evaluation.
- Repair budget should be bounded, for example max three transform iterations on burned days.
- No final certification on Databento/ThetaData alone.
- Vendor history can be used for exploration/pretraining and eventually CV training only after the canonical transform passes confirmation.
- Final threshold/certification evidence needs recorder-native or otherwise same-game confirmation.
- Fills/rewards should use raw quotes under conservative rules, not canonical mids.

## Questions Fable Must Answer

Please produce a concrete test design, not just critique.

1. What is the exact canonical transform v1?
2. Which canonical features are included in v1?
3. Which candidate features are intentionally excluded from v1?
4. What exact L0 tests should Codex implement first?
5. What exact L2 source-discriminator test should Codex implement first?
6. What are the first-pass admit/repair/reject thresholds?
7. How should the three burned recorder days be used without self-grading?
8. What is the routing decision schema?
9. What files/artifacts should Codex produce?
10. What should Codex explicitly not do?
11. What evidence would justify moving to L1 probes?
12. What evidence would invalidate canonical transform v1?
13. What should be measured now versus deferred to sealed recorder days?
14. How many fresh sealed days are required for initial confirmation, and what regime constraints matter?
15. How should this test design eventually feed into model training and paper-readiness?

## Proposed First Codex Goal Shape

Fable should critique and improve this.

```text
Goal: Protocol101 canonical minute-game v1 L0/L2 design audit.

Use existing burned paired days:
- 2026-06-30
- 2026-07-01
- 2026-07-02

Objective:
Build an audit-local canonical transform v1 and test whether candidate canonical model-facing features transfer across Databento/ThetaData and IBKR recorder data without source leakage or opportunity-correlated bias.

Do not train final models.
Do not run threshold selection.
Do not run feature uplift CV.
Do not paper trade.
Do not call broker APIs.
Do not download paid data.
Do not change promotion/defaults.
Do not edit runtime or launchd settings.

Required tests:
- L0 field divergence and bias tests.
- L2 source-discriminator test with day-level controls and same-source baseline.

Required output:
- feature admit/repair/reject table.
- measured divergence distributions.
- source-discriminator report.
- routing decision:
  - canonical_v1_design_pass
  - canonical_v1_repair_iteration_needed
  - canonical_v1_rejected_bias_irreducible
  - canonical_v1_insufficient_artifacts

Highest allowed claim:
canonical transform v1 L0/L2 design audit complete.
```

## What Fable Should Push Back On

Fable should be adversarial about:

- Whether option mid features are actually stable enough.
- Whether the source discriminator can be fooled by date/session effects.
- Whether L0 bias tests can be computed without leaking labels.
- Whether internal Greeks from canonical mids are a legitimate feature path.
- Whether deep wings should be excluded up front.
- Whether existing burned days are enough for design.
- Whether this path is just masked-v2 under a different name.
- Whether the final system can survive IBKR execution/fill realism.

## Desired Fable Deliverable

Please return:

1. A verdict on the canonical minute-game v1 plan.
2. A revised exact feature list.
3. A revised L0/L2 test design.
4. Concrete pass/fail criteria.
5. A first Codex `Goal` prompt that can be executed immediately.
6. A note on what evidence will be needed after L0/L2.

Do not answer only with "collect 40-60 days." If additional recorder data is required, separate:

- design data,
- initial confirmation data,
- final paper-readiness certification data.

The user needs a path that starts now, uses existing Databento/ThetaData data intelligently, and still avoids training on a fake game.
