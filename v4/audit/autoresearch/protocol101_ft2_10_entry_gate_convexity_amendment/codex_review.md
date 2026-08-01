# FT2-10 Entry-Gate Convexity Amendment — Codex Review

Status: **ENDORSED WITH A REQUIRED STATISTICAL CLARIFICATION**  
Review role: Codex reviews Claude's proposal before implementation  
Pre-amendment authority SHA-256: `edcbee06ebfc5ac3a26fa13da043754589ba55fbbd11b207906e19459d4103f3`

## Verdict

The convexity diagnosis is correct and the entry gate should move from an
outcome-tail q10 positivity test to a central-tendency expected-upside test.
The proposal is implementable only after clarifying that an ordinary split-
conformal lower prediction bound is **not** a confidence bound on a conditional
mean. Applying an individual-outcome residual quantile to a mean forecast
recreates the same pessimistic outcome-tail gate and remains unsatisfiable.

The amended gate therefore uses a squared-error conditional-mean head and a
one-sided, session-clustered 90% lower confidence correction for its disjoint-
calibration mean residual. This is called a calibrated lower confidence bound
on expected upside, not an outcome-conformal bound. The correction is estimated
from equally weighted calibration-session residual means, preserving the
session as the independent unit and preventing row-rich sessions from
dominating.

## 1. Convexity diagnosis

Confirmed on the frozen 45-session Option-D tensor. Among 120,229 physically
eligible development rows, realized fee-adjusted MFE-return q10 is negative at
every horizon:

| Horizon | q10 MFE return | Median | Mean | Positive share |
|---|---:|---:|---:|---:|
| h3 | -16.10% | -0.97% | +2.85% | 47.14% |
| h5 | -14.65% | +1.89% | +8.39% | 55.48% |
| h10 | -12.60% | +7.72% | +18.94% | 65.14% |
| h20 | -10.83% | +14.58% | +35.60% | 72.29% |
| h45 | -8.92% | +24.55% | +66.69% | 77.83% |
| h90 | -7.22% | +33.64% | +108.11% | 80.94% |
| remaining session | -6.67% | +50.87% | +212.24% | 83.27% |

Realized profit-area-return q10 is exactly zero at every horizon. Because the
current rule requires all four q10 axes to be strictly positive, the MFE axes
are negative and the profit-area axes are zero. The gate is structurally
unsatisfiable independently of model quality.

The diagnosis is specifically about the positivity **gate**. Retaining q10
for ordering contracts that already pass a satisfiable gate remains a coherent
robust-ranking choice.

## 2. Correct central-tendency criterion

The expected/mean statistic is preferable to the median for this convex
strategy because it preserves the contribution of the large favorable tail
the strategy exists to monetize. Median is useful as a report-only robustness
diagnostic but would optimize a different, less convex operating objective.

The exact amended gate is:

1. Fit conditional-mean MFE and profit-area heads in dollars and return for
   every registered horizon using the unchanged causal features and unchanged
   fee-adjusted labels.
2. On the unchanged disjoint calibration sessions, compute per-row residuals
   `realized - predicted_mean`, reduce them to one equally weighted mean per
   session, and take a deterministic one-sided 90% lower confidence bound on
   the mean residual using a session-cluster bootstrap.
3. Add that correction to each predicted conditional mean. This is the
   calibrated conservative lower bound on expected upside.
4. Across every causally available registered horizon, require the arithmetic
   mean of those lower bounds to be strictly greater than zero on all four
   axes: MFE dollars, MFE return, profit-area dollar-minutes, and profit-area
   return-minutes. Labels are already fee-adjusted, so strict zero is the
   fee-clearing threshold; no extra 5% threshold is introduced.
5. Rank passers using the existing calibrated-q10 percentile and deterministic
   tie-break sequence. Ranking is unchanged.

The calibration is global per head for the walking-skeleton dry-run. A formal
campaign may use a preregistered phase-specific session-cluster correction only
when its existing minimum-evidence rule is met; otherwise the same-head global
correction is required.

## 3. Horizon decision

All causally available registered horizons are retained. Excluding h3/h5 is
not supported:

- Their realized means are positive after fees (+2.85% and +8.39%). Their
  negative q10 values diagnose the old statistic, not dead expected upside.
- The consolidated authority explicitly retains h3/h5 because 3–8 minute
  resolution is an owner emphasis.
- In a same-family one-seed mean-head prototype, the session-clustered lower-
  bound gate passes 71.32% of eligible replay rows with all horizons and 70.75%
  after excluding h3/h5. Exclusion does not improve satisfiability.
- Removing horizons would broaden the amendment beyond the single entry-gate
  criterion.

## 4. Prototype falsification check

The same Option-D HGB family, causal features, 20 fit sessions, 20 disjoint
calibration sessions, and four replay sessions were used for a review-only
mean-head prototype:

| Gate support | Replay passes | Pass rate |
|---|---:|---:|
| Raw conditional means | 3,545 / 4,191 | 84.59% |
| Session-clustered 90% lower confidence means | 2,989 / 4,191 | 71.32% |
| Individual-outcome q90 error subtraction | 0 / 4,191 | 0.00% |

This both supports the proposed operating concept and falsifies the tempting
but incorrect implementation that relabels an outcome-tail bound as a mean
confidence bound.

## 5. Second-order effects

The entry-gate change does not imply that the complete composer will trade.
The following unchanged downstream gates may bind:

- The selected-cluster gap must exceed its calibrated uncertainty margin.
- The conservative MFE/error-margin rule may double-count conservatism after
  the new lower-bound gate and must be reported separately if it blocks.
- The selected contract's q90 normalized-regret upper bound must be <= 0.10.
- The action-conditioned gate remains mandatory. The preceding Option-D run
  had only 19 realized WAIT outcomes versus the frozen minimum of 50, so this
  gate was unavailable and would force WAIT even if upstream gates passed.

These are findings to measure in the required rerun, not authority to weaken
or bypass downstream rules in this amendment.

## Scope decision

Approved semantic change: the FT2-10 Stage-2 positive-after-fee entry gate and
the mean-head/calibration support strictly necessary to compute it.

Unchanged: RLAC targets, labels, folds, fees, simulator, D48, D49, exit design,
guardrail screen, q10 ranking, graph topology, downstream thresholds, and
action-conditioned minimum evidence.

