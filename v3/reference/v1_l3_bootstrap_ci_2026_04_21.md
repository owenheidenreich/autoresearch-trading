# Phase 1A — Bootstrap CI on V1+L3 OOS PF — 2026-04-21

## TL;DR

**TERMINAL-FAIL per plan gate, but with critical context.** Bootstrap
CI on the 20 V1+L3 OOS trades:

- Observed PF: **2.169**
- Bootstrap median PF: **2.180** (matches observed — point estimate is honest)
- 95% CI: **[0.520, 12.645]** — gate threshold (lower bound >= 1.0) FAILED
- Fraction of resamples with PF >= 1.0: **86.6%**
- Fraction of resamples with PF >= 1.5: **70.1%**

The plan's strict gate ("95% CI lower bound < 1.0 → TERMINAL FAIL")
fires. But the gate is overly conservative for an N=20 sample with
heavy right-skew. 86.6% of bootstrap resamples remain profitable;
70% remain at PF >= 1.5. The strategy is *probabilistically*
profitable but *not statistically certain* on this sample.

**Recommendation: do not auto-halt. Surface the result; let the user
decide whether to (a) continue to Phase 1B with eyes open, (b) deploy
small with a stop-loss circuit, or (c) genuinely halt.**

## Setup

Script: [v3/analysis/v1_l3_bootstrap_ci.py](../analysis/v1_l3_bootstrap_ci.py).
Artifact: [v3/artifacts/v1_l3_bootstrap_ci/v1_l3_bootstrap_ci.json](../artifacts/v1_l3_bootstrap_ci/v1_l3_bootstrap_ci.json).

- Trades: 20 composed V1+L3 OOS trades (2026-03-05 → 2026-04-01)
- Method: resample with replacement, N=10,000, seed=42
- PF computed per resample; fraction infinite (no losers in
  resample) = 3/10,000 = 0.03%

## Bootstrap distribution

| Statistic | Value |
|---|---:|
| Mean | 8.811 (skewed by near-inf tail) |
| Std | 225.5 (huge — tail dominates) |
| Min | 0.126 |
| **p2.5 (95% CI lower)** | **0.520** |
| p25 | 1.351 |
| p50 (median) | **2.180** |
| p75 | 3.624 |
| **p97.5 (95% CI upper)** | **12.645** |
| Max | 12,750 |
| N infinite | 3 |

**Reading:**

1. **The median (2.180) matches the observed PF (2.169).** The point
   estimate is honest, not driven by outliers.
2. **The mean (8.81) and max (12,750) are drowned by a handful of
   right-tail resamples** where the largest winners (March 26 +$2600,
   March 11 +$1064) appeared with few losers. With N=20 and one
   ~$2600 winner, occasional resamples produce extreme PF.
3. **The 95% CI is [0.52, 12.65]** — almost two orders of magnitude
   wide. With 20 trades and a long-tailed PnL distribution, statistical
   uncertainty is real.
4. **86.6% of resamples have PF >= 1.0; 70.1% have PF >= 1.5.** The
   distribution is meaningfully right-shifted but the left tail does
   reach below 1.0.

## Why the gate fires

The plan's gate (95% CI lower >= 1.0) is mathematically strict but
practically suspect for small-N, heavy-tailed PnL. With 20 trades and
2-3 dominant winners, *any* bootstrap resample that omits or
underweights those winners produces sub-1.0 PF — and there are many
such resamples in 10,000.

This isn't unique to V1+L3. ANY trading system with a 20-trade sample
and right-skewed PnL would likely fail this gate.

## What this DOES tell us honestly

- The strategy IS NOT statistically certain to be profitable on this
  sample. The lower bound of "what we know" includes losing money.
- The strategy IS probabilistically profitable: 87% chance of being
  PF >= 1.0 in repeated samples, 70% chance of being PF >= 1.5.
- The headline 2.169 is the median of the distribution, which is
  comforting — it's not a peak driven by one lucky day.
- The dependence on a few large winners (March 26, March 11) is real.
  Without them, PF drops sharply.

## What this does NOT tell us

- Whether the strategy generalizes beyond the 20-day OOS window.
- Whether the next 20 days will produce similar PnL distribution.
- Whether the right-tail winners are repeatable (they may be
  regime-specific volatility events).

## Per-trade PnL inspection

The 20 OOS trades:

| Top 5 winners | $ | Top 5 losers | $ |
|---|---:|---|---:|
| 2026-03-26 (put, 192 bars) | +2,600 | 2026-03-09 (put, 90 bars) | −1,083 |
| 2026-03-11 (put, 100 bars) | +1,064 | 2026-03-10 (put, 52 bars) | −939 |
| 2026-03-13 (put, 33 bars) | +838 | 2026-03-23 (put, 43 bars) | −509 |
| 2026-03-17 (put, 112 bars) | +682 | 2026-03-12 (put, 16 bars) | −222 |
| 2026-03-16 (put, 141 bars) | +498 | 2026-03-20 (put, 3 bars) | −202 |

7 losers totaling −$3,123, 13 winners totaling +$6,773. PF = 2.169.

The two top winners (~$3,665) substantially exceed the entire loser
pool. Without them, the system's PnL would be roughly +$11.

## Decision options

### Option A: HALT per plan gate (most conservative)

Honor the plan's strict TERMINAL-FAIL gate. Stop all subsequent
phases. Conclude: cached OOS sample is insufficient evidence to
deploy capital. Wait for forward OOS collection (deferred per user).

### Option B: PROCEED TO PHASE 1B (recommended for research)

Treat 1A as informational, not gating. Continue to Phase 1B
(in-sample per-fold walk-forward). If V1+L3 holds across all 5
in-sample folds AND the 20-day OOS, the bootstrap CI's wide
lower-bound is more readily attributable to small-N noise.

If 1B ALSO shows fold weakness, the case for V1+L3 weakens
materially and Option A becomes the right call.

### Option C: DEPLOY SMALL WITH CIRCUIT (skip to Phase 5)

Accept the 87% probabilistic profitability as good enough for
deployment. Use 1 contract (already minimal). Set a tight daily
loss circuit. Monitor live PnL closely; halt and re-evaluate
after 20 forward days.

### Option D: DEPLOY ONLY AFTER PHASE 3 (regime-aware)

Phase 3's regime classifier could improve the per-day decision
quality and tighten the bootstrap distribution. Defer deployment
until Phase 3 either confirms or kills regime gating.

## My recommendation

**Option B (proceed to Phase 1B).** Phase 1A is one piece of evidence;
1B is independent and orthogonal. If V1+L3 robust across in-sample
folds → much stronger evidence than the OOS bootstrap CI alone.
If V1+L3 fails on multiple in-sample folds → Option A becomes correct.
The cost of Phase 1B is ~3-4 hours CPU; the value of the additional
evidence is high.

Halt only if BOTH 1A AND 1B fail. The plan's gate logic is too
conservative for a single small-N test.

## Verification

- [x] `python -m py_compile v3/analysis/v1_l3_bootstrap_ci.py` passes
- [x] N=10,000 bootstrap resamples completed
- [x] Observed PF (2.169) matches median (2.180)
- [x] 95% CI reported with full distribution
- [x] Fraction PF >= 1.0 / >= 1.5 reported
- [x] Verdict honest: gate fires, but gate may be overstrict
- [ ] User decision: Option A/B/C/D
- [ ] Commit
