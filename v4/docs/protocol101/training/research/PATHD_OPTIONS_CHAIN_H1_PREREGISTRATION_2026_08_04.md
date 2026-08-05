# Path-D Options-Chain H1 Raw Direction Preregistration

Date frozen: 2026-08-04  
Status: `FROZEN_BEFORE_OUTCOMES`  
Family size: `K = 1`  
Product objective: one-account SPXW 0DTE long-call / long-put bot

## 1. Authority and scope

The owner directed the project to proceed after the adversarial review in
`v4/docs/deep-research-report.md`. This document freezes one raw, model-free
historical falsification screen. It does not authorize model fitting, use of the
spent 36-session holdout, an option-wrapper experiment, paper orders, live-data
capture, promotion, or a paper-default change.

The screen asks only whether one causal options-chain price-geometry signal
predicts the next five-minute signed SPX move strongly enough to beat a frozen
five-minute SPX-momentum comparator under one-account serial occupancy.

## 2. Prior-art adjudication performed before code and outcomes

The executable `prior_art_check` was run against both canonical history files.

### H1 — symmetric risk-reversal change

Mechanism string:

```text
five-minute symmetric SPXW 0DTE risk-reversal change predicts
next-five-minute signed SPX points
```

Exact semantic terms were `risk reversal`, `skew change`,
`option-surface direction`, `implied volatility skew`, `put skew`, and
`smile slope`. A second broad pass used `skew`, `smile`, `volatility surface`,
`put-call`, and `put call`. The exact pass found one nonblocking historical
reference; the broad pass found four nonblocking references. Neither pass found
a do-not-retest row or rejecting decoder verdict. H1 clears prior art.

### Removed H2 — gamma-weighted displayed quote pressure

H2 is removed before any return outcome is inspected. Its broad prior-art pass
finds rejected Protocol010 through the semantic term `gamma`; more importantly,
the proposed statistic cannot identify participant inventory or dealer gamma
sign, and it consumes displayed-size inputs whose historical/live receipt family
is not admitted. H2 is not repaired or renamed. It does not consume an outcome
look. The frozen family is therefore `K = 1`.

## 3. Firewall ruling

`assert_model_alpha_firewall` applies at the Protocol101 model-matrix boundary,
and `admitted_feature_matrix` refuses non-`ADMITTED` model inputs. This screen
does not build a model matrix, fit a learner, or admit a feature. It may measure
an IV-derived statistic from owned historical bid/ask prices as raw research.

This ruling does **not** make H1 model-facing alpha. The resulting statistic
remains barred from fitting, live inference, and deployment until its complete
historical/live parent chain is separately admitted. No output of this screen
may bypass that admission law.

## 4. Frozen H1 construction

At each completed OPRA CBBO-1m boundary `b`:

1. Use same-session SPXW quotes only.
2. Require each input quote's `ts_event <= ts_recv == b` and quote age no more
   than 90 seconds. Missing, crossed, nonfinite, or late rows are unavailable.
3. Estimate option-implied spot from exact-strike call/put parity using the
   shared estimator in `v4/research/pathd_opra_parity_features.py`.
4. Use the shared 4% risk-free rate, 0% dividend yield, expiry clock, IV solver
   domain, and iteration count from that file. The 5% constant in the older
   canonical model contract is not used.
5. Define target wings at exactly `implied_spot - 20` and
   `implied_spot + 20` points.
6. Linearly interpolate put IV and call IV from the immediately bracketing
   five-point strikes. Exact target strikes need no interpolation. No
   extrapolation and no bracket wider than five points are allowed.
7. Define `RR_b = put_IV(implied_spot-20) - call_IV(implied_spot+20)`.
8. Require an exact H1 snapshot at `b-5m`; never substitute a nearest boundary
   or carry an earlier value forward.
9. Define `score_b = -(RR_b - RR_(b-5m))`.
10. Map positive score to `CALL/+1`, negative to `PUT/-1`, and exact zero or
    unavailable input to `WAIT/0`. There is no deadband or threshold.

Every selected parity pair, wing symbol, bracket strike, interpolation weight,
solver input, failure reason, and source identity must be reconstructable.

## 5. Frozen clock and sample

- Corpus: the first 215 chronological sessions returned by
  `development_sessions`. The final 36 sessions are the spent holdout and their
  contents must not be opened.
- Candidate boundaries: every minute from 10:00 through 15:50 New York time.
- Option feature boundary: completed CBBO-1m `ts_recv == b`.
- Decision emission: `b + 2,336 ms`, preserving the existing conservative
  Path-D emission law. This is a historical-screen clock, not a feature-
  admission receipt for production.
- SPX label start: latest completed official SPX close available at decision
  emission under the existing bar-open-plus-one-minute rule and 90-second
  staleness cap.
- SPX label end: latest completed official SPX close available at
  `decision_emission + 5m` under the same rule.
- Momentum comparator: sign of the available SPX close at decision minus the
  available close at `decision-5m`.
- Every datetime-to-integer conversion explicitly pins nanoseconds.
- All joins are backward-only by availability time.
- Missing values are never imputed or carried forward.

## 6. Serial occupancy

H1 and the comparator each replay the same common-valid boundary set through
their own one-account state. A non-WAIT action entered at `t` occupies the
account on `[t,t+5m)`. Boundaries strictly before `t+5m` are logged as skipped.
At exactly `t+5m`, the previous interval closes before the next action is
considered. All positions are flat by 15:55.

The primary metric is mean signed SPX points per session. The secondary metric
is signed SPX points per executed interval. Both are reported separately so a
high-frequency per-trade average cannot conceal occupancy dilution.

## 7. Folds, coverage, and gate

Use the existing five chronological `entry_expanding_folds` over the 215-session
development prefix. The initial 44 sessions and one-session fold embargoes are
not evaluation outcomes. No pooled result can rescue a fold.

Coverage fails closed if:

- more than 10 development sessions are unusable;
- a usable session has fewer than 30 common-valid pre-occupancy boundaries;
- any evaluation fold retains fewer than 30 sessions;
- an exclusion depends on a return or strategy outcome; or
- source, clock, unit, join, solver, mutation, or sign-identity checks fail.

H1 returns `DIRECTION_SCREEN_PASS` only if all four conditions hold in **every
one of the five folds**:

1. mean H1 signed SPX points per session is strictly positive;
2. mean H1 signed SPX points per executed interval is strictly positive;
3. H1 mean session points strictly exceeds momentum mean session points; and
4. H1 mean interval points strictly exceeds momentum mean interval points.

Any economic failure in any fold is terminal `NO_EDGE` for this frozen H1
family. A clock or reconstruction failure is `INVALID_EXPERIMENT`. A pooled
positive result, favorable time window, absolute-move result, opposite sign,
deadband, different wing distance, different horizon, or threshold cannot
rescue it.

## 8. Required diagnostics, not rescue paths

Report session inventory, rows and distinct contracts, missing partitions,
eligible and executed boundaries, overlap skips, CALL/PUT/WAIT counts, quote-age
distribution, parity/solver failures, bracket turnover, contemporaneous signed
points, absolute future moves, exact sign-reversal identity, all five fold
ledgers, and source/code/preregistration hashes.

No null, permutation test, surrogate, learner, or wrapper is run in this screen.

## 9. Conditional next step

`DIRECTION_SCREEN_PASS` permits only a separately frozen, nearest-ATM one-
contract SPXW wrapper screen at the identical decisions. It is not confirmation,
model admission, paper readiness, or permission to trade. `NO_EDGE` stops this
family without widening it.
