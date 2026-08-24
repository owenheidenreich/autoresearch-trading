# Unconditional SPX race, winner path, and IV clock — 2026-08-23

## Verdict

**[VERIFIED] The earlier favourable-touch rate was materially too generous for a stop-constrained
interpretation.** At 09:31 over 20 minutes, the motivating call cell reproduces exactly: +10 was
observed before -5 on **240/1,011 = 23.7% [21.2%, 26.5%]** of sessions, -5 arrived first on
**495/1,011 = 49.0% [45.9%, 52.0%]**, and neither barrier was observed on
**276/1,011 = 27.3% [24.6%, 30.1%]**. The full surface now exposes those three states for eight
favourable and eight adverse barriers, every declared clock/horizon, both sides, and both eras plus
pooled history.

**[VERIFIED] Barrier geometry, more than call-versus-put direction, controls the race.** Over 20
minutes, a symmetric 5-point race gives the favourable side a 28.1%–44.9% share across the selected
clock/side cells below. Requiring +10 before allowing only -5 cuts that to 9.1%–23.7%. Requiring +20
before -10 cuts it to 1.3%–8.7%, with 63.5%–88.7% unresolved. Exact swapped-threshold call/put counts
hold over all 38,400 cells, so these are underlying path-geometry statements, not directional alpha.

**[VERIFIED] The winners still have meaningful conditional magnitude, but most do not arrive
immediately.** For +10-before-5 winners over 20 minutes, selected pooled median overshoot is
3.2–6.7 SPX points and the 90th percentile is 10.2–20.7 points. An eventual +10 touch typically takes
9–13 minutes at the median and 17–20 minutes at the 90th percentile. Those distributions narrow the
oracle maximum-touch upper bound, but they are still underlying-path inputs rather than option
payoffs.

**[VERIFIED] The alarming unconditional adverse-first count does not mean most eventual winners
first breached the stop.** In the 09:31 call example, 282 sessions eventually reached +10; 240 did
so before -5 and 42 only after -5. Thus **85.1% [80.5%, 88.8%]** of eventual +10 paths survive a
strict -5 underlying stop, even though -5 arrives first in 495 sessions overall. The denominators
answer different questions. Across the selected clocks/sides, a strict 5-point stop preserves
80.3%–98.2% of eventual +10 paths; a 10-point stop preserves 93.1%–98.2%.

**[VERIFIED] A flat 13% IV does not describe the corpus's ATM clock shape.** The primary
side-balanced ATM10 median is **9.96% [9.61%, 10.45%] at 09:35**, bottoms near
**8.86% [8.66%, 9.38%] at 13:30**, reaches **10.64% [10.31%, 11.11%] at 15:30**, and rises to
**14.27% [13.96%, 14.67%] at 15:55**. A paired same-strike closest-ATM sensitivity has the same
qualitative clock shape and differs by at most 0.585 volatility points in median level. All 1,011
sessions are present at all 16 clocks. The primary ATM10 filter excludes 41 solver-bound candidate
rows across 41 session-clocks, beginning at 13:30. The sensitivity's whole-ladder pairing filter
excludes 1,024 candidate rows across 824 session-clocks, beginning at 09:31; neither filter loses a
session-clock estimate.

**[INFERRED] In owner language, the underlying gives a long option a plausible race only when its
required favourable move is commensurate with the adverse room and available time.** A 5-versus-5
race frequently resolves and is approximately balanced by construction. A +10-versus-5 or
+20-versus-10 demand is hostile: the adverse barrier arrives first far more often than the favourable
one, and larger late/midday moves are usually not observed at all within 20 minutes. Extending the
horizon resolves more paths but does not reverse the asymmetric barrier disadvantage.

**[UNKNOWN] This still is not option expectancy, and no cell is proven “unwinnable regardless.”** No
historical option P&L, bid/ask exit path, percentage stop, theta/charm/color repricing, contract
selection, or loss magnitude was read. `NEITHER` means neither *underlying* barrier was observed; it
does not prove an option died from theta. The measured ATM curve cannot supply strike-specific IV for
25-, 40-, or 60-point OTM contracts. Therefore a break-even success rate and expected dollars remain
unknown, and no strategy or entry rule follows.

**ADOPT NOTHING.** This is an explicitly owner-authorized, unconditional continuation of Job 48,
not a signal test, option-outcome run, alpha experiment, or strategy proposal.

## Evidence and population

**[VERIFIED]** Canonical evidence is the self-hashed
[attempt-001 receipt](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/receipt.json)
(`12c51f2d7dc3453108fde93ac1c2a5e12ed2d6350be36f0ac36090b20b2cd703`), the complete
[race surface](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/race_surface.csv),
[overshoot distribution](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/overshoot_distribution.csv),
[time-to-event distribution](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/time_to_event_distribution.csv),
[stop compatibility table](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/stop_compatibility.csv),
[conditional quantiles](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/conditional_quantiles.csv),
[IV clock table](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/iv_by_time.csv),
[paired IV changes](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/iv_clock_change.csv),
and [readable tables](../../../v4/audit/autoresearch/unconditional_spx_race_2026_08_23_attempt001/readable_tables.md).

**[VERIFIED]** Independent post-run verification reproduced the receipt self-hash; bytes, SHA-256,
and declared row counts for all 15 artifacts; bytes and hashes for four upstream receipts; and bytes
and hashes for every source named by the 2,028-row combined manifest. The manifest binds exactly
1,014 parity-tape and 1,014 ladder files. It records 2,022 analyzed family/session files and six
whole-family exclusions—the same three frozen sessions excluded independently from tape and ladder.
All 12 semantic-freeze sources reverified against anchor
`71463cc0eeb4e242e43307e19a926ea4e258fd45a254c389e5a27110923893c4`.

**[VERIFIED]** The analyzed population is 1,011 pre-reservation sessions: 768 backfill-era and 243
owned-era, from 2022-06-01 through 2026-07-30. The known 2023-06-26, 2023-10-19, and 2023-10-25
whole-book freezes are excluded before tape prices or ladder IV rows are opened; vendor-padded
2022-11-25 is absent upstream. No session on or after the 2026-08-06 confirmation cutoff is present
or used.

**[VERIFIED]** The receipt's `outcome_conditioning=false` and
`NONE_EXACT_CLOCK_AND_HORIZON_ONLY` refer to entry/window selection and external signal, label, or
option-outcome reads. They do not erase the explicitly declared post-entry event denominators:
overshoot conditions on favourable-first, while time-to-M and pre-M adverse excursion condition on
an eventual M touch; every such denominator is carried in the machine tables.

**[VERIFIED]** Path entry is the exact parity observation at clock `t`; each window uses future
snapshots `t+1` through `t+H`, including an exact 16:00 endpoint and refusing any cell that would end
after 16:00. The parity tape contains one scalar snapshot per minute (`O=H=L=C`, volume zero), so a
touch is first observed one-minute crossing. Continuous-time touches and within-minute order are not
detectable.

**[VERIFIED]** The path tables contain 38,400 three-state race rows, 384,000 fixed-threshold
overshoot rows, 142,320 time-CDF rows, 38,400 stop rows, and 384,000 conditional-statistic rows. The
IV tables contain 97,056 session-clock-estimand rows plus 2,304 level and 2,304 paired-change rows.
QC reproduced every probability and Wilson endpoint from counts, enforced exact schemas and grids,
proved state partitions and threshold/horizon monotonicity, proved swapped call/put identities, and
required the strict-stop survivor count to equal the race favourable-first count at every J.

**[VERIFIED]** Every probability cell uses one Bernoulli value per session. Conditional means use a
session-value t interval; quantiles use exact binomial order-statistic intervals. When a sparse tail
cannot support a finite 95% endpoint, the machine row says `MEASURED_WITH_OPEN_CONFIDENCE_BOUND`
rather than clamping to the sample maximum; fixed-threshold CDF/survival rows retain Wilson
intervals. These are pointwise session-unit intervals. They are not adjacent-session serial bands,
simultaneous family bands, best-cell tests, causal era comparisons, or equivalence tests.

## The three-state race

**[VERIFIED]** Selected pooled 20-minute cells are below. Brackets are pointwise 95% session-unit
intervals. `Neither` is always in the denominator and means neither underlying barrier was observed
by the horizon.

| Start ET | Side | Race | Favourable first | Adverse first | Neither |
|---|---|---|---:|---:|---:|
| 09:31 | Call | +5 / -5 | 44.9% [41.9, 48.0] | 43.2% [40.2, 46.3] | 11.9% [10.0, 14.0] |
| 09:31 | Call | +10 / -5 | 23.7% [21.2, 26.5] | 49.0% [45.9, 52.0] | 27.3% [24.6, 30.1] |
| 09:31 | Call | +20 / -10 | 6.5% [5.2, 8.2] | 28.7% [26.0, 31.5] | 64.8% [61.8, 67.7] |
| 11:30 | Call | +10 / -5 | 11.6% [9.7, 13.7] | 34.7% [31.8, 37.7] | 53.7% [50.6, 56.8] |
| 13:30 | Call | +10 / -5 | 9.1% [7.5, 11.0] | 31.2% [28.4, 34.1] | 59.7% [56.7, 62.7] |
| 13:30 | Call | +20 / -10 | 2.3% [1.5, 3.4] | 11.0% [9.2, 13.1] | 86.7% [84.5, 88.7] |
| 15:30 | Call | +10 / -5 | 10.1% [8.4, 12.1] | 34.3% [31.5, 37.3] | 55.6% [52.5, 58.6] |
| 15:30 | Put | +10 / -5 | 13.2% [11.2, 15.4] | 33.5% [30.7, 36.5] | 53.3% [50.2, 56.4] |
| 15:30 | Call | +20 / -10 | 2.4% [1.6, 3.5] | 13.6% [11.7, 15.9] | 84.0% [81.6, 86.1] |

**[VERIFIED]** More time resolves `NEITHER`, as it must, but the asymmetric race remains adverse.
For a 09:31 call, +10/-5 moves from 23.7% favourable / 49.0% adverse / 27.3% neither at 20 minutes
to **34.4% [31.6%, 37.4%] / 61.0% [58.0%, 64.0%] / 4.5% [3.4%, 6.0%]** at 90 minutes. At
13:30, +20/-10 moves from 2.3% / 11.0% / 86.7% at 20 minutes to
**11.5% [9.7%, 13.6%] / 38.0% [35.0%, 41.0%] / 50.5% [47.5%, 53.6%]** at 90 minutes. The longer
horizon raises both resolved states; it does not convert tight adverse room into a favourable race.

**[INFERRED]** The useful terrain boundary is therefore not a clock ranking. It is a compatibility
test between required move, adverse room, and hold. Five points each way often has a real two-sided
race. A contract/economic scenario that effectively requires +20 while tolerating only -10 has very
little 20-minute underlying support at any selected clock. This statement screens geometry only; it
does not select a contract or authorize an entry.

## Conditional magnitude and time

**[VERIFIED]** The table below fixes +10-before-5 over 20 minutes. Overshoot uses the maximum
favourable snapshot through the *whole remaining horizon* after conditioning on favourable-first;
reversal after the touch does not erase the maximum. Time and pre-M adverse excursion condition on
any eventual +10 touch, whether it arrived before or after -5.

| Start ET | Side | Favourable-first n | Overshoot q50 [95%] | Overshoot q90 [95%] | Time-to-10 q50 [95%] | Time-to-10 q90 [95%] | Pre-10 adverse q90 [95%] |
|---|---|---:|---:|---:|---:|---:|---:|
| 09:31 | Call | 240 | 5.0 [3.9, 5.9] pts | 17.5 [15.6, 20.3] pts | 10 [9, 11] min | 18 [17, 19] min | 7.6 [5.6, 9.1] pts |
| 09:31 | Put | 237 | 6.7 [5.5, 8.2] pts | 20.7 [15.8, 27.1] pts | 9 [8, 10] min | 17.1 [17, 19] min | 7.2 [5.8, 9.4] pts |
| 13:30 | Call | 92 | 3.2 [2.1, 5.0] pts | 16.6 [12.8, 28.8] pts | 10.5 [9, 12] min | 18 [17, 19] min | 4.6 [3.9, 9.1] pts |
| 13:30 | Put | 109 | 3.6 [2.7, 4.5] pts | 10.2 [9.0, 17.5] pts | 12 [10, 14] min | 19 [17, 20] min | 2.7 [2.2, 3.9] pts |
| 15:30 | Call | 102 | 4.7 [3.1, 7.1] pts | 13.8 [12.6, 19.6] pts | 13 [10, 15] min | 20 [18, 20] min | 3.7 [2.9, 8.2] pts |
| 15:30 | Put | 133 | 3.9 [2.9, 5.5] pts | 16.1 [11.9, 30.4] pts | 11 [9, 13] min | 18 [17, 19] min | 3.8 [3.2, 4.7] pts |

**[VERIFIED]** The fixed-threshold tail confirms that the conditional mean is not enough. Among
selected +10-before-5 winners, **36.5%–59.1%** overshoot by at least another 5 points,
**11.9%–32.9%** overshoot by at least 10, and **1.8%–11.0%** overshoot by at least 20. Every one of
those proportions has its own Wilson interval in the complete CSV. Extreme q99 point estimates are
large, but their upper 95% order-statistic endpoints are open in these conditional samples; the data
do not justify a finite tail cap.

**[INFERRED]** Winner paths are economically heterogeneous: many barely clear M, while a small tail
continues much farther. Most eventual +10 moves also consume roughly half or more of a 20-minute
hold. That makes a terminal maximum-touch calculation an upper bound twice over—it assumes both an
oracle exit at the maximum and ignores the option decay incurred before a late move arrives.

**[UNKNOWN]** The overshoot cannot be multiplied by delta or repriced with one IV number to obtain
historical P&L. Delta, gamma, theta, charm, color, spread, and IV evolve along the option path; none of
those historical contract paths was read here.

## What the adverse path says about stops

**[VERIFIED]** Pre-M adverse excursion is `max(0, -signed_path)` through the first +M snapshot. A
first-minute hit therefore has zero adverse excursion. A strict -J underlying stop survives only
when this value is `< J`; an exact touch at -J is killed. The identity
`stop survivor count = favourable-first race count` passes at every M/J/clock/horizon/scope.

| Start ET | Side | Eventual +10 n | Survive -5 [95%] | Survive -10 [95%] |
|---|---|---:|---:|---:|
| 09:31 | Call | 282 | 85.1% [80.5, 88.8] | 94.3% [91.0, 96.5] |
| 09:31 | Put | 290 | 81.7% [76.9, 85.7] | 94.5% [91.2, 96.6] |
| 13:30 | Call | 102 | 90.2% [82.9, 94.6] | 96.1% [90.3, 98.5] |
| 13:30 | Put | 111 | 98.2% [93.7, 99.5] | 98.2% [93.7, 99.5] |
| 15:30 | Call | 112 | 91.1% [84.3, 95.1] | 97.3% [92.4, 99.1] |
| 15:30 | Put | 139 | 95.7% [90.9, 98.0] | 97.8% [93.8, 99.3] |

**[VERIFIED]** The 09:31 call denominator mechanism is explicit. The race has 495 adverse-first
sessions, but only 42 of the 282 eventual +10 sessions first touched -5; the other 453 adverse-first
paths never reached +10 by the horizon. This reconciles the 49.0% unconditional adverse-first rate
with 85.1% conditional stop survival and rules out an aggregation defect.

**[INFERRED]** An underlying stop tighter than five points conflicts with a nontrivial minority of
eventual +10 winners, especially near the open. Ten points preserves substantially more winners.
That is a stop-compatibility input, not a stop recommendation.

**[UNKNOWN]** A -5 or -10 SPX move cannot be mapped to a -40% option stop without the selected
contract, entry premium, Greeks, IV/spread path, and clock. No percentage option stop is validated or
adopted.

## Implied volatility by time of day

**[VERIFIED]** `self_iv` is the corpus's causal quote-time solve from bid/ask midpoint, parity spot,
strike, side, and minutes remaining. The primary session-clock value takes the call median and put
median separately inside `|moneyness| <= 10` points, then averages the two sides so unequal node
counts cannot reweight the level. Finite values at least five minutes from expiry and strictly inside
the pinned solver bounds `(1%, 500%)` are admitted. One session gets one vote.

| Start ET | Sessions | ATM10 p10 [95%] | ATM10 median [95%] | ATM10 p90 [95%] | Mean [95%] | Closest-pair median [95%] |
|---|---:|---:|---:|---:|---:|---:|
| 09:35 | 1,011 | 6.67% [6.50, 6.79] | 9.96% [9.61, 10.45] | 19.29% [18.19, 20.07] | 11.72% [11.36, 12.07] | 9.94% [9.61, 10.44] |
| 12:30 | 1,011 | 5.55% [5.31, 5.80] | 8.87% [8.55, 9.21] | 16.70% [15.73, 18.02] | 10.43% [10.07, 10.79] | 8.91% [8.51, 9.18] |
| 13:30 | 1,011 | 5.50% [5.35, 5.73] | 8.86% [8.66, 9.38] | 17.78% [16.92, 18.53] | 10.78% [10.37, 11.19] | 8.90% [8.61, 9.37] |
| 15:30 | 1,011 | 6.69% [6.42, 6.88] | 10.64% [10.31, 11.11] | 20.55% [19.55, 21.30] | 12.53% [12.11, 12.95] | 10.55% [10.12, 11.01] |
| 15:50 | 1,011 | 9.34% [9.13, 9.47] | 14.01% [13.56, 14.40] | 24.12% [22.99, 25.32] | 15.80% [15.34, 16.27] | 13.76% [13.24, 14.17] |
| 15:55 | 1,011 | 10.11% [9.93, 10.28] | 14.27% [13.96, 14.67] | 23.70% [22.96, 24.69] | 15.98% [15.58, 16.39] | 13.68% [13.30, 14.19] |

**[VERIFIED]** On paired sessions, the median ATM10 change from 09:35 is
**-1.19 volatility points [-1.34, -1.03] at 13:30**, **+0.39 [0.22, 0.54] at 15:30**, and
**+3.80 [3.59, 3.99] at 15:55**. Calls and puts show the same broad U-shaped clock. The paired
same-strike nearest-ATM sensitivity tracks the primary, making a strike-grid switch or unequal side
count an insufficient explanation for the shape. The primary ATM10 filter's 41 solver-bound
candidate-row exclusions begin at 13:30 and are sparse. The sensitivity applies the same interior-IV
filter across the entire stored ladder before nearest paired-strike selection, so its broader
diagnostic is 1,024 excluded candidate rows across 824 session-clocks beginning at 09:31. Neither
filter removes a session-clock estimate.

**[INFERRED]** The flat 13% companion assumption is above the measured ATM median through 15:45 and
below it in the final ten minutes. It also hides a wide, right-skewed cross-session distribution.
Replacing the flat number with the clock median may improve an ATM-level scenario, but that
recalculation belongs on the separately owned option-economics side.

**[UNKNOWN]** The persisted ladder is hard-bounded at +/-25 points. Forty- and sixty-point OTM IV are
absent, and the boundary -25 node is clipped rather than a full 25-point bucket. The ATM10 curve must
not be silently assigned to the companion's 25-, 40-, or 60-point OTM contracts. Strike-specific IV
for those scenarios requires a separately authorized full-chain reconstruction; it is not estimated
or extrapolated here.

## Era description

**[VERIFIED]** The machine tables report every path and IV figure separately for 768 backfill and
243 owned sessions. At 09:35 over 20 minutes, call +10-before-5 rises from
**18.8% [16.1%, 21.7%]** in backfill to **26.3% [21.2%, 32.2%]** in owned, while adverse-first also
rises from 45.2% to 55.6% and neither falls from 36.1% to 18.1%. The put favourable-first share moves
from **21.6% [18.8%, 24.7%]** to **29.2% [23.9%, 35.2%]**, with the same higher-resolution pattern.

**[VERIFIED]** ATM10 median IV is lower in the owned era at every selected readable clock: 09:35 is
**10.50% [9.93%, 11.04%]** backfill versus **8.71% [8.30%, 9.61%]** owned; 13:30 is
**9.86% [9.21%, 10.32%]** versus **7.19% [6.66%, 7.56%]**; 15:30 is
**11.47% [10.94%, 12.09%]** versus **8.62% [8.04%, 9.29%]**.

**[INFERRED]** The early path difference is consistent with the prior finding that fixed SPX-point
moves resolve more often at a higher index level. It is not evidence that the owned era offers a
cleaner directional race: both favourable and adverse states rise as `NEITHER` falls.

**[UNKNOWN]** Stability or drift is not identified. Acquisition source changes exactly with date,
the years at both ends are partial, the intervals are pointwise rather than serial/simultaneous, and
no equivalence margin was declared. Source and calendar cannot be separated with this corpus.

## Failure behavior and project health

**[VERIFIED]** The wrapper refuses a pre-existing attempt before mutation. After creating a new
attempt, source archival is inside the caught region: an archival, freeze, upstream, data, analysis,
QC, or artifact-write failure preserves the directory, all available archived sources, traceback
`run.log`, partial artifacts, and a self-hashed `failure_receipt.json`; CLI exits 1. A targeted test
proves the missing-data failure receipt, archived source/log hashes, traceback, and overwrite refusal.
The successful attempt archives the wrapper plus both custom analysis modules.

**[VERIFIED]** Targeted tests pass for clock boundaries, defect exclusion before reads, three-state
semantics, swapped call/put identities, monotonicity, overshoot/time/stop conditioning, strict stop
touches, sparse quantile bounds, exact Wilson/schema mutation refusal, side-balanced/paired IV,
solver-bound exclusion, IV population inference, and wrapper failure preservation. `check_project.py`
reports one front door, registered work, valid links, no v4 imports, and no stray planning files. The
canonical `v5/tests` suite passes **1,229 tests** with the same 105 pre-existing warning-class
emissions; the targeted race suite contributes 15 passing tests.

## What this could not determine

- **[UNKNOWN] Historical option expected P&L or break-even hit rate.** Underlying race, overshoot,
  time, and pre-M adverse distributions do not provide option win/loss dollars.
- **[UNKNOWN] A percentage option stop.** SPX points do not map to contract percentage drawdown
  without a selected contract and its full price/IV/Greek path.
- **[UNKNOWN] Strike-specific 25/40/60-OTM IV.** ATM10 is measured; the required wings are clipped or
  outside stored support.
- **[UNKNOWN] Continuous-time barrier order.** Scalar minute snapshots can miss touches and cannot
  order two within-minute events.
- **[UNKNOWN] Conditional opportunity.** No feature, signal, state, score, or label conditions a
  window. An authorized signal might enrich or worsen any unconditional cell.
- **[UNKNOWN] A best clock, global significance, equivalence, or stability.** The grid is descriptive
  and pointwise, not a multiple-comparison selection exercise.
- **[VERIFIED] No fit, option outcome/P&L read, alpha-ledger charge, reserved-session use,
  vendor/broker contact, download, spend, live action, order, strategy proposal, or adoption
  occurred. ADOPT NOTHING.**
