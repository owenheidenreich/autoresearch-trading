# FT2-04 Path-Label Freeze v2

Node: `FT2-04-PATH-LABEL-FREEZE`

Outcome: `labels_frozen_v2`

Prepared: 2026-07-29

Scope: definitions, role manifests, and synthetic verification only. This node
did not compute any label value or opportunity statistic from real market data.

## Why v2 exists

The FT2-20 parallel design review identified three defects in the original
label contract:

1. A decision at completed minute `t` was priced from the same minute.
2. Minutes without an executable bid were erased instead of represented as
   adverse path states.
3. One embargo session per outer fold remained in the census role.

The owner amended the consolidated authority to use next-completed-minute
execution, conservative no-bid valuation, a `$1.00` D49 soft-close floor, and
explicit embargo exclusion. The amended authority SHA-256 is:

`2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`

The complete pre-v2 FT2-04 packet remains byte-preserved under
`superseded/v1_pre_tplus1_20260729/`.

## Frozen two-step timing law

For a flat-state BUY decision at completed minute `t`:

- The order is committed at `t` and reserves the one-position slot.
- It fills at completed minute `e=t+1` using the executable ask `A_e`.
- Cash and premium accounting occur at `e`.
- If no executable `A_e` exists, the candidate has no valid entry label.
- The first descriptive future mark is `u=e+1=t+2`.

For an open-state EXIT decision at completed minute `v`:

- It fills at completed minute `x=v+1` using executable bid `B_x`.
- Occupancy and cash release at `x`.
- A 15:54 EXIT fills once at 15:55 and suppresses a second forced-flat event.

The last legal entry decision is 15:29, filling at 15:30. From 15:30 onward the
runtime is manage/exit-only. The last learned exit decision is 15:54, filling at
the 15:55 forced-flat boundary.

## Economic primitives

- Contract multiplier `M = 100`.
- Round-trip fee overlay `F = $3`, applied exactly once.
- Premium at risk `= A_e * M`.
- Fee-adjusted dollar PnL at a future mark:
  `PnL$(u) = (B_u - A_e) * M - F`.
- Fee-adjusted return:
  `R(u) = PnL$(u) / (A_e * M)`.
- Strict fee-adjusted profit:
  `B_u > A_e + F/M`.

Every path-property target is represented in both dollars per contract and
return on entry premium.

## Horizon and clock definition

Finite horizons are `3, 5, 10, 20, 45, 90` minutes plus
`remaining_session`.

For finite `h`:

`W(t,h) = {u : e < u <= e+h} intersect {u <= 15:55 ET}`, where `e=t+1`.

Therefore:

- a fully observed `h`-minute horizon contains exactly `h` descriptive marks;
- the first mark is `t+2`;
- primary time-to-first-profit is measured from the actual entry fill `e`;
- decision-relative time-to-first-profit is report-only.

Near close, the window is censored at 15:55. It is never extended, shortened
silently, or populated with invented future marks.

## No-bid law

A missing, stale, nonpositive, or otherwise non-executable bid is not dropped.
For every such minute:

- option value is conservatively `$0`;
- `PnL$(u) = -(A_e*M) - F`;
- the minute is underwater;
- it contributes full-loss depth to drawdown and underwater-burden metrics;
- it breaks positive runs;
- it cannot be a first-profit minute;
- no earlier bid is carried forward.

At 15:55, forced-flat uses that exact boundary state. If no executable bid
exists, the option realizes at `$0`; an earlier executable bid may not be
substituted.

Future evaluation must also emit an MNAR sensitivity comparison between
no-bid-as-loss and no-bid-excluded labels. Candidate ranking may not depend on
which convention is used.

## Six path-property families

The frozen families are:

1. Early drawdown: minimum and lower-tail path PnL over 3/5/10 minutes.
2. Time to first real profit: first strictly positive fee-adjusted mark.
3. Pre-profit adverse excursion: worst path state before first profit.
4. Underwater burden: depth, duration, integral, and longest underwater run.
5. Profitable-window stability: positive fraction, positive runs, and
   one-minute jitter sensitivity.
6. Upside: MFE, upper-tail quantiles, and positive area.

The exact machine definitions are in `label_spec.json`.

## Synthetic timing examples

Assume a BUY decision at 10:00, an actual ask fill at 10:01 of `$2.00`, and
future completed-minute bids:

| Time | Bid | Role | Fee-adjusted PnL |
|---|---:|---|---:|
| 10:01 | 1.95 | entry-fill minute, not a descriptive mark | n/a |
| 10:02 | 1.80 | first descriptive mark | `-$23` |
| 10:03 | no bid | conservative full-loss state | `-$203` |
| 10:04 | 2.10 | first real profit | `+$7` |
| 10:05 | 2.40 | later upside | `+$37` |

For the 3-minute horizon, the marks are 10:02, 10:03, and 10:04. Primary
time-to-first-profit is 3 minutes after the 10:01 fill. The no-bid state is
included in early drawdown and underwater burden and is never treated as a
profitable or executable exit.

For a 15:54 EXIT decision, the fill is the 15:55 bid. If no 15:55 bid exists,
the position realizes at zero option value and is not valued again by a second
forced-flat event.

## Oracle rules

FT2-05 may compute only these transparent, hindsight label-side references:

1. Best executable bid by horizon.
2. Hold to the exact 15:55 forced-flat boundary.
3. First strict fee-adjusted profit, otherwise exact-boundary forced flat.

These are opportunity ceilings and baselines, not model inputs or learned
policies. Any chosen oracle exit-fill minute `u` corresponds to an EXIT decision
at `u-1`.

Serial oracle replay obeys:

- actual `t+1` entry asks and exit bids;
- one open position;
- Simulator-v5 account continuity;
- D48 using the actual fill ask;
- D49 using the actual fill ask and realized session loss;
- the `$1.00` entry-premium soft-close floor;
- the 5% realized daily breaker;
- the 15:29 last-entry-decision boundary;
- the exact 15:55 forced-flat law.

If no otherwise eligible contract priced at least `$1.00` fits the remaining
D49 budget, the session soft-closes permanently to WAIT for new entries.

## Role firewall

The governed corpus has 301 sessions. The census role is:

`corpus - outer-test union - protected holdout - every fold embargo session`

Counts:

- Outer-test union: 225.
- Protected holdout: 30.
- Fold embargo sessions: 5.
- Lawful census sessions: 45.
- Census intersections with each forbidden role: 0.

The census dates are 2025-01-02 through 2025-03-10. The five excluded embargo
dates are 2025-03-11, 2025-07-08, 2025-09-11, 2025-11-14, and 2026-01-26.

`intersection_proof.json` proves the exact role equation and zero overlap.
The verifier's synthetic test suite passes 12 tests, including embargo,
holdout, outer-test, incompleteness, and ambiguous-role failures.

This early-2025 census is not regime-complete. FT2-05 must stratify its outputs
and may use this role only for feasibility and power guidance, never to certify
generalization.

## Deferred measurement

Historical minute data cannot identify a sub-minute fill. When lawful
development-class recorder evidence is next inspected, a report-only study may
compare the first executable quote around decision `+5s` against this contract's
next-completed-minute quote. Purchasing sub-minute historical data is deferred
to a separate owner decision in campaign 2.

## Integrity

- No real label values or census statistics were computed here.
- No model was trained, tuned, selected, or promoted.
- No outer-test or protected-holdout rows were read as market data.
- No recorder, broker, paid-data, runtime, launchd, or paper path was touched.
- Simulator v5 source was not modified.

## Highest allowed claim

> FT2-04 v2 freezes next-completed-minute execution, conservative no-bid path
> semantics, exact 15:55 forced flat, and a 45-session embargo-clean census
> role. It is ready for the bounded FT2-05 census rerun.
