# Unconditional SPX move terrain — 2026-08-23

## Verdict

**[VERIFIED] The SPX tape often supplies the required *underlying move* early in the day, but it
rarely supplies the large move demanded by a cheap far-OTM contract late in the day.** In the fixed
13%-IV companion scenario, the scale-adjusted move required by a 25-point-OTM contract on a
20-minute hold was observed in 68.4%–74.8% of sessions at 09:31/10:30, depending on clock and side.
For a 60-point-OTM contract it was observed in 55.3%–65.7%. By 15:30 those ranges were 9.2%–11.4%
and 0.2%–0.8%, respectively. Every figure is an exact-session share with the pointwise session-unit
95% interval in the tables below.

**[INFERRED] Underlying-motion scarcity therefore does not, by itself, kill the whole long-option
class.** It is a serious screen against cheap, far-OTM, late-day contracts. It is not a profitability
finding for the early cells.

**[UNKNOWN] This census cannot determine whether any cell is “worth” buying or has positive expected
P&L.** The companion calculation supplies a zero-P&L *move threshold*, not a break-even *hit
probability*. A required-move touch is only a necessary condition under the fixed repricing scenario.
Expected value additionally needs the sizes and timing of wins and losses, terminal spot, IV changes,
actual spread/fills, and an exit law. A hit probability and a move threshold cannot simply be
multiplied into expected P&L.

**[UNKNOWN] The terrain is not established as stable or drifting.** Fixed-point rates differ
materially in some cells, but much of the early gap is mechanical index-level scaling. Remaining
changes have mixed signs, ordinary pointwise intervals are not a simultaneous drift test, 2022 and
2026 are partial years, and source is perfectly confounded with date. No equivalence margin exists,
so “no detected difference” would not prove stability either.

**ADOPT NOTHING.** This is an unconditional market-structure census, not a signal, entry rule,
contract selector, or strategy proposal.

## Evidence and population

**[VERIFIED]** Final evidence is the self-hashed [attempt-004 receipt](../../../v4/audit/autoresearch/unconditional_spx_move_terrain_2026_08_23_attempt004/receipt.json)
(`e0a1732f5e3c0352b6458eed2053c5b527eb1a33dd1dafcacd604a19c9c0afe4`), the complete
[excursion grid](../../../v4/audit/autoresearch/unconditional_spx_move_terrain_2026_08_23_attempt004/excursion_grid.csv),
[race grid](../../../v4/audit/autoresearch/unconditional_spx_move_terrain_2026_08_23_attempt004/race_grid.csv),
[contract necessary-condition grid](../../../v4/audit/autoresearch/unconditional_spx_move_terrain_2026_08_23_attempt004/contract_necessary_condition_grid.csv),
[comparison grid](../../../v4/audit/autoresearch/unconditional_spx_move_terrain_2026_08_23_attempt004/comparison_grid.csv),
and [readable tables](../../../v4/audit/autoresearch/unconditional_spx_move_terrain_2026_08_23_attempt004/readable_tables.md).
The receipt self-hash, every artifact and upstream-evidence hash, and every one of the 1,014 tape-file
size/hash bindings were independently rechecked after the run.

**[VERIFIED]** The tape contained 1,014 sessions from 2022-06-01 through 2026-07-30. The three known
interior-freeze sessions—2023-06-26, 2023-10-19 and 2023-10-25—were excluded whole before their
prices were read. Vendor-padded 2022-11-25 was already absent. The analyzed population was 1,011
sessions: 768 backfill-era and 243 owned-era; yearly counts were 144/239/245/241/142 for 2022–2026.
No session on or after the 2026-08-06 confirmation cutoff was present or used.

**[VERIFIED]** Each tape row is one parity-spot snapshot, not an intraminute OHLC candle. Tape bar
labels run 09:30–15:59 ET, while the actual observation minutes run 09:31–16:00; bar `t` contains the
snapshot from `t+1`. Windows use exact observation time and future snapshots `t+1` through `t+N`.
No timestamp is snapped, filled, truncated or encoded as a miss.

**[VERIFIED]** The fixed grid contains 100 observable clock/horizon cells across 16 exact starts,
eight horizons `{5,10,15,20,30,45,60,90}`, eight favourable thresholds
`{2,5,10,15,20,30,40,50}`, both directions, and adverse race barriers `{2,5,10,20}`. The adverse
barriers were fixed before reading outcomes because they span a tight wrong-way touch, medium stop,
and material loss without turning the grid into a threshold search. Late clock/horizon combinations
ending after 16:00 are explicitly `UNOBSERVABLE_PAST_16_00` in the receipt, not omitted losses.

**[VERIFIED]** Every reported cell has exactly one eligible window per session. Wilson intervals
therefore use session Bernoulli observations rather than treating overlapping intraday windows as
independent. Newcombe intervals compare disjoint session groups. They are descriptive pointwise 95%
intervals; they do not model adjacent-session serial dependence or support best-cell, simultaneous
family, significance-ranking, equivalence, or global drift claims.

**[VERIFIED]** QC passed on 32,000 excursion rows, 64,000 race rows, 14,000 companion rows and 36,528
comparison rows: zero denominator drift, zero threshold/horizon monotonicity violations, exact
call-favourable = put-adverse symmetry, complete race partitions, zero impossible same-snapshot
ties, and no gain-first count above its favourable-hit count. A separate raw-tape recomputation of
the pooled 09:31/20-minute call cell found 282/1,011 +10-point hits and reproduced its +10-before-5
partition exactly: 240 gain-first, 495 adverse-first and 276 neither.

## What the underlying actually does

**[VERIFIED]** Selected pooled favourable-excursion cells are below. Brackets are pointwise 95%
session-unit intervals; the complete declared grid is in the CSV.

| Start ET | Horizon | Move | Call/up | Put/down |
|---|---:|---:|---:|---:|
| 09:31 | 10m | 10 pts | 15.2% [13.1, 17.6] | 16.9% [14.7, 19.3] |
| 09:31 | 20m | 20 pts | 7.1% [5.7, 8.9] | 9.0% [7.4, 10.9] |
| 09:31 | 45m | 20 pts | 17.3% [15.1, 19.8] | 19.5% [17.2, 22.0] |
| 10:30 | 20m | 10 pts | 18.8% [16.5, 21.3] | 19.9% [17.5, 22.5] |
| 12:30 | 20m | 10 pts | 8.4% [6.9, 10.3] | 11.6% [9.7, 13.7] |
| 14:30 | 20m | 10 pts | 10.2% [8.5, 12.2] | 10.4% [8.7, 12.4] |
| 15:00 | 20m | 10 pts | 11.0% [9.2, 13.1] | 13.0% [11.0, 15.2] |
| 15:30 | 20m | 10 pts | 11.1% [9.3, 13.2] | 13.7% [11.8, 16.0] |
| 15:50 | 10m | 10 pts | 10.4% [8.7, 12.4] | 11.1% [9.3, 13.2] |

**[VERIFIED]** The clock shape is U-like rather than a monotone decay: a 10-point move in 10 minutes
was 15.2%/16.9% at 09:31, fell to 4.2%/4.1% at 12:30, and returned to 10.4%/11.1% for the final
15:50–16:00 window. Longer horizons raise raw touch probability monotonically, while larger moves
lower it monotonically. Those identities passed over the full grid.

**[INFERRED]** In owner language: ten points is an ordinary possibility over 20–45 minutes, especially
near the open, but 20–40 points is a tail over short holds. A contract needing 30–50 points late in
the day is asking the underlying for a rare event before its own option economics are considered.

## The adverse race matters

**[VERIFIED]** `P(+M first)` is unconditional: unresolved/neither sessions remain in its denominator.
These are first *observed one-minute snapshot* crossings, not continuous-time barrier order.

| Start ET | Horizon | Race | Call: +M first | Put: +M first | Neither |
|---|---:|---:|---:|---:|---:|
| 09:31 | 20m | +10 before −5 | 23.7% [21.2, 26.5] | 23.4% [20.9, 26.2] | 27.3% [24.6, 30.1] / 26.0% [23.4, 28.8] |
| 09:31 | 45m | +10 before −10 | 38.3% [35.3, 41.3] | 38.4% [35.4, 41.4] | 23.3% [20.8, 26.0] / 23.3% [20.8, 26.0] |
| 12:30 | 20m | +10 before −5 | 8.0% [6.5, 9.8] | 10.4% [8.7, 12.4] | 60.3% [57.3, 63.3] / 60.0% [57.0, 63.0] |
| 14:30 | 45m | +10 before −10 | 22.1% [19.6, 24.7] | 21.4% [18.9, 24.0] | 56.6% [53.5, 59.6] / 56.6% [53.5, 59.6] |
| 15:30 | 20m | +10 before −10 | 10.8% [9.0, 12.8] | 13.5% [11.5, 15.7] | 75.8% [73.0, 78.3] / 75.8% [73.0, 78.3] |

**[VERIFIED]** For the independently checked 09:31 call race, +10 before −5 occurred in 23.7%, but
−5 arrived first in 49.0% and neither arrived in 27.3%. Raw favourable-touch probability therefore
does not describe the path a stop-constrained long option experiences.

**[UNKNOWN]** No option stop, theta-based exit, or path-dependent option valuation was measured, so
the SPX race does not itself say when a contract would have been stopped, decayed out, or profitable.

## Necessary-condition join to the companion contract scenario

**[VERIFIED]** `STATUS.md` resolves a conflict in the prompt: **$17.92 is ES futures friction**, not
the option round trip. The companion option module uses $3.08 fees plus one crossed spread, or
$13.08 ATM and $23.08 OTM under its 0.10/0.20-point spread assumptions. It fixes SPX=6,800, IV=13%,
and OTM distances `{0,5,10,15,25,40,60}`; these are scenario assumptions, not reconstructed
historical surfaces. Another companion-label correction is mechanical: 330 minutes to a 16:00 close
means **10:30 ET**, while 09:31 has 389 minutes remaining.

**[VERIFIED]** The table below holds for the companion 20-minute scenario and reports the historical
share whose favourable SPX excursion met the scenario requirement after scaling that requirement by
each session's entry SPX level. This percentage version is the primary pooled-history comparison.

| Start ET | 25 OTM call | 25 OTM put | 60 OTM call | 60 OTM put |
|---|---:|---:|---:|---:|
| 09:31 | 74.8% [72.0, 77.4] | 72.6% [69.8, 75.3] | 65.7% [62.7, 68.5] | 65.0% [62.0, 67.9] |
| 10:30 | 70.6% [67.7, 73.3] | 68.4% [65.5, 71.2] | 57.3% [54.2, 60.3] | 55.3% [52.2, 58.3] |
| 12:30 | 56.3% [53.2, 59.3] | 55.0% [51.9, 58.0] | 29.5% [26.7, 32.4] | 30.3% [27.5, 33.2] |
| 14:30 | 38.5% [35.5, 41.5] | 33.8% [31.0, 36.8] | 6.6% [5.3, 8.3] | 6.0% [4.7, 7.7] |
| 15:00 | 25.5% [22.9, 28.3] | 26.7% [24.1, 29.5] | 2.4% [1.6, 3.5] | 2.3% [1.5, 3.4] |
| 15:30 | 9.2% [7.6, 11.1] | 11.4% [9.6, 13.5] | 0.2% [0.1, 0.7] | 0.8% [0.4, 1.6] |

**[VERIFIED]** The mechanism is contract geometry, not a tape anomaly. On this 20-minute scenario,
the 25-OTM requirement grows from about 2.27–2.29 points at 09:31 to 14.89–14.92 points at 15:30.
The 60-OTM requirement grows from 3.82–3.91 points to 45.98–46.31 points. The cheap far-OTM contract
has already lost most of its delta and must travel much farther to overcome $23.08 of fixed friction
and decay. The raw excursion grid's monotonicity and the direct cell recomputation rule out an
aggregation reversal as the source of the decline.

**[VERIFIED]** The $2,000 signed cap excludes the 09:31 ATM contract in this scenario. The first
under-cap call is 10 points OTM at $1,813; the first under-cap put is 5 points OTM at $1,937. At
10:30 the first under-cap contract is 5 points OTM on both sides. From 12:30 onward ATM is under the
cap. This confirms the companion finding that the capital cap mechanically moves the early action
set away from the lowest-hurdle strike; it does not amend the owner's cap ruling.

**[INFERRED]** The useful screening conclusion is asymmetric. Early, the under-cap 25–60 OTM
requirements are not rare enough for the underlying distribution alone to rule them out. Late, the
60-OTM requirement is so rarely touched that it has almost no terrain support. Nearer-money late
contracts have much smaller hurdles and cannot be rejected by borrowing the far-OTM result.

**[UNKNOWN]** A move threshold solved at the terminal hold time and an intrahold maximum do not form
a payoff distribution. A touch can occur early and reverse before the clock exit; IV can move;
spread can vary; a minute snapshot can miss a touch; and a non-touch can lose anything from a small
amount to nearly all premium. Without conditional win/loss magnitudes there is no break-even success
rate `L/(W+L)`, so neither “profitable” nor “structurally dead” follows from these hit shares.

## Era comparison

**[VERIFIED]** Fixed-point and scale-adjusted comparisons tell different parts of the mechanism. For
a 09:31 call to touch +10 within 20 minutes, backfill was 23.3% and owned was 42.4%, a descriptive
+19.1pp [12.3, 26.0] difference. But the scale-adjusted 25-OTM companion requirement over the same
clock/hold was 74.5% versus 75.7%, +1.2pp [−5.2, 7.1]; the 60-OTM version was 65.4% versus 66.7%,
+1.3pp [−5.7, 7.9]. The higher owned-era SPX level explains much of the apparent early fixed-point
increase.

**[VERIFIED]** Scaling does not make every later difference disappear. At 14:30/45 minutes, the
scale-adjusted 25-OTM call requirement fell from 32.2% backfill to 16.5% owned, −15.7pp
[−21.0, −9.6], while the corresponding put fell 27.3% to 20.6%, −6.8pp [−12.4, −0.5]. These are
pointwise descriptive contrasts among thousands of correlated cells, not a corrected family verdict.

**[VERIFIED]** The requested raw endpoint comparison is mixed by side and clock. For +10 within 20
minutes at 09:31, calls rose 34.7% in partial 2022 to 50.0% in partial 2026, +15.3pp
[3.8, 26.2], while puts moved 45.8% to 40.1%, −5.7pp [−16.9, 5.7]. A June–July common-support
comparison is included in the machine grid, but it has only 40 versus 41 sessions and correspondingly
wide intervals.

**[UNKNOWN]** A causal “market drift” conclusion is not identified. Backfill ends exactly where
owned data begins, so acquisition source and time cannot be separated. Mixed signs, partial-year
composition, serial dependence not modeled by the pointwise intervals, and grid multiplicity further
prevent a stable/drifting binary verdict. The only defensible state is `STABILITY_UNKNOWN`.

## What this could not determine

- **[UNKNOWN] Historical option expected P&L.** No historical contract was selected or repriced from
  its observed IV, spread, bid/ask path, or executable exit.
- **[UNKNOWN] Continuous-time touch and barrier order.** Snapshot OHLC has no intraminute range, so
  observed touch probabilities are lower bounds on any-touch probabilities and ambiguous intraminute
  double touches cannot be resolved.
- **[UNKNOWN] Conditional opportunity.** No feature, signal, prior state, label, or option outcome
  conditioned a window. This census neither proves nor disproves that a separately authorized signal
  can enrich a rare move.
- **[UNKNOWN] Global clock/era significance or equivalence.** Intervals are session-unit pointwise
  descriptions, not moving-block serial bands or simultaneous family bands; no equivalence margin
  was declared.
- **[VERIFIED] No fit, alpha charge, reserved-session use, vendor/broker contact, download, spend,
  order, or strategy adoption occurred.** The result is terrain only. **ADOPT NOTHING.**
