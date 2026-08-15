# Emission-allowance re-derivation rule — pre-registration

**Status: DRAFTED 2026-08-12, AWAITING OWNER SIGNATURE. Not in force. No value has been re-derived and
none may be until this is signed.**

## What this is for in one sentence

The number that says how long the bot must wait after a minute closes before it is allowed to act on that
minute was measured on the wrong data feed, and we now have the right feed — but the rule for what to do
with it has to be fixed *before* anyone computes the answer, or the answer picks the rule.

## The knob

[`research/knobs.py`](../research/knobs.py) records `emission_lag_ms` as `UNCERTIFIED` at **2,336 ms**.

The clock law it serves is `decision_emission = t + emission_lag_ms`: after minute `t` closes, a decision
using minute `t`'s data may be emitted only that many milliseconds later. The current value is the p99 of
**five ThetaData samples**
([evidence](../../v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/shared_emission_lag.json),
`theta_p99_ns = 2335230000`, `theta_sample_count = 5`). ThetaData is not the feed the option features come
from, and five samples cannot support a percentile.

Its unfreeze condition reads:

> A multi-session Track-A OPRA capture re-deriving the allowance from the same stream the features come
> from. **It may not be lowered unless the lowering rule is pre-registered first.**

The Track-A capture completed 2026-08-12 and delivers the first half. This document is the second half.

## Which direction is dangerous

Lowering is the optimistic direction, and that is why the condition guards it.

| Change | Effect on the bot | Risk |
|---|---|---|
| **Raise** the allowance | Decides later, having waited longer for data | Understates edge. Conservative |
| **Lower** the allowance | Decides sooner, assuming data has arrived | **Backtests may read data that had not actually arrived — look-ahead** |

A backtest that emits a decision before the data was really there produces an edge nobody can trade. That
is the same class of defect as ledger row 183, arriving through the clock instead of through a feature.

## The honest limit on this pre-registration

**This document is not blind, and pretending otherwise would be worse than saying so.** The certification
issued 2026-08-12 already published the candidate inputs, and they are repeated in
[STATUS §13](../STATUS.md#13-track-a-capture-2026-08-06-failure-and-08-10-arming): envelope p50/p99/max of
**523.641 / 926.294 / 1004.346 ms** on `CBBOMsg:rtype=193`, plus the full per-window table. Anyone writing
a rule today can see roughly where each candidate rule lands.

So the protection a normal pre-registration gives — the rule cannot be reverse-engineered from the answer —
is **not available for the present-day decision**. What is still genuinely protected, and what this
document exists to fix in advance:

1. **Which statistic** defines the allowance (median, p99, max, or max-with-margin).
2. **Which record classes** count toward it.
3. **How truncated windows are treated.**
4. **That the rule is applied once**, with no second attempt on a different statistic if the first answer
   is unwelcome.
5. **The rule for a future, wider capture** — §5 below — which *is* written blind, because the sessions it
   governs do not exist yet.

## 1. The choice of statistic decides the direction, not the data

This is the single most important fact for the signature, and it is why the decision cannot be delegated
to whoever runs the command.

| Candidate rule | Reads | Versus the incumbent 2,336 ms |
|---|---|---|
| p99 of CBBO-1m arrivals | 926.294 ms | **Lowers it by ~60%** |
| Max of CBBO-1m arrivals | 1,004.346 ms | Lowers it by ~57% |
| Max across all captured classes | 1,406.018 ms (CBBO-1s, 08-10 open) | Lowers it by ~40% |
| 4 x max of CBBO-1m, borrowing the staleness guard's factor | ~4,017 ms | **Raises it by ~72%** |

(The certification's staleness guard is `max(10 s, 4 x p99)`. The factor 4 is borrowed from it; the
statistic is not, for the reason in the next paragraph.)

The measured evidence straddles the incumbent. **Choosing the statistic chooses the direction.** A rule
picked after seeing this table is a rule picked for its answer, which is precisely what the unfreeze
condition forbids — so the statistic must be chosen on principle, from what the number is *for*.

**What it is for:** guaranteeing that when the bot acts on minute `t`, minute `t`'s data is actually
present. That is a near-worst-case guarantee, not a typical-case one. A p99 allowance is wrong for this by
construction: it means one decision in a hundred fires against data that had not arrived. Central
tendency is the wrong family of statistic for a safety allowance.

## 2. Three reasons the measured numbers are a *lower bound*

Each of these makes the true allowance larger than what the capture reports. None is a defect in the
receipt; all are limits of what three windows in one week can see.

**Truncation at the window edge.** The freshness receipt records `interval_ends: 19` against
`expected_interval_ends: 21` — **two expected minute boundaries produced no CBBO-1m records at all**. A
record that would have arrived after the recorder stopped is absent from the sample rather than counted as
slow, so every percentile is conditional on having arrived inside the window. (This also corrects the
earlier reading of the 58.2% worst-window coverage as market sparsity; among the boundaries that did
report, coverage was 96.98%.)

**Only one record class was certified.** The latency receipt covers `CBBOMsg:rtype=193` alone. The same
capture recorded OHLCV-1m reaching **897.938 ms** at p99 and one-second CBBO reaching **1,406.018 ms** at
its max — higher than any CBBO-1m observation. Today every admitted feature reads only CBBO-1m, so the
receipt matches the ledger; the moment a barred class is admitted, an allowance derived from CBBO-1m alone
silently under-covers it.

**Three sessions, one week, one regime.** The issuance already records sample-max exceedance at **14.3%**
against a daily p95 and **3.0%** against a daily p99. No CPI, FOMC, opex or holiday-shortened session is in
the sample.

## 3. The rule, for signature

### Rule A — applies now

> The emission allowance is re-derived as the **maximum observed arrival lag, taken over every record
> class read by a feature admitted in the current ledger, over every usable window in the current
> capture, multiplied by a safety factor of 4, and rounded up to the next whole millisecond.** It is then
> adopted **only if it is greater than the incumbent value**. If it is lower, the incumbent is retained
> and the axis is recorded as settled-with-retention.

Fixed terms, so that no discretion survives to the moment of computation:

| Term | Definition |
|---|---|
| **Usable window** | A window the issuance counts as evidence. A window whose observed interval-ends are fewer than expected is **truncated**: usable for raising, excluded from any lowering |
| **Record class** | Every class read by a feature `ADMITTED` in the ledger at derivation time — auto-widening if a bar lifts, so the allowance can never fall behind the ledger |
| **Statistic** | Maximum, not a percentile. §1 |
| **Safety factor** | **4**, the same shape the certification's staleness guard already uses. Fixed here, not at run time |
| **Ratchet** | Adopt only if higher. Removes the incentive to tune, because no admissible rule choice can lower the number |
| **One shot** | Applied exactly once per capture. A second derivation on a different statistic is a protocol violation, reported in STATUS |

**Why the ratchet.** Because this document is not blind (see above), the only fully defensible present-day
rule is one that *cannot* produce the answer a motivated reader would want. Rule A settles the axis — the
allowance becomes backed by the right feed — while making a reduction structurally impossible on evidence
this thin.

**Expected consequence, stated so the signature is informed:** on the published inputs Rule A raises the
allowance rather than lowering it, to roughly four seconds. The "may not be lowered" clause therefore never
binds today, and signing Rule A is signing a *more* conservative clock, not a looser one. No value is
adopted by this document — the derivation is a separate run against the receipt bytes, and its output is
what enters `knobs.py`. Since the inputs are already public this consequence is foreseeable rather than
secret, and hiding it from the signature would be theatre, not protection.

### Rule B — written blind, for a future capture

> The allowance may be **lowered** only from a capture meeting all of: **at least 20 sessions**, spanning
> **at least two calendar months**, including **at least one CPI or FOMC session** and **at least one
> monthly opex session**, with **no truncated windows**, and covering **every record class read by every
> admitted feature**. The lowered value is the **maximum observed arrival lag across that capture times a
> safety factor of 2**, floored at **1,000 ms**, adopted only if lower than the incumbent.

Rule B is a genuine pre-registration: none of the sessions it governs exists, so it is written without
sight of its inputs. Its preconditions cannot be met by the 2026-08 capture, so signing it authorizes
nothing today. It exists so that the *next* capture has a lowering rule that predates it — the exact
protection unavailable now.

## 4. What signing does and does not do

**Does:** fix the statistic, the class scope, the truncation handling, the safety factor, the ratchet, and
the future lowering rule. Permits one derivation run.

**Does not:** authorize training, unfreeze any other knob, change the issued certification, alter the
admitted feature count, or move any gate. `emission_lag` binds at **G4**, which is blocked behind a G1 pass
the owned corpus cannot produce. **Nothing is unblocked by signing this.** It is done now only because the
evidence is fresh and the rule must precede the computation.

## 5. Expiry

This rule expires with the receipt that feeds it, **2026-11-10**. A derivation after that date requires a
current capture and a re-signature.

## Signature

- Drafted: **2026-08-12**, by Claude Opus 5, on owner instruction.
- Rule A — **UNSIGNED**
- Rule B — **UNSIGNED**

*Drafting this document computed no new allowance, contacted nothing, and changed no runtime state.*
