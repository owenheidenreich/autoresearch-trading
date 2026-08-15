# Draft amendments to the compact shared lifecycle protocol — awaiting owner signature

**Status: DRAFT. Nothing here is adopted.** Prompted by the 2026-08-15 external adversarial review
(reports under `external-review/chatgpt-research/8-15-26/`) and calibrated by the capacity
measurement in
[`CAPACITY_KNOWN_ANSWER_2026_08_15`](../../research/findings/CAPACITY_KNOWN_ANSWER_2026_08_15.md).
Each amendment tightens the frozen protocol; none loosens a gate, adds a retry, or expands the search
space. If signed, they apply to `COMPACT_SHARED_LIFECYCLE_PROTOCOL_V1.md` as a V2 revision.

## A1 — Known-answer power gate (measurement-backed, decisive)

> Before any real target is constructed, the complete frozen pipeline must demonstrate, at its actual
> per-fold training sizes, at least 80% recovery of the smallest effect the final gate is intended to
> accept and a null false-pass rate of at most 5%, in a declared known-answer harness. Failure ends
> the experiment as `UNDERPOWERED` without reading real economics, exactly as G1's precedent orders.

Consequence today: the 120-parameter fit is refused — the V2 receipt measures 23–40% small-edge
recovery across every training size the proposed backfill can produce. This amendment converts that
from an argument into standing machinery.

## A2 — Capacity by measurement, not ratio (measurement-backed)

> No parameter budget may be authorized from an observations-per-parameter ratio or from an evidence
> budget computed on a corpus larger than the fit's own training data. A capacity claim for a given
> architecture, training size and training law is established only by measured recovery in the
> known-answer harness of A1.

Consequence: retires the 20:1 rule and the row-43 full-corpus convention for this packet's pipelines;
the ratio produced an answer measured to be wrong in the optimistic direction.

## A3 — Research-exposure firewall on score blocks (referee-verified, independent of A1)

> Before any acquired outcome is opened, every session is classified `DEVELOPMENT` or `CONFIRMATION`
> in a hashed exposure ledger. Sessions whose economics, labels, predictions or diagnostics informed
> any design decision — including the 251-session quote corpus used throughout 08-13/08-14 — are
> `DEVELOPMENT` and are excluded from inferential scoring. If no chronologically valid untouched score
> set remains, the result is exploratory and can neither authorize nor close the strategy class.

Verified fact behind it: chronological sorting places the heavily-reused 251 sessions inside the late
score blocks of the frozen design.

## A4 — Serial daily breaker (referee-verified conflict with row 21)

> The daily loss limit binds the serial account ledger. Once realized daily loss plus the worst-case
> remaining loss of any open position reaches the controlling breaker, no new entry is legal that
> session. Permission for two entries per session does not override the breaker.

## A5 — Duration-matched exit control (referee-found gap; echoes the stopwatch finding)

> On identical outer-fold entries, the learned exit must beat both a fixed-clock ladder and a control
> matched to its own out-of-fold holding-time distribution, paired, at midpoint and at the touch.
> Otherwise exit value is attributed to time in position, not to the model.

## A6 — Quote admissibility and decision clock (referee-found gap)

> Freeze before outcomes: no forward-filled quotes, bounded quote age, non-crossed bid/ask, positive
> displayed size, deterministic duplicate/revision handling, and entry/exit at the first eligible
> quote strictly after the order request. A decision for the bar ending at time t may read only
> records received by the signed arrival boundary; still-forming bars and later revisions are
> unavailable.

## A7 — Primary estimand and clustered inference (referee-found gap)

> The primary outcome is serial account P&L per scored session, with no-trade sessions counted as
> zero. All absolute and paired intervals resample whole sessions. Alpha, the multiplicity family,
> weighting, pairing and zero-trade handling are frozen before outcomes. Per-trade means are
> diagnostic only.

## A8 — Complete semantic freeze and closure scope (referee-found gaps)

> The post-acquisition declaration hashes the acquired-session manifest, QC and exclusion rules, the
> eligible-contract mask, feature and target code, quote/fee/settlement laws, model source,
> preprocessing, optimizer, seed, checkpoint rule, fold boundaries, control matcher, null generator
> and inference code; any semantic difference cancels the experiment. A failed run closes only the
> declared staged specification; it may close the long-option branch only if a joint entry-exit
> known-answer test first shows the staged law can recover an interaction edge.

## Not adopted from the referee

- Its amendment 5 (divergence-register gate) already exists as repo machinery
  (`research/divergence.py` with blocking axes tracked in STATUS §15); binding it into this packet's
  declaration is covered by A8's hash list rather than a new clause.
- Its per-fit *linear-scaled* budget formula is superseded by A2 — measurement replaced the formula it
  proposed as well as the one it attacked.

## What signing does and does not do

Signing adopts tighter law. It does not authorize a fit, a purchase, or vendor contact; the
120-parameter fit remains refused by A1 on today's evidence, and any future data request must arrive
with a rehearsal that passes.
