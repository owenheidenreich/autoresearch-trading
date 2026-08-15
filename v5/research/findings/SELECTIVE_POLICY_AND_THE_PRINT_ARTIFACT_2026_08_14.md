# The model learns, and what it learns is the tape

**2026-08-14.** The first learned selective policy this project has built, and the reason its
profit is not real.
[print-label receipt](../../../v4/audit/autoresearch/selective_policy_2026_08_14/receipt.json) ·
[fair-label receipt](../../../v4/audit/autoresearch/selective_policy_fair_2026_08_14/receipt.json) ·
code: [dataset](../../ops/build_decision_dataset.py), [policy](../../ops/train_selective_policy.py) ·
tests: [causality](../../tests/test_decision_dataset.py), [occupancy](../../tests/test_selective_policy.py)

## What this changes

**The experiment cannot be run correctly on the trade corpus, and that is a fact about the data
rather than about the strategy.** A model free to choose its own moment will harvest the noise in
last-trade minute bars before it finds anything about the market, because that noise is five times
larger than the toll it is trying to beat. Settling this needs the quote corpus, which we own.

## What was built

The system the owner described: a model that looks at greeks, time of day, price action and volume,
decides **whether** to trade at all, picks its own contract, and holds one position at a time.

- **2,379,757 candidate trades** over 1,045 sessions — every contract that actually printed at each
  five-minute decision point, out to 60 points of moneyness on both sides. Roughly 2,277 candidates
  per session, and the model may take none of them.
- **Thirty causal features**, including per-contract and chain-wide **volume**, which the corpus has
  carried since download and which no screen in this project had ever read.
- **The chain is open.** The near-ATM band was a charter choice, and the measured break-even is not
  flat across it, so the model chooses moneyness rather than inheriting it.
- **Cost scales with the contract**, interpolated through the six measured points of the moneyness
  study. A flat round trip across an open chain teaches a model to buy the cheapest paper on the
  board, which is how the percentage-excursion entry failed in the opposite direction.
- **Selectivity is the object of study.** Rather than "take the top 5%", the threshold that achieves
  a target hit rate on the *training* fold is carried to the next fold unchanged.
- **One position at a time**, so the equity curve is one an account could have carried.

## It learns

Against a plain random selection it is not close: **52.0% hit rate against 31.8%**, and the more
selective it is asked to be, the better it does — a monotone curve, not one lucky cell.

It also survives the control that matters. A random selection matched on *side, delta and premium* —
same instrument, random moment — loses at every operating point. On the print label the model made
**+$20.9/trade against −$47.1** for that matched control, and it was **positive in all four
out-of-sample years** while the control was negative in all four.

That is the strongest result this project has produced. It is also wrong.

## And what it learns is the tape

These are last-trade prints, not quotes. A contract trades across a spread, so a print lands
sometimes on the bid and sometimes on the ask. Put/call parity gives an independent value for every
contract from the other side of its own strike, and the gap between the two — the **parity
residual** — has a standard deviation of **$102.60**. The average round trip is **$19.79**. *The
noise in the entry price is five times the cost the strategy is trying to beat.*

Sorting all 2.34M candidates by that residual shows what it is worth:

| Print sat | Residual | Net on prints | Net at fair value |
|---|---:|---:|---:|
| most below fair value | −$114.4 | **+$4.0** | **−$100.3** |
| middle | −$0.8 | −$22.5 | −$23.2 |
| most above fair value | +$73.3 | **−$50.3** | **+$19.3** |

The relationship **inverts**. Buying prints that landed on the bid looks profitable and is not: it
books the half-spread that only someone who did not have to pay it could collect.

The model found this before it found anything else. Its mean parity residual is **−$20.4** against
**−$2.0** for the matched control, and scoring the identical trades at parity-averaged value turns
**+$20.9 into −$2.1**. The edge equals the distortion it selected.

### Retraining on the clean label does not fix it — it inverts it

Trained on parity-averaged value instead, the model posts **+$21.8 to +$61.3/trade** with intervals
clearing zero. Its residual is now **+$37.6 to +$77.1** — it has learned to select contracts whose
*parity twin* print is stale, which understates the entry and inflates the same statistic from the
other side.

The relationship is mechanical:

> **net = parity residual − $16.5, with a standard deviation of $1.3** across cells whose residuals
> span +$37.6 to +$77.1.

A market edge does not track a price distortion to within a dollar and a half. That is an identity.

## Why this was not caught by the usual controls

Every guard this project has built was in place and passed. The features are causality-tested by
mutating everything after the decision minute and asserting no feature moves. Validation is
chronological. Thresholds come from the training fold. The shuffled-label null is negative. The
composition-matched control is negative. Four of four out-of-sample years are positive.

**None of that can see this defect, because the contamination is in the price, not in the features
or the split.** The label itself is measured through the spread. It is the same lesson as the
2026-08-13 leak in a new place: the control that finds a defect is the one aimed at the mechanism,
and a defect nobody has named yet has no control pointed at it.

## Two defects found in the machinery

**A precision floor of fifty rows lets noise set the threshold.** The shuffled-label null found a
chance precision spike near the top of its own scores, latched a threshold there, and posted
+$113/trade on 224 trades. Raised to 2,000 rows, with a regression test.

**The plain random reference was not comparable.** It drew from the whole chain — delta 0.02 paper at
$1,109 — while the model held delta 0.66 paper at $3,287, which credits the model for picking an
instrument rather than a moment. The matched-composition control replaced it, and it is the control
that did the work here.

## What is actually open

Under fair pricing the model earns **+$39.5/trade gross against a $41.6 round trip** — and that
figure is itself measured through contaminated prints, so it is an upper bound rather than a result.
The gross edge is 1.20% of premium and the toll is 1.27%. Because both scale with premium, moving to
cheaper contracts does not help: net stays between −$2 and −$8 across the whole chain.

**The one measurement that would settle it uses data we already own.** The quote corpus carries bid,
ask and size every minute for 251 sessions. Priced there — entry at the ask, exit at the bid — no
amount of selection can conjure a spread that is not there, because the spread is charged rather
than inferred. If the timing signal survives that, it is real. If it does not, this class is closed
properly rather than by argument.

The cost is 251 sessions instead of 1,045, which is the whole of the trade-off: four times less data,
and a price that cannot be gamed.

## Honest limits

- The parity fair value is itself built from prints, including forward-filled ones, so it reduces
  bounce rather than removing it. It is a better estimate, not a clean one.
- 2022 sits inside the 300-session warm-up and is never scored, so the one bear year is untested.
- The model chose contracts averaging $3,287 of premium — 33% of a $10,000 account, far outside the
  charter's ticket ceiling. Even a real edge at that size is not deployable as configured.
- Deep in-the-money 0DTE contracts are thin. Whether the interpolated round trip is right for them is
  not established; the moneyness study measured spreads, not fills, at those strikes.
- The exit is a fixed 30-minute clock. No exit model has been fitted on this stream yet.
