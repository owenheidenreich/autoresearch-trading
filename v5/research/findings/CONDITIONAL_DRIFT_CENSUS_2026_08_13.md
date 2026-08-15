# There is no state where buying pays: a census instead of another rule

**2026-08-13.** One measurement, no model, no rule, no selection.
[Receipt](../../../v4/audit/autoresearch/conditional_drift_2026_08_13/receipt.json) ·
code: [`ops/measure_conditional_drift.py`](../../ops/measure_conditional_drift.py) ·
[tests](../../tests/test_conditional_drift.py)

## What this changes

**The proposed entry-edge search (job 34) would have searched a space this measurement shows is
empty, and would have taken about forty runs to say so.** Across 375 declared cells covering every
observable the owned corpus supports, not one has a positive expected payoff that survives
correction. The job should be re-aimed before it is run, not run.

## Why a census rather than another screen

Every screen this project has run tested a **rule**: pick a signal, act on it, score the result. A
rule that loses closes only itself, which is why six campaigns have closed and the space still feels
open. Each negative invites "but you did not try *this* rule."

A trading rule is a **function of observable state** — of things the bot can see at the minute it
decides. So ask the question one level up:

> Is there **any** observable state in which the expected gross payoff of holding a long near-ATM
> SPXW 0DTE option is positive?

If the answer is no, then no function of those states can have positive expected gross either. Not
an entry rule, not an exit rule, not the two combined, not a model of any depth. **One pass closes
the whole rule space; forty rules close forty rules.**

This is the same argument that caps the optimal-stopping exit at zero. For any two decision times
`t_in <= t_out` on a process that drifts down, the expected gain is at most zero. A causal entry is a
decision time exactly as an exit is, so the cap applies to entries too. The job-34 packet reads the
exit's `gross ~ $0` as proof the exit is solved and profit must therefore come from the entry. That
inference does not hold: the result is a property of **the price process**, not of the exit half of
the trade, and it binds both halves equally. The only escape is a state where the process does not
drift down — which is what this measures.

## What was measured

| | |
|---|---|
| Corpus | 1,045 sessions, 2022-06-01 to 2026-07-31, **27,025 slots** |
| Payoff | fixed-horizon hold, 15 and 60 minutes, **no exit model** so nothing can attenuate the number |
| Sides | long call, long put, long straddle, scored separately |
| States | 9 single observables + 2 declared crosses, **375 cells** in total |
| Grid hash | `3e6eb89e5efe...`, fixed before the run |

The states span the complete set of things that move a long option's value — the clock, realised
magnitude, signed direction at two horizons, the price of volatility, implied against realised, where
spot sits in the session range, the session's width, and the expiry calendar — plus the two crosses
most likely to interact (clock x richness, magnitude x direction).

**Every cut point is causal.** Quartile boundaries come from a trailing window of prior sessions
only, never from the whole sample, so every cell assignment is one the bot could have made at the
decision minute. Clock and calendar states use fixed declared buckets.

## The answer

| | |
|---|---:|
| Cells whose lower bound clears zero, **after** correcting for the grid | **0 of 375** |
| Cells whose lower bound clears zero, uncorrected | **1 of 375** |
| Cells chance alone predicts would clear, uncorrected | **~9.4** |
| Cells that are **significantly negative** | **77 of 375** |

Read the last three rows together, because that is where the finding is. If the true payoff were
merely *zero* everywhere — unpredictable, a fair coin — roughly nine cells would clear zero by luck
and roughly nine would fall below it by luck. Instead **one** cleared and **77** fell below.

**The drift is not zero and unpredictable. It is reliably negative nearly everywhere we can look.**

Unconditionally, and reproducing the published variance premium to the digit:

| Hold | Long call | Long put | Straddle |
|---|---:|---:|---:|
| 15 min | −0.30% | −0.79% | **−0.55%** |
| 60 min | −0.63% | −2.21% | **−1.42%** |

That the pipeline lands exactly on the independently measured −0.55% and −1.42% is the positive
control: it is not silently computing something else.

### The one cell that cleared, reported because it did

15-minute hold, long call, quietest prior hour crossed with the strongest hourly up-move: **+$37.84,
CI [+5.19, +72.97], 200 slots**. Its z is **2.14** against the **3.65** this grid requires. It is one
cell out of 375 when chance predicts nine, so the honest reading is noise. It is recorded rather than
dropped, and it is the only thing in the census that would justify a pre-registered re-test.

### Do the observables carry any information at all?

A separate and better-powered question, because a state can carry information without any cell
clearing a profit bar. Eleven of 66 state/side/hold tests separate payoffs beyond chance — the clock,
day of week, and where spot sits in the session range, mostly on the straddle.

**They separate payoffs into cells that are all negative.** The information present is information
about *how much you lose*, not about when you win. That distinction matters for what to do next: this
is not an underpowered sample failing to see an edge, it is a sample that sees structure clearly and
the structure is a cost.

## Two defects found in this measurement's own machinery

Both were caught by disbelieving a clean-looking result, and both are now regression-tested.

**A never-forgetting window turns a cell into a calendar bucket.** The first draft cut quartiles on
every prior session. The corpus spans a 7.08x range of move dispersion across its years, so 2026
slots were being cut on mostly-2022 quantiles and piled into the top cells — the "high volatility"
cell would have quietly meant "recent". Fixed with a trailing 250-session window; the receipt now
records each cell's share in its busiest year (median 0.27, max 0.47).

**A NaN in a permutation null reads as evidence for the hypothesis.** Sessions have different slot
counts, so the session-swap null leaves gaps. One NaN turns a cell sum into NaN, the statistic into
NaN, and the comparison `null >= observed` into `False` — which counts a **failed** draw as a vote
for the state. Unmasked, that pinned six states at `p = 0.000`. This is the same class of defect as
the shuffled-label null that could not see feature leakage: **a control that fails open is worse than
no control**, because it produces confident wrong answers rather than obvious blank ones.

## Honest limits

- **This closes the states the owned corpus supports, not all states.** Order-flow imbalance, the
  event calendar, cross-asset context and chain-wide skew are not in this data. Skew and term
  structure exist only on the 251-session quote corpus, not the 1,045-session trade corpus.
- **It closes the long side.** Everything here is the buyer's side, which the charter restricts the
  bot to. The seller's side of these same numbers is positive before costs and was already measured
  as smaller than the spread.
- Declared crosses are two, not all pairs. A high-order interaction invisible to both the crosses and
  the dispersion tests is not ruled out, only made unlikely.
- The underlying is a put/call parity estimate from traded option prices, not an SPX print, and
  prices are last trades, so they carry bid/ask bounce.

## What this says about the next job

The detection floor on this corpus is **$27.5/trade for a single pre-registered test** and **$51.1
for a grid this size** — 2.6% and 4.9% of the median cell premium. The instrument's entire measured
mispricing is 0.55% per fifteen minutes, and the round trip is 2.20% of the same premium.

**The binding constraint is not prediction and is no longer measurement. It is that the toll is
several times the prize, and the bot is on the paying side of it.** No entry signal changes a ratio.
