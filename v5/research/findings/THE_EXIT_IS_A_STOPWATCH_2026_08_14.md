# The exit was never solved — it was a stopwatch

**2026-08-14.** Re-running the exit at the touch, and the load-bearing claim underneath it.
[receipt](../../../v4/audit/autoresearch/quoted_exit_2026_08_14/receipt.json) ·
code: [paths](../../ops/build_quoted_exit_dataset.py), [policies](../../ops/train_quoted_exit.py) ·
[tests](../../tests/test_quoted_exit.py)

## What this changes

**The fitted optimal-stopping exit has no demonstrated skill. Everything it achieves, a five-minute
stopwatch achieves, and the stopwatch achieves slightly more.** The conclusion "the exit is worth
about zero" survives; the claim "the exit is solved by optimal stopping, therefore every dollar must
come from the entry" does not. The number was right and the reasoning under it was not.

## Why it needed re-running

On 2026-08-13 optimal stopping drove gross from −$7.3 to −$0.2 and was declared solved. That number
became the premise for the whole entry-edge programme.

It was measured on last-trade minute bars, and the value iteration is `V = max(price, C)` — an
**explicit maximum over the price series**. A maximum over a noisy series is inflated by the noise,
and on this corpus the put/call-parity residual has a standard deviation of **$102.60** against a
**$19.79** round trip. Selling at a high print is, mechanically, selling a print that landed on the
ask. The entry model measured the same way turned out to be doing exactly that, so the exit had to be
checked in the same place.

## What was done

**32,976 trades over 251 sessions** rebuilt on the owned quote corpus: every five minutes, the
nearest-to-the-money call and put, held up to sixty minutes. Entries are deliberately dumb, because
the exit is the object of study.

`fit_continuation` is **imported unchanged** from [`train_optimal_exit`](../../ops/train_optimal_exit.py),
so the value iteration, feature list, sweep count and hyperparameters are provably identical and the
only thing that differs is the price series. Two runs:

* **bid** — an honest exit: bought the offer, sold the bid, fees on top.
* **mid** — no spread and almost no bounce, which separates *cost* from *noise*.

And one control the original study did not have: a ladder of **fixed clocks**. A rule that exits after
eight minutes on average must be read against simply holding for eight minutes.

## The result

Gross per trade, 19,834 scored trades:

| Policy | at the bid | at the mid | held |
|---|---:|---:|---:|
| hold 5m | **−$15.8** | **−$0.2** | 5.0 |
| hold 8m | −$15.9 | −$0.2 | 8.0 |
| hold 15m | −$18.4 | −$2.5 | 15.0 |
| hold 30m | −$22.3 | −$6.0 | 30.0 |
| hold 60m | −$24.7 | −$7.2 | 56.9 |
| **optimal stopping** | −$16.5 | −$3.0 | 7.9 |
| random exit minute | −$17.8 | −$1.4 | 29.0 |
| shuffled-label null | −$25.4 | −$8.9 | 34.1 |
| **perfect foresight** | **+$602.2** | **+$623.1** | 23.7 |

Read the fitted rule against the clock that holds for the same time, not against the thirty-minute
one it was originally compared to. Paired trade by trade, on identical trades:

| Fitted rule minus… | at the bid | at the mid |
|---|---:|---:|
| hold 5m | −$0.8 [−5.5, +4.2] | −$2.8 [−7.5, +1.9] |
| hold 8m | −$0.6 [−5.8, +4.8] | −$2.8 [−7.7, +2.2] |
| hold 30m | +$5.8 [−4.7, +15.8] | +$3.0 [−8.1, +12.7] |
| random minute | +$1.3 [−11.4, +12.9] | −$1.6 [−14.4, +9.6] |

**Every comparison is indistinguishable.** The fitted rule is not measurably better than a stopwatch
at any setting, nor than exiting at random. Its point estimate is *worse* than a five-minute clock at
both price sources.

Its whole apparent advantage — "+$5.8 against a thirty-minute hold" — is holding for less time. A
five-minute stopwatch collects **+$6.5** of that same advantage at the bid and **+$5.8** at the mid,
more than the model does, and it needs no features, no fit and no walk-forward.

One detail worth keeping: the print study's headline figure, optimal stopping at **−$0.2** gross, is
reproduced almost exactly by a **five-minute stopwatch on clean prices** (−$0.2).

## What is true, restated

* **Short holds beat long holds.** Monotone: 5m ≈ 8m > 15m > 30m > 60m, at both price sources. That
  is theta, and it is a fact about the instrument rather than a fitted policy.
* **No adapted exit rule beats zero.** At the mid, with the spread removed entirely, the best policy
  on the board returns **−$0.2**. Not a small profit — zero.
* **The gap to perfect foresight is enormous and untouched.** The oracle keeps **+$623** at the mid.
  Every causal rule tested captures **none** of it. That is not a near miss; it is the whole prize
  sitting in a place no non-anticipating policy has reached.
* **The spread is what makes it negative.** Mid to bid costs about **$16** per trade on a $1,100
  contract, which is the entire difference between −$0.2 and −$15.8.

## What this does to the entry argument

The entry-edge programme rested on: *the exit is solved and produces gross ≈ $0, so every dollar must
come from the entry.* Both halves are now wrong in the same way.

The exit is not solved — it is null, and no better than a stopwatch. The entry, measured at the touch,
is also null. That is not two separate failures; it is one observation seen twice. Prices that no
adapted entry rule and no adapted exit rule can beat are prices behaving like a martingale, and the
single measured deviation from that — the variance premium — accrues to the seller, which the charter
bars.

## Honest limits

- 251 sessions of one year, against the print study's 745 sessions of four. The direction is
  consistent at both price sources and across five clock settings, but the sample is one regime.
- Every paired comparison is indistinguishable rather than decisive. The claim is that the fitted rule
  shows **no advantage** over a stopwatch, not that it is proven worse.
- Exit at the bid is the worst case for a taker. A resting exit would pay less spread and take adverse
  selection instead, separately measured at −$74 to −$86 and worse.
- The entry here is deliberately unskilled. This measures the exit given a random entry, which is what
  the original study measured; it does not measure an exit conditioned on a good entry, and no such
  entry currently exists to condition on.
- The oracle's +$623 is not an attainable target. It is the value of the running maximum, which no
  non-anticipating rule can reach; it bounds the prize rather than promising it.
