# All three fixes, a leak I nearly reported, and a clean null

**2026-08-13**, owner-approved: fix the exit objective, give the entry greeks, address sub-minute data,
and keep everything runnable on a live OPRA feed.

[Optimal exit](../../../v4/audit/autoresearch/optimal_exit_2026_08_13/receipt.json) ·
[full system](../../../v4/audit/autoresearch/full_system_2026_08_13/receipt.json) ·
code: [greeks](../greeks.py), [optimal exit](../../ops/train_optimal_exit.py),
[full system](../../ops/train_full_system.py) ·
tests: [greeks](../../tests/test_greeks.py), [optimal exit](../../tests/test_optimal_exit.py)

## One paragraph

All three were done. The exit, reformulated as optimal stopping, is **the right formulation and it works**:
it drives a randomly-entered trade's gross profit and loss from −$7.3 to **−$0.2**, holding 2.8 minutes
instead of 30. That is the ceiling on an exit working a random entry — **it removes the option's decay
exactly and stops**, which means every dollar of profit must come from the entry. The entry now reads
greeks, computed from price so the same function runs live. And with all three fixes in place the combined
system produces **nothing**: the best cell is −$29.6 a trade against −$26.5 for random selection. Along the
way the system briefly showed **+$119 a trade**, and that number was a one-minute look-ahead leak which the
shuffled-label control did not and could not catch.

## Fix 1 — the exit as optimal stopping

The old objective predicted "the best price still to come" and left when it fell below a threshold. It is
positive at 90% of decision points, so the rule barely fired. The correct question is not *will a better
price appear* but **is this price better than what following my own policy from here would get me** — an
optimal-stopping problem with a standard solution.

Fitted value iteration: `V = max(price, C)` where `C` is the learned value of continuing, swept three
times from "exit immediately everywhere". The policy needs no threshold: **exit the first minute the price
is at least the predicted continuation value.**

| Policy | Minutes held | Gross/trade | Net/trade |
|---|---:|---:|---:|
| hold to horizon | 30.0 | −$7.3 | −$30.3 |
| random exit | 15.5 | −$5.0 | −$27.9 |
| null: shuffled labels | 20.3 | −$6.4 | −$29.4 |
| **optimal stopping** | **2.8** | **−$0.2** | **−$23.2** |
| *oracle* | *13.6* | *+$350.8* | *+$327.8* |

**This is the cleanest result of the session.** The exit takes gross profit and loss to **zero and stops
there**, in a third of the holding time the threshold version needed for the same figure. That is exactly
what an optimal exit on a zero-drift decaying asset should do, and it is a strong sign the formulation is
right rather than merely better-tuned.

It also settles something. With a random entry, **the best possible exit gets you to free, not to
profitable.** The round trip does not enter the stopping decision at all — it is paid once whenever the
trade closes — so it cannot be exited around. Profit has to come from the entry.

## Fix 2 — greeks, computed rather than taken

Implied volatility by bisection, then delta, gamma, vega and theta — [`v5/research/greeks.py`](../greeks.py),
34 tests including put/call parity and an IV round trip, plus a vectorised path pinned to agree with the
scalar one.

**They are computed from price, and that is a parity decision.** The owned quote corpus carries a
`greek_source` column precisely because greeks are modelled: vendors disagree, and a vendor's value at
09:35:00 is not necessarily what the live system would hold at 09:35:00. A greek recomputed from the
option's own price, a parity spot, the strike and the clock cannot diverge, because **the same function
runs on both sides**. Every input is available on a live OPRA feed at decision time. Theta is quoted per
**minute**, not per day, because a per-day theta on a contract with an hour left is not actionable.

Solved on 99.9% of 1,494,261 rows. Median IV 0.093; median theta 0.256% of premium per minute.

**They bought nothing.** The entry model on greeks alone reached correlation +0.31 with the dollar
excursion and produced no economic edge — because that correlation is largely *contract size*: a $2,000
contract has a bigger dollar excursion than a $200 one, which is arithmetic, not skill. Measured directly,
`corr(entry premium, dollar excursion) = +0.297`.

## Fix 3 — sub-minute data, bounded before buying it

The described exit runs at one second; everything here is minute resolution. Rather than buy data to find
out what that costs, the owned corpus bounds it:

| Ceiling | Value |
|---|---:|
| oracle on minute **closes** — a price held for a whole minute | $321.4 |
| oracle on intra-minute **highs** — perfect timing inside the minute | $384.9 |
| **what perfect sub-minute timing can add** | **$63.5, 16% of the ceiling** |

A real model captures a fraction of a ceiling, and the optimal-stopping exit captured 2% of the
minute-resolution one. **Sub-minute data is not the big lever**, and buying it before the entry works would
be spending money to refine a policy whose gross profit is already zero. It stays proposed, not requested.

## The leak

With greeks and one path feature, the top-5% cell showed **+$119.1 a trade, CI [+104, +136], 70.9%
winners, clearing zero.** It was wrong.

`underlying_return_3m` on a trade's first recorded row is `spot(entry+1) / spot(entry) − 1` — **the
underlying's move in the minute after the entry decision.** Measured, it correlates **+0.19** with the
trade's dollar outcome. The model was being told which way the market went after it bought.

Removing it takes the same cell from **+$119.1 to −$29.4**.

**The shuffled-label null did not catch this, and could not.** Permuting the label destroys the thing a
leak would leak *to*, so the control comes back clean while the feature is still poisoned. The only control
that finds this class of defect is a **timestamp audit of every feature**, asking of each: could the bot
have known this in the minute it decided? That audit is now encoded as a test rather than left to whoever
remembers.

## The combined system, with everything causal

Entry: greeks plus underlying path features, all from windows ending **at** the entry minute. Exit: optimal
stopping. Label: dollar excursion net of the round trip. Chronological throughout.

| Take | Entry | Trades | Gross | Net/trade | 95% CI |
|---:|---|---:|---:|---:|---:|
| 5% | **model** | 2,055 | −$6.6 | **−$29.6** | [−38.9, −20.7] |
| 5% | random | 1,481 | −$3.5 | −$26.5 | [−33.5, −19.4] |
| 5% | null (shuffled) | 1,561 | +$3.2 | −$19.8 | [−27.4, −11.8] |
| 25% | model | 7,067 | +$0.2 | −$22.8 | [−26.5, −19.0] |
| 25% | random | 7,404 | −$2.4 | −$25.4 | [−28.1, −22.7] |
| 50% | model | 12,293 | −$0.3 | −$23.3 | [−25.5, −21.2] |

**Nothing clears zero anywhere, and the shuffled null beats the model at the most selective setting.** That
is the signature of no signal.

**And the earlier +$2.80 does not reproduce.** The previous run's best cell was +$2.8 with a confidence
interval of [−$33.6, +$42.1]; under the better exit formulation the same selection gives −$29.6. It was
inside its own noise band and should have been read that way at the time. I called it "break-even, not
profitable", which was right, but I should have weighted the interval more heavily than the point estimate.

## Where this leaves the question

The gross column is the honest summary. **The optimal exit reliably produces a gross profit and loss of
about zero** — at 25% selectivity, +$0.2; at 50%, −$0.3; on random entries, −$0.2. The system is, to within
measurement error, **free to run and worth nothing**, and the $23 round trip is the entire loss.

Three things follow, and none of them is another model:

1. **No entry signal tested so far produces positive gross.** Direction, magnitude, cheapness, excursion in
   two units, with and without greeks — all null once the leak is out.
2. **The exit is solved.** Optimal stopping removes the decay exactly, and the remaining ceiling is
   foresight nobody can have.
3. **The binding constraint is unchanged and now triply confirmed**: the round trip is $23.00 measured, and
   nothing found in this project generates gross profit larger than it.

## Honest limits

- Entries remain one contract per side every fifteen minutes near the money, thirty-minute window.
- Minute resolution; the sub-minute bound is a ceiling, not a measurement of what finer data would deliver.
- The $23 round trip is measured on 1.9 million contract-minutes of the owned quote corpus and carried into
  2022–2025 sessions whose spreads were never observed.
- Value iteration is three sweeps with one function approximator; a deeper solve could differ, though the
  gross figure sitting on zero suggests it has converged to the right place.
- Nothing here has passed a gate, and no known-answer campaign has been run, because nothing reached the
  point of deserving one.
