# The design works. The label was wrong, and the cost is still bigger than the edge.

**2026-08-13**, owner-approved. The first trained entry and exit models this project has built, and the
first non-negative number it has produced.

[Exit model](../../../v4/audit/autoresearch/exit_model_2026_08_13/receipt.json) ·
[entry and exit](../../../v4/audit/autoresearch/entry_and_exit_2026_08_13/receipt.json) ·
code: [dataset](../../ops/build_exit_dataset.py), [exit](../../ops/train_exit_model.py),
[combined](../../ops/train_entry_and_exit.py) · [tests](../../tests/test_exit_model.py)

## One paragraph

Both halves of the described design **learn something real**. The exit model predicts what a trade still
has left with an out-of-fold correlation of **+0.379**; the entry model predicts a contract's excursion at
**+0.310**, and selecting the top 5% raises the realised excursion from 39% to 74%. But the design as
stated optimises the **wrong quantity**: "largest premium-changing move" read as a *percentage* selects
cheap contracts, whose dollar moves are small and whose fixed round trip is proportionally huge — those
trades lose **more** than randomly chosen ones. Re-labelled in dollars net of cost, the combined system
reaches **+$2.80 a trade at the top 5% selectivity**, the first positive figure in this project, with a
confidence interval of **[−$33.60, +$42.10]**. It is break-even, not profitable, and the gap it still has
to cross is almost exactly the **$23 round trip**.

## The exit model alone

At every minute of an open trade, predict the best price still to come; leave when little is left. 1,494,261
decision points, 35,586 trades, chronological folds.

| Policy | Minutes held | Net/trade | vs hold |
|---|---:|---:|---:|
| hold to horizon | 30.0 | −$30.1 | — |
| random exit | 15.5 | −$27.5 | +$2.6 |
| **trained exit, 60th-percentile cut** | **12.5** | **−$22.9** | **+$7.2** |
| null: shuffled labels, matched hold | 13.3 | −$26.8 | +$3.3 |
| *oracle: best minute close* | *13.6* | *+$329.3* | *+$359.4* |

**Prediction/label correlation +0.379** — genuine skill, ten times what the two-sided model managed.

Two readings. The exit is worth **+$7.2 a trade**, of which about $3.9 survives the matched-holding-time
null, so roughly half is information and half is simply holding for less time on a decaying asset. And
+$7.2 against a $359 prize is **2% of the ceiling**. Note what that $7.2 buys: hold-to-horizon is −$7.1
gross of cost, the trained exit is **+$0.10 gross**. **The exit model recovers the option's decay almost
exactly and leaves nothing for the spread.**

## The entry model, and the label that was wrong

The design says: pick the contract likely to make the largest premium-changing move in the smallest time.
Read literally, that is a percentage excursion. Measured:

| Take | Entry label | Realised excursion | Net/trade | vs random |
|---:|---|---:|---:|---:|
| 5% | **percentage** | **74.2%** | **−$31.2** | −$8.1 |
| 5% | random | 39.1% | −$23.1 | — |
| 5% | **dollars, net of cost** | 49.2% | **+$2.8** | **+$25.9** |
| 10% | percentage | 66.1% | −$34.9 | −$4.7 |
| 10% | dollars | 45.2% | −$23.5 | +$6.7 |
| 25% | percentage | 54.6% | −$26.9 | −$1.7 |
| 25% | dollars | 41.0% | −$19.4 | +$5.8 |
| 50% | dollars | 40.6% | −$21.8 | +$3.9 |

**The percentage-labelled model succeeded at its task and lost money doing it.** It nearly doubled the
excursion — 74.2% against 39.1% — and finished $8 a trade *behind* random selection. A 74% move on a $200
contract is $148; a 39% move on a $1,000 contract is $390; and the $23 round trip falls on both alike.
Optimising the percentage is optimising for cheap contracts.

**Re-labelled in dollars, the same machinery beats random and beats its own shuffled null at every
selection rate**: +$25.9, +$6.7, +$5.8, +$3.9 against random, and +$47.7, +$8.2, +$8.3, +$5.2 against the
null. The consistency across all four rates is what makes it more than a cherry-picked cell.

## What that adds up to, stated conservatively

The best cell is **+$2.80 a trade at the top 5%**, 891 trades, 48.6% of them profitable against 42.0% for
random. Its confidence interval is **[−$33.60, +$42.10]** and it does not clear zero. Eight label × rate
cells were inspected. **This is a break-even system, not a profitable one**, and it should not be described
as anything more.

What makes it worth continuing rather than closing:

- both halves show out-of-fold predictive skill against proper chronological validation;
- the improvement over random and over shuffled nulls is **consistent in sign at every selection rate**,
  not confined to the best cell;
- the residual gap is **the transaction cost**, which is a measured quantity rather than an unknown.

At the measured $23 round trip the top-5% cell is +$2.8. The same trades at a **$12** round trip would be
**+$13.8**, and at $6, **+$19.8**. Every dollar of execution improvement is a dollar of edge here, and this
is now the third independent line of work to land on execution as the binding constraint.

## What I got wrong before this, and what still stands

I twice called this question closed. Both times the conclusion was correct **for the policy I had
measured** and wrong about the policy being proposed:

- the variance-premium work priced a **fixed-horizon** trade and could not see a path-dependent exit;
- the direction and magnitude screens tested **declared rules**, and a declared exit is measurably worse
  than no exit at all here, so their nulls said nothing about a trained one.

What still stands: the option is priced above delivery, the spread is ~2.8% of premium and worst late in
the session, and passive execution is adversely selected. None of that is overturned. The correction is
that a trained path-dependent policy sits close enough to break-even that those costs are now the whole
question rather than an academic one.

## What I would do next

1. **Optimise the exit objective properly.** The current label — best price still to come — is optimistic
   by construction and produces a policy that rarely fires on its own scale. A policy trained against the
   *decision* (hold versus exit under the continuation policy) is the correct formulation and is likely
   worth more than a feature search.
2. **Feed the entry model better features.** It currently sees the underlying's recent path and the
   contract's price. It sees **no greeks**, no implied volatility, no order-flow, and no chain-wide state,
   all of which the described design assumes.
3. **Sub-minute data.** Everything here is minute resolution. The described exit runs at one second, and
   whether that helps — closer to the peak — or hurts — more whipsaw — is genuinely untested.
4. **Then, and only then, a known-answer campaign.** Nothing here has passed a gate.

## Honest limits

- **Nothing clears zero.** The single positive cell is inside a wide interval and is the best of eight.
- **Entries are one contract per side every fifteen minutes**, near the money, with a thirty-minute window.
  Other bands and windows are untested.
- **Minute resolution**, trade prints, and a $23 round trip carried from the owned quote corpus.
- **The exit model is scored on trades the entry model chose**, so the two are not independent; a
  known-answer campaign would need to freeze both together.
- **Eight cells were inspected** and no multiplicity correction is applied to the headline figure, which is
  another reason to read it as break-even rather than positive.
