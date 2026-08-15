> **This finding retracts the scope of "Why 0DTE does not work here", written
> earlier the same day.** That work priced a *fixed-horizon* policy — buy at one
> minute, sell at a fixed minute later — and concluded the instrument was
> efficiently priced. The conclusion is correct **for that policy** and does not
> transfer to a scalp with a dynamic exit, which is a different payoff
> distribution. I closed the question too early.

# The exit is the strategy, and the corpus had been telling me so all along

**2026-08-13.** [Receipt](../../../v4/audit/autoresearch/scalp_exits_2026_08_13/receipt.json) ·
[code](../../ops/measure_scalp_exits.py) · [tests](../../tests/test_scalp_exits.py)

## One paragraph

Every option measurement this project has made used only the **close** of each
minute bar. The corpus also carries **high and low**, and nobody had ever read
them. Reading them says that a near-ATM 0DTE contract reaches a best price of
**+28.2% (median) within thirty minutes**, that 60.9% of contracts touch +20%,
and that the peak arrives after a median of **12 minutes**. Selling at the best
price a contract *held for a whole minute* would earn **+$321 a trade** after the
measured round trip. Holding to a clock earns **−$32**. The prize for exiting
well is roughly **$350 a trade on a $989 premium**, and every naive exit rule
captures **none** of it.

## What the highs and lows say

49,928 trades across 1,045 sessions, near the money, thirty-minute window:

| | |
|---|---:|
| Best price reached, mean | **+44.3%** |
| Best price reached, median | **+28.2%** |
| Worst price reached, mean | −35.1% |
| Median minutes to the peak | **12** |
| Share touching +20% | **60.9%** |
| Share touching +50% | 29.7% |
| Share touching +100% | 10.3% |

A contract that moves 28% in twelve minutes is exactly the thing a scalp exists
to catch. None of this is visible in a close-to-close measurement, which is all
this project had ever done.

## The ceiling, and what naive rules capture of it

| Exit rule | Net/trade | When right | When wrong | Break-even | Winners |
|---|---:|---:|---:|---:|---:|
| hold to horizon | −$31.8 | +$334.5 | −$398.2 | 54.35% | 38.6% |
| take 20%, stop 20% | −$31.2 | +$48.3 | −$110.7 | **69.62%** | 47.7% |
| take 50%, stop 30% | −$35.0 | +$185.0 | −$255.0 | 57.95% | 37.5% |
| stop 30%, let winners run | −$30.4 | +$233.8 | −$294.6 | 55.75% | 32.1% |
| trailing 30% | **−$26.4** | +$148.9 | −$201.8 | 57.54% | 29.3% |
| *oracle: best of first 5 min* | *+$127.2* | | | | *81.1%* |
| **oracle: best minute close** | **+$321.4** | +$551.8 | +$91.1 | — | **81.5%** |
| *oracle: absolute high* | *+$384.9* | | | | *91.1%* |

Two readings, and both matter.

**The prize is large.** The gap between the best declared rule and the
conservative oracle is **$348 a trade**, 35% of the average premium. Even perfect
foresight restricted to the **first five minutes** earns +$127.

**Every declared rule makes the bar worse, not better.** A 20% target with a 20%
stop takes the break-even from 54.35% to **69.62%** — it caps winners at +$48
while losers still cost −$111, because a 20% stop sits well inside the −35% of
noise these contracts routinely show. Fixed levels are not merely unhelpful here;
they are actively harmful. That is precisely the argument for a *trained* exit
rather than a declared one.

## The honest discount on the oracle

The oracle earns **+$91 a trade even when the direction call was wrong**. That is
pure foresight: the contract bounced at some point in thirty minutes and the
oracle sold into it. **No causal model can systematically harvest a bounce it
cannot see coming**, so the attainable fraction of $321 is certainly well below
one, and possibly small.

What the oracle establishes is not that money is there for the taking. It is that
**the dispersion is large enough that exit skill has room to matter** — which the
fixed-horizon work implicitly assumed away by never letting the exit vary.

## What I got wrong, precisely

The earlier finding measured the variance risk premium and found the option
priced above delivery by 0.55% over fifteen minutes, against a 2.83% round trip,
and called the question closed. That measurement is sound and I do not withdraw
it. What I withdraw is the **scope**: it prices a policy that buys and sells on a
clock, and it says nothing about a policy that buys and sells on the *path*.

The variance premium is a statement about the **terminal** distribution. A scalp
lives on the **path**, and the path carries a median 28% excursion that the
terminal distribution does not show. Both facts are true at once; I let the first
one answer a question it was not about.

## What is now open, and testable

1. **Train the exit model.** At each minute in a trade, decide hold or exit from
   causal features — unrealised profit and loss, minutes elapsed, distance to
   break-even, the contract's own recent path, the underlying's path. Score
   chronologically out of fold and measure what fraction of the $321 it
   captures. This is the experiment that decides the strategy.
2. **Measure the excursion by moneyness and by time of day.** The entry model's
   job is to pick the contract that will make the largest premium move soonest,
   which is a *different label from direction* — and this project has only ever
   tested direction labels. The excursion is that label and it is now computable.
3. **Get sub-minute data if the exit model looks promising.** The described
   design watches price at one-second resolution; this measurement is
   minute-resolution and cannot speak to what the extra speed buys.

## Honest limits

- **Minute resolution.** A one-second exit cannot be simulated here. Whether
  finer data helps or hurts is untested: it allows a closer approach to the peak
  and also more whipsaw against tight stops.
- **Prices are trade prints.** The high may be a single aggressive print at the
  far side of a wide spread, which is why the close-based oracle is the one
  quoted.
- **The round trip is $23.00**, measured on 1.9 million near-ATM contract-minutes,
  and is charged once per trade.
- **Entries are unconditional**, every fifteen minutes, near the money. No entry
  signal is applied, so "when right" and "when wrong" are conditional labels
  rather than a policy's results.
- **The oracle is not attainable and is not a target.** It is a ceiling, and the
  gap between it and a real model is the whole open question.
