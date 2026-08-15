# Occupancy is not the lever, and the account is too small to trade it anyway

> **SUPERSEDED IN ITS NUMBERS 2026-08-13.** Every payoff below was measured on a slot population that silently dropped 18.7% of slots — the ones where the underlying barely moved, averaging **-$186.90 per trade**. That was a look-ahead filter I introduced, and it moved break-even 3.27 accuracy points in the flattering direction. Corrected figures and the mechanism: [direction screen and the look-ahead](DIRECTION_SCREEN_AND_A_LOOKAHEAD_I_INTRODUCED_2026_08_13.md). The *comparative* conclusions here survive where both arms were filtered identically; every absolute level does not.

**Phases 1 and 2 of job 24, measured 2026-08-13**, on 909 usable 0DTE sessions and the exit-price
convention settled the same day.
[Occupancy receipt](../../../v4/audit/autoresearch/hold_occupancy_2026_08_13/receipt.json) ·
[risk receipt](../../../v4/audit/autoresearch/occupancy_risk_2026_08_13/receipt.json) ·
[code](../../ops/measure_hold_occupancy.py), [risk code](../../ops/check_occupancy_risk.py) ·
[tests](../../tests/test_hold_occupancy.py), [risk tests](../../tests/test_occupancy_risk.py)

## One sentence

The plan's central claim — that trading a 15-minute serial clock instead of a 60-minute one is worth
**3 accuracy points**, more than every other lever this project has found combined — **is not true**: it
was an artifact of the exit-price convention Phase 0 disproved, and the real spread across every hold from
5 to 60 minutes is **0.69 points**.

## What the plan expected, and what the corrected measurement gives

Every break-even in the plan's table was computed by dropping contracts that stopped printing before their
exit minute. [Phase 0](EXIT_PRICE_CONVENTION_2026_08_13.md) measured that convention to be wrong, and
wrong in a way that punishes long holds hardest: **5.8% of 60-minute contracts stop printing against 2.9%
of 5-minute ones**, and the ones that stop are 92% winners. Dropping them therefore taxed the 60-minute
cell about twice as heavily as the 5-minute cell, and that differential tax *was* the occupancy gain.

| Hold | Plan: trades/session | Plan: break-even | Plan: provable at | **Now: trades/session** | **Break-even** | **Provable at** |
|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 77 | 52.73% | 53.57% | **36.1** | **51.88%** | **52.52%** |
| 10 min | — | — | — | **22.0** | **51.01%** | **51.83%** |
| 15 min | 25 | 52.65% | 54.12% | **16.0** | **51.01%** | **51.99%** |
| 30 min | 12 | 52.82% | 54.95% | **8.8** | **50.69%** | **52.05%** |
| 60 min | 6 | 54.10% | 57.11% | **4.9** | **50.43%** | **52.28%** |

Three things changed, and only one of them is bad news.

**The bar fell everywhere.** The hardest cell now asks for 52.52% and the easiest for 51.83%, against the
plan's 53.57% to 57.11%. The 60-minute design this project has assumed since the beginning needs
**52.28%, not 57.11%** — a correction of 4.8 accuracy points, and by far the largest single number this
session produced.

**Occupancy stopped being a lever.** The spread from 5 to 60 minutes is 0.69 points and it is not even
monotonic: the best cell is **10 minutes at 51.83%**, and 60 minutes at 52.28% is only 0.45 points worse.
Trading twelve times as often buys nothing worth having.

**Occupancy was overstated as well.** `385 / hold` assumed every slot could be priced. Measured, the
5-minute clock delivers **36.1 trades a session, not 77**. Almost all of the shortfall is the parity spot
sitting on the same strike at both ends of a 5-minute window — 39.4 slots a session — which is a limit of
reading the spot off a five-point strike grid rather than a session the bot could not have traded. That
means 36.1 understates true 5-minute occupancy; it does not rescue the cell, because even doubling `n`
moves its provable bar only to about 52.3%, still behind 10 minutes.

**Trades within a session are very nearly independent.** The measured intra-session correlation of
direction is 0.000 to 0.031, so the design effect runs 1.00 to 1.12 and the effective sample is 95% to
100% of the raw trade count. This was the one assumption in the plan's table that held.

## Phase 2: the charter risk check, which rules out every cell at a $10,000 account

The plan required that a cell survive the **5% daily circuit breaker** and the **50% survival floor**
before it could be declared, and said plainly that the risk limits are not the part worth amending. They do
not survive.

The check simulates a year of sessions 20,000 times per cell, resampling **measured per-trade profit and
loss in dollars**. It works in dollars because the charter buys one contract: the money at stake is
whatever that contract costs, so the **account size** is what sets risk per trade, not a chosen fraction.
A near-ATM 0DTE contract costs about **$1,050**, which is 10.5% of a $10,000 account.

> **Corrected 2026-08-13, after the numbers below were first published.** The resampling this section
> rests on drew from a hundred-point quantile grid, whose draws can never exceed the 99.5th percentile.
> Option payoffs are heavily right-tailed, so that understated the mean winning trade by 6% to 19% while
> leaving the bounded losing side accurate to within 1% — a systematic bias against the strategy. It is
> replaced by equal-probability strata, whose resampled mean is exact
> ([`strata_means`](../../ops/measure_hold_occupancy.py), regression tests in
> [`test_hold_occupancy.py`](../../tests/test_hold_occupancy.py)). The figures below are the corrected
> ones. **The direction of the finding is unchanged and its severity is lower**: the first version
> reported 96–100% of years breaching the survival floor at a $10,000 account, and the true figure is
> **0.0%**, because an account that cannot afford the ticket stops trading rather than losing the money.

At $10,000, holding to the horizon, and assuming the strategy achieves exactly the accuracy a screen could
prove:

| Hold | Nominal trades/session | Realised | Occupancy kept | Breaker fires | Years breaching the floor |
|---:|---:|---:|---:|---:|---:|
| 5 min | 36 | **5.5** | 15.3% | 8.6% of sessions | 0.0% |
| 10 min | 22 | **3.6** | 16.3% | 8.6% | 0.0% |
| 15 min | 16 | **2.8** | 17.2% | 8.3% | 0.0% |
| 30 min | 9 | **1.5** | 16.8% | 7.0% | 0.0% |
| 60 min | 5 | **0.8** | 16.3% | 5.6% | 0.0% |

**One losing trade is most of a 5% day.** At a 10.5% ticket and a typical wrong-side outcome near −58% of
premium, a single loss costs about 6.1% of the account. Every cell therefore keeps only about a sixth of
its nominal occupancy, and the 60-minute cell does not reliably get even one trade a session. The account
is not destroyed — it stops trading, because after a drawdown one contract no longer fits under the 13%
ceiling — but the occupancy the plan is buying never happens.

Against three criteria — breaker firing on at most 1 session in 20, no year breaching the survival floor,
and at least half the nominal occupancy actually traded — the smallest account that passes is **$250,000**
at the provable accuracy for every cell, and $500,000 for most cells if the strategy has no skill at all.
Either way the occupancy design needs an account **25 times larger** than the one the charter amendment
was written against.

The third criterion had to be added during the work and the reason is worth recording: without it a
$10,000 account "passes" both charter limits, because tripping the breaker after one trade keeps it from
ever losing half its capital. It passes by not trading. A cell chosen for its occupancy has to deliver it.

The ladder is not monotonic, and the reason matters. A $25,000–$100,000 account trades **more** than a
$10,000 one and is therefore exposed to more of the dispersion, while still being too small for a
near-ATM ticket to be a small share of capital. The protection at $10,000 is not safety, it is
inactivity.

## The declared stop: a result this corpus cannot support

The charter's 13% ticket is justified *by* the −30% exit, so the risk check was also run with it. Measured
on trade prints, that exit lowers the break-even at every hold — by 0.34 points at 5 minutes and **2.29
points at 60**, taking the 60-minute cell to 48.14%.

**That number is not usable and is not carried into any conclusion.** It says a coin flip on near-ATM 0DTE
premium is profitable after costs, which would be a money machine, and it contradicts this project's own
[quote-based measurement](../../../v4/audit/autoresearch/stop_effectiveness_2026_08_13/receipt.json), which
found the same exit to be worth nothing and non-monotonic across levels — that is, noise.

Two candidate artifacts were tested and **both ruled out**:

- *Stale paths protect winners.* A contract that stops printing has a frozen path that cannot trigger a
  stop, and Phase 0 showed the ones that stop printing are winners. Measured on 2,402 recent 60-minute
  trades: the stop fires on 96.3% of wrong-side contracts and 29.7% of right-side ones, but the frozen
  paths are only **19 trades of 2,402**. Too small to explain the effect.
- *Filling at the price that triggered the stop.* Re-running with the exit taken at the **next** minute's
  price, the earliest a real order could be worked, moves the break-even by 0.09 points — 46.53% to
  46.62%. Not the explanation either.

The mechanism is therefore unproven, and this project's standing rule is that a large or surprising result
is a suspected bug until its mechanism is shown. Everything above is stated on **hold-to-horizon** numbers,
which are the conservative pair.

The same probe surfaced something that does need saying: on the most recent 250 sessions the 60-minute
hold-to-horizon break-even is **49.36%**, below a coin flip, against 50.43% pooled and 54.00% in 2022. Long
0DTE premium has been getting cheaper relative to realised movement across this corpus. That is a statement
about the instrument in a particular regime, not an edge, and any candidate frozen on the pooled number is
being judged against an average of five materially different years.

## What this means for the plan

- **Phase 0's failure condition did not fire**, and its correction improved every bar by 1 to 5 points.
- **The plan's reason for existing did fire.** Occupancy is worth 0.69 points, not 3, so "trade a
  15-minute serial clock instead of a 60-minute one" is no longer a change worth making.
- **The plan's second stated failure condition also fired.** The risk check rules out the occupancy cell
  at the charter's account size — and rules out every other cell too, including the 60-minute design the
  project already had.

## What is explicitly not concluded

That no edge exists. Nothing here evaluates a policy, fits a model, or searches a threshold. Every figure
is a property of the instrument: what a right or wrong direction call is worth in near-ATM 0DTE premium,
and what the charter's own limits do to an account trading it.

## Honest limits

- **The round trip is $25, carried from the owned quote corpus** and applied to 2022–2025 sessions whose
  spreads were never observed.
- **The risk simulation resamples trades independently within a session.** A day that trends against every
  position is under-represented, so the real breaker rate is at least as high as reported and probably
  higher.
- **It also resamples from a summary of the payoff distribution**, not from the trades themselves. That
  summary is now equal-probability strata, which reproduce the measured mean exactly; the first version
  used a quantile grid, which did not, and the correction is recorded at the top of the Phase-2 section
  rather than quietly applied.
- **The provable-accuracy column is a screening bracket**, computed for a single pre-registered hypothesis
  at one-sided 95% and 80% power. A declared family must still compute its own power, and the receipt also
  carries the figures under the measured G1 conjunction penalty, which adds 0.6 to 1.9 points.
- **The parity spot is a proxy** read from traded call and put prices on a five-point strike grid, and it
  is what makes 5-minute occupancy hard to measure.
