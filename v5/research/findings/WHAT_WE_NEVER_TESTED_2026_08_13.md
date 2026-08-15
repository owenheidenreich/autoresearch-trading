# "No edge" was never measured. Three things were, and none of them is that.

> **SUPERSEDED IN ITS NUMBERS 2026-08-13.** Every payoff below was measured on a slot population that silently dropped 18.7% of slots — the ones where the underlying barely moved, averaging **-$186.90 per trade**. That was a look-ahead filter I introduced, and it moved break-even 3.27 accuracy points in the flattering direction. Corrected figures and the mechanism: [direction screen and the look-ahead](DIRECTION_SCREEN_AND_A_LOOKAHEAD_I_INTRODUCED_2026_08_13.md). The *comparative* conclusions here survive where both arms were filtered identically; every absolute level does not.

**Written 2026-08-13**, in answer to a direct owner challenge: why is the answer always that we cannot
trade a model, when people trade 0DTE every day?

[Time-of-day receipt](../../../v4/audit/autoresearch/time_of_day_2026_08_13/receipt.json) ·
[code](../../ops/measure_time_of_day.py) · [tests](../../tests/test_time_of_day.py)

## The correction that matters most

**This project has produced one measured negative result about 0DTE, not six.** Ledger row 181 measured a
minute-cadence long-premium model at about −$13 per trade before costs. That is a real negative.

Everything since stopped for a different reason. Both G1 attempts were closed `UNDERPOWERED`, and both
findings state plainly that the profit and loss was **never computed**. Job 15 concluded "measurable, not
winnable" about an ES design. The occupancy and charter work of 2026-08-13 measured what the *instrument*
pays and what an account can carry.

**No signal has ever been tested for directional edge on the option corpus.** Not once. The answer has not
been "there is no edge"; it has been "we cannot see one from here", and those are different sentences that
have been getting the same treatment.

## What is being missed, in order of size

### 1. Profitable and provable are different numbers, and everything here is calibrated to the second

Near-ATM, 60-minute hold, on 909 sessions, with the aggressive round trip charged: break-even is **50.43%**
and the accuracy a pre-registered screen could certify at 95% confidence is **52.28%**. The gap is 1.85
accuracy points, and it is where a normal trading business lives:

| True accuracy | Profit per year, 4.87 trades/session | On a $50k account | Certifiable? |
|---:|---:|---:|---|
| 51.0% | $8,808 | 17.6% | no |
| 51.5% | $16,586 | 33.2% | no |
| 52.0% | $24,365 | 48.7% | no |
| 52.5% | $32,144 | 64.3% | yes |

A strategy at 51.5% accuracy returns roughly a third of a $50,000 account per year and **cannot be
distinguished from noise** on every session this project owns. The gate is not wrong about that — it is
answering "can we prove it", correctly. The error has been letting that answer stand in for "can we trade
it".

This is the whole of "but people trade 0DTE all the time". They are operating in a band that this
project's measurement standard declares invisible.

### 2. The cost assumption is the most pessimistic available, and nothing has ever attacked it

Every number the project has produced charges **$25**, the measured cost of crossing the full spread both
ways. The measured fee-only floor is **$3.08**. Backing the cost out of the recorded payoffs:

| Hold | Gross when right | Gross when wrong | Break-even at $25 | at $14 | at $3.08 |
|---:|---:|---:|---:|---:|---:|
| 5 min | $260 | $229 | 51.88% | 49.64% | 47.40% |
| 15 min | $379 | $344 | 51.01% | 49.48% | 47.98% |
| 60 min | $653 | $614 | 50.43% | 49.57% | **48.70%** |

**Execution quality is worth 1.7 to 4.5 accuracy points** — more than hold length, occupancy, the exit rule
and the moneyness band put together, every one of which this project has spent weeks on. It has never been
treated as a variable.

Note also what the gross columns say: **being right pays more than being wrong costs, before costs.** That
is option convexity, and it is why the fee-only break-even sits below a coin flip. It is not a free lunch —
$3.08 requires resting passive and accepting non-fills, and a bot that must get filled pays more — but the
true cost is somewhere in that range and has never been measured for an actual order policy.

### 3. Every measurement is unconditional, and no trader trades unconditionally

The whole framework asks: what if you bought a near-ATM contract every N minutes, all day, every day? That
is deliberately signal-free, which is the right way to characterise an instrument and the wrong way to
characterise a strategy. Three consequences, all declared and none tested:

- **Magnitude is assumed independent of correctness.** The 2026-08-12 threshold derivation declares this
  assumption explicitly. A signal that fires only when a large move is coming earns far more per correct
  call than the measured pair implies, and the entire break-even shifts.
- **Selection is assumed absent.** A real design takes a handful of setups a session. Refusing to trade is
  free, and abstention quality is registered in this project as a first-class property, but no measurement
  here permits it.
- **Both sides are always traded.** The pair of nearest call and nearest put is what makes the conditional
  payoff computable, and it is not a policy anyone would run.

### 4. Time of day was never examined, and it moves the number

Measured on 1,045 sessions, near ATM, break-even by entry hour:

| Entry hour | 15-min hold, mean premium | Break-even at $25 | at $14 | 60-min hold at $25 |
|---|---:|---:|---:|---:|
| 09:00 | $1,529 | 51.80% | 50.65% | 50.83% |
| 11:00 | $1,190 | 51.21% | 49.63% | 50.48% |
| 13:00 | $925 | 50.67% | 49.01% | 50.72% |
| **15:00** | **$529** | **49.90%** | **48.30%** | — |

**The break-even falls all day**, by 1.9 points from the open to the last hour at a 15-minute hold, and the
straddle gross rises monotonically from +$16 to +$51 — realised movement increasingly outruns what the
options charge for it as expiry approaches. That is the gamma effect 0DTE traders talk about, and it is
visible in this corpus.

The last hour is also the cheapest place to stand: a near-ATM contract costs **$529** there against $1,529
at the open, which is 5.3% of a $10,000 account rather than 15.3%. **The account constraint and the
accuracy constraint both improve into the close**, and every measurement this project has made started at
09:35.

## What I am and am not confident about

**Confident**, because it is measured, reproducible and cross-checked against a second corpus:

- the conditional payoff pair for near-ATM 0DTE, on 1,045 sessions;
- the break-even that follows from it at a stated cost;
- the arithmetic relating account size, ticket size and ruin.

**No information at all**, because it has never been tested:

- whether any signal predicts SPX direction well enough to clear 50.4%;
- what a conditional, selective policy does to the payoff pair;
- what an actual order policy costs, as opposed to the two endpoints;
- anything about the last trading hour as a design rather than as a measurement.

The confident list characterises the racetrack. It says nothing about whether any horse can run.

## The constructive change this suggests

The 95% one-sided standard has been applied uniformly, as though every decision carried the same stake. It
does not. Matching the standard to the size is ordinary practice and this project has never done it:

| Confidence | Provable at, 909 sessions | With the free 2013 extension (~2,300) |
|---|---:|---:|
| 99% | 52.79% | — |
| 95% — the only one ever used | 52.28% | 51.60% |
| 90% | 52.01% | 51.43% |
| 80% | 51.68% | 51.22% |

Against a 50.43% break-even, a screen at 80% confidence on an extended corpus needs **51.22%** — a margin
of 0.79 points, rather than the 1.85 the current standard demands. A candidate clearing the weaker bar
would trade at a correspondingly smaller size, and forward sessions would do the rest of the work.

**The case against**: this project has a documented history of exactly the failure a strict standard
prevents — the label-leakage retraction is in the ledger, and the owner has rejected score-optimising
shortcuts before. Weakening a standard right after two failures is what a rationalising agent does. The
defence is that the weakening must be paid for in position size, and that the standard and the stake should
move together in both directions.

## What this finding does not claim

That an edge exists. That anything should be traded. That the gate machinery is wrong — it is not, it is
answering the question it was built to answer. Only that the question it answers has been mistaken for a
different one.
