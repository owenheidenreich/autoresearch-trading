# Higher-occupancy direction screen — feasibility, and the answer it produced

**Job 15 in [`STATUS.md`](../../STATUS.md). Opened and answered 2026-08-12.**

**One sentence for the owner: more trades per session very nearly fixes the measurement problem — but
every design that becomes measurable requires a skill level that would make this one of the best intraday
strategies in existence, so the honest recommendation is not to declare a family.**

## Why this job existed

Ledger row 186 closed G1 as `UNDERPOWERED` and named the few genuinely new alternatives. One was **"a
materially higher-occupancy design, since more independent decisions per session lower the per-decision
effect required and the current one-trade-per-session clock forbids that."** Row 185 attached a condition:
such a family needs **"its own occupancy-specific MDE receipt"** before it may be frozen.

So the receipt came first, deliberately, before any family existed — the design cannot be chosen to fit a
result it has not seen.

## What was measured

[`feasibility.py`](feasibility.py) reads owned ES bars on the 247-session index (the 254 owned sessions
minus the seven the product cannot trade), and for each horizon computes the dispersion of
**non-overlapping** close-to-close moves under a serial one-position clock — the same occupancy law the
frozen G1 game uses. It computes noise only. No rule, sign, threshold, or policy return appears anywhere
in it, and the loader's own reservation guard confirms every session is development data.

Two checks were run before trusting any of it:

1. **The pipeline reproduces the published measurement review exactly.** At the fixed 09:35 slot it returns
   session SDs of **13.486 / 18.099 / 24.807** points at 15/30/60 minutes — the review's numbers to three
   decimals.
2. **Trades are independent enough for the arithmetic.** Lag-1 autocorrelation between consecutive
   non-overlapping moves inside a session is **−0.030 to +0.038** across all six horizons. The measurement
   review found −0.055 at 15 minutes; the assumption stays slightly conservative at every horizon, so the
   surface is not flattered by correlated trades.

### A correction to a published number, in the optimistic direction

The measurement review's higher-occupancy row (0.859 points/trade at 6 trades/session, 0.413 at 26) applied
the **opening** window's volatility to **every** trade of the day. The opening 15 minutes has SD 13.486;
the average all-day 15-minute window has SD **8.898**. The open is 1.52x more volatile than a typical
intraday window, so spreading trades through the session samples quieter periods and the true per-trade
noise is about a third lower than published.

This makes the surface *more* favourable than the review's table. Per this project's own rule that
surprising results are suspected bugs until their mechanism is proven, the mechanism is stated plainly:
intraday volatility seasonality. It is not a new discovery — it is why the review's shortcut was a shortcut.

## The surface

247 sessions, 80% power, 95% familywise, friction 0.358 points/round trip. `m` is the declared family size,
which sets the Bonferroni correction. "Binds" names which of the two bars is higher — the economic one
(beat friction) or the measurement one (be visible at all).

| Horizon | Trades/session | Move SD | m=2 MDE | Binding constraint | Required edge/trade | As share of SD | Session friction |
|---:|---:|---:|---:|---|---:|---:|---:|
| 5 min | 75.7 | 5.237 | 0.107 | **friction** | 0.358 | **6.8%** | 27.1 pts |
| 10 min | 37.9 | 7.339 | 0.213 | measurement | 0.586 | 8.0% | 13.6 pts |
| 15 min | 24.9 | 8.898 | 0.318 | measurement | 0.875 | 9.8% | 8.9 pts |
| 20 min | 18.9 | 10.339 | 0.424 | measurement | 1.167 | 11.3% | 6.8 pts |
| 30 min | 12.0 | 12.467 | 0.643 | measurement | 1.770 | 14.2% | 4.3 pts |
| 60 min | 6.0 | 18.012 | 1.314 | measurement | 3.618 | 20.1% | 2.1 pts |

The conjunction penalty applied is **1.38–2.76x**, not the 2–4x measured for G1. G1's measured penalty
included Bonferroni across its eighteen members, and this surface already applies Bonferroni explicitly
through `m`; carrying 2–4x over unchanged would charge the same correction twice. The eighteen-member
factor is 3.612/2.487 = 1.452, leaving the residual for the parts not modelled here — the 4-of-5 fold rule
on both absolute and paired results, the session-block bootstrap, and the paired comparator. **This
decomposition assumes the penalty is multiplicative and separable, which is an approximation, not a
measurement.** It is a screening bracket, and row 184 still requires a full known-answer campaign for any
family that reaches declaration.

**Consistency check:** applied to G1's own design — one trade per session at the opening slot — this model
gives 4.3–8.5 / 5.7–11.5 / 7.9–15.7 points/session at 15/30/60 minutes. G1's campaign measured floors of 8
(M3) and 16 (M1, JOINT). The bracket covers the measured values, which is the strongest available evidence
that the surface is not merely optimistic arithmetic.

## What it says

**The good news is real.** Occupancy does what row 186 said it would. G1's frozen family needed an edge
**22–88x the cost bar**. At a five-minute horizon with a family of six or fewer, the measurement floor
drops **below** the friction bar — the first cell in this project's history where a screen would be
testing the market rather than testing its own power. That is a genuine, large improvement and it
vindicates the occupancy idea.

**The bad news is what the required edge implies.** The lowest requirement anywhere on the surface is
**6.8% of one move's standard deviation, per trade, sustained**. Held over 75.7 trades a session for a
year, that is an implied annualised Sharpe ratio of about **9**. Every other cell is worse: 7.8 at ten
through sixty minutes, and higher for larger families.

That number is the finding. A design is only interesting if the edge it demands is one a real strategy
could plausibly reach, and a sustained Sharpe near 9 is not a marginal strategy — it is a world-class one.
A screen built here would be asking whether we have found something exceptional, which is the same
question G1 asked and a poor use of the one corpus we own.

### The trap, stated once

The two constraints move in opposite directions with horizon, and there is no window where both are
satisfied:

- **Friction as a share of the move falls with horizon** — 6.8% at five minutes, 2.0% at sixty. Long
  horizons are where an ordinary edge could clear costs.
- **The measurement floor rises with horizon** — 0.107 points at five minutes, 1.314 at sixty. Short
  horizons are where 247 sessions can see anything.

Where a realistic edge could exist, the owned year cannot measure it. Where the owned year can measure,
friction is too large a share of the move for a realistic edge to clear it. **Occupancy moves along this
curve; it does not escape it.** That is a property of ES friction against ES volatility on 247 sessions,
and no family declaration changes it.

## Recommendation

**Do not declare a higher-occupancy family.** It would be a correctly-built screen asking a question whose
only affirmative answer is implausible, and it would spend the analysis budget that the two genuinely
open routes deserve.

The routes this surface does *not* close, and which remain open in row 186's terms:

- **A different instrument with better friction relative to its volatility.** The trap above is a ratio,
  and this study measured it only for ES. An instrument whose round-trip friction is a materially smaller
  share of its move at a measurable horizon would sit somewhere this surface has never looked. That is a
  data question, not a modelling one.
- **More sessions.** Unchanged from row 186 — the floor scales as 1/sqrt(n) and nothing here alters that.

## What this job did not do

Declare or freeze a family; compute any policy, rule, sign, threshold, or return; inspect any economics;
touch option data; read reserved sessions; or fit anything. The receipt is
[`feasibility_receipt.json`](feasibility_receipt.json) and every figure above reproduces from it by
re-running `PYTHONPATH=. ./.venv/bin/python v5/work/occupancy-feasibility/feasibility.py`.
