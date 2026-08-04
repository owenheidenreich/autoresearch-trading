# Gross-Expectancy Feasibility Study (2026-08-03)

The recommended next step out of the [Phase-1 close-out](PATHD_PHASE1_CLOSEOUT_2026_08_03.md): **a study,
not a training run.** Does any instrument/horizon reachable through IBKR have a friction hurdle low enough
that a directional edge could survive? Measured from the existing corpus — **no paid download, no
training, no broker.**

> ## ⚠ CORRECTED 2026-08-03 after adversarial review — read this first
>
> Codex adversarially reviewed this study
> ([`CODEX_ADVERSARIAL_REVIEW_2026_08_03.md`](CODEX_ADVERSARIAL_REVIEW_2026_08_03.md), `321b3bbd`) and
> **refuted two claims. Claude independently reproduced the refutations and accepts both.**
>
> **The "116.2% = mathematically impossible" claim is WITHDRAWN.** It was an artefact of using the
> **median** |move| inside an **expected-value** formula. An EV breakeven is magnitude-weighted, so it
> requires *mean* |move|. Recomputed: mean |move| = $43.86 → **80.19%**; one independent first window per
> trajectory → **76.48%**. One-minute aggressive 0DTE is still deeply unattractive, but **"impossible" does
> not follow**, and no scalar win rate is a feasibility theorem.
>
> **The "ES is the ONLY reachable cell" claim is WITHDRAWN.** Under the same EV screen, aggressive options
> reach **55.51% at 30m** and **53.88% at 60m** — comparable to ES. This document also contradicted itself:
> it declared "≤55% reachable" and then listed passive options at 60m as exactly **55.0%** while calling ES
> the only reachable cell.
>
> **What SURVIVED every attack: horizon dominates instrument choice** (C7) — robust across all three
> estimators (116.2%→57.16% median, 80.19%→53.88% mean, 76.48%→53.78% first-window). That is the finding
> to keep.
>
> The corrected estimator-sensitivity table is in [Estimator sensitivity](#estimator-sensitivity) below.

## Headline

**The fatal choice in Path-D was the CADENCE, not the instrument.** The same SPXW 0DTE option held for
60 minutes rather than 1 clears a far lower bar on every estimator tried — 116.2%→57.2% (median),
80.2%→53.9% (mean), 76.5%→53.8% (first-window). Friction is roughly fixed per round trip while the move
being chased grows with horizon. **Trading less often is worth more than any model improvement measured
anywhere in this project.**

Path-D combined the highest-friction instrument with the shortest horizon — the worst available corner.

## Estimator sensitivity

A scalar "required win rate" is a **screen, not a theorem**. It assumes a symmetric ±M sign bet; options
are convex and the real action is enter/wait, not long/short. The estimator choice moves the answer by
tens of points:

| Instrument / entry | Horizon | median screen | **mean-payoff screen** | first-window / non-overlap |
|---|---|---|---|---|
| SPXW aggressive | 1m | 116.20% | **80.19%** | 76.48% |
| SPXW aggressive | 30m | 61.03% | **55.51%** | 59.46% |
| SPXW aggressive | 60m | 57.16% | **53.88%** | 53.78% |
| ES (assumed $17) | 15m | 54.25% | **52.82%** | 54.00% |
| ES (assumed $17) | 30m | 52.96% | **52.01%** | 52.83% |
| ES (assumed $17) | 60m | 52.06% | **51.42%** | 52.00% |

**ES's advantage over options is far narrower than this study originally claimed, and it is entirely
contingent on the unmeasured $17 friction constant.** Codex quantified that contingency:

| ES spread | Friction | 15m | 30m | 60m |
|---|---|---|---|---|
| 1 tick | $17.00 | 54.25% | 52.96% | 52.06% |
| 2 ticks | $29.50 | 57.38% | 55.13% | 53.58% |
| 4 ticks | $54.50 | 63.63% | 59.48% | 56.61% |

**At 2 ticks ES is no better than passive options; at 4 ticks it is worse.** Codex also established via a
read-only `metadata.get_cost` probe (no purchase, no range request) that GLBX quote schemas are **priced,
not free** — `mbp-1` $0.897, `tbbo` $0.822, `bbo-1s` $0.073, `bbo-1m` $0.0023 for a single session — so
confirming ES friction is a **paid-data hard stop requiring owner authorization**.

## Method

The feasibility question is *not* "does this instrument have positive drift." Nothing reliably does — if it
did, no model would be needed. The question is **how big an edge must be to clear friction**.

For a symmetric move of size `M` with round-trip friction `F`, a trader right with probability `p` earns
`EV = (2p−1)·M − F`. Breakeven therefore requires:

```
required_win_rate = 50% + F / (2 · M)
```

That converts friction into one number comparable across instruments: **the accuracy you must achieve
before making a cent.** `M` is the median absolute move over the holding horizon, measured per contract.

Reading guide: **≤55% reachable · 55–60% hard · 60–70% implausible · >70% dead.**

## Results

| Instrument | Horizon | median \|move\| | drift | friction | hurdle | **required win %** |
|---|---|---|---|---|---|---|
| **ES future** | 1m | $50.00 | +0.13 | $17.00 \* | 0.340 | 67.0% |
| **ES future** | 5m | $125.00 | +0.47 | $17.00 \* | 0.136 | 56.8% |
| **ES future** | **15m** | $200.00 | +1.65 | $17.00 \* | 0.085 | **54.2%** |
| **ES future** | **30m** | $287.50 | +2.65 | $17.00 \* | 0.059 | **53.0%** |
| **ES future** | **60m** | $412.50 | +4.87 | $17.00 \* | 0.041 | **52.1%** |
| VX future | 1m | $20.00 | −0.14 | $55.00 \* | 2.750 | 187.5% |
| VX future | 15m | $50.00 | −2.29 | $55.00 \* | 1.100 | 105.0% |
| VX future | 60m | $90.00 | −6.85 | $55.00 \* | 0.611 | 80.6% |
| SPXW 0DTE (passive) | 1m | $20.00 | −0.22 | $18.56 | 0.928 | 96.4% |
| SPXW 0DTE (passive) | 15m | $80.00 | −2.74 | $18.56 | 0.232 | 61.6% |
| SPXW 0DTE (passive) | 60m | $185.00 | −5.70 | $18.56 | 0.100 | **55.0%** |
| **SPXW 0DTE (aggressive)** | **1m** | **$20.00** | **−0.22** | **$26.48** | **1.324** | **116.2%** ← Path-D |
| SPXW 0DTE (aggressive) | 15m | $80.00 | −2.74 | $26.48 | 0.331 | 66.5% |
| SPXW 0DTE (aggressive) | 60m | $185.00 | −5.70 | $26.48 | 0.143 | 57.2% |

\* ES/VX friction is **assumed** from published tick structure, not measured — OHLCV carries no quotes.
Options friction is **measured** on 156,950 candidates.

## Findings

**1. Path-D sat in the worst possible corner.** It combined the highest-friction instrument with the
shortest horizon. Required win rate 116.2%. Every other measured combination is easier.

**2. Horizon is the dominant lever — bigger than instrument choice.** Holding SPXW 0DTE for 60 minutes
instead of 1 minute drops the required win rate from 116.2% → 57.2% on the same instrument with the same
fill law. Passive entry at 60m reaches **55.0%**. Friction is roughly fixed per round trip while move size
grows with horizon, so the hurdle falls as `1/√horizon`-ish. **Trading less often is worth more than any
model improvement measured in this project.**

**3. ES futures at 15–60 minutes are the most reachable thing in the corpus** — 52.1–54.2%. That is
genuinely in the range where a real directional edge could survive costs. ES friction is ~$17 round trip
on ~$300,000 notional (0.006%), versus 4.68% of premium for 0DTE options — roughly a **700× difference in
relative cost**.

**4. VX futures are dead at every horizon measured.** Best case 80.6% at 60m. $55 round-trip friction is
too large relative to VIX's move scale.

**5. Drift is not an edge anywhere.** ES shows small positive drift (+$0.13 to +$4.87), options and VX
negative. ES's is noise-scale relative to a $412 median move over the same horizon, and 261 sessions of a
rising market is not evidence of tradable drift. **Do not read the drift column as free money.**

## Correction — drift-adjusted hurdle (added 2026-08-03, Claude)

The `required_win_rate` formula above accounts for **friction only**. Drift must be added, and it applies
**asymmetrically**:

- **Options: theta is a one-way tax.** It works against you whether you hold calls or puts, so `|drift|`
  must be added to the hurdle.
- **Futures: drift is two-sided.** A strategy that goes both long and short sees it roughly cancel, so
  no adjustment (do not credit ES's positive drift — that is a long-only artefact of a rising sample).

| Instrument | Horizon | friction-only | **drift-adjusted** |
|---|---|---|---|
| ES future | 30m | 53.0% | **53.0%** |
| ES future | 60m | 52.1% | **52.1%** |
| SPXW 0DTE (passive) | 30m | 57.7% | **59.5%** |
| SPXW 0DTE (passive) | 60m | 55.0% | **56.6%** |
| SPXW 0DTE (aggressive) | 60m | 57.2% | **58.7%** |

**The corrected gap between the best options cell (56.6%) and the best ES cell (52.1%) is 4.5 points.**
The conclusions above are unchanged in direction; the options hurdle is simply harder than first tabled.

## What this study does NOT say

- **It does not say ES is profitable.** It says the *hurdle is reachable*. Nothing here demonstrates any
  edge exists — only that friction would not automatically eat one.
- **It does not license a training run.** Establishing that a hurdle is clearable is a precondition, not a
  finding. The Phase-1 preconditions still bind: new features, pre-registration under family maxT, forward
  validation on fresh live paper.
- **It does not evaluate risk.** ES at 60m carries gap and overnight exposure that 0DTE intraday does not.
  Required-win-rate says nothing about drawdown, tail shape, or margin.

## Limitations (stated plainly)

1. **ES/VX friction is assumed, not measured.** OHLCV has no quotes. ES being ~always 1 tick wide is well
   established, so the assumption is reasonable, but confirming it needs GLBX quote data. If ES is
   materially wider than 1 tick at the times you would trade, the 52–54% figures degrade.
2. **The symmetric-move model is a first-order screen.** Real distributions are fat-tailed, and options
   are convex — their payoff is asymmetric by construction. The metric ranks instruments; it does not
   price them.
3. **Longer horizons mean fewer trades.** 60-minute holds allow ~6 opportunities/day versus 390 at minute
   cadence. Lower capacity, and per-trade edge must be larger to matter in aggregate.
4. **VX coverage is thin** — 88 sessions versus 261 for ES.
5. **Options data is 0DTE-only.** Every option row in the corpus is same-day expiry (verified: one distinct
   expiry per session file). Longer-dated options could not be evaluated.

## What would require a paid download (owner authorization — HARD STOP)

The single most valuable unmeasured cell is **longer-dated SPX/SPXW options (1DTE, weekly, monthly)**.
The corpus cannot answer it. Theory says theta per unit time falls sharply with tenor while spreads widen
only modestly, so the hurdle should improve — but that is a prediction, not a measurement. Acquiring it is
a paid Databento request and a CLAUDE.md hard stop; **not recommended until the free ES result is either
confirmed or discarded.**

## Recommendation (revised after adversarial review)

1. **Neither ES nor passive 0DTE is uniquely preferred.** The original "ES is the only reachable cell"
   ranking did not survive. Both branches need a **matched, execution-aware, pre-registered comparison**
   under executable fill assumptions before either is trained.
2. **Do not revive SPXW 0DTE at one-minute aggressive cadence.** This survives the review — it is
   unattractive on every estimator (76–116% depending on method). "Impossible" was withdrawn; "don't"
   was not.
3. **ES friction is unmeasured and is a paid-data hard stop.** GLBX quote schemas are priced, not
   entitled. Do not purchase without owner authorization. Until measured, treat ES's edge over options as
   unproven — at 2 ticks it disappears entirely.
4. **Any 0DTE revisit must be at 30–60 minute holds with an executable fill model**, and must carry the
   honest passive-improvement range (+$2.31 to +$22.76 depending on fill assumption), not the optimistic
   touch-model point estimate.

## Reproduction

`feasibility_study.py` (session scratchpad); outputs `feasibility_result.{txt,csv,json}`. Inputs are the
committed corpus at `/Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31/raw/databento/` and the
Phase-1 trajectories. Options move sizes come from measured 1-second CBBO bid paths; ES/VX from 1-minute
OHLCV closes.

*Signed: Claude Opus 5 — 2026-08-03.*
