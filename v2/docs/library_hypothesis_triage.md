# Library-Driven Hypothesis Triage

**Purpose.** Replace the "many features, many stories" framing with one
coherent 0DTE market thesis that can be tested mechanically before any new
ML work.

## Source stack

Primary inputs for this pass:

- [Option Volatility and Pricing](</Users/gduby/Documents/The Library/Options/Option Volatility and Pricing_ Advanced Trading Strategies and Techniques.pdf>)
- [Hull - Options, Futures And Other Derivative Securities](</Users/gduby/Documents/The Library/Options/Hull-Options_ Futures And Other Derivative Securities_ 5Th Ed.pdf>)
- [Trading Price Action Trends](</Users/gduby/Documents/The Library/Options/Trading Price Action Trends_ Technical Analysis of Price Charts Bar by Bar for the Serious Trader.pdf>)
- [TradingThe10OclockBulls](</Users/gduby/Documents/The Library/Day Trading/TradingThe10OclockBulls.pdf>)
- [TA_Multiple_Timeframes](</Users/gduby/Documents/The Library/Technical Analysis/TA_Multiple_Timeframes.pdf>)
- [How To Use S_P 500 Futures To Get A Heads Up On Stock Price Action](</Users/gduby/Documents/The Library/General Trading/How To Use S_P 500 Futures To Get A Heads Up On Stock Price Action  (2001).pdf>)

The books are used as a **hypothesis engine**, not a training corpus.

## Current repo truth

The live feature contract is **79 base features**, not 52.

- `v2/data.pt`: `X.shape == X_sim.shape == (382920, 79)`
- `feature_names`: 79 names
- `v2/docs/feature_schema.md` is stale and still describes a 52-feature contract

Live feature groups from `v2/pipeline/compute_features.py`:

- 32 price / market features
- 14 session-structure features
- 12 option / Greeks features
- 13 surface features
- 8 flow features

This triage uses the live artifact and code as truth, not the stale doc.

## Comparative scorecard

Scoring rubric, 1-5 each:

1. point-in-time definability
2. data availability
3. feature coherence
4. mechanical-test clarity
5. relevance to 0DTE SPX long calls/puts

| Family | PT definability | Data availability | Feature coherence | Mechanical clarity | 0DTE relevance | Total |
|---|---:|---:|---:|---:|---:|---:|
| Reversion | 5 | 5 | 5 | 5 | 5 | **25** |
| Continuation | 4 | 4 | 3 | 4 | 4 | **19** |
| Vol expansion | 3 | 2 | 2 | 2 | 5 | **14** |

## Family notes

### 1. Reversion

**Market story.** The opening auction stretches too far away from fair value,
fails to continue, and reverts back toward VWAP / session open / opening-range
structure.

**Why the books support it.**

- `TradingThe10OclockBulls` centers the first 30 minutes and the 10:00 ET
  reversal window as meaningful intraday structure.
- `Trading Price Action Trends` provides the vocabulary for opening reversal,
  failed breakout, climax, and retracement back toward structure.
- `Option Volatility and Pricing` supports using higher-delta contracts for
  short-horizon directional expressions where theta punishes waiting.

**Best-fit live features.**

- VWAP distance and reclaim state
- opening gap and session-open distance
- first-15-minute acceptance / close position
- bar conviction and exhaustion
- gamma / theta / spread quality

**Missing data.**

- Helpful but not required: ES / NQ leader-laggard confirmation
- Helpful but not required: event calendar flags

**Why it wins.**

- strongest overlap between literature, current data, and a clean mechanical
  baseline
- easiest to falsify honestly
- easiest to support with a tight 6-15 feature shortlist

### 2. Continuation

**Market story.** The market discovers a real trend from the open or after a
break of early structure, then follows through in the same direction.

**Why the books support it.**

- `Trading Price Action Trends` is full of breakout, pullback, follow-through,
  and trend-resumption logic
- `TradingThe10OclockBulls` supports opening-range breakout structure
- multiple-timeframe framing fits trend continuation better than pure mean
  reversion

**Best-fit live features.**

- breakout confirmation
- ib_break / ib_extension
- trend_5min / ema_cross / macdh_slope
- volume_ratio / force_index / flow ratios

**Missing data.**

- ES / NQ leadership matters more here than for pure reversion
- higher-timeframe structure is only partially represented in the current
  feature set

**Why it loses to reversion.**

- current repo features can describe continuation, but not as tightly as they
  can describe failed extension and reclaim
- breakout continuation for 0DTE longs is more sensitive to missing
  leader-laggard and event context

### 3. Volatility expansion

**Market story.** A catalyst or regime shift produces a large realized move and
the long option wins because realized expansion outruns theta / paid premium.

**Why the books support it.**

- Natenberg and Hull make clear that volatility, gamma, theta, skew, and the
  speed of the move drive option value
- this family is the most theoretically "options-native"

**Best-fit live features.**

- atm_iv / iv_percentile / vrp
- gamma_pressure / aggregate_charm
- skew and slice-surface features
- realized vol and range expansion

**Missing data.**

- event calendar and release timing
- richer term structure / surface regime context
- better futures / internals context around expansion days

**Why it loses now.**

- too much of the story depends on missing or weakly represented data
- hardest to turn into one clean mechanical baseline without inventing extra
  context
- easiest way to drift back into "broad bag of features" thinking

## Winner

**Winning thesis: opening-structure reversion for SPX/SPXW 0DTE long calls and
puts.**

More specifically:

> After the opening auction extends too far away from fair value and fails to
> continue, the first high-quality opportunity is the reclaim / reject back
> toward VWAP and the opening range. Express that move with higher-delta 0DTE
> SPX long premium, not far-OTM convexity.

This is the best match to:

- the books
- the current repo's point-in-time features
- a compact feature family
- a testable mechanical baseline

## Consequences

1. The next implementation target is **one reversion strategy card**, not three
   half-alive hypotheses.
2. The next feature pass is a **pruning pass**, not a feature expansion pass.
3. ML is deferred until after the reversion baseline exists. Its first role is
   regime filtering or entry ranking, not end-to-end discovery.
4. Continuation and vol expansion remain live future families, but they are not
   the next baseline.

## What gets explicitly deprioritized

- journal-imitation as the main thesis
- broad end-to-end ML over all available features
- event-vol / vol-expansion baselines without event data
- continuation-first baselines that lean on trend features before a clean
  reversion test has been run
