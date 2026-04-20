# Options Theory Notes — Pickles 0DTE SPX Long Primer

**Purpose.** This file is the compact theory layer for the Pickles long-only
research pass. It does **not** replace the journal as the primary source of
truth. Its job is to sharpen the interpretation of Pickles' 0DTE debit-call /
debit-put decisions using the selected library stack.

## Reading order and role

| Order | Source | Role in this pass |
|---|---|---|
| 1 | [Option Volatility and Pricing](</Users/gduby/Documents/The Library/Options/Option Volatility and Pricing_ Advanced Trading Strategies and Techniques.pdf>) | Primary reference for near-expiry `delta`, `gamma`, `theta`, IV behavior, and strike-selection tradeoffs |
| 2 | [Hull - Options, Futures And Other Derivative Securities](</Users/gduby/Documents/The Library/Options/Hull-Options_ Futures And Other Derivative Securities_ 5Th Ed.pdf>) | Sanity-check reference for Greeks and expiry mechanics |
| 3 | [TradingThe10OclockBulls](</Users/gduby/Documents/The Library/Day Trading/TradingThe10OclockBulls.pdf>) | Context for why the 1000 EST window can matter as an intraday decision boundary |
| 4 | [Trading Price Action Trends](</Users/gduby/Documents/The Library/Options/Trading Price Action Trends_ Technical Analysis of Price Charts Bar by Bar for the Serious Trader.pdf>) | Vocabulary for reversal / retracement / continuation / first-touch reject setups |
| 5 | [TA_Multiple_Timeframes](</Users/gduby/Documents/The Library/Technical Analysis/TA_Multiple_Timeframes.pdf>) | Formalizes Pickles' stated multi-timeframe stack |
| 6 | [How To Use S_P 500 Futures To Get A Heads Up On Stock Price Action](</Users/gduby/Documents/The Library/General Trading/How To Use S_P 500 Futures To Get A Heads Up On Stock Price Action  (2001).pdf>) | Supports the ES / NQ → SPX lead-lag interpretation used throughout the journal |
| 7 | [Bollinger on Bollinger Bands](</Users/gduby/Documents/The Library/Technical Analysis/Bollinger on Bollinger Bands - John Bollinger (2002).pdf>) | Supports Pickles' repeated use of 4H Bollinger context |
| 8 | [Intermarket Technical Analysis](</Users/gduby/Documents/The Library/Technical Analysis/Intermarket Technical Analysis_ Trading Strategies for the Global Stock, Bond, Commodity, and Currency Markets.pdf>) | Context layer for DXY / rates / sector / cross-asset alignment |
| 9 | [Trading VIX Derivatives](</Users/gduby/Documents/The Library/Voltality and Vix/Trading VIX Derivatives_ Trading and Hedging Strategies Using VIX Futures, Options, and Exchange Traded Notes.pdf>) | Reference only when VIX term-structure or event-vol questions arise |

The stack is intentionally narrow. Broad beginner options books, spread-income
books, and generic investing texts are out of scope for this phase.

## 0DTE debit-long principles

### 1. Delta is first-order, gamma is second-order, theta is the cost of being early

For bought 0DTE SPX longs, the first question is still directional: was the
underlying move right or wrong? That is primarily a `delta` question. Near
expiry, `gamma` matters because delta changes fast once the move starts working.
`Theta` matters because the premium decays brutally when the move stalls.

This lines up with the user's framing and with the journal:

- Pickles treats 0DTE longs as **directional scalps**, not as passive option
  holds.
- The journal repeatedly pairs entries with language like `quick in & out`,
  `scale out`, `flat by 1115`, or `TP within opening drive`.
- That behavior is more consistent with high-delta intraday exposure than with
  lottery-ticket convexity.

**Implication for this project:** when the journal names the side and timing but
not the exact strike, future backtests should sweep **fixed delta buckets**
instead of inventing a fake precise strike rule from prose.

### 2. Strike selection should match the hold horizon

The journal suggests three broad strike-selection regimes:

| Use case | Theory implication | Best starting assumption for testing |
|---|---|---|
| Quick intraday scalp to VWAP / opening-candle high | Favor cleaner underlying mapping and less theta noise | ATM to slight ITM, roughly `0.45-0.60 delta` |
| Momentum breakout to the next named level | Some convexity is acceptable if continuation is expected | Near-ATM to slight OTM, roughly `0.30-0.50 delta` |
| Event-rejection wick with fast exit into opening drive | Avoid paying for far-OTM lottery convexity when IV is rich | Higher delta than a pure lotto, closer to ATM |

These are **testing defaults**, not claims that Pickles always traded those exact
deltas. The corpus often states the strike, rarely the delta, and sometimes
neither. That is why the future Fork-A backtests should compare a small delta
grid rather than overfit to one inferred strike rule.

### 3. IV matters, but not all IV questions are equally important

For this long-only phase, the biggest IV distinction is:

- **Before** a news release or before the opening imbalance resolves, bought
  premium can be expensive and fragile.
- **After** the rejection / reclaim / breakout is confirmed, the intraday
  realized move can dominate the remaining IV premium paid.

This matters most for:

- **Row 5** in [pickles_digest.md](/Users/gduby/Documents/autoresearch-trading/v2/docs/pickles_digest.md): the news-wick long into opening drive
- any future event-day backtest where the entry occurs immediately after CPI /
  PPI / FOMC-style volatility shocks

**Practical reading:** do not treat event mornings as "IV makes longs bad."
Treat them as "you must avoid paying for direction before the market has shown
which side of the event candle it will respect."

## Time-of-day and price-action translation

### 4. 1000 EST is a regime checkpoint, not magic in the mystical sense

The `1000 MAGIC TIME` idea in the journal is consistent with a common intraday
auction pattern:

- the opening imbalance has had time to show whether it can continue
- the first 15m structure is complete
- laggards and leaders across ES / NQ / RTY are easier to compare
- traders reassess whether the opening move was discovery or just inventory
  adjustment

This is why the `TradingThe10OclockBulls` and `Trading Price Action Trends`
references belong in the stack. Their value here is not "follow these books'
rules literally." Their value is giving formal language to what Pickles is
already narrating:

- failed continuation
- reversal after the opening drive
- first-touch reject / second-touch reclaim
- breakout with follow-through versus rejection back into range

### 5. Multi-timeframe context is part of the setup, not extra garnish

Pickles explicitly works across:

- 3m for trading
- 5m for trend quality / Heiken Ashi
- 10m / 15m / 30m for structure
- 1h / daily / weekly for context and levels

This reinforces a key research conclusion from the digest:

- a future ML fork that only swaps the current feature list without adding
  proper multi-timeframe context would miss a large part of what Pickles is
  actually using
- the time-of-day windows (`opening 15m`, `1000`, `1030`, `top of hour`,
  `lunch`, `1130 Europe close`, `power hour`) are not just clock features;
  they interact with which timeframe is currently informative

## Context filters that likely matter

### 6. ES / NQ lead-lag matters for SPX long entries

The journal repeatedly uses ES and NQ to decide whether the SPX long should
exist at all. That makes the SPX-futures heads-up book relevant even if the
trade object is always bought `SPX/SPXW` premium.

The theory takeaway is simple:

- SPX options are the execution vehicle
- ES / NQ are often the faster context feed
- if a future mechanical rule or supervised label ignores the leader / laggard
  relationship, it may flatten an important part of Pickles' real decision
  process

### 7. Bollinger, VIX, rates, DXY, and sectors are filters, not standalone entry rules

The supporting books matter here because Pickles uses these items mostly as
context filters:

- `Bollinger`: overextension / mean-reversion context, especially on higher
  timeframes
- `Intermarket`: DXY / yields / sector breadth alignment
- `Trading VIX Derivatives`: VIX regime and curve-shape context

These filters help answer "should this long setup be trusted today?" more than
"where is the exact entry tick?" That matches the digest's classification:
many of these belong in Tier-1 day or Tier-2 setup routing before they belong
in a pure entry-timing model.

## Setup-specific implications

| Canonical setup | Theory implication |
|---|---|
| Row 1 — VWAP SUPPORT quick-in-out | High-delta exposure makes the most sense because the trade thesis is short-horizon and price-level specific |
| Row 2 — 1000 MAGIC TIME reversal | The time window matters as much as the indicator stack; slight OTM is only justified if continuation is expected immediately |
| Row 3 — Supply-zone break with confluence | Breakout continuation can justify some convexity, but the confluence filter is likely more important than fine strike precision |
| Row 4 — Opening 15m bounce continuation | A failed opening move is a price-action concept first, options concept second |
| Row 5 — News-wick opening-drive long | Entry timing relative to the event candle matters more than generic "IV is high" language |
| Row 6 — Weekly VWAP bounce | This is a context + level trade; the option should express the bounce, not rescue a weak thesis |

## Fork implications

### Selected fork: Fork A

Under the plan's priority rubric, **Fork A wins now** because
[pickles_digest.md](/Users/gduby/Documents/autoresearch-trading/v2/docs/pickles_digest.md)
already surfaced at least one `High`-confidence mechanical candidate, and in
fact surfaced two (`Row 1` and `Row 3`).

The theory layer supports that choice:

- Rows 1 and 3 are directional, level-based, and short-horizon enough to test
  mechanically
- the missing screenshot PnL does not matter because the simulator supplies the
  outcome
- the options theory does **not** reveal a hidden blocker that would force
  Fork C ahead of Fork A

### What to carry forward into Fork A

- Test a small **delta bucket grid** rather than one inferred strike rule.
  Starting grid: `0.30`, `0.40`, `0.50` absolute delta, mapped to the nearest
  executable contract in the sidecar snapshot.
- Keep exits faithful to the journal thesis first:
  - level-based exits
  - flatten-on-failure behavior
  - quick-in / quick-out bias
- Treat VIX / rates / DXY / sector alignment as filters only if the underlying
  data is actually present. Do not backfill them with hand-wavy proxies.

### Why Fork C stays alive

[journal_supervision_feasibility.md](/Users/gduby/Documents/autoresearch-trading/v2/docs/journal_supervision_feasibility.md)
still shows that action-only supervision is viable. If the mechanical tests for
Rows 1 and 3 fail, the next-best use of this theory file is to constrain Fork C:

- day labels should capture context quality
- setup labels should respect time-of-day windows
- action labels should emphasize side, entry timing, and flatten / scale-out
  behavior rather than pretend the journal carries reliable PnL weights

## Bottom line

The theory stack supports a simple interpretation of Pickles' 0DTE longs:

- they are **timed directional expressions**, not generic option bets
- `delta` and structure matter first
- `gamma` is the accelerator when the move works
- `theta` is the tax for being early or wrong
- time-of-day and multi-timeframe context are part of the setup, not optional
  metadata

That is consistent with the journal and consistent with choosing `Fork A`
before `Fork C`.
