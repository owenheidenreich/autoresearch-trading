# Pickles Trading Journal — Extracted Domain Knowledge for SPX 0DTE Bot

> **Source:** 218 Discord journal files, Oct 2023 – Jul 2024
> **Extracted:** March 2026
> **Primary instruments:** SPX/SPXW 0DTE options, ES/NQ/RTY futures, VIX, CL (crude oil)

---
/Users/gduby/Documents/picklesGPT/pickles
[this document requires all related documents. related meaning, related to SPX options.]
this must be used in ART²


## 1. Entry Rules

### Pre-Market Analysis (Daily Ritual)
- **OVN Inventory bias:** Determine if overnight inventory is LONG or SHORT based on where price spent most time relative to GLOBEX OPEN. This sets the initial lean.
- **Session structure:** Read ASIA → EURO → US PRE-MARKET sessions. If EURO prints expanded range beyond ASIA = strong directional signal. If EURO stays inside ASIA range = breakout setup.
- **Asset class alignment:** Check CL (crude), 10YR yields, DXY before open. Divergence across asset classes = chop until resolved. Alignment = trending day.
- **Sector analysis:** Check XLK, XLF, XLY, XLC, XLE, XLB, XLU, XLP, XLV, XLRE. When XLU (utilities) is strongest sector = NOT bullish, crabby day. When XLE is green and strongest = NOT bullish for equities. Top 5 sectors need to be green for bullish day.

### Opening Mechanics
- **Wait for first 15m candle to close.** Bears must maintain selling pressure BEYOND the first 15m candle to confirm bearish trend. Same logic for bulls.
- **Opening drive shakeout:** Let the opening drive shake out profit-takers first, then lean in the direction indicated by OVN inventory.
- **VWAP relationship:** If ES/NQ have not closed above VWAP since RTH open = bearish lean. Price needs acceptance above VWAP for bullish confirmation.
- **First touch VWAP rejection:** First touch of VWAP often gets rejected (lots of resting orders). Enter LONGS on the retracement after successful 2nd test.

### 10:00 AM EST "Magic Time" (0955-1010 EST Window)
- **This is a critical inflection point.** The 10:00 AM candle frequently decides the session's direction.
- **Counter-trend longs** are prime at Magic Time if internals support it.
- **If too much divergence across big 4 indexes, Magic Time may not happen.**
- **If Magic Time fails = sit on hands** and wait for the next setup.

### Specific Entry Triggers
- **VWAP rejection + volume spike = directional trade.** "ES RETRACE back to VWAP, if fail to hold the LVN underneath will not hold sellers back."
- **IB (Initial Balance) break:** At 10:30 EST, the IB is set (high/low of first 60 min). A break above/below IB = directional move.
- **NQ above VWAP will give more juice to ES** — trade the laggard when the leader confirms direction.
- **Overextension beyond +2 VWAP = DO NOT LONG.** Wait for pullback to +1 to +1.5 VWAP. Vice versa for shorts above +2 VWAP.
- **Supply zone breakout:** If ES breaks through a supply zone with volume, GO LONG up into the next major level.
- **Bounce off MACRO LEVELS:** Multi-year static levels (e.g., ES 4705 from Nov 2021) generate tradeable bounces.
- **Accumulation patterns:** "Who is thinking all these BOTTOM WICKS on ES is an ACCUMULATION PHASE" — multiple bottom wicks = buyers accumulating.

### Confluence Requirements
- **Pickles loves confluence.** Never enter on a single signal. Combine: levels + VWAP + internals + VIX + sector rotation + PAPER (bonds) direction.
- **Counter-trend entries need higher confluence:** "will be entering COUNTER-TREND LONGS at TOP OF HOUR / ~5-10 min into the next 30m candle" — uses time cycle confirmation.

---

## 2. Exit Rules

### The Holy Gospel: "Always Take Profits Off the Table"
- **This is the #1 rule.** Repeated in virtually every journal entry. Treat it as inviolable.
- **There is always another trade.** Don't diamond-hand positions.

### Take Profit Mechanics
- **Credit spreads:** Target 50-80% profit before PM session if FOMC/events pending, or let go 100% worthless on high-confidence days.
- **TP before noon = 50%.** TP after noon = 80%. (Lasher's rule that Pickles agrees with.)
- **Long options (0DTE):** Exit on first test of target level. Do NOT wait for perfect top/wick.
- **Trailing stops on swing positions:** Use OCO (one-cancels-other) trailing stops starting at TP level.
- **Scaled exits on swing trades:**
  - TP 1 at first target: exit 20% of position
  - TP 2 at second target: exit 60% of position
  - TP 3 remainder with trailing stop from TP 2 as SL

### Stop Loss Rules
- **Single leg 0DTE longs: mental stop loss.** "get out when I see the trade go against me." No fixed % — reads the tape.
- **Credit spread short leg blown: get out of the shorts, keep the longs** to offset losses.
- **CCS trigger:** When short strike is breached, SL on the short legs only fires. The long legs stay open to potentially ride the move and offset losses.
- **"I'll never recommend throwing more money at a losing position"** — but will add SAME spread contracts to raise break-even to minimize NET LOSS (defense, not doubling down).
- **Set SL so they don't go back up in premium.** On weekly positions nearing expiration, set stops to protect decay profits.

### Time-Based Exits
- **Be totally FLAT on 0DTE by lunch if not confident** in PM session direction.
- **"FOMC or 35-50% profits, whichever occurs first"** — events override greed.
- **Watch top of each hour** for potential trend shifts. "Reaching top of the hour, SHORTS/LONG PUTS is tempting here."
- **Power Hour (last 60 min): Beware.** "I'm total trash at POWER HOUR." Generally avoid or be very small.

---

## 3. Risk Management

### Position Sizing
- **Static $100K account.** All weekly profits get vested elsewhere. Capital deployed determined by MAX LOSS.
- **MAX LOSS cannot exceed account size** but should be as close as possible. E.g., MAX LOSS $8,000/IC × 12 ICs = $96,000 deployed.
- **MAX LOSS determines position size, NOT risk tolerance.** Active defense/hedging occurs LONG before MAX LOSS is reachable.
- **Weekly premium model:** 100% YTD gains in 6 months = ~107% ROI. Designed for 92% win rate / max 4 breaks per year.
- **Average 7 minutes active management per week** on the weekly model.
- **Keep 50%+ margin cushion:** If IC uses $8K margin, keep additional $4K reserve (total $12K). Keeps the position/portfolio liquid.

### Defense Strategies (When Short Strike Threatened)
1. **Close out early and eat minimal loss.** "Closing out of a SHORT entirely is ALWAYS an acceptable defense."
2. **Debit spread (Ghetto Spread):** Buy single-leg LONG while still OTM so premium is cheap. Convert into ghetto spread when it goes ITM.
3. **Back ratio:** More LONGS than SHORTS to achieve NET POSITIVE DELTA. Calculate: how many LONGS needed to exceed total SHORT DELTA.
4. **Convert to Call/Put Ladder:** Add rungs (additional short strikes) for NET CREDIT while maintaining directional lean.
5. **Roll the shorts:** Last resort — roll to further-out strikes.
6. **Long futures to hedge SHORT SPX:** "TOTALLY in favor of HEDGING a SHORT SPX POSITION by going LONG FUTURES." Use MES for delta control (5 delta per MES vs 50 per ES).
- **Defense should NOT exceed 30-70% of original credit received.**
- **"A proper defense does not need to MAKE profit, it needs to STOP THE BLEEDING."**

### Anti-Martingale
- **"I never doubled down. Doubling a losing position is asking to get blown up."**
- **"Take those gamblers methods to Vegas, it has no place in trading."**
- **Adding to move break-even up is DEFENSE, not a winning strategy** — "a defense to keep from losing - very different perspective there."
- **"Having to average in to move the BREAK-EVEN up is not a winning strategy, it's a defense to keep from losing."**

### Weekly Model Risk
- Model designed to tolerate 4 breaks/year (92% win rate).
- **"JUN is the month most likely to break the WEEKLY MODEL"** — FOMC + CPI convergence.
- **If short strike breached but never CLOSED ITM, it can still survive.** Premium evaporates quickly on brief breaches.

---

## 4. Market Structure

### VWAP Framework
- **VWAP is the single most important intraday level.** Everything revolves around it.
- **5-minute VWAP** is the primary timeframe for VWAP bands.
- **+1 to +2 VWAP = overextended.** Wait for pullback to +1 to 1.5 before entering longs.
- **-1 to -2 VWAP = overextended short.** Wait for bounce to -0.5 to -1 before entering shorts.
- **Price spending all time between +1 and +2 = strong trend up.** Don't fight it.
- **Weekly VWAP (WVWAP):** Used for multi-day context. Being well above/below = potential reversion.
- **VWAP + Volume Profile confluence = highest-probability trades.**

### Volume Profile (VP)
- **POC (Point of Control):** Highest volume price level. Acts as magnet. "WEEKLY POC is at 4986 ES."
- **VALUE AREA (VA):** 70% of traded volume. Below VAL = bearish. Above VAH = bullish.
- **HVN (High Volume Node):** Like rush hour traffic — price moves slowly through these zones.
- **LVN (Low Volume Node):** Like driving the interstate at 3 AM — price zips through these fast. Serves as STRONG S/R.
- **Value Area shift higher = BULLISH signal.** Means buyers are lifting their auction.
- **Volume cluster building above previous range = new support forming.**
- **Swiss-cheese volume gaps:** Price either re-auctions to fill OR leaves them behind — if left behind, can serve as launch pad.

### Key Structural Concepts
- **IB (Initial Balance):** High/low of first 60 minutes of RTH (set at 10:30 AM EST). Break = continuation potential.
- **PS (Previous Session) CLOSE/MID/HIGH/LOW:** Key reference levels.
- **GLOBEX OPEN:** The OVN starting price. Spending most of OVN below = bearish inventory.
- **OVN H / OVN L:** Overnight high/low — key pre-RTH reference points.
- **MACRO LEVELS:** Multi-year static levels (e.g., major round numbers like 5000 SPX, 4700 ES). Generate massive reactions.
- **FIB levels:** Use .5 and .618 of prior session H/L for retracement entries. ".618 FIB of the WEDS/THURS GLOBEX H/L."
- **Single prints:** Areas where price moved through quickly with low volume. Often get re-auctioned.
- **Quarterly Volume Profile:** Broader context for value area identification.

### Session-Based Structure
- **ASIA session → EURO session → US PRE-MARKET → RTH.** Each session builds on the prior.
- **EURO AM split down from ASIA AM** = bearish lean. EURO expanded range beyond ASIA = strong signal.
- **EURO CLOSE (11:30 AM EST):** Frequently causes trend reversal. "Beware beware the magic of EURO CLOSE."
- **Re-auction:** The tendency to fill prior session's move. "Wait to see if opening 30m wants to re-auction the entire OVN move."

---

## 5. Time-of-Day Rules

### When TO Trade
- **9:30-10:00 AM (Opening Drive):** Watch, don't immediately trade. Let shakeout happen.
- **10:00 AM Magic Time (0955-1010):** Prime entry window. Counter-trend longs if internals support.
- **10:30 AM IB Set:** After IB, can trade breakout or fade.
- **AM Session (9:30-12:00):** This is the primary trading window. Most trades happen here.

### When NOT to Trade
- **12:00-2:00 PM (Lunch Chop):** "Lunch time for pickles." Price alerts set, not actively watching.
- **After 3:00 PM (Power Hour):** "I'm total trash at POWER HOUR." "Beware POWER HOUR." Biggest candles of the day often in last hour — too volatile/unpredictable.
- **After 12:00 PM if you've won in AM:** "If you can secure your gains by noon, do it."
- **Too late for credit spreads:** "1200 EST premium will have decayed too much to make assuming the risk worthwhile."
- **No need to trade every day:** "There is no need to trade everyday, even if you are a day-trader." Written multiple times.
- **Triple witching Fridays:** Trade super small. Elevated volume but not necessarily directional.
- **Mid-week holidays:** "Do not follow normal rules for a 5 day trading week."
- **VIXperation days:** "Expect an un-responsive VIX, do not rely on it too much."

### Top-of-Hour Effect
- **Watch the clock at every hour mark.** Trend often shifts at top of each hour.
- **10:00 AM, 11:00 AM, 11:30 AM (EURO close), 12:00 PM, 1:00 PM, 2:00 PM** — each can reverse trend.
- **"LONGS should be looking for the exit at top of hour."**

### Calendar Awareness
- **MOPEX (Monthly Options Expiration):** Whipsaw expected. Dealers repositioning.
- **Triple/Quad Witching:** Elevated volume, not necessarily directional. Treat as MOPEX on steroids.
- **VIXperation (usually Wed before MOPEX):** VIX contracts settle — produces VIX-specific distortion.
- **End of Quarter (EoQ) flows:** Rebalancing/rotation — "if the market were to cycle downwards on a non-macro catalyst, it'd fall within the post-triple-witching window."
- **JPM Collar Roll:** Happens quarterly on the last Friday of the quarter. Causes violent moves in final 20 min.
- **Short trading weeks:** "Short trading weeks do not follow normal rules."

---

## 6. Options-Specific Knowledge

### 0DTE SPX Options
- **Primary instruments:** 0DTE SPX calls/puts (directional) and 0DTE credit spreads (income).
- **IBKR has overnight SPX options** — "IBKR has OVN SPX and that's a total game-changer."
- **TOS for charts/scripting/option chain analysis, IBKR for execution.**
- **Premium on 0DTE evaporates quickly.** Even deep breaches of short strikes can see premium collapse in a 5m candle.

### Credit Spread Mechanics
- **CCS (Call Credit Spread / Bear Call Spread):** Sell calls higher, buy calls even higher. Profit if price stays below short strike.
- **PCS (Put Credit Spread / Bull Put Spread):** Sell puts lower, buy puts even lower. Profit if price stays above short strike.
- **IC (Iron Condor):** CCS + PCS married together. Profit if price stays in range.
- **Short strike selection:** Based on PS HIGH/OPEN, weekly S/R levels, or 10-15 delta.
- **Use PS MID for short strike** on credit spreads.
- **Premium efficiency:** "Why be in drawdown when I can be on the RIGHT side of WRITING PREMIUM" — delay entering last contracts to capture pumped-up midweek premium.

### Advanced Spread Strategies
- **Ghetto Spread:** Buy single-leg LONG while OTM, then sell a SHORT CALL married to it when it goes ITM. NET PROFIT regardless of outcome.
- **Call/Put Ladder:** Multiple rungs of short strikes for stacked NET CREDIT. Defense mechanism AND income tool.
- **Albatross Spread:** Super-wide iron condor. Profit on range/theta. Primary instrument in low-VIX.
- **Gut Iron Spread:** Similar to iron butterfly. Makes money if price stays OUTSIDE a certain range. Used paired with Albatross.
- **Short Straddle/Strangle:** For earnings events (NVDA, etc.) — target 30-50% TP on IV crush.
- **Back Ratio:** More LONGS than SHORTS (e.g., 3:4 or 3:5). Used for defense and VEGA plays.

### Greeks in Practice
- **Delta as position sizing:** "Use the TOTAL DELTA of all SHORTS" vs "how many LONGS to achieve more NET DELTA." Use MES = 5 delta, ES = 50 delta for futures hedging.
- **Theta harvesting:** Primary income strategy in low-VIX/crab markets. "Let THETA eat while the market tries to figure out which one is right."
- **Vega / IV:** In low-VIX environments, buy longer-dated (30+ DTE) VIX calls. Near events (FOMC), IV expansion means write premium just before release to capture IV crush.
- **"Don't try to time the VIX bottom... VIX can stay suppressed far longer than expectation."**
- **"I never look at options flow models where someone bought $10m calls because someone SOLD him $10m of those calls."** — Options flow is ambiguous because both sides exist.

### VIX Mechanics
- **VIX under 13 = under-VIXed.** "Trading directionally will be difficult — settle for lower gains."
- **VIX over 15 = healthy.** Good for options sellers AND directional traders.
- **VIX backwardation (current spots above near-term futures) = SPICY environment.**
- **VIX futures roll:** The upcoming VX contract MUST equal current spot at expiration. Usually the current contract gets ripped to match the future contract. Plan for that direction.
- **VIX acceptance above key levels (13.00, 13.80, 14.00, 15.00) = NLOD likely.**
- **VIX rejection at key levels = thesis invalidated.**

### Earnings Plays
- **"Both NVDA CALLS & PUTS are horrendously over-valued"** — prefer WRITING premium, not buying.
- **PM session entry:** Leg into credit spreads during PM session to maximize IV/VEGA pump before earnings.
- **Target 30-50% TP on IV crush** next morning. Close at opening bell.
- **15-25 wide credit spreads** for earnings to maximize premium capture.

---

## 7. Indicators & Data Used

### Market Internals
| Indicator | Usage |
|-----------|-------|
| **TICK** | "Green TICKS are weak on this run-up" = weak conviction. Deep green ticks + above VWAP = strong bullish. Negative ticks all day = setup for EOD rally. |
| **INTERNALS** | Broad term for TICK, breadth, $ADD. "INTERNALS are crap for bulls." |
| **Sector heatmap** | XLK, XLF, XLY, XLC, XLB, XLE, XLU, XLP, XLV, XLRE — rotation signals. |
| **S5TH** | Market breadth (% of S&P 500 above some threshold). |

### Volume & Structure
| Indicator | Usage |
|-----------|-------|
| **VWAP (5m)** | Primary intraday anchor. Bands at ±1, ±2. Overextension beyond ±2 = reversion. |
| **Weekly VWAP (WVWAP)** | Multi-day context. Well above = potential pullback. |
| **Volume Profile** | POC, VAH, VAL, HVN, LVN — primary structural framework. |
| **Quarterly VP** | Big-picture value area. |

### Volatility
| Indicator | Usage |
|-----------|-------|
| **VIX** | Core. Under 13 = suppressed, over 15 = healthy. Backwardation = spicy. |
| **VX (VIX Futures)** | Term structure for directional VIX bias. |
| **Bollinger Bands (BB)** | Oversold bounce off lower BB = long bias. |
| **IV (Implied Vol)** | 0DTE SPX IV around 17% = low. Can change dramatically in minutes around events. |

### Price Structure
| Indicator | Usage |
|-----------|-------|
| **Fibonacci (.5, .618)** | Retracement entries from prior session H/L. |
| **Heikin-Ashi (HA)** | 5m HA candles to gauge trend strength. "When HA candles get shorter / no longer flat-bottom = exit." |
| **FVG (Fair Value Gaps)** | "I'm not much for FVGs" but acknowledges they sometimes set up. |

### GEX / Dealer Positioning
| Indicator | Usage |
|-----------|-------|
| **GEX** | "GEX prepping for a re-visit back to LOD." "GEX wanting to pull SPX to 4520." Used for delta-neutral magnet targeting. |
| **JPM Collar** | Quarterly collar roll — tracks strikes, counter delta hedges, and post-roll behavior. "Observed behavior post roll: liq 30 points then recapture 20." |
| **Dealer repositioning** | VIXperation, MOPEX, triple witching = dealers repositioning hedges. If shifted higher = bullish. If shifted lower = downside protection. |

### Cross-Asset
| Indicator | Usage |
|-----------|-------|
| **10-Year yield** | Rising yields in pre-market = not bullish for equities. |
| **DXY (Dollar Index)** | Strength typically bearish for equities. Correlated with XLF. |
| **CL (Crude Oil)** | CL up + XLE leading = NOT bullish for broad equities. CL down + equities flat = divergence to be resolved. |
| **PAPER (Bonds/Treasuries)** | "PAPER is bullish for EQUITIES" when yields drop. Monitor 3MO, 6MO, 2YR, 10YR, 20YR, 30YR. |

### Correlation
- **ES & NQ have 96% positive correlation since the late 90s.** "One of these is wrong and will correct." When they diverge, trade the correction.
- **RTY (Russell 2000) often front-runs:** When RTY is weakest, look for ES/NQ to follow. When RTY prints NHOD while others don't, it may drag them up or get corrected.

---

## 8. Anti-Patterns (Things to Avoid)

### Trading Behavior
- **"Sitting on hands is very much a position you can take."** Repeated frequently. Not trading IS a trade.
- **Don't front-run:** "Don't front-run that possibility until you actually see BUYERS lose control."
- **Don't LONG at +2 VWAP.** Wait for pullback.
- **Don't try to trade STRONG TREND DAYs** if you're not experienced. "Not everyone can trade a STRONG TREND DAY."
- **Don't trade the bias, trade the chart.** "LONG BIAS at RTH shot. This is why we trade the chart, not the bias."
- **Don't chase after a miss:** "TEES up a perfectly good trade, but by the time I saw it, premium had already left and it wasn't worthwhile. There is always another trade."
- **Don't hold losers hoping for magic:** "Don't let greed convince you to try to capture 100% for both, take the green and RUN."
- **Don't trade when A+ setups aren't there:** "This week will be a tough week to trade, sitting on hands is a position anyone can take."

### Structural Mistakes
- **Don't enter credit spreads after noon (0DTE).** Premium decayed too much to justify the risk.
- **Don't assume every day is a trading day.** Holidays, mid-week holidays, VIXperation all distort.
- **Don't fight the trend:** "Trend is your friend." Above +2 VWAP = overextended but don't short it.
- **Don't rely on options flow models.** For every buyer there is a seller — flow is ambiguous.
- **Doubling down is gambling.** Period. If you're adding to a losing trade, you're defending, not betting — know the difference.

### Environment-Specific
- **In low VIX (<13):** "Settle for lower gains & lower momentum / slower trends. Don't be expecting triple digit % bangers."
- **On FOMC days:** Be FLAT or very small before 2:00 PM. IV can go from triple digits to 40% in 6 minutes.
- **On quad witching / MOPEX:** Elevated volume ≠ tradeable direction. Watch for whipsaw.
- **Short trading weeks:** Normal rules don't apply — especially mid-week holidays.
- **When XLU is strongest sector:** Do NOT go long. Defensive rotation in play.

### Post-Trade Psychology
- **"I consider it a losing trade because it didn't make money & the thesis failed. The PnL will see it as a draw."** — A breakeven exit from a failed thesis is a LOSS, not a win.
- **If 1 for 3 on trades:** "Log off and avoid losing any more money." Know when to stop.

---

## 9. Key Phrases & Trading Wisdom (Pickles' Own Words)

### Holy Gospel & Mantras
> **"Always take profits off the table."** — The Holy Gospel. Repeated in every single journal entry.

> **"Wait for your A+ setups."** — Daily sign-off line.

> **"Trade the chart, not the bias."** — Daily sign-off line.

> **"There is always another trade."** — Said when exiting early or missing entries.

> **"Sitting on hands is very much a position you can take."** — On non-trading days.

### On Risk & Defense
> **"A proper defense does not need to MAKE profit, it needs to STOP THE BLEEDING."**

> **"Closing out of a SHORT entirely is ALWAYS an acceptable defense, and sometimes the best one — eat a minimal loss instead of a huge one, learn from it, and move on."**

> **"if the SHORT-LEG of a CREDIT SPREAD is blown up, get out, and keep the LONG-LEG to offset the loss."**

> **"I'll never recommend throwing more money at a losing position. This is no longer a winnable trade but a trade I'm trying not to lose."**

> **"Having to average in to move the BREAK-EVEN up is not a winning strategy, it's a defense to keep from losing - very different perspective there."**

> **"Take those gamblers methods to Vegas, it has no place in trading."** — On martingale/doubling down.

> **"Anyone doing spreads that sit there and allow MAX LOSS without lifting a finger to stop it deserves to lose every penny."**

### On Market Behavior
> **"Money can be made even in a boring, flat, crab market — gotta deploy the correct strategy to fit the market condition."**

> **"When XLK is green with AAPL & GOOG at -3% that means the rest of the tech sector is greener than California on renewables."**

> **"GEX prepping for a re-visit back to LOD."**

> **"A LOW VOLUME NODE is driving on the interstate at 3 in the morning — you zip right through. A HIGH VOLUME NODE is driving during rush hour — you are going nowhere slowly."**

> **"Beware beware the magic of EURO CLOSE."**

> **"Above +2 VWAP is OVEREXTENDED. Despite what hindsight may imply is an easy day, not everyone can trade a STRONG TREND DAY."**

### On Psychology
> **"There is no need to trade everyday, even if you are a day-trader."**

> **"Amazing what a few percentage points per week can do for a port over time."**

> **"No worthwhile spread came without a healthy amount of CLENCH."**

### On Process
> **"Every OPEN is about putting those market 'puzzle pieces' together to form a HIGH CONFIDENCE ACTIONABLE TRADE."**

> **"WATCH THE CLOCK."** — Top-of-hour effect.

> **"This is the DUMBEST way to make 100k."** — After CCS blew up, SL triggered on shorts, kept longs which then went 100x. Luck ≠ strategy.

---

## 10. Weekly Premium Model (The "Range" Model)

### Structure
- SPX weekly iron condors (ICs) with short strikes at model-generated support/resistance levels.
- Opened Sunday night / Monday AM, fully scaled in by Tuesday AM.
- Model produces a specific number of contracts (typically 8-13) based on weekly premium available.
- Some contracts held back for midweek entry to capture elevated premium.
- Hold until Friday expiration — target 100% worthless (full premium capture).
- **107% ROI in first 6 months, tracking for 207% annualized.**
- **Zero breaks in 26 weeks** (first half 2024).
- **Average 7 minutes active work per week.**

### Risk Framework
- Model is geared for 92% win rate / 4 breaks per year.
- MAX LOSS determines position size (as close to $100K as possible without exceeding it).
- Defense deployed when short strikes approach 75% of total range width.
- Multiple defense options ranked from conservative to aggressive (see §3 above).

### Weekly Levels
- R1/R2/R3 and S1/S2/S3 published every Sunday for ES.
- Beyond R3/S3 = "TA takes a back seat to an outlier TREND DAY — switch to trading based on VOLUME & PRICE ACTION."

---

## 11. Instruments & Platforms

| Platform | Use |
|----------|-----|
| **IBKR** | Primary execution — OVN SPX access is key advantage |
| **TOS (ThinkOrSwim)** | Charts, scripting, option chain analysis — "TOS is #1 if you are a coder" |
| **tradingview** | Charting (contract rollover annotation) |

| Instrument | Use |
|------------|-----|
| **SPX/SPXW 0DTE** | Primary income (credit spreads) + directional trades |
| **ES** | Futures trading, directional + hedging |
| **NQ** | Futures directional trades (NQ longs are a specialty) |
| **MES** | Precise delta hedging (5 delta per contract vs 50 for ES) |
| **RTY** | Reads as front-runner or divergence signal |
| **VIX** | Volatility thesis + VIX credit spreads (~30DTE) |
| **CL** | Multi-week crude oil futures positions |

---

## 12. Data Events Calendar

| Event | Time (EST) | Impact |
|-------|-----------|--------|
| **CPI** | 8:30 AM | Major — trend day potential |
| **PPI** | 8:30 AM | Major — initial spike then reversal common |
| **NFP (Non-Farm Payrolls)** | 8:30 AM | Major — bounce off MACRO levels |
| **FOMC Rate Decision** | 2:00 PM | Major — flatten by 1:45 PM |
| **FOMC Press Conference** | 2:30 PM | Major — directional volatility |
| **FOMC Minutes** | 2:00 PM | Medium — usually pre-priced from speakers |
| **Jobless Claims** | 8:30 AM | Medium |
| **PMI** | 9:45 AM or 10:00 AM | Medium |
| **Treasury Auctions** | 11:30 AM / 1:00 PM | Medium-High (especially 10Y, 30Y) |
| **FED Speakers** | Various | "Injection of VOLUME and/or VOLATILITY" |
| **VIXperation** | Market Close (usually Wed before MOPEX) | VIX unresponsive that day |
| **MOPEX** | 3rd Friday | Elevated volume, whipsaw expected |
| **Triple/Quad Witching** | 3rd Friday of quarter month | Treat as MOPEX on steroids |
| **JPM Collar Roll** | Last Friday of quarter | Violent moves in final 20 min |
| **EoQ Flows** | Last 2 weeks of quarter | Rotation/rebalancing |

---

## Summary: The Pickles Framework in One Page

1. **Pre-market:** Read OVN inventory, sessions (ASIA→EURO), cross-asset (CL, 10Y, DXY), sectors, VIX.
2. **Open:** Let first 15m candle settle. Wait for VWAP relationship to clarify.
3. **10:00 Magic Time:** Prime entry. Counter-trend if internals support, trend-follow if IB breaks.
4. **Trade AM Session (9:30-12:00):** This is where 80% of profit comes from.
5. **Prefer THETA/spreads in low-VIX, DELTA in high-VIX.** Match strategy to environment.
6. **Confluence required:** Never single-indicator entries. Levels + VWAP + internals + VIX + sectors.
7. **ALWAYS take profits.** Holy Gospel is non-negotiable.
8. **Defend or exit — never hope.** If short strike threatened, deploy planned defense or close.
9. **Know when NOT to trade.** Lunch, power hour, VIXperation, mid-week holidays, no A+ setup = sit out.
10. **"There is always another trade."**
