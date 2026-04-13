# 0DTE SPX Options Trading — Domain Knowledge Reference

> **Source:** Extracted from trading books library + practitioner knowledge
> **Last Updated:** March 14, 2026

---

## 1. Library Inventory

### Day Trading (19 books)
- 18 Trading Champions Share Their Keys To Top Trading Profits (1996)
- 25 Rules Of Day Trading (2003)
- A Complete Guide to Day Trading (2008)
- Advanced Techniques in Day Trading (practical guide)
- After-Hours Trader
- Day Trading Ebook (2000)
- High Probability Short Term Trading Strategies (2005)
- How To Spot A Trend (2000)
- How To Day Trade eBook
- How to Day Trade for a Living
- Ken Wolff — Trading On Momentum: Advanced Techniques for High Percentage Day Trading
- The Day Trader's Bible (2001)
- The Greatest Trade Ever (John Paulson)
- The Secret Of Selecting Stocks For Immediate And Substantial Gains (2000)
- Timing the Market (2006)
- Trades About to Happen (2013)
- Trading The 10 O'clock Bulls
- Trading in the Zone — Mark Douglas

### Options (28 books)
- A Complete Guide to the Futures Market (technical analysis, options, spreads)
- Dan Passarelli — Trading Option Greeks
- Get Rich With Options (2009)
- Hull — Options, Futures and Other Derivative Securities (5th Ed)
- Natenberg — Option Pricing and Volatility: Advanced Strategies and Trading Techniques (1994)
- Option Pricing Models (2007)
- Option Spread Strategies (2009)
- Option Strategies: A Quick Guide (2010)
- Option Strategies: Profit-Making Techniques for Stock, Stock Index, and Commodity Options
- Option Trading: Pricing and Volatility Strategies (2010)
- Option Volatility and Pricing: Advanced Trading Strategies and Techniques (Natenberg, updated)
- Options and Options Trading: A Simplified Course (2004)
- Options Trading Strategies: Complete Guide
- Options Trading for the Conservative Investor (2010)
- Options Trading QuickStart Guide
- Options Bible
- Put Option Strategies for Smarter Trading (2010)
- Statistics of Financial Markets (2013)
- The Complete Guide to Option Pricing Formulas (2007)
- The Complete Guide to Option Strategies
- The Complete Volume Spread Analysis System Explained
- The Options Course (2005)
- Think Like an Option Trader
- Trade Options Online (2009)
- Trading Binary Options: Strategies and Tactics
- Trading Options For Dummies (2008)
- Trading Price Action Trends (Bar by Bar)
- Your Options Handbook

### Volatility & VIX (11 books)
- Options for Volatile Markets (2nd Ed)
- Put Option Strategies for Smarter Trading (turbulent markets)
- Stock Market Volatility (2009)
- The Little Book of Sideways Markets
- The Swing Trader's Bible: Strategies to Profit from Market Volatility
- Trading Against the Crowd: Profiting from Fear and Greed
- Trading Regime Analysis: The Probability of Volatility (2009)
- Trading VIX Derivatives: Trading and Hedging Strategies (Futures, Options, ETNs)
- VIX (standalone reference)
- Volatile Markets Made Easy
- Volatility Trading (2013) — Euan Sinclair

### Technical Analysis (63 books)
*(Highlights only — full list available)**
- Volume Price Analysis (Anna Coulling)
- Technical Analysis of the Financial Markets (John Murphy, 1999)
- The New Trading for a Living (Alexander Elder, 2014)
- Price Action Breakdown
- Bollinger on Bollinger Bands
- MIDAS Technical Analysis
- Trading Price Action Trends (Al Brooks)
- Managing Risk with Technical Analysis

### Risk Management (8 books)
- Quantitative Financial Risk Management
- Stock Market Math: Essential Formulas
- Mastering the Stock Market: High Probability Market Timing
- Financial Risk Management: Applications in Market, Credit, Asset and Liability Management

### General Trading (notable)
- The Market Maker's Edge (2000)
- Trading Systems and Methods (2013)
- Keene on the Market (2013)

---

## 2. Top 5 Books for 0DTE SPX Options Trading

### 1. **Natenberg — Option Volatility and Pricing: Advanced Trading Strategies and Techniques**
**Why:** The definitive reference for options pricing mechanics. Covers BSM in practical depth, volatility surface construction, how Greeks interact, and — critically — how all of these behave as expiration approaches. Chapter on "The Effect of Time" is essential for understanding 0DTE gamma/theta dynamics. Every market maker learns from this book.

### 2. **Dan Passarelli — Trading Option Greeks**
**Why:** Focused entirely on how Greeks drive P&L. Covers second-order Greeks (charm, vomma, vanna) that dominate 0DTE behavior. Explains gamma scalping, delta-neutral hedging, and how market makers adjust positions — directly maps to understanding dealer GEX mechanics.

### 3. **Volatility Trading (2013) — Euan Sinclair**
**Why:** Quantitative approach to volatility as an asset class. Covers realized vs implied vol, vol forecasting models, variance risk premium, and vol regime detection. His framework for thinking about vol surface dynamics is directly applicable to building features for a 0DTE model.

### 4. **Trading VIX Derivatives — Russell Rhoads**
**Why:** Deep coverage of VIX futures term structure, VIX-SPX basis, volatility of volatility, and how VIX products move relative to SPX. Understanding VIX mechanics is essential for 0DTE because intraday vol regime shifts directly impact gamma P&L and dealer hedging behavior.

### 5. **Volume Price Analysis — Anna Coulling**
**Why:** Covers the relationship between volume and price that reveals institutional activity. Volume profile, point of control, value area concepts are directly relevant to identifying support/resistance zones where 0DTE options accumulate open interest and where dealer hedging creates self-reinforcing price magnets.

---

## 3. 0DTE Greeks Behavior (<8 Hours to Expiry)

### 3.1 Theta (Time Decay)

**Standard behavior:** Theta decays options value over time, accelerating as expiration approaches.

**0DTE specifics:**
- Theta is **not linear** — it follows approximately $\theta \propto \frac{1}{\sqrt{T}}$ where T is time to expiry
- At market open (6.5h to expiry), ATM SPX options with ~15 IV points might have **$2-4 of theta decay remaining**
- By 2pm ET (2h to expiry), theta acceleration is **extreme** — ATM options can lose 30-50% of remaining value per hour
- **Theta cliff:** Between 3:00-3:30pm ET, ATM options can drop from $3 to $0.50 in minutes if SPX doesn't move
- OTM options: theta is lower in absolute terms but higher as a percentage of premium
- **Practical implication:** Theta is the primary profit source for sellers and the primary enemy of buyers. Holding long 0DTE through lunch is a massive theta drag.

**Theta curve approximation for 0DTE ATM:**
$$\theta(t) \approx \frac{S \cdot \sigma}{2\sqrt{2\pi \cdot T(t)}}$$
where T(t) is annualized time remaining. At T = 1/252 (1 day), this is large. At T = 1/2016 (1 hour), it's ~2.8x larger.

### 3.2 Delta

**Standard behavior:** Delta measures directional exposure, ranges from 0 to ±1.

**0DTE specifics:**
- ATM delta starts ~0.50 at open but becomes increasingly **binary** as expiry approaches
- **Delta acceleration:** A 5-point SPX move at 9:45am might change delta by 0.05. The same move at 3:30pm changes delta by 0.30+
- Near expiry, delta approaches a **step function**: 0 below strike, 1 above strike
- This creates **pin risk** — options near ATM oscillate rapidly between deep ITM and OTM
- **Practical implication:** Delta is not a stable hedge ratio on 0DTE. A position that is 20-delta at 2pm can be 80-delta by 2:15pm on a 10-point move.

### 3.3 Gamma

**The most important Greek for 0DTE.**

**Standard behavior:** Gamma measures rate of delta change. Highest ATM, decreases with distance from strike.

**0DTE specifics:**
- Gamma is **inversely proportional to sqrt(time)**: $\Gamma \propto \frac{1}{S \cdot \sigma \cdot \sqrt{T}}$
- At 9:30am open, ATM gamma might be 0.02-0.04
- By 3:00pm, ATM gamma can reach **0.10-0.20** — a 5-10x increase
- **Gamma spike zone (3:00-4:00pm):** ATM gamma approaches infinity as T→0. This is where the most violent 0DTE moves happen.
- **Gamma concentration:** The closer to expiry, the narrower the "gamma band" — only strikes within ±5 points of SPX have meaningful gamma
- **Gamma flip:** The level where aggregate dealer gamma switches from long to short. Above the flip, dealers buy dips and sell rips (stabilizing). Below the flip, dealers sell dips and buy rips (destabilizing).
- **Practical implication:** Gamma is both the opportunity and the risk. High gamma means a correctly-timed 10-point move can turn a $1 option into $8. But short gamma means the same move destroys a sold spread.

### 3.4 Vega

**Standard behavior:** Vega measures sensitivity to implied volatility changes.

**0DTE specifics:**
- Vega is **near zero** for most of 0DTE — with hours to expiry, there isn't enough time for vol changes to matter much
- Exception: ATM options at market open still have meaningful vega (~$0.03-0.05 per vol point)
- **Vomma (vol of vol):** More important than vega on 0DTE. Rapid IV spikes (fear events, data releases) can temporarily re-inflate 0DTE premiums
- **Vol crush:** After scheduled events (CPI, FOMC), IV collapses instantly. 0DTE holders eating vol crush on top of theta = double devastation
- **Practical implication:** Vega matters most in the first 2 hours. After noon, gamma and theta dominate. IV level changes still serve as a regime indicator.

### 3.5 Charm (Delta Decay / DdeltaDtime)

**Often overlooked but critical for 0DTE.**

- Charm measures how delta changes with time (not price movement)
- For OTM options: charm pushes delta toward 0 as time passes (they become more OTM)
- For ITM options: charm pushes delta toward 1
- **0DTE magnitude:** Charm is massive. An option that's 25-delta at noon can be 5-delta by 2pm without any price change
- **Dealer impact:** As delta decays via charm, dealers who sold those options need to unwind hedges → net selling pressure for calls, buying pressure for puts
- **Practical implication:** Charm-driven delta decay is a directional signal — it creates predictable dealer flows in the afternoon

### 3.6 Speed (DgammaDspot)

- Third-order Greek: how gamma changes with spot price
- On 0DTE with extreme gamma, speed determines how quickly a position goes from manageable to catastrophic
- **Practical implication:** Speed captures "how fast things get out of control" — relevant for stop-loss sizing

---

## 4. Volatility Concepts

### 4.1 Implied Volatility Surface

The IV surface is a 3D structure: **strike × expiry × IV level**.

**Key features for 0DTE:**
- **Skew (vertical):** Puts trade at higher IV than calls (downside fear premium). For SPX, the 25-delta put might be 3-5 vol points above the 25-delta call.
- **Term structure (horizontal):** 0DTE IV is usually **lower** than 30-day IV in calm markets (steep term structure) but can **invert** during fear events.
- **Smile:** The curvature of IV across strikes. On 0DTE, the smile tends to be **steeper** because extreme moves are proportionally more likely in short time frames.
- **ATM IV level:** The anchor for 0DTE pricing. At the 0DTE timescale, a 15 IV means SPX is expected to move ±0.94% (about ±45 points) for the day, but most of that move has already occurred by afternoon.

### 4.2 Realized vs Implied Volatility

$$\text{Variance Risk Premium} = IV^2 - RV^2$$

- On average, IV > RV (sellers collect premium). This is the structural edge for premium sellers.
- **On 0DTE:** The VRP is compressed — less buffering time. But the edge still exists because of:
  - Retail overpaying for lottery tickets (far OTM calls)
  - Hedging demand keeping put IV elevated
  - Market maker spread capturing
- **Practical implication:** When RV >> IV (market moving more than options imply), the regime is dangerous for sellers. When IV >> RV (market choppy but options expensive), it's a seller's market.

### 4.3 Volatility Regimes

From Sinclair's framework:

| Regime | VIX Range | 0DTE Impact |
|--------|-----------|-------------|
| **Low vol** | <15 | Tight ranges, theta wins, gamma not dangerous |
| **Normal** | 15-20 | Standard behavior, moderate ranges |
| **Elevated** | 20-30 | Wider ranges, gamma becomes dangerous, skew steepens |
| **Crisis** | >30 | Extreme gamma, limit moves possible, liquidity vanishes |

**Regime transitions are more dangerous than the regime itself.** A move from VIX 14→22 is more destructive than a steady VIX 25 environment.

**Practical implication:** VIX rate-of-change matters more than the VIX level itself for 0DTE risk.

### 4.4 Intraday IV Dynamics

- IV is **not constant** throughout the day even on 0DTE
- Opening 30 min: IV often elevated (uncertainty premium), then settles
- Pre-data release: IV spikes (anticipation)
- Post-data release: IV crush
- Power hour (3:00-4:00pm): IV can spike on gamma-driven moves
- **Sticky strike vs sticky delta:** On 0DTE, the vol surface tends to operate in **sticky strike** mode (IV at a given strike stays roughly constant) rather than sticky delta (IV moves with the spot)

---

## 5. Dealer Mechanics (GEX, Market Maker Hedging)

### 5.1 What is GEX (Gamma Exposure)?

GEX = aggregate gamma exposure of options market makers across all strikes and expirations.

$$\text{GEX} = \sum_{\text{strikes}} \left( \text{OI}_{\text{calls}} \cdot \Gamma_{\text{call}} - \text{OI}_{\text{puts}} \cdot \Gamma_{\text{put}} \right) \times 100 \times S^2 \times 0.01$$

**Why it matters:** Market makers hedge by buying/selling the underlying (SPX futures, SPY, ES) to stay delta-neutral.

### 5.2 Positive vs Negative GEX

**Positive GEX (dealers long gamma):**
- Dealers must **sell** when price rises, **buy** when price falls
- This creates a **dampening effect** — price tends to revert to the GEX center
- Characterized by: low realized vol, tight ranges, mean-reversion
- SPX tends to "pin" to high-OI strikes
- **For 0DTE sellers:** This is the ideal environment — range-bound with theta working

**Negative GEX (dealers short gamma):**
- Dealers must **buy** when price rises, **sell** when price falls
- This creates an **amplifying effect** — moves accelerate
- Characterized by: high realized vol, trend days, breakouts
- SPX can move 50+ points intraday with momentum
- **For 0DTE sellers:** Extremely dangerous — a sold spread can go from +$2 to -$20 in minutes

### 5.3 Gamma Flip Level

The price level where aggregate dealer gamma switches sign.

- **Above gamma flip:** Positive gamma environment (stable, range-bound)
- **Below gamma flip:** Negative gamma environment (volatile, trending)
- The flip level moves daily based on options OI distribution
- **Typical location:** Usually near the put wall (highest put OI strike)
- **Practical implication:** Gamma flip level relative to current price is a stability measure. Above the flip = stable, below = volatile.

### 5.4 Charm Flows (Afternoon Dealer Hedging)

As time passes and delta decays (charm):
1. Dealers who sold OTM calls that are now decaying → they unwind their long delta hedges → **sell pressure**
2. Dealers who sold OTM puts that are now decaying → they unwind their short delta hedges → **buy pressure**
3. **Net effect depends on the put/call OI distribution at nearby strikes**

This creates predictable afternoon flows:
- If SPX is above the "call wall" (highest call OI): charm creates selling pressure, pulling price back
- If SPX is near the "put wall": charm creates buying support
- **Result: SPX tends to gravitate toward max pain / high-OI zones in the afternoon**

### 5.5 Pin Risk and Magnet Strikes

**Pin risk:** Near expiry, SPX can "stick" to a high-OI strike because:
1. Dealers are heavily hedged at that strike
2. As price approaches, delta changes require massive hedge adjustments
3. The hedge adjustments themselves push price back to the strike

**0DTE pinning behavior:**
- Strongest in the last 90 minutes
- Most common at round numbers (5900, 5950, 6000) where OI clusters
- Breaks when a large order overwhelms the pinning force
- **Practical implication:** Distance to nearest high-OI strike and historical pinning frequency indicate where SPX is likely to settle.

### 5.6 Dark Pool and Institutional Flow

- Large institutional trades execute in dark pools (off-exchange)
- These don't show in the tape until after execution
- **DIX (Dark Pool Indicator for SPX):** When dark pool buying is elevated, it's contrarian bullish
- **GEX + DIX combo:** Low GEX + high DIX = institutional accumulation in volatile environment = likely reversal

---

## 6. Volume Profile Concepts

### 6.1 Structure

Volume Profile shows the distribution of traded volume at each price level over a time period.

**Key levels:**
- **POC (Point of Control):** The price with the most traded volume — the "fairest" price where most participants agreed
- **Value Area (VA):** The range containing 70% of volume — typically ±1σ
- **Value Area High (VAH):** Upper edge of the value area
- **Value Area Low (VAL):** Lower edge of the value area

### 6.2 VP for 0DTE

**Developing profile (current session):**
- If price is inside VA: range-bound expectation, good for selling premium
- If price breaks above VAH: initiative buying, potential trend day
- If price breaks below VAL: initiative selling, potential breakdown

**Naked POC (unfilled):**
- A POC from a previous session that price hasn't revisited
- Acts as a **magnet** — 80%+ probability of being tested within 3 sessions
- **For 0DTE:** If a naked POC exists within the expected daily range, it's a high-probability target

**Volume distribution shapes:**
- **P-shape:** Volume concentrated at highs → accumulation day → bullish
- **b-shape:** Volume concentrated at lows → distribution day → bearish
- **D-shape (normal):** Balanced, centered → range day → theta wins
- **Double distribution:** Two POCs → trend reversal day

### 6.3 VP + Options Synergy

The intersection of Volume Profile and Options OI creates **institutional consensus zones:**
- Where high volume profile POC aligns with high options OI strike = **strong magnet**
- Price is very likely to settle near these zones by close
- **Practical implication:** Where high volume profile POC aligns with high options OI = strong price magnet

---

## 7. Risk Rules for 0DTE

### 7.1 Position Sizing

**From Natenberg and practitioner consensus:**

- **Max risk per trade:** 1-2% of account per trade (not per contract)
- **0DTE adjustment:** Because 0DTE moves are faster and stops can gap, use **0.5-1% max** per trade
- **Kelly Criterion adaptation:** $f^* = \frac{p(b+1) - 1}{b}$ where p = win rate, b = avg win/avg loss
  - For 0DTE credit spreads with 70% win rate and 1:3 risk/reward: $f^* = \frac{0.70 \times 4 - 1}{3} = 0.63$ → but Kelly overbets, use half-Kelly: **~30% of theoretical max**
- **Notional exposure:** Never exceed 10% of account in 0DTE notional

### 7.2 Stop Methodology

**For sold spreads (credit):**
- **Time-based stop:** If the position is -50% of credit by noon, close. Afternoon recovery is unreliable.
- **Delta-based stop:** If short strike delta exceeds 0.40, close regardless of P&L
- **Price-based stop:** If SPX breaches the short strike by $width/3, close
- **Never hold to zero:** A $3 credit spread has max loss $97 per contract. Letting "hope" manage the position → blown accounts

**For bought spreads (debit):**
- **Time-based**: If breakeven hasn't been reached by 2pm, close for salvage value
- **Cut losses at 50%:** If debit paid was $2 and it's worth $1, close — time is the enemy

### 7.3 Max Daily Loss

- **Hard stop:** -3% of account → done for the day, no revenge trades
- **Soft warning:** -1.5% → reduce size by 50%, last-trade-of-the-day mode
- From Mark Douglas's Trading in the Zone: **the primary purpose of risk management is to preserve the ability to trade tomorrow**

### 7.4 Event Risk

**Non-negotiable rules:**
- **No 0DTE positions through FOMC:** IV crush + large moves = unbounded risk
- **CPI/PPI/NFP:** Either be flat before or accept the gamma risk. CPI at 8:30am → positioning should happen pre-market
- **Triple witching / monthly OpEx:** Unusual pinning behavior, larger-than-normal gamma effects
- **Lunch chop (11:30am - 1:30pm ET):** Low volume, erratic moves, wide bid-ask. Reduce size or avoid.

### 7.5 Liquidity Rules

- **Bid-ask spread:** Never trade 0DTE SPX options with spread > $1.00 (ATM) or > $0.50 (for SPXW)
- **Maximum width:** Single-leg stops need $0.30 or less spread to execute reliably
- **Time-of-day:** Best liquidity at open (9:30-10:30) and close (3:00-4:00). Worst at lunch.
- **Slippage assumption:** Budget 1 tick ($0.05) per leg, so a spread has $0.10 slippage round-trip

---

## 8. Key Formulas

### 8.1 Black-Scholes for 0DTE Repricing

$$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$
$$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$

Where:
$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

**0DTE note:** When T < 1/252, the $\sqrt{T}$ term is tiny, making $d_1$ and $d_2$ very sensitive to small changes in S. This is why gamma explodes near expiry.

### 8.2 GEX Calculation

$$\text{GEX}_{\text{strike}} = \Gamma(K,T) \times \text{OI}(K) \times 100 \times S^2 \times 0.01$$

**Total GEX:**
$$\text{GEX}_{\text{total}} = \sum_{K} \text{GEX}_{\text{calls}}(K) - \sum_{K} \text{GEX}_{\text{puts}}(K)$$

(Convention: assume dealers are net short puts, net short calls → negation on puts)

### 8.3 Expected Move

$$\text{Expected Move}_{0\text{DTE}} = S \times \sigma_{\text{ATM}} \times \sqrt{\frac{T_{\text{remaining}}}{252}}$$

At open (T = 1/252): $\text{EM} = S \times \sigma / \sqrt{252} \approx S \times \sigma \times 0.063$

Example: SPX 5900, IV 15%: EM = 5900 × 0.15 × 0.063 = ±$55.7

### 8.4 Theta Acceleration

Theta at time T remaining:
$$\theta_{\text{ATM}} \approx \frac{-S \cdot \sigma}{2\sqrt{2\pi T}}$$

Ratio of theta at 1 hour vs theta at 4 hours:
$$\frac{\theta_{1h}}{\theta_{4h}} = \sqrt{\frac{4}{1}} = 2.0$$

So ATM theta **doubles** every time remaining is quartered.

---

## 9. Summary: Key Principles of 0DTE Trading

1. **Time is the dominant variable on 0DTE.** More than any indicator, where you are in the day determines the regime.
2. **Gamma is king.** Every 0DTE strategy is either exploiting or managing gamma risk.
3. **Dealers are the market.** Their hedging creates predictable flows (GEX positive = dampening, GEX negative = amplifying).
4. **Vol regime changes everything.** A strategy tuned for VIX 13 will blow up at VIX 25. The model must be regime-aware.
5. **Liquidity is not constant.** Bid-ask spreads widen at lunch and during stress. Execution quality varies.
6. **Risk management is not optional.** The maximum loss on any single 0DTE trade must be capped regardless of conviction.
7. **The afternoon is different from the morning.** Charm flows, gamma spike, and pinning create a fundamentally different microstructure after 2pm.
8. **Avoid trading during events.** FOMC, CPI, NFP — the edge disappears and the risk is unbounded.
9. **VRP exists but is thin.** Selling premium works statistically but requires strict discipline. One uncapped loss erases many wins.
10. **Pin risk is real.** Near expiry, high-OI strikes become magnets. Don't fight the pin unless you have strong evidence.
