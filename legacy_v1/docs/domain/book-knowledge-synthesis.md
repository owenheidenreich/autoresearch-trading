# Book Knowledge Synthesis — 7 Books for SPX 0DTE Long Options

Compiled 2026-03-27 from deep reading of 7 trading books. Organized by actionable theme, not by book. Each finding tagged with source.

---

## 1. Greeks Near Expiry (0DTE Critical)

### Gamma Explosion at ATM
- ATM gamma at 92 days: ~0.08. At 7 days: ~0.35. At 0DTE: maximum possible. [Passarelli]
- OTM/ITM gamma → 0 at expiry. Only ATM spikes. [Passarelli]
- Short-term ATM with low IV = highest gamma of any option config. [Passarelli]
- Gamma distribution at 7d: ATM ~0.35, one strike OTM ~0.15, two strikes OTM ~0.05, three+ ~0.00. [Passarelli]
- **For 0DTE: effectively ONE active gamma strike (ATM).** OTM5/OTM10 are near-zero gamma binary bets. [Passarelli]

### Theta Acceleration
Passarelli's 10-day ATM theta countdown:
| Days to Expiry | Theta |
|----------------|-------|
| 10 | 0.075 |
| 5 | 0.106 |
| 2 | 0.171 |
| 1 | 0.443 |

- Final day theta = 6x day-10. For 0DTE, theta accelerates throughout the session. [Passarelli]
- Theta decay is ~1/sqrt(T) — accelerates nonlinearly. [Passarelli, Operating Manual]
- ATM holding from 10am-2pm costs ~60-70% of remaining time value. [Passarelli]

### Delta Behavior at Expiry
- Delta becomes binary: ITM→1.0, OTM→0.0. [Passarelli]
- ATM delta swings violently as underlying crosses strike — a few SPX points can flip 0.30 to 0.70. [Passarelli]
- Charm: OTM 0DTE option with 0.35 delta at 10am → may have 0.15 delta by 2pm purely from time. [Passarelli]

### Vega Near Expiry
- 7-day ATM vega ~0.014. For 0DTE: vega is essentially irrelevant. [Passarelli]
- IV must change ~7 points to move option $0.10. [Passarelli]
- Confirms: price changes driven by delta/gamma/theta, not IV. [Passarelli, Sinclair]

### Gamma/Theta Breakeven
```
Breakeven daily move ≈ sqrt(2 × Theta / Gamma)
```
If theta=$50, gamma=0.04 → need $50 underlying move (~0.9% at SPX 5800). [Passarelli]

### Delta-Gamma P&L Formula
```
dV = [Delta + Gamma/2] × dS
```
Core formula for estimating option P&L from underlying movement. [Passarelli]

---

## 2. Volatility & The Variance Premium

### The Headwind: IV > RV for Indices
- Average VIX - 30d realized vol spread: **~3.09 points**. [Sinclair]
- This is INDEX-ONLY (not present for individual stocks). [Sinclair]
- **We are fighting this premium by being long options.** [Sinclair]

### Variance Premium by VIX Regime
| VIX Level | Avg IV-RV Spread | Implication |
|-----------|-----------------|-------------|
| < 20 | +27.75% | Expensive to be long |
| 20-30 | +19.78% | Moderately expensive |
| 30-40 | +13.73% | Getting cheaper |
| 40-50 | +9.26% | Near fair |
| > 50 | **-11.52%** | **Premium INVERTS — good for longs** |
[Sinclair]

### When to Buy Options
- VIX > 50: premium inverts, long options have statistical edge. [Sinclair]
- VIX below EWMA(0.95) = WORST time to buy options. [Sinclair]
- VIX above MA = options are relatively cheaper. [Sinclair]
- Narrow IV-HV spread = cheap options = better entry. [Rhoads]

### Volatility Regime Classification
| VIX Range | Regime | Action |
|-----------|--------|--------|
| < 15 | Tight | Small premiums, strong signals only |
| 15-25 | Normal | Standard sizing |
| 25-35 | Elevated | Widen stops 1.5x, reduce size 50% |
| > 35 | Crisis | Reduce size 75% OR mean-reversion only |
[Rhoads]

Bull regime median vol: 12.1%. Bear regime: 21.6%. [Sinclair]

### Rule of 16
```
Expected daily SPX move = VIX / 16
```
- VIX 16 → 1% daily move. VIX 32 → 2%. VIX 48 → 3%. [Rhoads]
- Use for stop-loss calibration and position sizing. [Rhoads]

### VIX Term Structure
- **Contango** (normal, ~80% of days): front-month futures < back-month. [Rhoads]
- **Backwardation** (crisis): front > back. In 2008, ~50% of days. [Rhoads]
- **VXV/VIX ratio**: >1.0 = normal. <1.0 = crisis/inverted. [Rhoads, Sinclair]
- IVTS (VIX/VXV): >1.0 = backwardation = historically good for long vol. [Sinclair]

### Volatility Properties
- Vol clusters — today's vol predicts tomorrow's. [Sinclair]
- Vol mean-reverts to long-term average. [Sinclair]
- Negative returns ~13% larger than positive for SPX (leverage effect). [Sinclair]
- Intraday returns have THINNER tails than overnight — favorable for 0DTE. [Sinclair]
- VIX rises 2-3x the SPX drop magnitude on worst days. [Rhoads]
- Volume and volatility strongly correlated (r=0.85 for SPY). [Sinclair]

### Volatility Estimators
| Estimator | Efficiency vs Close-to-Close |
|-----------|------------------------------|
| Close-to-Close | 1x (baseline) |
| Parkinson (high-low) | ~5x |
| Garman-Klass (OHLC) | ~7-8x |
| Yang-Zhang | Most efficient |
[Sinclair]

### GARCH(1,1) Forecast
```
sigma_t^2 = omega + alpha × r_{t-1}^2 + beta × sigma_{t-1}^2
E[sigma_{t+tau}^2] = V + (alpha+beta)^tau × (sigma_t^2 - V)
```
Produces exponential mean reversion toward long-term variance V. [Sinclair]

---

## 3. Volume-Price Analysis

### Core Rule: Effort vs Result (Wyckoff's Third Law)
Volume = effort. Price spread = result. They must match. [Coulling]

### Volume-Price Confirmation
| Price | Volume | Meaning |
|-------|--------|---------|
| Rising | Rising | Valid uptrend — stay long |
| Falling | Rising | Valid downtrend — stay short |
| Rising | Falling (3+ bars) | **Weak uptrend — reversal coming** |
| Falling | Falling (3+ bars) | Selloff losing steam — bounce coming |
[Coulling]

### Critical Anomalies (Highest-Value Signals)
| Signal | Volume | Price | Action |
|--------|--------|-------|--------|
| Fake move / trap | < average | Wide spread candle | Do NOT trade in direction |
| Exhaustion | > 2x average | Narrow spread candle | Reversal imminent |
| Stopping volume | > 2x average | Narrow + deep wick | Insiders halting decline |
| Successful test | < 0.7x average | Hammer/star in test zone | Breakout imminent |
| Buying climax | > 3x average | 2+ hammers at range bottom | Bullish reversal starting |
| Selling climax | > 3x average | 2+ shooting stars at top | Bearish reversal starting |
| Valid breakout | > 1.5x congestion avg | Close beyond range | Enter breakout direction |
| Fake breakout | < congestion avg | Poke beyond range | Expect reversal back |
[Coulling]

### Volume Profile Concepts
- **POC (Point of Control):** Price with highest accumulated volume = "fair value." Market gravitates toward it. [Coulling]
- **HVN (High Volume Nodes):** Price magnets — consolidation zones, strong S/R. [Coulling]
- **LVN (Low Volume Nodes):** Transition zones — price moves through quickly. [Coulling]
- **Value Area:** 70% of session volume. Inside = consolidation. Outside = trending. [Coulling]

### Volume Features for Model
1. **volume_price_divergence** — consecutive bars where price/volume directions disagree. >3 = strong reversal. [Coulling]
2. **candle_body_volume_ratio** — (body/avg body) / (volume/avg volume). >>1 = fake move. <<1 = exhaustion. [Coulling]
3. **stopping_volume_detector** — volume >2x avg AND spread <0.5x avg in downtrend. [Coulling]
4. **breakout_volume_confirmation** — volume ratio on breakout bar. >1.5 = confirmed. <0.8 = fakeout. [Coulling]
5. **volume_climax_detector** — volume >3x avg AND deep wick (>60% of range). [Coulling]

### Time-of-Day Volume
- Opening bars = amateur panic/euphoria → trap moves with wide spreads on low volume. [Coulling, Elder]
- Closing volume = professional consensus → more meaningful signals. [Elder]
- "High volume" threshold = 1.25x the 2-week average. [Elder]

---

## 4. Technical Indicators as Features

### MACD-Histogram (Elder's #1 Signal)
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
MACD-H = MACD Line - Signal Line
```
- MACD-H slope is the single most important signal. [Elder]
- Indicator Seasons: Spring (rising below zero = BUY), Summer (rising above = HOLD), Autumn (falling above = SELL), Winter (falling below = HOLD SHORT). [Elder]
- Divergences best 20-40 bars apart, 2nd peak ≤50% of 1st, must cross centerline between. [Elder]
- "Hound of Baskervilles": failed divergence → powerful breakout in opposite direction. [Elder]

### RSI
```
RS = avg_up / avg_down over N periods
RSI = 100 - 100/(1+RS)
```
- Recommended period for intraday: 7-9 bars (not standard 14). [Elder]
- Overbought >70, Oversold <30. [Elder]
- 5% Rule: adjust reference lines so RSI spends only 5% of time beyond each. [Elder]

### Force Index
```
Force Index = Volume × (Close - Close_prev)
```
- 2-bar EMA: buy when negative in uptrend, short when positive in downtrend. [Elder]
- 13-bar EMA: above zero = bulls, below = bears. [Elder]
- Combines price change AND volume into single number. [Elder]

### Impulse System
- Green: EMA rising AND MACD-H rising → no shorting. [Elder]
- Red: EMA falling AND MACD-H falling → no buying. [Elder]
- Blue: mixed → either direction. [Elder]
- Blue after Red = bears losing power. [Elder]

### Stochastic Oscillator
```
%K = (Close - Low_n) / (High_n - Low_n) × 100 (n=5)
```
- Oversold <20, Overbought >80. Best in ranges, not trends. [Elder]

### Bollinger Bands
- 21-bar EMA ± standard deviation. [Elder]
- Narrow bands (low width) precede breakouts. [Elder]
- Band width = (upper-lower)/middle as volatility squeeze detector. [Elder]

### ATR & Channel
```
True Range = max(H-L, |H-C_prev|, |L-C_prev|)
ATR = EMA of True Range (N=14)
```
- 1 ATR from price = normal pullback. 2 ATR = extended. 3 ATR = extreme reversal zone. [Elder]

### Triple Screen System (Multi-Timeframe)
For 1-min bot with 120-bar window:
| Screen | Timeframe | Indicators |
|--------|-----------|------------|
| 1 (Trend) | 5-min (aggregate 5 bars) | EMA(13/26) slope, MACD-H slope |
| 2 (Oscillator) | 1-min (native) | Force Index 2-bar, RSI(7) |
| 3 (Entry) | 1-min (native) | Trailing stop, EMA penetration |
[Elder]

---

## 5. Risk Management

### Position Sizing

#### Elder's 2% Rule
- Never risk more than 2% of account on single trade. Pros use 0.5-1%. [Elder]
- Iron Triangle: `max_contracts = (account × 0.02) / (entry - stop)`. [Elder]

#### Elder's 6% Rule
- Stop ALL new trades when monthly losses + risk on open trades ≥ 6%. [Elder]
- With 2% per trade → max 3 simultaneous positions. [Elder]
- Daily equivalent for 0DTE: stop when daily losses + open risk ≥ 6%. [Elder]

#### Kelly Criterion
```
f = r / sigma^2  (continuous outcomes)
f = (p×w - q×l) / (w×l)  (binary outcomes)
```
- Full Kelly: 33% chance bankroll halves before doubling. [Sinclair]
- Half-Kelly standard practice. Quarter-Kelly for conservative. [Sinclair]
- Need 100+ trades before Kelly sizing is reliable (std(f)=0.097 at 100 trades). [Sinclair]
- Bayesian edge adjustment: `p_adjusted = (w+1)/(N+2)`. [Sinclair]

#### VIX-Based Sizing
- VIX 15 → 1x base size. VIX 30 → 0.5x. VIX 45+ → 0.25x or sit out. [Rhoads]
- Scale stop-loss with VIX/16 expected daily move. [Rhoads]

#### Douglas Rules
- Size must be comfortable with worst-case (losing ALL trades in sample). [Douglas]
- No size increase after wins — leads to "boom-and-bust" cycle. [Douglas]
- Measure over 20+ trade windows, not individual trades. [Douglas]

### Stop-Loss Methods

#### ATR-Based Stops
- Minimum 1 ATR from entry. 2 ATR is safer. [Elder]
- Trailing 2-ATR stop at every bar. [Elder]

#### SafeZone Stop
1. Measure all downside penetrations below prior bar's low over 10-20 bars. [Elder]
2. Average penetration depth. [Elder]
3. Stop at 2x average penetration for longs, 3x for shorts. [Elder]

#### Greek-Aware Stops
```
Adverse move for X% loss ≈ X% × premium / delta
```
For 0DTE ATM: gamma cushions adverse moves (delta shrinks as you lose). [Passarelli]

#### VIX-Calibrated Stops
```
Base stop = (VIX/16) × SPX_price × strike_delta_fraction
```
- In contango: 1.0x base. In backwardation: 1.5-2.0x (wider). [Rhoads]

### Daily Loss Limits
- 5% daily loss → block new entries. [Douglas, Operating Manual]
- 10% session loss → kill switch. [Douglas, Operating Manual]
- EOD flatten all positions at 4:00 PM ET. [Douglas]

### Profit Protection
- Take first third at small reliable level. [Douglas]
- Take second third at S/R target. Move stop to breakeven. [Douglas]
- Let final third run. [Douglas]
- Elder: protect 33% of open profit with trailing stop after breakeven. [Elder]

---

## 6. Time-of-Day Trading Rules

### Morning (9:35-10:30 ET) — Best Window
- Strongest trending period. [Elder, Operating Manual]
- Theta per minute is lowest (more total time remaining). [Passarelli]
- Best window for long 0DTE entries. [Passarelli]
- Opening bar volume = amateurs. Trap moves common (wide spread, low volume). [Coulling, Elder]

### Lunch (11:30-13:30 ET) — Avoid
- Volume drops, trends stall. [Elder]
- Oscillators give false signals. [Elder]
- Theta bleeds with no gamma scalping opportunity. [Passarelli]
- Model's v8.1 collapse to lunch-only trades was a dysfunction. [Project History]

### Afternoon (13:30-15:00 ET) — Caution
- Only enter with strong directional catalyst. [Passarelli]
- Theta per minute becomes extreme. [Passarelli]
- Breakeven move becomes unrealistically large. [Passarelli]

### Close (15:30-16:00 ET) — Extreme Gamma
- Gamma peaks for ATM options. [Passarelli, Operating Manual]
- Market maker hedging creates pin risk or amplified moves. [Sinclair]
- Closing volume = professional consensus, more meaningful. [Elder]

### Weekly Patterns
- Bull markets: lows Mon/Tue, highs Thu/Fri. Bear: reversed. [Elder]

---

## 7. Probabilistic Framework

### Casino Model [Douglas]
- Operate like a casino, not a gambler.
- Take EVERY valid signal (don't cherry-pick).
- Evaluate over 20+ trade sample sizes.
- Accept random distribution of wins/losses.

### Five Fundamental Truths [Douglas]
1. Anything can happen.
2. Don't need to know what happens next to make money.
3. Random distribution between wins/losses for any edge.
4. An edge = higher probability, not certainty.
5. Every market moment is unique.

### Rules: Hard-Coded vs Adaptive [Douglas]
- **Hard-coded (immutable):** Stop-loss execution, daily loss limits, position size caps, EOD flatten, pre-define risk before entry.
- **Adaptive (model-learned):** Entry identification, direction selection, profit targets, stop distance, time-of-day preferences.

### Fear-of-Missing-Out / "Would I Do It Now?" [Douglas, Passarelli]
- At any point during a trade: "If I had no position, would I enter NOW at current prices?"
- If no → exit. This is exactly what the gate head's exit decision should embody. [Passarelli]

### OTM Calls Are the Worst Buy [Sinclair]
- Lottery-ticket seekers overpay. Returns worsen the farther OTM.
- **Prefer ATM.** Confirmed by gamma concentration at ATM near expiry. [Passarelli, Sinclair]

---

## 8. New Feature Candidates (Prioritized)

### High Priority (Strong Evidence from Multiple Books)
1. **MACD-Histogram slope** — Elder's #1 signal, Indicator Seasons. [Elder]
2. **Force Index (2-bar EMA)** — combines price + volume, entry timing. [Elder]
3. **Volume-price divergence counter** — 3+ bars = strong reversal signal. [Coulling]
4. **Candle body-volume ratio** — effort vs result anomaly detection. [Coulling]
5. **VIX/16 normalized expected move** — stop calibration, position sizing. [Rhoads]
6. **IV-HV spread** — option cheapness/richness indicator. [Sinclair, Rhoads]
7. **Gamma/theta ratio** — cheap vs expensive gamma for long positions. [Passarelli]
8. **RSI(7)** — fast oscillator for 1-min bars. [Elder]

### Medium Priority (Good Evidence, Single Source)
9. **VIX term structure slope** — contango/backwardation regime signal. [Rhoads]
10. **Impulse System color** — EMA slope + MACD-H slope composite. [Elder]
11. **Volume profile POC distance** — how far from "fair value." [Coulling]
12. **Breakout volume confirmation** — >1.5x = real, <0.8x = fake. [Coulling]
13. **Stopping volume detector** — high volume + narrow spread in downtrend. [Coulling]
14. **EMA value zone position** — above/in/below EMA(13)/EMA(26) zone. [Elder]
15. **Multi-timeframe trend agreement** — Triple Screen concept. [Elder]

### Lower Priority (Interesting but Less Proven for 0DTE)
16. Bollinger Band width (squeeze detector). [Elder]
17. Stochastic %K for range-bound detection. [Elder]
18. ADX for trend strength measurement. [Elder]
19. OBV trend direction divergence. [Elder]
20. GLD/VIX ratio signal (<2.75 buy, >6.25 sell). [Rhoads]

---

## 9. Architecture Validation

### What the Books Confirm About Our Design
1. **ATM preference is correct.** Gamma concentration at ATM near expiry is extreme. [Passarelli]
2. **Gate head exit is critical.** Passarelli's "Would I Do It Now?" + Douglas's pre-defined risk = gate head architecture. [Passarelli, Douglas]
3. **Disabled value head is correct.** Douglas: "trying to know what happens next" is the fundamental error. Sinclair: vega near zero at 0DTE. [Douglas, Sinclair]
4. **Risk head is exactly right.** Pre-defining risk (stop_pct, size_frac) before each trade is THE most important discipline. [Douglas, Elder]
5. **Stop-loss > model_exit priority is correct.** "Only the best traders cut losses without reservation or hesitation." [Douglas]
6. **Morning bias is correct.** Theta per minute lowest, trending strongest. [Passarelli, Elder]
7. **Daily loss limits (5%/10%) well-calibrated.** Close to Elder's 2%/6% framework. [Elder, Douglas]

### What Suggests Improvements
1. **Afternoon theta penalty.** Heavily penalize entries after 2pm — breakeven move becomes unrealistically large. [Passarelli]
2. **Realized vs implied vol comparison.** If intraday realized vol > IV, conditions favor long gamma. Feature or gate condition. [Passarelli, Sinclair]
3. **Variance premium awareness.** We fight ~3 vol points. Edge must come from timing + direction + exit speed. [Sinclair]
4. **Partial profit-taking.** Scale out in thirds at predefined levels. [Douglas]
5. **VIX regime-conditional sizing.** Inverse scaling with VIX level. [Rhoads]
6. **Event day IV crush risk.** Avoid long entries immediately before data releases. Enter after IV crush if direction confirmed. [Passarelli]

---

## Sources
- **Passarelli** — "Trading Option Greeks" (Dan Passarelli)
- **Sinclair** — "Volatility Trading" 2nd Ed (Euan Sinclair, 2013)
- **Rhoads** — "Trading VIX Derivatives" (Russell Rhoads)
- **Coulling** — "Volume Price Analysis" (Anna Coulling)
- **Elder** — "The New Trading for a Living" (Alexander Elder)
- **Douglas** — "Trading in the Zone" (Mark Douglas)
- **Stock Math** — "Stock Market Math" (Thomsett) — limited relevance, basic TA formulas only
