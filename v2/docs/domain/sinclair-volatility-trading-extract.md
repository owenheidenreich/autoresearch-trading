# Sinclair "Volatility Trading" (2nd Ed, 2013) -- Actionable Extraction

Source: Euan Sinclair, "Volatility Trading", 2nd Edition, Wiley 2013.
Extracted for: SPX 0DTE long options transformer model (ART2).

---

## 1. Volatility Forecasting

### Core Properties of Volatility (Stylized Facts)
- **Volatility clusters.** Squared and absolute returns show significant autocorrelation. "Tomorrow's volatility will be the same as today's" is a robust heuristic. Autocorrelation of absolute returns > squared returns.
- **Volatility mean-reverts.** Short-term vol reverts to long-term mean. VIX annualized daily vol = 0.96, weekly = 0.84, monthly = 0.59 (1990-2011). The declining vol-of-vol across timescales confirms mean reversion.
- **Volatility is log-normally distributed.** Heavily right-skewed -- vol spends much more time in low states than high states.
- **Leverage effect (asymmetry).** Vol rises on declines, falls on rallies. Average negative daily return for SPY is ~13% larger than average positive return. Particularly pronounced for indices. This is NOT explained by financial leverage -- it persists across asset classes with positive expected returns.
- **Vol clustering is stronger in developed markets, more pronounced in bear markets** but decays faster in bears. During crashes, autocorrelation decays fastest.
- **Volume and volatility are strongly correlated** (r=0.85 for SPY daily range vs volume, r=0.68 for absolute returns vs volume).

### Bull vs Bear Volatility Regimes (S&P 500, 1990-2011)
| Regime | Median 30-day Vol |
|--------|-------------------|
| Bull (above 200-day MA) | **12.1%** |
| Bear (below 200-day MA) | **21.6%** |

### EWMA Model
```
sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2
```
Lambda typically 0.9 to 0.99. Simple but ignores mean reversion -- forecast for all future days is the same.

### GARCH(1,1) Model
```
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
```
Where:
- Long-term variance V = omega / (1 - alpha - beta)
- gamma + alpha + beta = 1
- Typical fit (MSFT): omega=0.00000505, alpha=0.053, beta=0.884

**Forecast at horizon tau:**
```
E[sigma_{t+tau}^2] = V + (alpha + beta)^tau * (sigma_t^2 - V)
```
This produces exponential mean reversion toward V. GARCH cannot produce humped term structures seen in markets.

### Volatility Cones
Place current IV and HV in percentile context over rolling windows (e.g., 30-day, 60-day). Selling 1-month IV at 35% because it's in the 90th percentile over 2 years is more sensible than relying on GARCH point forecasts. **Context matters more than point estimates.**

### Quick Conversion Rules
- **Daily move to annualized vol:** multiply average |daily return| by ~20 (not 16)
- Precisely: average_move = 0.04986 * sigma * S, roughly sigma*S/20
- Volatility of 16% annualized corresponds to ~0.8% average daily return (not 1%)

### Volatility Estimators (Efficiency vs Close-to-Close)
| Estimator | Efficiency | Notes |
|-----------|-----------|-------|
| Close-to-Close | 1x (baseline) | 30 samples gives 95% CI of +/- 25% of true value |
| Parkinson (high-low) | ~5x | Biased LOW due to discrete sampling (0.55x at N=5, 0.86x at N=100) |
| Garman-Klass (OHLC) | ~7-8x | Uses open, high, low, close |
| Yang-Zhang | Most efficient | Handles overnight jumps |

**Parkinson estimator:**
```
sigma = sqrt(1/(4N*ln2) * sum(ln(hi/li))^2)
```

**Key insight:** Most kurtosis in stock returns comes from overnight returns (earnings, halted trading). Intraday kurtosis is significantly lower.

### Sampling Error Formula
```
var(s) = sigma^2 / (2N)
```
30 daily observations: standard deviation of vol estimate = sigma/sqrt(60) ~= 13% of true value. This is why 30-day rolling vol is noisy.

---

## 2. VIX Regime Classification

### VIX-Based Variance Premium by Regime (Jan 1990 - May 2012)

| VIX Level | Days in Sample | Avg IV-RV Spread |
|-----------|---------------|-----------------|
| VIX < 20 | 3,102 | **+27.75%** |
| 20 < VIX < 30 | 1,925 | **+19.78%** |
| 30 < VIX < 40 | 407 | **+13.73%** |
| 40 < VIX < 50 | 111 | **+9.26%** |
| VIX > 50 | 56 | **-11.52%** |

**Critical finding:** The variance premium is PROPORTIONALLY LARGER when VIX is low. When VIX > 50, implied is actually BELOW realized on average. Mean reversion expectation dominates at high VIX.

### VIX Futures Term Structure as Signal
- VIX < 20: futures curve upward sloping 78% of the time (but downward 22% -- anomalies create trading opportunities)
- VIX 40-50: curve upward sloping 46% of the time (not always backwardated even at extreme levels)
- VIX futures basis (front - cash) averaged +0.66 points (2006-2011), range: low decile -1.40, high decile +2.01

### Implied Volatility Term Structure (IVTS) Signal
```
IVTS = VIX / VXV
```
Where VXV = CBOE 3-month implied vol index.
- IVTS > 1.0 = downward sloping = backwardation = historically VIX tends to rise
- IVTS < 0.91 = steep contango = bearish vol outlook

**Portfolio weights by IVTS (Donninger 2011):**
| IVTS Level | VXX Weight | VXZ Weight |
|------------|-----------|-----------|
| <= 0.91 | -0.60 | +0.40 |
| 0.91-0.97 | -0.32 | +0.68 |
| 0.97-1.05 | -0.25 | +0.75 |
| > 1.05 | -0.10 | +0.90 |

Result: 99% annualized return, Sharpe 2.62, max drawdown 12%.

### Strong Negative Correlation: VIX Changes vs S&P Returns
This is the most exploitable regularity for directional options trading. S&P down -> VIX up, and vice versa.

---

## 3. Variance Risk Premium

### Core Finding
**Index implied volatility is systematically higher than subsequent realized volatility.** This is the "variance premium" -- the most reliable source of beta in volatility trading.

- Average VIX - 30d realized vol spread: **~3.09 points** (Figure 4.5, 2006-2008 data)
- VIX is almost always above 30-day rolling realized vol
- **This premium is an INDEX effect -- it does NOT persist for individual equities** (Table 11.1 shows Dow stocks have mixed results selling vol)

### Profitability of Selling Index Vol (QQQ, 2000-2010)

| Strategy | Ann. Return | Sharpe | Max DD |
|----------|------------|--------|--------|
| Sell 10-delta strangle | 41.6% | 1.32 | 20.6% |
| Sell 20-delta strangle | 93.3% | 1.21 | 34.2% |
| Sell 30-delta strangle | 64.1% | 1.14 | 32.9% |
| Sell ATM straddle | 58.6% | 1.14 | 32.7% |
| Sell 20-delta, VIX < 35 only | 41.7% | 1.30 | 31.4% |
| Sell 20-delta, VIX < MA only | 69.8% | **1.68** | **10.3%** |
| Sell 30-delta risk reversal | 22.2% | 0.69 | 26.9% |

**Key insights for our LONG options bot:**
1. The variance premium works AGAINST long options buyers on average
2. To profit buying options, we need TIMING ALPHA that overcomes this headwind
3. The premium narrows when VIX is high (>40) and inverts when VIX > 50 -- those are the best times to be long
4. VIX below its EWMA (decay=0.95) is the worst time to buy options

### Sources of the Premium
1. **Correlation premium:** Index vol contains implied correlation; dispersion trading (short index vol, long component vol) is profitable with similar Sharpe ratios. Crashes increase correlation.
2. **Skewness premium:** ~50% of the excess return from selling OTM puts comes from the correlation between returns and volatility (realized skewness). Kozhan, Neuberger, and Schneider (2011).
3. **Insurance premium:** Sellers are compensated for providing downside protection
4. **OTM call premium:** Lottery-ticket seekers overpay for OTM calls. Returns to long calls worsen the farther OTM they are.

### Bakshi-Madan Model for Variance Spread
```
sigma_rn^2 - sigma^2 ~= -gamma * sigma * skew + (gamma^2/2) * sigma^2 * (kurtosis - 3)
```
Where gamma = risk aversion parameter. Higher negative skewness and higher kurtosis widen the spread.

### Implication for 0DTE Long Bot
**We are fighting the variance premium by being long.** The edge must come from:
- Timing (buy when premium is thin: high VIX, VIX > MA, IVTS > 1.0)
- Direction (correct prediction of underlying moves that trigger gamma profits)
- Exit speed (capture gamma scalp before theta bleeds it away)

---

## 4. Greeks Near Expiry

### Instantaneous P/L for a Hedged Long Option
```
P/L = (1/2) * S^2 * Gamma * (sigma_realized^2 - sigma_implied^2) * dt
```
Or equivalently:
```
P/L = Vega * (sigma_realized - sigma_implied)
```
The first form shows P/L accrues each bar proportional to gamma; the second shows total P/L over life.

### Gamma-Vega Relationship
```
Vega = sigma * T * S^2 * Gamma
```
As T -> 0 (0DTE), vega collapses but gamma explodes for ATM options. This is the 0DTE trader's weapon.

### ATM Option Approximation
```
C ~= (1/sqrt(2*pi)) * S * sigma * sqrt(T)
```
ATM straddle vega ~= (2/sqrt(2*pi)) * S * sqrt(T)

### Path Dependency Is Critical
- Even if you correctly forecast realized vol, discrete hedging creates P/L variance proportional to:
```
sigma_PL ~= (pi/4) * Vega * sigma / sqrt(N)
```
where N = number of hedging intervals.
- **Timing of moves matters hugely.** A jump near expiration has maximum impact because gamma is largest at-the-money near expiry.
- Finishing near the strike at expiration maximizes P/L for both long AND short gamma positions (gamma is largest there).

### Hedging Volatility Choice Matters
| Position | Market | Hedge Vol Bias |
|----------|--------|---------------|
| Short Gamma | Trending | Low (hedge less, let deltas run) |
| Short Gamma | Range Bound | High (hedge more) |
| Long Gamma | Trending | High (hedge less) |
| Long Gamma | Range Bound | Low (hedge more, capture scalps) |

**For 0DTE long options without delta hedging:** We are purely long gamma. Our profit comes from the underlying moving MORE than implied vol suggests. We want big moves, especially near expiry when gamma is highest.

### Market Maker Feedback at Expiry
When market makers are long gamma near a strike at expiry, they buy below the strike and sell above -- compressing realized volatility. This HURTS long gamma positions. When they are short gamma, the reverse happens (amplifying vol). This pin risk is a real 0DTE consideration.

---

## 5. Position Sizing (Kelly Criterion)

### Kelly Formula for Continuous Outcomes
```
f = r / sigma^2
```
Where:
- f = fraction of bankroll to risk
- r = expected return of the trade (over risk-free rate)
- sigma^2 = variance of the trade outcome

**This is the core formula.** For binary outcomes: f = (p*w - q*l) / (w*l)

### Growth Rate at Fraction of Kelly
```
GR = f - (f^2 * r^2) / (2 * sigma^2)
```
Maximized at f=1 (full Kelly) where GR_max = r^2 / (2*sigma^2). Growth rate is ZERO at f=2 (2x Kelly).

### Critical: Asymmetry of Over/Under-betting
- **Betting more than Kelly is worse than betting less.** At f=1+x, growth equals f=1-x but with MORE volatility.
- Betting 2x Kelly = zero expected growth (equivalent to not trading at all, but with massive variance).

### Drawdown Probabilities

| Fraction of Kelly | P(double before halve) |
|-------------------|----------------------|
| 1.0 (full) | 66.7% |
| 0.8 | 73.9% |
| 0.6 | 83.4% |
| 0.4 | 94.1% |
| 0.2 | 99.8% |

**At full Kelly, 1/3 chance your bankroll halves before it doubles.** This is why half-Kelly or quarter-Kelly is standard practice.

### Fractional Kelly Rationale
1. **Pragmatic middle ground** between max growth and acceptable drawdowns
2. **Bayesian uncertainty acknowledgment.** Half-Kelly = averaging your edge estimate with zero (the possibility you have no edge)
3. **There is NO utility function that fractional Kelly maximizes.** It's a practical hack, not theoretically optimal.

### Parameter Estimation Error
Naive win rate estimate (e.g., 6/10 = 0.6) **overestimates** the true probability. Bayesian correction:
```
p_adjusted = (w + 1) / (N + 2)
```
For 6 wins in 10: p = 7/12 = 0.583 (not 0.6).

Kelly ratio accounting for estimation uncertainty:
```
f = (2w - N) / (N + 2)
```
After 100 trials, naive is off by only 2%. **But variance of f is large: for 6/10 wins, std(f) = 0.274.**

After 60/100 wins: std(f) = 0.097. Need ~100 trades minimum before Kelly sizing is reliable.

### What Is Bankroll?
Bankroll = amount you can lose before the strategy is abandoned. NOT haircut/margin. NOT total account value (unless you'd stop trading entirely).

For someone with $1M trading account and $5M total net worth: could use bankroll=$5M with multiplier=0.1 instead of bankroll=$1M with multiplier=0.5.

### Mean-Reversion Optimal Entry
For a mean-reverting process with normal deviations:
```
Optimal entry = 0.75 * sigma from the mean
```
This maximizes total P/L = 2T * S * (1 - N(S)), where S = distance from mean in std devs. Verified empirically on VIX: peak P/L near 0.75 sigma. Err on the side of caution (trade slightly less often).

---

## 6. Vol Surface / Skew

### PCA of the Vol Surface
Principal component analysis shows:
1. **Parallel shift** (level change) = 65-80% of all vol surface movement
2. **Slope change** (skew tilt) = most of remainder
3. Higher-order shape changes are much less important

**Implication:** Getting the VOL LEVEL right is far more important than modeling smile shape.

### Sticky Strike vs Sticky Delta
- **Sticky strike:** Vol of a given strike is unchanged as underlying moves. Observed in range-bound markets.
- **Sticky delta:** Vol curve moves with underlying so same-delta options keep same vol. Observed in trending markets.
- Neither rule is universally correct. Derman (1999) on S&P 500 options.

### Index Skew Is Steeper Than Single Stocks
Because index vol = f(component vols, correlations), and correlations increase in crashes:
```
sigma_index = sqrt(sum(wi^2 * sigma_i^2) + 2*sum(wi*wj*rho_ij*sigma_i*sigma_j))
```
Even if all components have flat vol surfaces, the index exhibits a smile if crash-correlated.

### Skew as a Normalized Signal
Dividing all strikes' vols by ATM vol gives a remarkably stable curve across time. Scale = vol(delta)/vol(ATM):
- 10-delta put: ~1.46x ATM
- 20-delta put: ~1.27x ATM
- ATM: 1.00
- 20-delta call: ~0.84x ATM
- 10-delta call: ~0.78x ATM

### Why Smiles Exist
1. End users buy downside protection (puts) and sell calls against long stock
2. Market makers hedge dynamically, amplifying the effect
3. Implied correlation premium (index effect)
4. Actual returns are negatively skewed and fat-tailed
5. Takeover premium in upside strikes (single stocks)

### Skewness and Kurtosis Measurement
```
Skewness: mu_3 = (1/N) * sum((xi - x_bar)^3) / sigma^3
  Var(mu_3) ~= 6/N

Kurtosis: mu_4 = (1/N) * sum((xi - x_bar)^4) / sigma^4
  Var(mu_4) ~= 24/N
```
S&P 500 daily excess kurtosis (1950-2011): **21.3**. 24 days below -5%, 17 days above +5%. Oct 19, 1987: -20.47% (probability under normal: ~10^-88).

### Corrado-Su Smile Model
Extends BSM by adding skewness and kurtosis:
```
C = C_BSM + mu_3 * Q3 + (mu_4 - 3) * Q4
```
Where Q3, Q4 are functions of the BSM variables. This maps between implied vol and physical moments.

---

## 7. Risk Management

### Stop-Loss Philosophy
**"Adding arbitrary price-based stops to a trading system is a poor idea."** For mean-reverting volatility trades, the rationale STRENGTHENS as the trade moves against you (until the regime changes).

- Exit when your thesis is wrong, not when price hits an arbitrary level
- For trending instruments: cut losses (consistent with being wrong)
- For mean-reverting instruments: consider adding (consistent with trade getting better)
- **But always evaluate: has the regime changed?** The real danger is that the "mean" has shifted.

### Drawdown Rules of Thumb
- **10% drawdown:** causes "marketing issues and uncomfortable questions"
- **30% drawdown:** "starts to raise questions of survival"
- Drawdowns are a fact of life. Pre-decide the plan for getting out.
- Maximum daily loss should be tracked and compared to expectations

### Key Risk Principle
**"Never estimate the magnitude of risks from within the same model that you priced them with."** BSM is for finding trades, NOT for controlling risk. Tail risk must be evaluated separately.

### Hedging Frequency vs P/L Variance
P/L standard deviation for hedged option position:
```
sigma_PL ~= (pi/4) * Vega * sigma / sqrt(N)
```
Doubling hedging frequency reduces P/L variance by ~30%. But transaction costs increase linearly.

### Dynamic Hedging Cannot Handle Jumps
"Dynamic hedging cannot be relied on. It must be combined with static hedging using other options. Only this can offset the risks associated with jumps."

For 0DTE: The underlying WILL have discontinuous moves (economic releases, large orders). Long options position provides natural protection against adverse jumps (limited downside).

### Position Limits
"A trade should be big enough that the profits mean something, but not so big that the losses are catastrophic. If this optimum size can't be found, the trade probably doesn't have enough edge to begin with."

---

## 8. Quantitative Formulas and Thresholds for Features/Rules

### Feature Candidates from Sinclair

| Feature | Formula/Threshold | Use |
|---------|------------------|-----|
| VIX regime | <20, 20-30, 30-40, 40-50, >50 | Variance premium direction |
| VIX vs MA | VIX < EWMA(0.95) | Best time to sell vol; worst for buying |
| IVTS | VIX/VXV | >1.0 = backwardation, good for long vol |
| IV-HV spread | IV - 30d realized vol | Narrower spread = cheaper options |
| Avg spread | ~3.09 for S&P 500 | Baseline; deviation from this = signal |
| Vol percentile | Current vol in vol cone | >90th percentile = expensive; <10th = cheap |
| Bull/bear regime | Price vs 200-day MA | Bear: median vol 21.6%; Bull: 12.1% |
| Vol of vol | Annualized VIX vol | Daily: 0.96, Weekly: 0.84, Monthly: 0.59 |
| Parkinson vol | sqrt(1/(4N*ln2) * sum(ln(H/L)^2)) | 5x more efficient than close-to-close |
| Leverage asymmetry | avg(neg returns) / avg(pos returns) | ~1.13 for SPY; higher = more skewed market |
| Kelly fraction | r / sigma^2 | Position sizing; use half-Kelly in practice |
| Mean-revert entry | 0.75 sigma from mean | Optimal entry for mean-reverting processes |
| Kurtosis | S&P 500 excess kurtosis = 21.3 | Fat tails are extreme; plan for 5-sigma moves |

### Rules Derived from Sinclair

1. **Long options are fighting the variance premium (~3 vol points for S&P).** Edge must come from timing + direction.
2. **The premium inverts (favors long) when VIX > 50.** At VIX > 40, spread drops to ~9%.
3. **VIX below its EWMA = worst time to buy options.** Sell 20-delta with VIX<MA filter: Sharpe 1.68, max DD 10.3%.
4. **Vol clustering means today's vol predicts tomorrow's.** Absolute return autocorrelation is significant at lag 1 and decays slowly.
5. **Negative returns are ~13% larger than positive returns for SPX.** Build asymmetric directional priors.
6. **Overnight returns carry most of the kurtosis.** Intraday (our domain) has thinner tails -- good for 0DTE.
7. **At expiry, gamma is maximized ATM.** Pin risk from market maker hedging compresses or amplifies realized vol.
8. **Optimal mean-reversion entry: 0.75 sigma.** Applicable to VIX mean reversion trades and potentially to entry timing.
9. **Half-Kelly with Bayesian edge adjustment.** After 100 trades: std(f)=0.097. Need 100+ trades before trusting Kelly sizing. Always use p_adjusted = (w+1)/(N+2).
10. **Out-of-the-money calls are the worst buys.** Lottery seekers overpay. Returns worsen the farther OTM. Prefer ATM.
