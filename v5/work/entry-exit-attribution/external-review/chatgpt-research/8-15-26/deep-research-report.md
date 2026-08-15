# SPX/SPXW 0DTE: Who Makes Money, How, and Whether a Long-Only Retail Edge Exists

## Bottom line and ranked findings

**Single-source flag:** † means I found only one direct empirical source for that specific claim. A paper that merely cites the original does not count as independent confirmation. I distinguish **observed P&L** from **backtested return**, **risk-premium evidence**, and **exchange/practitioner description**.

**First — I found no rigorous published or working-paper evidence, through August 15, 2026, of a positive-net-expectancy strategy that stays inside your exact box: a small retail account, SPX/SPXW 0DTE only, one long call or put at a time, no short option leg, no delta hedge, and decisions made from one-minute-or-slower data.** The closest papers reporting positive results all leave that box in at least one decisive way: they sell premium, use multi-leg structures, dynamically hedge delta, switch between long and short, trade variance rather than direction, or exploit order-book execution rather than forecasting the option's subsequent return. [1][3][4][5][6] citeturn18search0turn26view1turn9view0turn25view2turn22search3

**Second — the best-documented economic edge is on the risk-bearing/selling side, but it is small relative to tail risk and highly sensitive to transaction costs.** Vilkov's SPXW study, extended through January 2026, finds a positive same-day variance-risk premium, but describes its unconditional monetization as economically small; many simple implementations deteriorate materially after spreads and fees. Its strongest conditional result is a *put-ratio* strategy with gross Sharpe 1.18 and net Sharpe 0.93, while a diversified basket falls from 1.12 gross to 0.82 net. Those are multi-leg, model-timed strategies whose signal can reverse the position sign—not long-only calls or puts. The same study reports substantial 1% expected-shortfall and drawdown risk. † [4] citeturn8view0turn9view0turn9view1turn10view1turn10view2

**Third — direct retail evidence is unusually unfavorable to the strategy class you specified.** Beckmeyer, Branger and Gayda's transaction-level SPX study finds more than $125 million of aggregate identified-retail 0DTE losses from February 2021 through September 2023, with losses concentrated in single-leg positions; buying, high-implied-volatility choices and single-leg trades perform particularly poorly. Independently, Bogousslavsky and Muravyev's trader-level data find average 0DTE option-trade returns around −4.6%, little difference between index and non-index 0DTE trades, and a typical retail option trade resembling exactly the object you are testing: purchase of a short-dated call or put, often tied to the S&P 500, held for hours. [1][3] citeturn18search0turn26view1

There is an important qualification: the SEC's Fu-Li-Musto-Pearson paper discovered an OPRA trade/quote sequencing defect that can make a post-trade quote appear to precede the trade. It specifically notes that this affects the kind of effective-spread calculation used by Beckmeyer et al. Therefore **the Beckmeyer paper's claim that more than $90 million of the losses were transaction costs should not be treated as settled**. The SEC evidence does *not* establish that retail long 0DTE trading is profitable; rather, it shows that retail-sized traders frequently use passive limit orders and can transact much more cheaply than a naïve “always cross the entire displayed spread” assumption implies. [1][2] citeturn19view3turn20view0turn20view1

**Fourth — “dealers make the money” is directionally too crude.** The strongest direct microstructure evidence shows counterparties to customer passive orders earning *small* positive realized spreads, often less than half a tick, not extracting the whole quoted spread. Meanwhile Cboe's exchange-level position data show customer buys and sells sufficiently balanced that net dealer 0DTE gamma is usually modest relative to underlying S&P futures liquidity. The new 2026 factor evidence is more economically precise: compensation attaches to warehousing gamma and absorbing jump/skewness exposure, while apparent factor-neutral 0DTE alpha becomes infeasible under even minimal transaction costs. † for the latter result. [2][6][8][9] citeturn20view2turn22search1turn26view0turn25view1

**Fifth — sub-minute structure is real, but the published 0DTE evidence locates it mainly in execution and liquidity provision, not in a magical sub-minute directional forecast available to a long-option buyer.** The SEC study needs event-level OPRA sequencing and explicitly evaluates a one-second limit-order/fallback protocol. By contrast, published positive risk-premium/timing work can operate with 30-minute option observations and one-minute underlying data. I found no SPX 0DTE study showing that a losing one-minute *long-only call/put* strategy becomes positive merely by moving to seconds or ticks. [2][4][5][6] citeturn19view3turn20view1turn8view0turn25view2turn22search3

**Sixth — your OPRA findings are not in tension with the literature. They are, in several places, a more restrictive and more directly relevant test than the published literature.** Your positive seller-side delivery gap agrees in sign with the documented intraday variance-risk premium; your failure to find a long directional state agrees with both retail-performance studies; your execution-cost result is directionally compatible with the marketable-order case in the SEC study, although the SEC demonstrates that patient passive execution can reduce that burden; and your warning about last-trade artifacts has an independent analogue in the SEC's demonstration that naïve OPRA event ordering can produce mistaken microstructure inferences. [1][2][3][4][5][6] citeturn18search0turn19view3turn26view1turn9view0turn25view2turn22search3

## Who is actually making money

The most important distinction is between **profit directly observed in participant transactions** and **a return premium inferred or backtested from market prices**. Public research does not give us audited Citadel/Optiver/Susquehanna-style SPX-0DTE desk P&Ls. Anyone claiming that a particular market-making firm earns a specific amount from 0DTE is therefore going beyond the public evidence.

| Participant | Best published evidence | What the evidence actually says | Mechanism | Available to $10k, long-one-contract, 1-minute retail? |
|---|---|---|---|---|
| **Retail long-option buyers** | Beckmeyer et al.; Bogousslavsky-Muravyev [1][3] | Directly observed retail trading is negative on average; single-leg/0DTE purchases are among the weakest categories. | They pay option risk premium plus execution costs and must forecast enough realized directional movement quickly enough to overcome both. | **Yes, the trade is available; no documented positive edge.** |
| **Systematic premium sellers / volatility sellers** | Beckmeyer et al.; Vilkov [1][4] | Selling/multi-leg retail decisions outperform buying in observed data; systematic SPXW tests find a positive but small 0DTE VRP and some profitable conditional short/multi-leg rules. | Collect implied-over-realized variance/jump compensation; theta is merely the accounting manifestation, not the economic source. | **Not under your constraint.** Defined-risk spreads are retail-accessible, but they require selling at least one option. |
| **Market makers / dealers** | SEC DERA; Dorion-Orłowski-Song; Cboe [2][6][9] | Counterparties earn small positive realized spreads; factor evidence identifies compensation for gamma/jump-skewness warehousing; no public firm-level SPX-0DTE P&L. | Spread capture, queue/price priority, inventory-risk premia, cross-option/futures hedging. | **No** as a single long option strategy. Passive orders can save costs, but that is not equivalent to a dealer book. |
| **HFT / electronic liquidity providers** | SEC DERA; broader options-wholesaler evidence [2][13] | High-speed competition matters for queue priority and picking off stale/passive quotes. Specific realized SPX-0DTE HFT firm P&L is not publicly identified. | Latency, automated repricing, liquidity provision/removal, inventory netting. | **No.** The infrastructure and strategy are different in kind, not merely faster charting. |
| **Institutional hedgers** | Fed daily-options study; Cboe flow research [7][8] | Institutions use daily options for event-precise insurance/risk transfer; event-spanning insurance is priced more richly. No evidence says the hedge itself has positive standalone expected P&L. | Pay a premium to transfer tail/event risk; value is portfolio insurance rather than option-trade alpha. | Buying the put is technically available, but **it is insurance, not documented positive standalone EV**. |
| **Institutional/quant relative-value traders** | Almeida-Freire-Hizmeri; Vilkov [4][5] | There are documented relative-value and conditional effects, but successful implementations use delta hedging, long/short decisions or multi-leg structures. | Relative mispricing, volatility-risk premia, cross-strike/cross-state conditioning. | **No under your stated constraint.** |

The two independent retail datasets are particularly important because they are closer to *realized behavior* than a simulated strategy. Beckmeyer et al. find that since mid-2022, 60–80% of identified retail 0DTE activity was single-leg, that retail favored ATM/slightly OTM contracts held several hours, and that aggregate losses were driven primarily by those single-leg positions. Bogousslavsky-Muravyev find that 0DTE trades underperform other option trades by roughly three percentage points and average about −4.6%; the average index-0DTE trade was only about $1,077, making the sample highly relevant to small-account behavior rather than institutional block trading. [1][3] citeturn18search0turn26view1

Cboe's 2025 characterization looks superficially different: it estimates retail at roughly 50–60% of SPX 0DTE volume and says more than 95% of trades are limited-risk—either outright longs or shorts embedded in spreads—with only about 4% naked-short. † Those figures are useful for **participant composition**, not profitability. Cboe does not demonstrate that the retail long trades make money, and its broader retail estimate is not directly comparable with Beckmeyer's auction-code identification of a subset of retail activity. [8] citeturn26view0

### Dealers and market makers

The best direct evidence of a dealer-like trading edge is much less spectacular than popular narratives imply. Fu et al. measure realized spreads from the customer's counterparty perspective at one second and one minute. For passive customer orders, those counterparties generally earn positive but tiny amounts: realized spreads are usually below half a tick, and by one minute can be below a penny in the later sample. † The authors interpret this as evidence of strong competition to trade against customers. [2] citeturn19view3turn20view2

That is **spread capture**, but it is not “buy at $2, sell it back to retail at $2.20.” Market makers are continuously pricing thousands of strikes/expirations, receiving both sides of flow, netting positions and changing futures hedges. The SEC data actually show market makers both providing and demanding liquidity; customer passive orders themselves can become the liquidity providers. [2] citeturn19view3

The emerging academic picture adds an inventory-risk component. Dorion, Orłowski and Song's August 2026 working paper finds that six systematic factors span 30-minute SPX option returns from 45 minutes to 15 days; warehousing gamma and absorbing jump/skewness exposure earn premiums, while residual apparent alpha is concentrated in 0DTE but cannot support a factor-neutral trading strategy after minimal costs. † This says the economically defensible dealer reward is compensation for intermediating difficult nonlinear risks—not a free arbitrage against a naive retail customer. [6] citeturn22search1turn22search3

Cboe's own position records further weaken the folklore that dealers are mechanically and massively short 0DTE gamma. In its 2023 analysis, average estimated market-maker net gamma exposure was $170–$670 million through the day, approximately 0.04–0.17% of daily S&P futures liquidity; at 3:30 p.m. the median was +$173 million, although the distribution could be substantially long or short. † Cboe's 2025 update again describes net gamma hedging as de minimis, at most about 0.2% of SPX daily liquidity. Because Cboe operates the SPX market, its participant-position information is unusually valuable; because Cboe also sells and promotes the product, its conclusions about benign market impact deserve to be identified as **exchange research, not independent P&L research**. [8][9] citeturn25view1turn26view0

### Systematic premium sellers

This is the participant class with the clearest positive return-premium evidence, but “selling 0DTE wins” is still too simplistic.

Beckmeyer et al. observe that retail decisions involving selling—and especially complex/multi-leg positions—perform substantially better than outright purchases. Their July 2024 draft reports median margin-adjusted returns of about 3% for retail put spreads and 3.3% for call spreads, while median single-leg returns are negative. † However, the SEC sequencing critique means Beckmeyer's detailed transaction-cost decomposition deserves caution. [1][2] citeturn18search0turn19view3

Vilkov supplies the broader price-based result through January 2026: there is a statistically positive intraday 0DTE variance-risk premium, but its unconditional magnitude is small; ATM-to-OTM call purchases and most put purchases lose on average, while selling slightly OTM options is profitable in a large majority of observations. † The study's stronger conditional results require strategy selection and/or sign changes. [4] citeturn9view0turn9view1

Crucially, this is **not a free high-win-rate income machine**. Under Vilkov's implementation assumptions, one-percent expected shortfall across candidate strategies is roughly 0.58–1.58% of SPX spot-equivalent exposure, worst days are severe, and drawdowns are material. † An iron-condor model that looks decent gross—Sharpe about 0.77—becomes negative net, about −0.20, whereas the better put-ratio model survives costs at roughly 0.93 net Sharpe. † [4] citeturn9view1turn10view1turn10view2

So the defensible statement is: **a small positive intraday risk premium appears persistent enough to detect statistically, but extracting it is strategy-specific, cost-sensitive and paid for by negative convexity/jump exposure.**

## Where the money comes from

### Variance and jump-risk premium

For a long option held to expiration, the buyer wins economically when realized movement, conditional on strike/direction, exceeds what was embedded in the purchase price by enough to cover execution. The seller is on the opposite side. The literature increasingly finds positive compensation for bearing same-day variance/jump exposure, although different papers parameterize it differently. Almeida-Freire-Hizmeri document a high intraday variance-risk premium, primarily associated with upside risk; Vilkov finds a small directly monetizable 0DTE VRP; Dorion-Orłowski-Song attribute option-return compensation to gamma and jump/skewness exposure. [4][5][6] citeturn9view0turn25view2turn22search3

That makes your **+0.55% of premium at 15 minutes and +1.42% at one hour priced-over-delivery result** economically orthodox rather than anomalous. The published papers do not validate your exact magnitude or your finding that most of it sits in the final two hours, but they agree on the central sign: on average, immediate convexity/variance protection is not being handed to the buyer for free. [4][5][6] citeturn9view0turn25view2turn22search3

The “theta” explanation often used in retail material is incomplete. Theta is an option-pricing sensitivity, not an independent source of economic profit. A short option can collect spectacular time decay and still lose if realized movement/jumps overwhelm the premium. The empirical edge is better described as **compensation for bearing variance, gamma and jump risk**, which naturally comes with occasional large losses. [4][6] citeturn9view1turn22search3

### Spread capture and liquidity provision

The SEC paper gives perhaps the cleanest numerical demonstration. For a call below $3 with a two-tick $0.10 spread, a customer who improves to the midpoint with a non-marketable order had an approximately 50% one-second fill probability in the earlier sample and 62% later. Under the paper's mechanical rule of crossing the market after one second if unfilled, expected cost was about $0.028 early and $0.021 late versus $0.05 from immediately submitting a marketable order. † Similar savings occurred across calls/puts and buys/sells. [2] citeturn20view1

This is highly relevant to your 2.8–4.7% premium round-trip cost, but it changes the interpretation rather than overturning your result. Your estimate represents an aggressive ask-to-bid round trip. A retail trader willing to expose a limit order can sometimes pay much less. **What the SEC does not show is that saving half a spread converts long-option expected returns from negative to positive.** It establishes an execution improvement, not a forecasting alpha. [2] citeturn19view3turn20view1

A small account can therefore access *one fragment* of professional microstructure—placing passive orders. It cannot thereby reproduce a market maker's business. To systematically harvest realized spreads requires repeated two-sided participation, queue management, inventory control and the ability to cancel/reprice before adverse selection dominates. The SEC explicitly describes competition with high-speed participants and price-time priority in SPXW. [2] citeturn19view3

### Gamma scalping

“Gamma scalping” should not be conflated with “market makers automatically win.” A delta-hedged **long-gamma** position benefits from sufficiently large realized movement relative to the volatility paid; a short-gamma book is the opposite. Dealers can be either long or short aggregate gamma depending on the day's customer flow, strikes, longer maturities and offsetting products. Cboe's actual participant-position data show the sign changing rather than being permanently short. [9] citeturn25view1

For your question, the key point is simpler: **long gamma requires buying the option, but profitable gamma scalping requires repeated delta hedging and sufficiently favorable realized-versus-implied variance.** That is already outside a “one long call or put, no other leg” rule. Moreover, the positive variance-risk-premium evidence says the average 0DTE buyer generally starts on the unfavorable side of that realized-versus-implied comparison. [4][5][6] citeturn9view0turn25view2turn22search3

### Latency and internalization

There is strong evidence in the broader U.S. options market that professional wholesalers have built a large business around retail flow: Bryzgalova, Pavlova and Sikorskaya find nearly 90% of option payment-for-order-flow payments concentrated among three wholesalers. † That result is **not SPX-0DTE-specific dealer P&L**, so it should not be used as proof that wholesalers earn a particular return from your contracts. [13] citeturn17search0turn17search7

For SPXW specifically, the SEC study is more useful. It shows that milliseconds/seconds matter to queue placement, limit-order fills and adverse selection. That is an edge in *how an order is executed*. There is no comparable direct study showing that a trader watching sub-second SPXW quotes can predict the subsequent multi-minute call or put return strongly enough to produce positive net long-option EV. [2] citeturn19view3turn20view1

## The strongest case for buying premium — and why it does not clear your bar

The fairest way to answer the long-premium question is to start with the studies that appear most favorable rather than dismissing them.

### Relative mispricing

Almeida, Freire and Hizmeri provide probably the strongest academic “something is mispriced in 0DTE” result. They find widespread violations of stochastic-dominance restrictions and construct trades intended to exploit relative mispricing; their overall 0DTE asset-pricing evidence includes highly profitable opportunities in earlier data. [5] citeturn25view2turn18search1

But this is **not evidence for your trade class**. Their implementation buys or writes delta-hedged options according to relative valuation rather than simply purchasing one call or put directionally. More damaging for the 2022–2026 question, the paper reports that the extraordinary profitability of the mispricing strategy is largely an earlier-period phenomenon and dissipates after daily 0DTE expirations make the market deeper and more integrated. † [5] citeturn14search7

This is an important negative result disguised as a positive one: **one of the strongest academic 0DTE anomalies gets weaker precisely when the modern five-days-a-week SPXW regime begins.**

### Event days

FOMC, CPI and payroll days are natural candidates for long-premium alpha because realized moves can be enormous. The strongest clean evidence does not say “buy the event straddle,” however. Londono and Samadi show that daily SPX options spanning CPI, FOMC, nonfarm payroll and GDP releases carry *higher ex-ante prices for price, variance and downside insurance* than neighboring expirations. † In other words, the options market recognizes scheduled-event risk and charges for it. [7] citeturn24view0

Beckmeyer et al. also find no special retail salvation on FOMC days: identified retail aggregate profits on FOMC announcement days are statistically indistinguishable from other days, while purchasing options with higher implied volatility predicts lower subsequent retail returns. † [1] citeturn18search0

An event can therefore create spectacular winners without creating positive expectation. The distinction is exactly the same one your “perfect exit +$623” calculation exposes: **large ex-post opportunity is not evidence that the opportunity is knowable ex ante.**

### Trend and momentum days

I found studies showing that 0DTE prices contain information about intraday market returns and studies finding conditional option-strategy predictability, but **none that establishes net positive post-2022 performance for a causal rule that only purchases one SPX 0DTE call on bullish states or one put on bearish states using minute-level information.** Almeida et al.'s intraday predictive relation is embedded in a broader risk-premium/relative-value framework; Vilkov's successful conditional rules trade strategies that can be long *or* short and are often multi-leg. [4][5] citeturn10view0turn10view1turn25view2

Vilkov is especially informative because his tests are not primitive unconditional selling rules. The paper uses out-of-sample predictive models, expanding or rolling estimation and only information observable at the trading decision. Yet the attractive results come from changing exposure according to the model and from multi-leg structures; direct return prediction is generally weak, with only some methods modestly surviving costs. † [4] citeturn10view0turn10view1

Thus your finding that **zero of 375 pre-declared causal states cleared positive expectation for long calls, puts or straddles** is not contradicted by published conditional-strategy successes. Your hypothesis class is materially narrower—and, for this question, more relevant.

### Tail protection

There is a perfectly legitimate reason to buy 0DTE puts even if their expected standalone P&L is negative: protecting a larger equity portfolio against a specific intraday shock. Londono-Samadi's result explicitly treats them as insurance. O'Donovan's 2026 paper further argues that the introduction of daily 0DTE expirations lowered the price of equity tail protection at longer horizons through customer substitution and dealer-intermediation channels. † [7][12] citeturn24view0turn12search0turn12search8

That can make the hedge economically valuable at the **portfolio** level while still having negative average standalone returns. It is therefore not an exception to your question about a trader whose objective is to make money buying the contract itself.

### The strongest evidence against

The negative evidence is unusually convergent:

Beckmeyer et al.: retail 0DTE losses are concentrated in single-leg positions, retail buys perform poorly, and buying expensive/high-IV contracts is particularly damaging. [1] citeturn18search0

Bogousslavsky-Muravyev: independently observed retail 0DTE trades average about −4.6%, with little distinction between index and non-index 0DTE trades; 0DTE underperforms longer-dated option trading. [3] citeturn26view1

Vilkov: buying many ATM-to-OTM calls and most puts loses on average, while the detectable same-day VRP sits on the selling side; profitable conditional implementations leave the long-single-leg class. † [4] citeturn9view0turn10view1

Dorion-Orłowski-Song: option returns compensate gamma and jump/skewness warehousing, and the tempting residual 0DTE “alpha” cannot sustain a factor-neutral trade after even minimal transaction costs. † [6] citeturn22search3

The SEC paper: transaction costs can be substantially reduced with intelligent passive orders, but its result is about cheaper implementation, **not positive post-execution long-option returns**. [2] citeturn20view1turn20view2

Your OPRA census then attacks a gap left by these papers: rather than asking whether retail loses in aggregate, you explicitly searched hundreds of pre-declared state variables for a *causal conditional exception* and found none. Nothing I found in the literature supplies a published exception inside your restrictions.

## What data resolution actually buys you

There is a useful separation between **resolution required to study price formation** and **resolution required to implement a documented expected-return strategy**.

| Evidence | Resolution actually used | What the high resolution buys | Does it document sub-minute long-call/put alpha? |
|---|---|---|---|
| SEC Fu et al. [2] | Event-level OPRA/L2 logic; explicitly evaluates fills after **one second** | Correct sequencing, queue position, fill probability, adverse selection, realized spreads | **No.** Execution-cost/liquidity-provision result. |
| Beckmeyer et al. [1] | Transaction-level Cboe/OPRA information | Retail identification, position/P&L measurement | **No.** Retail buys lose. |
| Vilkov [4] | **30-minute option bars**, one-minute underlying data | Intraday VRP and conditional strategy construction | **No.** Positive results are principally short/multi-leg/sign-changing. |
| Almeida-Freire-Hizmeri [5] | High-frequency/intraday option and underlying information, including minute-scale construction | Intraday pricing kernel, VRP, relative mispricing | **No post-2022 single-leg long-only result.** |
| Dorion-Orłowski-Song [6] | **30-minute option returns**, maturities down to 45 minutes | Factor/risk-premium decomposition | **No.** Residual factor-neutral alpha fails costs. |

Sources: [1][2][4][5][6]. citeturn19view3turn8view0turn25view2turn22search3

This is an important answer to the “maybe the alpha exists below one minute” hypothesis. **Yes, economically meaningful events occur below one minute. But the documented benefit is being first in a queue, avoiding a stale quote, controlling adverse selection or measuring the market correctly. It is not published evidence that future 15-, 30- or 60-minute option returns become directionally forecastable.**

The SEC paper is almost a controlled example. A customer improving a quote gets a substantial cost benefit if filled during a **one-second** window. The counterparty, meanwhile, earns a small realized spread. That is a genuine sub-minute edge class. But the payoff is created by supplying/demanding liquidity at advantageous prices—not by predicting that SPX will rise sufficiently for the call premium to outperform subsequently delivered movement. [2] citeturn20view1turn20view2

It also provides unusually strong support for your skepticism about trade-print backtests. The authors discover an endemic OPRA problem whereby the BBO generated *after* a trade can receive an earlier timestamp than the trade. They state that failure to resequence these events leads to mistaken microstructure calculations. † [2] citeturn19view3turn20view0

That is not literally the same phenomenon as your measured **~$100-per-contract standard deviation of last trade from fair value**, so I would not claim the SEC independently reproduces your statistic. What it independently demonstrates is the broader methodological point: **the raw apparent “price” and timestamp sequence in OPRA are not automatically a causally valid executable-price series.** A backtest can manufacture an edge without ever making an economically executable forecast. [2] citeturn19view3

Your observation that the median *best* future exit is +28% inside 30 minutes while no one-minute causal rule captures it fits the same distinction. High gamma creates a huge range of ex-post extrema. There is no published evidence I found that the extrema themselves become forecastable merely because one samples them more frequently. Published models with surviving performance forecast broader strategy states or volatility premia; they do not solve the causal “which local peak should I sell?” problem for a long call/put. [4][5] citeturn10view0turn10view1turn25view2

## Reconciliation with your OPRA results and the seller-side alternative

### Your measured variance premium

Your result—near-ATM straddles priced about 0.55% above subsequent delivery over 15 minutes and 1.42% over an hour, with the premium present each year 2022–2026—is **directionally corroborated** by the academic evidence for a positive intraday variance/gamma/jump-risk premium. [4][5][6] citeturn9view0turn25view2turn22search3

Your additional findings—that the premium shrinks over calendar time and concentrates in the final two hours—are more granular than the textual results I found. I would treat those as findings from your dataset rather than pretend the papers independently establish those exact magnitudes.

There is a plausible structural consistency with the market becoming more efficient. Fu et al. document narrowing SPXW spreads and increasing competition between 2020 and 2023; Almeida et al. report that a particularly strong 0DTE mispricing strategy becomes much less profitable after the daily-expiration regime develops. † for the Almeida regime-change result. [2][5] citeturn19view3turn14search7

### Your 375-state null result

Nothing in the papers I reviewed directly invalidates it.

Vilkov does obtain conditional predictive performance, but the strategy universe contains straddles, strangles, ratios, iron condors and other structures, and the predictive implementation can reverse signs—meaning it may be long a structure under one state and short it under another. † [4] citeturn10view0turn10view1

Almeida et al. use delta-hedged relative-value trades rather than naked single-leg direction. [5] citeturn25view2

The Fed event evidence tells us event options are priced richer, not that “FOMC call buying” is profitable. [7] citeturn24view0

The direct retail data finds no aggregate FOMC-day rescue and associates higher IV with lower subsequent retail returns. † [1] citeturn18search0

Therefore, finding a positive result in those broader papers and a zero in your 375-state census would not be a contradiction. They are testing different admissible trades.

### Your transaction-cost estimate

Your aggressive ask-entry/bid-exit burden of roughly 2.8–4.7% of premium is entirely capable of eliminating modest option-timing signals. The main revision suggested by the SEC evidence is that it should **not be interpreted as the minimum execution cost faced by every retail trader**. On two-tick options, patient limit-order protocols reduced expected one-way effective cost roughly by half in the SEC exercise. † [2] citeturn20view1

But consider the economic hurdle. If your pre-cost long-option delivery disadvantage is already negative, reducing a positive trading-cost term toward zero can make the trade *less negative*; it does not automatically make the underlying expectancy positive. This is exactly why the SEC paper and the retail-return papers can both be right: retail may receive substantially better executions than quoted-spread folklore suggests and still choose overpriced or poorly positioned long options. [1][2][3] citeturn18search0turn20view1turn26view1

That distinction also means your break-even directional accuracy calculation should ideally be reported under at least two execution regimes: immediate marketable execution and a realistically modeled passive-limit protocol. The literature gives reason to expect the numerical threshold to improve under the latter. It supplies no evidence that a known directional signal would then clear it. [2][4] citeturn20view1turn10view1

### How big is the documented seller edge?

This is where the literature is easy to oversell.

Vilkov describes the raw 10:00-to-close 0DTE variance-premium monetization as only around **0.0011% of the underlying** at the median—economically small before considering how much tail exposure is required. † [4] citeturn9view0

Short slightly OTM calls/puts and strangles win in a high fraction of observations, but high win frequency is not the same as high risk-adjusted return. † [4] citeturn9view0turn9view1

Model selection can improve some implementations: the put-ratio conditional strategy reaches about 0.93 net Sharpe and the top-three strategy basket about 0.82 net Sharpe. † But other apparently sensible selling structures fail the cost test—the reported iron-condor conditional Sharpe falls from roughly 0.77 gross to −0.20 net. † [4] citeturn10view1turn10view2

Tail exposure is not cosmetic: reported 1% expected shortfalls are approximately 0.58–1.58% of SPX spot-equivalent exposure depending on strategy, with meaningful drawdowns and severe worst observations. † [4] citeturn9view1

The documented seller-side edge should therefore be characterized as **small positive carry in exchange for negatively convex/jump-sensitive exposure, not a large free theta premium**. Defined-risk spreads make the loss finite and are structurally accessible to smaller accounts, but they also buy away some of the very tail exposure for which the market is paying the seller. And, as the iron-condor example shows, a defined-risk structure is not automatically positive after costs. [1][4][8] citeturn18search0turn10view2turn26view0

Practitioner material frequently advertises late-day theta harvest. Option Alpha, for example, presents empirical charts showing especially rapid decay late in the SPX 0DTE session. † That observation is compatible with both your final-two-hour finding and the academic VRP literature, but it is **not credible evidence of a trading edge by itself**: the provider sells options analytics/automation, and the analysis does not establish out-of-sample, risk-adjusted profitability net of all tail losses. [14] citeturn22search5

## Final verdict

**No. On the published evidence available through August 15, 2026, there is no documented positive-edge occupant inside the intersection**

> **small retail account + SPX/SPXW 0DTE + long-only + single call or put + same-day holding + minute-resolution causal information.**

That conclusion does **not** mean nobody makes money buying a 0DTE call or put. The return distribution obviously contains enormous winners. It means I found no rigorous evidence that a causal rule available to that trader makes the *distribution's expectation positive after realistic implementation costs*. The direct retail datasets instead put that class among the losers, while asset-pricing studies place positive expected compensation primarily with the parties warehousing variance, gamma, jump and skewness risk. [1][3][4][5][6] citeturn18search0turn26view1turn9view0turn25view2turn22search3

**There are documented edge classes in 0DTE, but every one I can substantiate violates at least one of your constraints:**

| Documented edge class | What must change from your setup |
|---|---|
| Variance/jump premium | **Sell** option exposure rather than only buy it. |
| Conditional premium-selling | Use **multi-leg structures and/or switch long/short**. |
| Relative-value mispricing | **Delta hedge**, trade both signs, often trade relative values across strikes. |
| Dealer/intermediary premium | Run **inventory plus dynamic futures/options hedges**. |
| Spread/queue capture | **Provide liquidity**, manage queue/adverse-selection risk at tick/second speed. |
| Event/tail insurance | Accept that the option is a **costly hedge**, not demonstrated positive standalone EV. |

Sources: [2][4][5][6][7]. citeturn19view3turn10view1turn25view2turn22search3turn24view0

Your own findings actually make the conclusion stronger because they test the precise loopholes that aggregate studies leave open. A positive seller-side delivery gap does not prove there can never be a sufficiently good directional forecast; your 375-state census then searches for one and finds none. Large within-trade excursions do not prove exits are forecastable; your causal exit tests find none. A backtest built on prints can appear to supply the missing alpha; your fair-value comparison finds print noise far larger than the economic hurdle, while the SEC independently shows that raw OPRA sequencing itself can generate erroneous microstructure inference. [2][4] citeturn19view3turn20view0turn10view0

The most defensible synthesis is therefore:

**The modern SPX 0DTE market does contain positive expected-return occupants. They are principally being paid to supply liquidity or to warehouse volatility/gamma/jump risk, and the edge is generally small enough that execution, hedging and tail control matter enormously. The literature does not identify an analogous positive premium for simply being long a same-day call or put.**

For a $10,000 account restricted to a single long contract and one-minute information, **the documented professional edges are not merely hard to reproduce; they are different trades.**

## Annotated references

**[1] Heiner Beckmeyer, Nicole Branger, Leander Gayda, “Retail Traders Love 0DTE Options… But Should They?”** SSRN working paper, July 2024 version. The most directly relevant study of identified retail SPX 0DTE trades. Finds aggregate retail losses above $125 million in its February 2021–September 2023 sample, especially in single-leg/buying activity; multi-leg and selling decisions perform better. Its detailed transaction-cost attribution must be read alongside the subsequent SEC sequencing critique in [2].  
Full link: https://ssrn.com/abstract=4404704 citeturn18search0

**[2] Lei Fu, Su Li, David K. Musto, Neil D. Pearson, “Hope at a Reasonable Price: Customer Use of Limit Orders in the 0DTE Market.”** SEC Division of Economic and Risk Analysis Working Paper, 2025. Critical microstructure paper. Finds extensive retail-sized passive liquidity provision in SPXW, materially lower costs from non-marketable limit orders, small realized spreads for counterparties, and—most importantly for backtesting—an endemic OPRA trade/quote sequencing problem.  
Full link: https://www.sec.gov/files/dera-hope-reasonable-prc-2503.pdf citeturn17search2turn19view3

**[3] Victor Bogousslavsky and Dmitriy Muravyev, “An Anatomy of Retail Option Trading.”** Trader-level retail options study. Particularly useful as independent corroboration because it does not rely solely on the Beckmeyer retail-identification design. Reports average 0DTE trade returns around −4.6%, with index 0DTE trades similarly weak; describes the typical retail option trade as a short-horizon option purchase often linked to the S&P 500.  
Full link: https://www.brettonwoodsskiconference.com/uploads/b/f9bfc8b0-0251-11ed-a646-3dea17112d2f/An%20Anatomy%20of%20Retail%20Option%20Trading.pdf citeturn26view1

**[4] Grigory Vilkov, “0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing.”** Working paper with an unusually useful reproducible research repository; current implementation sample extends through January 2026. Best source here for the size of the directly monetizable 0DTE variance premium, transaction-cost sensitivity, tail risk, and out-of-sample conditional multi-leg strategies. Positive findings are not evidence for long-only single-leg calls/puts.  
Paper: https://ssrn.com/abstract=4641356  
Replication/code: https://github.com/vilkovgr/0dte-strategies citeturn18search3turn7view0turn8view0

**[5] Caio Almeida, Gustavo Freire, Rodrigo Hizmeri, “0DTE Asset Pricing.”** Working paper on the intraday pricing kernel, variance-risk premium and stochastic-dominance violations in 0DTE SPX options. Important as one of the strongest “there really is exploitable mispricing” papers—but its relative-value implementation involves delta hedging/buy-write decisions rather than your long-only trade class, and the extreme profitability is reported to weaken after the daily-expiration market develops.  
Full link: https://ssrn.com/abstract=4701401 citeturn18search1turn25view2

**[6] Pierre Dorion, Piotr Orłowski, and coauthor Song, “The Factor Structure of 0DTE Option Returns.”** August 2026 working paper. Very recent and therefore flagged heavily as single-source evidence. Finds that gamma and jump/skewness warehousing earn risk premiums in 30-minute SPX option returns; residual 0DTE alpha is not implementable factor-neutrally after minimal transaction costs.  
Full link: https://ssrn.com/abstract=7149778 citeturn22search1turn22search3

**[7] Juan M. Londono and Mehrdad Samadi, “The Price of Macroeconomic Uncertainty: Evidence from Daily Options.”** Federal Reserve International Finance Discussion Paper, 2023. Clean evidence against the simplistic “just buy premium on known event days” story: S&P 500 options spanning CPI, FOMC, payroll and GDP releases charge higher ex-ante prices for price, variance and downside insurance.  
Full link: https://www.federalreserve.gov/econres/ifdp/the-price-of-macroeconomic-uncertainty-evidence-from-daily-options.htm citeturn24view0

**[8] Mandy Xu / Cboe, “0DTEs Decoded: Positioning, Trends, and Market Impact.”** May 2025 exchange research. Best treated as participant/flow evidence, **not performance evidence**. Cboe estimates retail at roughly 50–60% of SPX 0DTE trading, over 95% of trades in limited-risk form and only about 4% naked-short, while estimated net market-maker gamma hedging remains very small.  
Full link: https://www.cboe.com/insights/posts/0-dt-es-decoded-positioning-trends-and-market-impact/ citeturn26view0

**[9] Mandy Xu / Cboe, “Volatility Insights: Much Ado About 0DTEs — Evaluating the Market Impact of SPX 0DTE Options.”** September 2023. Uses Cboe participant-position information rather than inferred trade signs to estimate dealer gamma. Particularly useful for rejecting the assumption that market makers are mechanically always massively short gamma. Again, exchange research rather than independent academic P&L evidence.  
Full link: https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options/ citeturn25view1

**[10] Chukwuma Dim, Bjørn Eraker, Grigory Vilkov, “0DTEs: Trading, Gamma Risk and Volatility Propagation.”** Academic working paper on dealer inventory and hedging effects. Useful because it models dealers as intermediaries of a broader expiration inventory rather than assuming today's gross 0DTE volume equals today's new dealer gamma. Its market-impact conclusions differ from some competing 0DTE-volatility papers, so I do not use it as stand-alone proof of profitability.  
Full link: https://ssrn.com/abstract=4692190 citeturn6search15turn6search22

**[11] Jonathan Brogaard, Joon Ho Han, P. Y. Won, “Does 0DTE Options Trading Increase Volatility?”** Working paper representing the competing strand of the literature on the underlying-market effects of 0DTE hedging. Relevant mainly to prevent overclaiming consensus about dealer gamma's market impact; it does not establish a retail long-option alpha.  
Full link: https://ssrn.com/abstract=4426358 citeturn5search1

**[12] James O'Donovan, “0DTE Options and the Price of Tail Protection.”** 2026 working paper. Uses the May 2022 expansion to five-day-a-week SPX expirations to study how 0DTE availability changes the cost of longer-horizon tail insurance. Important distinction: cheaper insurance after market redesign is not evidence that buying the insurance has positive expected standalone P&L.  
Full link: https://ssrn.com/abstract=6836498 citeturn12search0turn23search5

**[13] Svetlana Bryzgalova, Anna Pavlova, Taisiya Sikorskaya, “Retail Trading in Options and the Rise of the Big Three Wholesalers,” Journal of Finance 78, 2023.** High-quality broader U.S. options-market evidence on retail flow and wholesaler concentration. I use it only for the institutional structure of options internalization/PFOF, not as evidence of SPX-0DTE wholesaler P&L.  
Full link: https://onlinelibrary.wiley.com/doi/10.1111/jofi.13285 citeturn17search0turn17search7

**[14] Robert Cernera and Kirk Du Plessis / Option Alpha, “The Truth About 0DTE Options Time Decay.”** Practitioner analysis illustrating the common empirical observation of sharply accelerating late-session decay in SPX 0DTE structures. Useful as descriptive practitioner evidence only; it is not an independent, risk-adjusted, out-of-sample profitability study and comes from a commercial options platform.  
Full link: https://optionalpha.com/blog/0dte-options-time-decay citeturn22search5