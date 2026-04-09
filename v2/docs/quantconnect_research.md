# QuantConnect: Full Assessment and Recommendation

## Context

We've been using QC's free tier to validate our trading signals and collect bid-ask spread data. The free tier only allows output via `self.Error()` (one message per backtest), no Jupyter/Research notebooks, and no data downloads. We've confirmed QC has real bid/ask, Greeks, IV, and volume for SPX options. The question is: does the $60/month Researcher tier unlock enough value to justify the cost?

## What Each Tier Gives Us

### Free Tier (what we have now)
- Cloud backtesting with full SPX options data (minute resolution)
- Access to all asset classes
- Output limited to one `self.Error()` per backtest
- No LEAN CLI, no local backtesting, no data downloads
- No Research/Jupyter notebooks (despite docs claiming otherwise)
- Workaround: export CSV from backtest output folder (hourly/daily only)

### Researcher Tier ($60/month)
- **LEAN CLI access** -- run backtests locally in Docker
- **Data downloads** via CLI -- SPX minute options data to local disk
- **Research notebooks** (Jupyter) -- interactive data exploration
- **1 live trading node** -- deploy strategies to trade live
- **API access** -- programmatic control
- Download cost: ~$0.15 per file (minute resolution). SPX options for 1 year of minute data: ~$113

### Team Tier ($120/month)
- Everything in Researcher plus tick/second resolution
- 10 concurrent backtest nodes
- We don't need this

## How Researcher Tier Would Help Our Project

### 1. Data Download Pipeline (HIGH VALUE)
With LEAN CLI, we could download SPX 0DTE options minute data with real bid/ask quotes, Greeks, and IV directly to our local machine. This would let us:
- **Replace or supplement Polygon data** with a second independent source that includes bid/ask (Polygon doesn't have this)
- **Get real Greeks** (Delta, Gamma, Theta, Vega, IV) per strike per minute. Our current pipeline estimates these from Polygon close prices. QC computes them properly.
- **Validate our feature pipeline** by comparing QC's computed Greeks against our estimates at scale (not just 4 days)

Cost estimate: ~$113 for 1 year of minute data + $60/month subscription = $173 first month, $60 ongoing.

### 2. Research Notebooks (HIGH VALUE)
Interactive Jupyter access to the full dataset. No more cramming results into a single `self.Error()` call. We could:
- Run proper statistical analysis on spread patterns (not just 5-day samples)
- Build whipsaw classifiers with full pandas/sklearn
- Explore IV surfaces across 1000+ days
- Test hypotheses interactively without writing full QCAlgorithm classes

### 3. Local Backtesting (MEDIUM VALUE)
Run LEAN engine locally in Docker. Faster iteration, no cloud queue, unlimited log output. But we already have our own backtester (simulator.py), so this is supplementary, not essential.

### 4. Live Trading Node (FUTURE VALUE)
Deploy a strategy to trade live through QC's infrastructure. Not needed now, but could be an alternative to our IBKR integration path. Supports IBKR as a brokerage.

## What Researcher Tier Does NOT Help With

- **Training the model.** We train on Akash H100 with PyTorch. QC uses C#/LEAN, not PyTorch. The data pipeline stays separate.
- **Real-time features for live trading.** We'd still need IBKR for live data feeds. QC's live node uses their own framework.
- **0DTE-specific filtering.** QC's `Expiration(0,0)` filter didn't work in our tests. We'd need to filter by expiry date post-download.

## Other Use Cases We Haven't Explored

### Greeks as Training Features
Our current `atm_gamma`, `atm_theta_per_bar`, and `charm_estimate` are estimated from Polygon price data. QC provides properly computed Greeks per contract per minute. If we downloaded this data, we could:
- Replace our estimated Greeks with real computed Greeks
- Add new features: per-strike IV skew, gamma exposure across the chain, term structure
- This could improve model accuracy since it's learning from real Greeks, not proxies

### Full Options Chain Analysis
QC has the complete chain (all strikes, all expirations) at minute resolution. We could:
- Study how the IV surface evolves intraday
- Measure dealer gamma exposure (GEX) from the chain
- Build a proper volatility surface model
- Study order flow patterns across strikes

### Walk-Forward Validation
With LEAN CLI, we could set up proper walk-forward validation: train on our data, then validate on QC's independent data programmatically (not one backtest at a time).

### Paper Trading
The live node supports paper trading through IBKR. We could run our model's signals through QC's execution engine for paper validation before going live.

## Recommendation

**Yes, get the Researcher tier.** $60/month is small relative to the value.

The biggest unlock is **data downloads with real Greeks and bid/ask**. Our current pipeline estimates Greeks from price data. QC computes them properly. Having real Greeks as training features could meaningfully improve model quality. The Research notebooks also eliminate the painful one-Error-per-backtest workflow we've been fighting.

The download cost (~$113 for a year of SPX minute options) is a one-time spend that gives us a permanent second data source with fields Polygon doesn't provide.

### Concrete first actions after subscribing:
1. Download 1 year of SPX 0DTE minute options data (trade + quote + Greeks)
2. Build a comparison pipeline: QC Greeks vs our Polygon estimates
3. If QC Greeks are materially different, integrate them as features
4. Run proper whipsaw analysis in Jupyter (50+ days, not 4)
5. Build IV surface model from the downloaded chain data

## Files to Update After Decision
- `v2/docs/quantconnect_findings.md` -- add tier assessment
- `v2/pipeline/build_v2_dataset.py` -- potentially add QC data ingestion path
- `v2/core/features.py` -- add QC-sourced Greek features if they differ from estimates

## Verification
- Subscribe to Researcher tier
- Install LEAN CLI: `pip install lean`
- Test download: `lean data download --dataset "US Index Options" --data-type "Bulk" --resolution "Minute" --start 20250101 --end 20250131`
- Verify downloaded data contains Greeks, bid/ask, IV
- Compare one day's Greeks against our Polygon estimates
