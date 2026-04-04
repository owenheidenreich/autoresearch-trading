# Data Audit Findings (2026-04-03)

## Finding 1: Narrow strike grid (CRITICAL)

prepare.py downloads only 14 of 209 available strikes from Polygon flat files.
ATM +/- 30pt in 5pt steps. SPX daily range is 20-100pt. After a 30pt move,
the grid is lopsided -- the model can't see contracts a real trader would use.

38.7% of tradeable bars have >20pt drift (4+ strikes of compression).

FIX: Wide-grid downloader extracts ATM +/- 100pt (82 contracts per day).
After 50pt move, still 50pt OTM coverage on each side.

STATUS: download_wide_grid.py built and running.

## Finding 2: Oracle labels are 100% winners (CRITICAL)

All 286,053 trade labels have P&L > 0. Zero losing trades in training.
PF=388, WR=84% are artifacts, not real performance.

FIX: Triple-barrier labeler with real losers (35-50% loss rate expected).

STATUS: not yet implemented. Waiting for wide-grid data.

## Finding 3: Moneyness drift (CRITICAL)

ATM strike fixed at opening. 84% of bars drift >5pt. 58% of "OTM5 call"
bars are actually ITM. Mean drift 19.3pt, max 462.8pt.

Call-put parity gives real-time SPX with 73.3% coverage. Confirmed working.

Features MUST be relative (moneyness %, normalized price) not absolute.
What the model learns at SPX 4300 must transfer to SPX 6500.

FIX: compute current moneyness from call-put parity.
STATUS: proven in data_enriched.pt, needs integration with wide grid.

## Finding 4: Spread costs (SIGNIFICANT)

Commission: $1.30 round trip = 0.13% on $10 option. Negligible.
Bid-ask spread: unknown from Polygon data. Real range ~$0.05-$0.30 ATM.
Old lookup table estimated 30-150 bps, off by 4-20x for some time periods.

Corwin-Schultz estimator: shows correct time-of-day pattern but overestimates
(mixes real price movement with spread). Median non-zero estimate: $0.49.

FIX: $0.30 round-trip spread + $1.30 commission as conservative fixed cost.
Corwin-Schultz as relative feature (model learns when expensive to trade).

## Finding 5: Per-strike OHLCV thrown away (FIXED)

Raw cache has high, low, volume, transactions per strike per bar. prepare.py
only kept close prices. 20.6% of bars have zero volume (illiquid).

STATUS: extracted to v2/data_enriched.pt (228 MB, 118 keys).

## Finding 6: Signal edge exists in volatility features (GO signal)

Baseline: always-put PF=1.106 (structural intraday put bias).

Features that beat baseline:
- option_spread_width: PF=1.855 (+0.748)
- session_range_pct: PF=1.435 (+0.328)
- rsi_7: PF=1.219, bollinger_position: PF=1.216
- atm_iv: PF=1.182, atm_gamma: PF=1.176

Pattern: high vol = calls win (gamma), low vol = puts win (theta).
Momentum has NO edge (ret_6 PF=0.81, worse than always-put).

Neural network combining volatility-regime features could achieve PF > 2.

## What's sound

- Option prices track the same contract all day (no switching)
- 39 features well-engineered for SPX price action
- 4 years history (999 trading days)
- Simulator correctly models same-contract entry/exit
- Infrastructure works end-to-end (Akash deploy, train, replay, score)
- Polygon flat files have 209+ strikes per day (we just need to extract more)
