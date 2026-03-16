# Future Features — Live Trading Readiness

## Recently Implemented (v3.1, March 2026)

### VIX Level ✅
Added as features 45-48: `vix_level`, `vix_change`, `vix_regime`, `vrp`. Uses real CBOE VIX
from IBKR (5-min bars). The model now has volatility regime awareness — critical because a
strategy tuned for VIX 13 will fail at VIX 25.

### OTM Strike Support ✅
Direction head expanded from 2 → 6 outputs: `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM,
PUT_OTM5, PUT_OTM10]`. OTM options downloaded at ATM±5 and ATM±10 strikes. 6 OTM/skew
features (indices 49-54) provide IV profile awareness. OTM P&L arrays enable trade simulation
across all strikes. 8 effective actions total.

### Intraday VIX ✅
Real-time VIX via IBKR 5-min bars. `vix_change` captures intraday VIX spikes and collapses.
`vrp` (variance risk premium) shows IV vs RV divergence in real time.

### EXIT Signal Strengthening ✅
EXIT labels now actively used in the loss function (EXIT_LOSS_WEIGHT = 0.3). Prior runs had
EXIT < 1%; the corrected loss targets gate=NO_TRADE on bars where exit labels indicate
profit targets reached.

## Features Still To Add

### GEX / Dealer Positioning
Whether dealers are long or short gamma determines if moves get dampened or amplified.
True GEX requires expensive data ($200/mo Options Depth). We can compute a **GEX
approximation** from publicly available open interest data if Polygon's option chain
snapshot API is available on our tier. Planned as Phase 5C:
- 4 features: `gex_level`, `gex_sign`, `gex_flip_dist`, `max_gamma_strike_dist`
- **Risk**: Polygon API tier may not include `list_snapshot_options_chain`. Test first.
- **Limitation**: Approximation only — uses start-of-day OI, assumes all options are
  dealer-short. Better than nothing for regime detection, not for precise levels.

### Market Internals (TICK, Breadth)
NYSE TICK measures uptick/downtick imbalance across all stocks. Breadth (advance/decline
ratio) shows whether moves have participation or are narrow. Both are leading indicators.
Planned as Phase 5D (deferred):
- 3 features: `tick_level`, `tick_extreme`, `tick_momentum`
- **Blocked on**: IBKR TICK-NYSE availability (needs testing)
- Can revisit with alternative sources (ThinkorSwim, TradeStation) if IBKR doesn't have it

### Mamba/SSM Architecture (Deferred)
At LOOKBACK=24 (2 hours of 5-min bars), Transformer is fine. Mamba's O(N) scaling advantage
only matters at 500+ sequence length. Not worth the complexity unless autoresearch experiments
show LOOKBACK=48/96 scoring better or the model plateaus on architecture.

## Features to Keep

### `day_of_week`
Empirically relevant. Monday/Friday have different dynamics (weekend risk premium,
weekly expiration hedging). Keep this.

## Features to Reconsider

### `gap`
May be misleading — gap behavior is highly regime-dependent and could introduce
noise rather than signal. Consider removing or replacing with gap-relative-to-ATR
for normalization.

## Training Run Findings

### v2 Run (March 15, 2026 — 86 experiments, 45 features, ATM only)
- **Low dropout (0.02-0.05)** consistently produced the best scores
- **D_MODEL 80-112** is the sweet spot. 64 too small, 128 too large.
- **DEPTH 5-6** — balanced transformer depth.
- **LOOKBACK 24** — full context window helps.
- **Score ceiling: 155.9** at experiment #52. 34 more experiments couldn't beat it.
- **27% failure rate** (Claude breaking data loading code). Safety guards added.
- **Win rate 28-32%**, compensated by PF 20-45x. Classic sniper pattern.
- **EXIT < 1%** — loss function never taught it.

### v3 Test Run (March 15, 2026 — 2 experiments, 49 features, ATM only)
- **0% failure rate** — safety guards working
- **Score: 10.2 → 21.6** in 2 experiments
- **EXIT actively used**: 37.8% exit_pct, 252 model-driven exits (buggy but functional)
- 49 features (but option IV features were all NaN due to SPX price scale bug — now fixed)

### Pending: v3.1 Run (55 features, ATM + OTM, ~900 days)
- EXIT loss corrected (targets gate=NO_TRADE, not TRADE)
- SPX price scale bug fixed — all option/OTM IV features will now work
- 6-class direction head for OTM strike selection
- ~2x more training data (900 days vs 446)
- Expected: higher score ceiling, meaningful EXIT usage, OTM strike diversity

## Next Steps (Priority Order)
1. Complete data.pt rebuild with extended history (Sept 2022 - present)
2. Local smoke test with TIME_BUDGET=30
3. 8-hour H100 production run with v3.1
4. Analyze OTM strike distribution and EXIT patterns
5. Test Polygon OI endpoint for GEX approximation (Phase 5C)
6. Paper trading with best model
