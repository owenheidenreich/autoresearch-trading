# autoresearch-trading v0.4 — Overhaul Plan

> Created: March 13, 2026

## Summary

Fix the broken historical options pipeline (28 constant features → real intraday Greeks computed via Black-Scholes on historical options bars from Polygon), then run the Karpathy-style autoresearch loop to find the optimal architecture, then build a live streaming pipeline for novel real-time data.

---

## What's Wrong (v0.3)

1. **28 dead features**: Used `list_snapshot_options_chain` (real-time-only endpoint). Historical training data has constant defaults (IV=0.20, delta=0.50, etc). Feature gating ignores these groups.
2. **Wrong endpoint**: Polygon Options Starter includes 2 years of historical options **aggregate bars** (`get_aggs` on option tickers like `O:SPY240415C00560000`). We never used this.
3. **Model overparameterized**: 450K params / 2,700 hourly samples = 166 params/sample. Overfitting guaranteed.
4. **Autoresearch loop never started**: All time spent on data prep bugs.

## The Fix

Use `get_aggs()` on historical option contract tickers to download real intraday options prices, then compute IV and Greeks via Black-Scholes at 5-minute resolution. All 105 features become dynamic.

---

## Phase 1: Historical Options Overhaul

### Step 1.1: Rewrite `prepare.py` options pipeline

**What changes:**
- Add Black-Scholes pricing, IV solver (Brent's method), Greeks computation
- Add option ticker construction (`O:SPY{YYMMDD}{C/P}{strike*1000:08d}`)
- Replace `_download_options_snapshots` + `_fetch_options_snapshot_for_date` with `_download_historical_options_bars`
- New function downloads ATM call + ATM put 0DTE bars for each trading day
- Computes IV, delta, gamma, theta, vega from option prices at 5-min resolution
- Returns DataFrame aligned to SPY 5-min timestamps (not daily snapshots)
- Update all 4 feature functions to use per-bar data instead of daily lookups
- Update cache version (`v3` → `v4`) to force re-download

**API cost:**
- ~500 trading days × 2 contracts (ATM call + put) = 1,000 API calls
- At 13s spacing: ~3.6 hours download time
- One-time cost, cached to disk

**What we get:**
- IV that changes intraday (not constant 0.20)
- Delta that shifts as SPY moves (not constant 0.50)
- Theta that accelerates toward expiry (not constant -0.01)
- Real volume data for flow features
- Real bid-ask spread for liquidity features
- 92,000 five-min bars × 105 **real** features

### Step 1.2: Test data download on Akash H100

- Deploy container, run `uv run prepare.py --start 2024-04-01`
- Expected: ~3.6 hours for options bars + ~30 min for equity bars
- Verify: options features change bar-to-bar (not constant within a day)
- Check Black-Scholes outputs are reasonable (IV 10-80%, delta 0.3-0.7 for ATM)

### Step 1.3: Right-size model (via autoresearch loop)

The autoresearch loop handles this automatically by trying different model sizes. Starting suggestions in `program.md`:
- Try `DEPTH` 3-8, `D_MODEL` 64-256
- Try 5-min resolution with shorter lookback instead of hourly with 36-bar lookback
- Let the AI agent explore architecture space

### Step 1.4: Run autoresearch loop (overnight, 8-12 hours)

- Claude Code reads `program.md` → modifies `train.py` → runs 5-min experiment → keeps/reverts → repeat
- ~12 experiments/hour × 8 hours = ~96 experiments
- Track in `results.tsv`
- Cost: ~$22-32 in H100 time
- Output: Optimized architecture + best val_sharpe

### Step 1.5: Evaluate and lock architecture

- Review `results.tsv` and `git log`
- Verify no overfitting (Sharpe < 2.5, trades 10-400, flat time > 30%)
- Commit winning `train.py` as the base for Phase 2

---

## Phase 2: Live Streaming Pipeline

### Step 2.1: Build `live_stream.py`

Real-time data collector during market hours (9:30-4:00 ET):
- Every 5 min: fetch 1 SPY bar + 1 options snapshot (NOW real-time works!) + QQQ/VIX
- Compute features incrementally (maintain rolling state)
- All 105 features dynamic in real-time
- ~5 API calls per 5-min interval = well within rate limits
- Optional: 1-min bars for higher resolution

### Step 2.2: Build `live_train.py`

Online learning during market hours:
- Load Phase 1 winning architecture as warm start
- Accumulate new bars from live_stream.py
- Every N bars: mini fine-tuning epoch
- Run inference → produce position signal
- Log predictions + actual outcomes

---

## Phase 3: Paper Trading (1 week)

- Run live_stream + live_train for 5 trading days
- Track position signals, hypothetical options P&L
- Compare historical-only vs live-trained model
- Evaluate: options-adjusted Sharpe, win rate, profit factor, drawdown

---

## API Budget (Phase 1)

| Item | API Calls | Time (13s/call) |
|------|-----------|-----------------|
| SPY 5-min bars | ~23 | ~5 min |
| SPY/VIX/QQQ/TNX daily | ~100 | ~22 min |
| VIX/QQQ/SPX 5-min | ~70 | ~15 min |
| Market internals | ~50 | ~11 min |
| **Options bars (ATM call+put per day)** | **~1,000** | **~3.6 hours** |
| **Total** | **~1,250** | **~4.5 hours** |

## Files Changed

| File | Change | Phase |
|------|--------|-------|
| `prepare.py` | Major rewrite: Black-Scholes, historical options bars, per-bar features | 1 |
| `program.md` | Update deployment details, add autoresearch tips | 1 |
| `train.py` | Modified by autoresearch loop (architecture search) | 1 |
| `live_stream.py` | NEW: real-time Polygon data collector | 2 |
| `live_train.py` | NEW: online learning with warm start | 2 |

## Black-Scholes Implementation

For each 5-min bar where we have an option price:

```
Given: option_price, S (SPY price), K (strike), T (time to expiry), r (risk-free rate)

1. Solve for IV: find σ such that BS(S, K, T, r, σ) = option_price
   (Brent's method on [0.01, 10.0])

2. Compute Greeks from IV:
   d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
   d2 = d1 - σ√T
   delta = N(d1)           [call]  or  N(d1) - 1    [put]
   gamma = n(d1) / (Sσ√T)
   theta = [-Sn(d1)σ/(2√T) - rKe^(-rT)N(d2)] / 365  [call]
   vega  = Sn(d1)√T / 100
```

Time to expiry for 0DTE: `T = max(minutes_to_close / (365 * 24 * 60), 1e-6)`

## Verification Checklist

### Phase 1
- [ ] Options features change bar-to-bar within a trading day
- [ ] IV values range 10-80% (not constant 0.20)
- [ ] Delta ranges 0.3-0.7 for ATM (not constant 0.50)
- [ ] Theta accelerates through the day (gets more negative)
- [ ] Autoresearch loop runs 50+ experiments
- [ ] Best val_sharpe > 0.0
- [ ] No overfitting red flags

### Phase 2
- [ ] live_stream.py fetches data without rate limit errors
- [ ] All features match Phase 1 format (compatible tensors)
- [ ] Position signals produced every 5 minutes during RTH

### Phase 3
- [ ] 5 days of paper trading logged
- [ ] Win rate > 50%, profit factor > 1.2
- [ ] Model stays flat > 30% of bars
