# v10 Plan: Book-Informed Feature Upgrade + Training Dynamics

**Created:** 2026-03-27
**Status:** PLANNING
**Baseline:** v9 model — score 19.56, replay PF 1.56, 158 trades, 44% WR, +93.2% return

---

## Executive Summary

The v9 model is profitable but predictions are effectively random (44% WR). Profit comes from R:R asymmetry (+32% avg winner vs -16% avg loser), not prediction quality. Deep analysis of 7 trading books identified specific LEADING indicators missing from our feature set that multiple authors independently identify as their highest-value signals. This plan executes two tracks sequentially:

- **Track A** — Add 6 book-derived features + expand OTM strike range (6→10 direction outputs) to give the model forward-looking information and wider strike expressiveness
- **Track B** — Improve training dynamics to better use all features (existing + new)

Each track has its own testing phase and comparison against previous models.

---

## Root Cause (from v9 research + book cross-reference)

**Why predictions are random:** Current 42 features are mostly LAGGING indicators (returns, RSI, Bollinger, VWAP distance) that describe where price WAS, not where it's going. The gate label `(best_pnl > 0)` asks the model to predict 1-min directional moves from this lagging data — an essentially impossible task.

**What the books say:** Multiple authors independently identify the same gap:
- Elder: "MACD-H slope is the single most important signal" — we don't have it
- Elder: Triple Screen requires multi-timeframe trend — we compute everything on 1-min only
- Coulling: Volume-price divergence is the highest-value LEADING signal — we don't detect it
- Passarelli: Gamma/theta breakeven ratio tells you if buying options is worth it — we have the components but not the ratio
- Sinclair: Variance premium means long options need TIMING alpha — our features don't encode timing quality

---

## Track A: Feature Additions (v10)

### A.1 Changes to prepare.py

**Feature count:** 42 → 47 (5 new features + replace RSI_14 with RSI_7)

| # | Feature Name | Formula | Source | Why It's Leading |
|---|---|---|---|---|
| 1 | `macdh_slope` | sign(MACD_H[i] - MACD_H[i-1]): +1 rising, -1 falling, 0 flat | Elder | Rate of change of momentum consensus. Slope change precedes price change. Indicator Seasons (Spring = buy, Autumn = sell) |
| 2 | `force_index_2` | EMA(2) of [Volume × (Close - Close_prev)], normalized by 20-bar avg | Elder | Integrates price change AND volume into single number. Negative in uptrend = pullback entry. Positive in downtrend = rally to short |
| 3 | `vol_price_diverg` | Count consecutive bars where price direction ≠ volume direction. Positive = bullish divergence, negative = bearish. Capped at ±10 | Coulling | 3+ bars of divergence = strong reversal signal. One of few reliably leading indicators |
| 4 | `effort_vs_result` | (body_size / avg_body_20) / max(volume / avg_volume_20, 0.1). Capped to [-3, 3] | Coulling | >>1 = big move on low volume = fake/unsustainable. <<1 = small move on high volume = exhaustion/accumulation |
| 5 | `trend_5min` | Slope of EMA(13) computed on 5-bar aggregated closes (5-min equivalent). Normalized by close | Elder Triple Screen | Screen 1 trend filter. Distinguishes "1-min pullback in uptrend" (buy) from "1-min rally in downtrend" (trap) |
| — | `rsi_7` (replaces `rsi_14`) | Standard RSI formula with period=7 instead of 14 | Elder | "Recommended period for intraday: 7-9 bars." RSI(14) is too sluggish for 1-min bars |

**Deferred to v11 (strong candidates, but limit scope):**
- `gamma_theta_ratio` — sqrt(2*theta/gamma) / expected_move. Deferred because it requires atm_theta and atm_gamma to be non-zero and non-NaN simultaneously, which needs validation.
- `impulse_color` — EMA slope + MACD-H slope composite. Deferred because `macdh_slope` + `trend_5min` capture the same information with more granularity.
- `breakout_volume_confirm` — volume ratio on breakout bar. Deferred: requires breakout detection logic that adds complexity.

### A.1b OTM Strike Expansion (6 → 10 Direction Outputs)

**Source:** Pickles (17-year trader, $100M+ net worth) almost exclusively buys OTM SPX calls, sometimes +30 OTM (6 strikes above ATM). His reasoning: limited downside because of cheaper premiums on OTM 0DTEs. He uses a $1M trading port.

**Current state:** Direction head has 6 outputs: `CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10`. We already download and compute P&L for OTM±15 and OTM±20 as "sidecar" data in prepare.py (`OTM_STRIKE_STEPS = (5, 10, 15, 20)`). The infrastructure exists — it's just not exposed to the model.

**v9 evidence OTM works:** Our own backtest shows OTM is profitable when selected correctly:
- BUY_CALL_OTM5: 14 trades, 50% WR, +$1,121
- BUY_CALL_OTM10: 10 trades, 40% WR, +$1,525
- BUY_PUT_OTM5: 2 trades, 100% WR, +$1,219

The old "-601% OTM cumulative" was from v8 with VALUE_EXIT corruption. v9 with clean exits shows the model CAN learn when OTM is appropriate.

**Why books were wrong here:** Sinclair's "OTM calls are the worst buys" is statistically true for RANDOM buyers. Pickles and our model are SELECTIVE buyers with confluence-based entry. The limited downside argument is real: a $0.50 OTM call costs 1/10th of a $5 ATM call, but on a trending day gamma kicks it to $5-$15. This IS the R:R asymmetry our model already exploits.

**New direction head:** 14 outputs (from 6) — full OTM ladder matching Pickles' range:

| # | Call Action | Strike | Put Action | Strike |
|---|---|---|---|---|
| 0 | CALL_ATM | ATM | PUT_ATM | ATM |
| 1 | CALL_OTM5 | ATM+5 | PUT_OTM5 | ATM-5 |
| 2 | CALL_OTM10 | ATM+10 | PUT_OTM10 | ATM-10 |
| 3 | CALL_OTM15 | ATM+15 | PUT_OTM15 | ATM-15 |
| 4 | CALL_OTM20 | ATM+20 | PUT_OTM20 | ATM-20 |
| 5 | CALL_OTM25 | ATM+25 | PUT_OTM25 | ATM-25 |
| 6 | CALL_OTM30 | ATM+30 | PUT_OTM30 | ATM-30 |

**Total actions:** DO_NOTHING + 14 entries + EXIT = **16 actions**

**Data cost:** Near zero. The flat file download (`prefetch_spxw_from_flatfiles`) pulls a **full day's option file** from S3, then filters for specific strikes — adding ±25/±30 costs zero extra downloads. For Polygon API days, ~2.5 min extra download time. Existing per-day caches that lack ±25/±30 will need re-download for those strikes only.

**Liquidity concern:** Deep OTM 0DTE options (25-30 points out) may have sparse bars on low-volume days. The existing NaN handling for OTM15/20 already covers this — missing bars produce NaN P&L which sniper_loss skips.

**Cache invalidation (REQUIRED):** Existing per-day caches in `spxw_chain/*.pkl` only contain ±5/±10/±15/±20 data. These MUST be deleted and re-downloaded to include ±25/±30 strikes. Steps:
1. Delete all files in `data/spxw_chain/` (forces re-download with new strike steps)
2. The flat file path (`prefetch_spxw_from_flatfiles`) re-downloads the full day file from S3 and extracts all needed strikes in one pass — no extra cost
3. The Polygon API path (`download_spxw_chain`) will re-download all strikes per day including the new ±25/±30 — adds ~4 extra API calls per day
4. This re-download is part of the `prepare.py` run and happens automatically once `OTM_STRIKE_STEPS` is updated and caches are cleared
5. Estimated time: ~30-60 min for ~300 days depending on data source (flat files are faster than API)

**Changes required:**

1. **prepare.py:**
   - `OTM_STRIKE_STEPS`: `(5, 10, 15, 20)` → `(5, 10, 15, 20, 25, 30)`
   - Add OTM25/OTM30 price arrays and P&L computation (same pattern as OTM15/20)
   - Promote ALL OTM strikes from sidecar to core model data
   - Add stopped P&L labels (tight/med/wide) for all new strikes for sniper_loss
   - Update direction label assignment for 14-class direction
   - Update `SIDE_ACTION_ORDER` and related constants
   - Invalidate & re-download cached days missing ±25/±30 data (or add incremental download for new strikes only)

2. **train.py:**
   - Direction head: 6 → 14 logits
   - Update `ACTION_*` constants for all 14 entry types + EXIT
   - Update `NUM_ACTIONS`: 8 → 16
   - Update `sniper_loss` direction targets for 14-class direction
   - Update `evaluate_trades()` to handle all OTM strikes with price lookups
   - Direction entropy bonus may need adjustment (14 classes vs 6)

3. **replay.py:**
   - Add OTM15/20/25/30 strike handling in backtest logic
   - Add price lookups for new strikes

4. **program.md:**
   - Update action space: 8 → 16 actions
   - Update direction head: 6 → 14 outputs

**This is a head output dimension change → MUST fresh start** (already required by feature count change).

### A.2 Implementation Details

**MACD-H slope (`macdh_slope`):**
```python
# Pre-compute outside main loop
ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
macd_line = ema_12 - ema_26
signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
macd_h = macd_line - signal_line
macdh_slope = np.sign(np.diff(macd_h, prepend=macd_h[0]))
```
- Add to `_NO_NORMALIZE`: Yes (already in [-1, 0, +1])
- NaN handling: First bar = 0

**Force Index 2-bar EMA (`force_index_2`):**
```python
# Pre-compute outside main loop
price_change = np.diff(close, prepend=close[0])
force_raw = volume * price_change
force_ema2 = pd.Series(force_raw).ewm(span=2, adjust=False).mean().values
# Normalize per-session by rolling 20-bar std
force_std = pd.Series(np.abs(force_ema2)).rolling(20, min_periods=1).std().values
force_index_2 = np.where(force_std > 1e-8, force_ema2 / force_std, 0.0)
force_index_2 = np.clip(force_index_2, -5.0, 5.0)
```
- Add to `_NO_NORMALIZE`: No (will be z-score normalized with other features)
- Session boundary: Reset EMA at session start

**Volume-price divergence (`vol_price_diverg`):**
```python
# Pre-compute outside main loop
price_dir = np.sign(np.diff(close, prepend=close[0]))  # +1 up, -1 down
vol_dir = np.sign(np.diff(volume, prepend=volume[0]))   # +1 rising, -1 falling
agreement = price_dir * vol_dir  # +1 = agree, -1 = disagree
# Count consecutive disagreements (reset on agreement)
diverg_count = np.zeros(N, dtype=np.float64)
for i in range(1, N):
    if agreement[i] < 0:  # disagreement
        diverg_count[i] = diverg_count[i-1] + price_dir[i]  # signed accumulation
    else:
        diverg_count[i] = 0.0
diverg_count = np.clip(diverg_count, -10.0, 10.0)
```
- Add to `_NO_NORMALIZE`: Yes (already bounded [-10, 10])

**Effort vs Result (`effort_vs_result`):**
```python
# Pre-compute outside main loop
body_size = np.abs(close - opn)
avg_body_20 = pd.Series(body_size).rolling(20, min_periods=1).mean().values
avg_vol_20 = pd.Series(volume.astype(float)).rolling(20, min_periods=1).mean().values
body_ratio = np.where(avg_body_20 > 1e-8, body_size / avg_body_20, 1.0)
vol_ratio_local = np.where(avg_vol_20 > 1e-8, volume / avg_vol_20, 1.0)
effort_vs_result = np.where(vol_ratio_local > 0.1, body_ratio / vol_ratio_local, 0.0)
effort_vs_result = np.clip(effort_vs_result, -3.0, 3.0)
```
- Add to `_NO_NORMALIZE`: Yes (already bounded)

**5-min Trend (`trend_5min`):**
```python
# Pre-compute outside main loop — aggregate 5-bar closes
close_5min = np.array([close[max(0, i-4):i+1].mean() for i in range(0, N, 5)])
# Pad back to N bars (each 5-min value covers 5 bars)
close_5min_expanded = np.repeat(close_5min, 5)[:N]
# EMA(13) on 5-min data, then expand
ema13_5min = pd.Series(close_5min).ewm(span=13, adjust=False).mean().values
ema13_expanded = np.repeat(ema13_5min, 5)[:N]
# Slope: change per bar, normalized by close
trend_5min = np.diff(ema13_expanded, prepend=ema13_expanded[0]) / np.maximum(close, 1.0)
trend_5min = np.clip(trend_5min, -0.01, 0.01)  # cap at ±1%
```
- Add to `_NO_NORMALIZE`: No (will normalize with other features)
- Session boundary: Reset at session start

**RSI(7) replacing RSI(14):**
- Change the RSI computation window from 14 to 7
- Keep feature name as `rsi_7` (update FEATURE_NAMES)
- Same [0, 1] scaling

### A.3 Other prepare.py Changes

- Update `FEATURE_NAMES` list (add 5, rename 1)
- Update `NUM_FEATURES` (automatic via `len(FEATURE_NAMES)`)
- Update `_NO_NORMALIZE` set for new features
- Pre-compute all new arrays BEFORE main loop (performance critical — learned from v9 session_cum_delta O(n²) bug)

### A.4 Fresh Start Procedure

Feature count change (42 → 47) **requires** fresh start per operating manual.

1. Archive current model: `cp training/best_model.pt archive/models/v9/`
2. Archive current train.py: `cp training/train.py archive/models/v9/`
3. Save score: `cp training/.best_score archive/models/v9/`
4. Delete `training/best_model.pt`
5. Reset `training/.best_score` to `-5.0`
6. Delete `training/.inner_loop_state.json`
7. Sync: `cp training/train.py training/best_train.py`
8. Update train.py feature count comment

### A.5 train.py Changes for Track A

Required structural changes:
- Update feature count comment (42 → 47)
- Direction head output: 6 → 14 logits
- `NUM_ACTIONS`: 8 → 16 (DO_NOTHING + 14 entries + EXIT)
- Add action constants: `ACTION_BUY_CALL_OTM15/20/25/30`, `ACTION_BUY_PUT_OTM15/20/25/30`
- Update `sniper_loss` to handle 14-class direction targets
- Update `evaluate_trades()` to handle all OTM entries with price lookups from data.pt
- Direction entropy bonus may need tuning (14 classes have higher max entropy than 6)
- No loss weight changes
- No new env vars beyond what's needed for OTM strikes

---

## Track A Testing Phase

### A.T1 Data Validation (local, no GPU)

Before deploying to Akash:

1. **Rebuild data.pt:** `python3 training/prepare.py`
   - Verify output: `data.pt` should have tensors with 47 feature columns
   - Verify no NaN in new features
   - Verify new feature distributions (print min/max/mean/std for each)

2. **Feature sanity checks:**
   - `macdh_slope`: should be roughly 33% each of {-1, 0, +1}
   - `force_index_2`: should be roughly normal, centered near 0
   - `vol_price_diverg`: should be mostly 0 with tails to ±5
   - `effort_vs_result`: should be mostly 0.5-2.0 with tails
   - `trend_5min`: should show clear positive/negative regimes across days
   - `rsi_7`: should be similar to rsi_14 but more responsive (wider spread)

3. **Feature correlation check:**
   - New features should NOT be >0.9 correlated with existing features
   - `macdh_slope` vs `ema_cross`: expect moderate correlation (~0.5), not redundant
   - `force_index_2` vs `bar_delta`: expect low correlation (<0.3) — different signals

### A.T2 Training (Akash GPU)

1. **Deploy:** `./infra/deploy.sh boot && ./infra/deploy.sh start`
2. **Upload:** Updated prepare.py, train.py, best_train.py, fresh data.pt
3. **Initialize:** `python3 tools/inner_loop.py init`
4. **Run 8-10 sequential experiments** — fresh start compounding
   - Each experiment warm-starts from previous KEEP
   - Default hyperparameters (no tuning yet — isolate feature impact)
   - ~6 min per experiment → ~60-90 min total
5. **Monitor:** `python3 tools/inner_loop.py status` between experiments

### A.T3 Validation

After training converges (score plateaus or best 3 consecutive experiments not promoted):

1. **Download best model:** via inner_loop.py (automatic on KEEP)
2. **Replay backtest:** `python3 training/replay.py --backtest --model training/best_model.pt`
3. **Record metrics:** PF, trades, WR, avg hold, exit reasons, direction breakdown

### A.T4 Comparison vs Previous Models

| Metric | v8 (37 feat) | v9 (42 feat) | v10 Target | Why |
|---|---|---|---|---|
| Replay PF | 1.43 | 1.56 | >1.60 | Better features → better entry timing |
| Win Rate | 40% | 44% | >45% | Leading indicators improve prediction |
| Trades/Day | 1.1 | 0.5 | 0.5-1.5 | Maintain selectivity |
| Avg Hold | 6 bars | 8 bars | 5-15 bars | Passarelli: need 5+ bars for theta compensation |
| Stop Loss Rate | 19% | 15% | <15% | Better entries → fewer stops hit |
| MODEL_EXIT Rate | 52% | 67% | >60% | Gate should remain primary exit |
| Max Drawdown | -12% | -10% | <-10% | Tighter through better selectivity |
| Train/Replay PF Gap | 3.8x | 4.4x | <3.0x | Better features → less overfitting |
| Direction Split | PUT heavy | 50/50 | Either | As long as both are profitable |
| OTM Trade % | ~20% | ~16% | 15-40% | Full ±30 OTM ladder gives model Pickles-like expressiveness |
| OTM P&L | +$4,520 (v8) | +$3,865 | >$3,000 | OTM trades should remain profitable |
| OTM15+ Usage | N/A | N/A | >0 | Model should discover when deeper OTM is valuable |
| Strike Diversity | 6 strikes | 6 strikes | 14 strikes | ATM + OTM5/10/15/20/25/30 × Call/Put |

**Success criteria for Track A:**
- Replay PF ≥ 1.50 (must not regress from v9)
- At least ONE of: higher WR, lower stop rate, better train/replay gap
- No critical anomaly flags

**Failure criteria:**
- Replay PF < 1.30 → revert to v9, investigate which feature is harmful
- 0 trades → gate collapsed, likely feature normalization issue
- Score < 5.0 after 8 experiments → features not helping, stop and diagnose

---

## Track B: Training Dynamics (v10.1)

**Prerequisites:** Track A complete. v10 model exists (whether improved or baseline).

Track B modifies train.py only. Warm start from Track A's best model.

### B.1 Time-of-Day Sample Weighting

**Source:** Passarelli (theta per minute lowest in morning, extreme in afternoon) + Elder (morning trends strongest, lunch avoid)

**Implementation:**
```python
# In sniper_loss or training loop, weight gate loss by time-of-day
# Use existing theta_pressure feature (0 morning → 1 close) or minutes_to_close
WEIGHT_TOD_MORNING = _env_float("WEIGHT_TOD_MORNING", 0.0, lo=0.0, hi=2.0)
# When > 0: upweight morning bars (theta_pressure < 0.3) by 1.0 + WEIGHT_TOD_MORNING
# Downweight lunch bars (0.3 < theta_pressure < 0.6) by 1.0 - WEIGHT_TOD_MORNING * 0.3
# Neutral for afternoon (theta_pressure > 0.6)
```

**Rationale:** Morning entries have better edge (lower theta, stronger trends). The model should learn MORE from morning bars and LESS from lunch chop. This doesn't change labels — just sample emphasis.

**Env var:** `WEIGHT_TOD_MORNING` (default 0.0 = no-op, try 0.5-1.0)

### B.2 Volume-Confirmed Sample Weighting

**Source:** Coulling (high volume = professional consensus, more meaningful signals) + Elder (volume confirms trends)

**Implementation:**
```python
WEIGHT_HIGH_VOL = _env_float("WEIGHT_HIGH_VOL", 0.0, lo=0.0, hi=2.0)
# When > 0: bars with volume_ratio > 1.25 get weight 1.0 + WEIGHT_HIGH_VOL
# Bars with volume_ratio < 0.5 get weight 1.0 - WEIGHT_HIGH_VOL * 0.3
```

**Rationale:** High-volume bars carry more information signal. Low-volume bars (lunch chop) are noise. Upweighting high-volume bars in loss computation teaches the model from better data.

**Env var:** `WEIGHT_HIGH_VOL` (default 0.0 = no-op, try 0.3-0.8)

### B.3 Regime-Balanced Batching

**Source:** Sinclair (variance premium varies by VIX regime, model must generalize across all)

**Implementation:**
```python
WEIGHT_REGIME_BALANCE = _env_float("WEIGHT_REGIME_BALANCE", 0.0, lo=0.0, hi=1.0)
# When > 0: during batch construction, stratify by VIX regime
# Ensure each batch has proportional representation of low/normal/elevated/crisis VIX bars
# Prevents model from specializing on the most common regime
```

**Rationale:** If 70% of training data is low-VIX, the model optimizes for low-VIX behavior and fails during regime transitions. Balanced batching forces generalization.

**Env var:** `WEIGHT_REGIME_BALANCE` (default 0.0 = no-op, try 0.3-0.7)

### B.4 Implementation Order

Track B changes are independent — test each in isolation to attribute impact:

1. **Experiment B1:** WEIGHT_TOD_MORNING=0.5 only
2. **Experiment B2:** WEIGHT_HIGH_VOL=0.5 only
3. **Experiment B3:** WEIGHT_REGIME_BALANCE=0.5 only
4. **Experiment B4:** Best combination from B1-B3

Each experiment is a single sequential run with warm start from v10 best.

---

## Track B Testing Phase

### B.T1 Training (Akash GPU)

- Warm start from v10 best model (no fresh start — only loss weight/sampling changes)
- 4 experiments (B1-B4), ~6 min each → ~30 min
- Monitor each for score improvement

### B.T2 Validation

For each kept experiment:
1. **Replay backtest** with same procedure as Track A
2. **Compare against v10 baseline** (Track A result)

### B.T3 Comparison vs All Models

| Metric | v8 | v9 | v10 (Track A) | v10.1 (Track B) |
|---|---|---|---|---|
| Replay PF | 1.43 | 1.56 | (measured) | (measured) |
| Win Rate | 40% | 44% | (measured) | (measured) |
| Trades/Day | 1.1 | 0.5 | (measured) | (measured) |
| Avg Hold | 6 | 8 | (measured) | (measured) |
| Stop Rate | 19% | 15% | (measured) | (measured) |
| Max DD | -12% | -10% | (measured) | (measured) |
| Train/Replay Gap | 3.8x | 4.4x | (measured) | (measured) |

**Success criteria for Track B:**
- Any single change (B1/B2/B3) improves replay PF over v10 baseline
- Combined (B4) shows additive improvement
- No regression in trade frequency or direction diversity

**Failure criteria:**
- All B1-B3 show no improvement → sample weighting not the bottleneck
- Any change causes score < v10 baseline → revert that specific change

---

## Timeline & Budget

| Phase | Duration | GPU Cost | Blocker |
|---|---|---|---|
| A.1-A.3: Implement features | 1-2 hours | $0 (local) | None |
| A.4: Fresh start prep | 10 min | $0 (local) | None |
| A.T1: Data validation | 30 min | $0 (local) | data.pt rebuild |
| A.T2: Training (8-10 exp) | 60-90 min | ~$5-8 (H100) | Akash deploy |
| A.T3-A.T4: Validation | 20 min | $0 (local) | Model download |
| B.1-B.4: Implement dynamics | 30 min | $0 (local) | Track A complete |
| B.T1: Training (4 exp) | 30 min | ~$3-4 (H100) | Akash deploy |
| B.T2-B.T3: Validation | 20 min | $0 (local) | Model download |

**Total estimated GPU cost:** ~$8-12
**Total wall clock:** ~4-5 hours (including local work)

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| New features introduce NaN/Inf | Validate all features in A.T1 before GPU spend |
| Feature correlation redundancy | Check pairwise correlations; drop if >0.9 with existing |
| Fresh start regresses from v9 | Archive v9 model; if v10 PF < 1.30 after 8 exp, revert |
| Track B changes are no-ops | All env vars default 0.0; warm start preserves baseline |
| Overfitting worsens with more features | Monitor train/replay PF gap; if gap > 5x, add dropout |
| Session boundary bugs in new features | Reset all session-dependent features at day boundaries |

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-27 | Path 2 selected (better data) | v9 research: features lag price, predictions random |
| 2026-03-27 | 6 features from 7-book synthesis | Multi-book evidence for each; all LEADING indicators |
| 2026-03-27 | RSI(14)→RSI(7) | Elder: "7-9 bars for intraday" |
| 2026-03-27 | Defer gamma/theta ratio to v11 | Requires NaN handling validation; lower priority |
| 2026-03-27 | Track A before Track B | Root cause is missing features, not training dynamics |
| 2026-03-27 | 8-10 experiments for Track A | Fresh start needs convergence; v8 took ~5 exp, budget 8-10 |
| 2026-03-27 | Expand OTM strikes: 6→14 direction outputs, full ±5/10/15/20/25/30 ladder | Pickles (17yr, $100M+) buys OTM calls almost exclusively, up to +30 OTM. v9 OTM already profitable. No reason to limit model's expressiveness — data cost is near zero (flat files contain all strikes) |
| 2026-03-27 | Corrected anti-OTM bias from book analysis | Sinclair's statistical argument doesn't apply to selective entry with confluence. Our own v9 data: OTM trades +$3,865 across 26 trades |

---

## References

- `docs/domain/book-knowledge-synthesis.md` — 7-book synthesis (450+ lines)
- `docs/domain/sinclair-volatility-trading-extract.md` — Variance premium, vol regimes
- `docs/domain/douglas-trading-zone-extraction.md` — Casino model, risk management
- `/tmp/elder-extraction-for-0dte-bot.md` — Triple Screen, MACD-H, Force Index, RSI(7)
- `/tmp/rhoads-extraction.md` — VIX regimes, Rule of 16, term structure
- `memory/research_v9_breakthrough.md` — v9 deep analysis, dead ends, 3 paths
- `training/program.md` — Model contract, architecture lock, loss policy
