# Audit: Section 2 -- Feature Engineering & Labels

**Date:** 2026-04-06
**Scope:** Feature accuracy, label computation, cost model consistency, data quality
**Architecture ref:** Section 2 of `v2/docs/architecture.md`

---

## Part A: Feature Accuracy (Estimated vs Real Data)

### A1: Which features are estimated?

6 of 55 base features are estimated from option prices rather than sourced from real market data.

**Greeks (3 features):**

| Feature | How we compute it | Source |
|---------|------------------|--------|
| `atm_gamma` | BS: `N'(d1) / (S * sigma * sqrt(T))` | `prepare.py:346` |
| `atm_theta_per_bar` | BS annualized theta / (252 * 390) | `prepare.py:348-350` |
| `charm_estimate` | Estimated dDelta/dT | `prepare.py` (derived) |

**Implied Volatility (3 features):**

| Feature | How we compute it | Source |
|---------|------------------|--------|
| `atm_iv` | Brent's method inverting BS from close price | `prepare.py:320-331` |
| `iv_skew` | put_iv - call_iv | derived |
| `iv_percentile` | Rank vs 60-day rolling history | derived |

### A2: QC validation attempt -- 7 tests, definitive result

We ran 7 QuantConnect backtests to validate our estimates.

**Definitive finding: QC does not compute Greeks or IV for SPXW 0DTE.**

All 90 readings across 5 days (Dec 17-23, 2025) returned IV=0.0, delta=0.0, gamma=0.0, theta=0.0, vega=0.0 for true SPXW 0DTE contracts. QC only computes Greeks for standard SPX weekly options (1+ DTE).

**This closes the QC free tier path. No external Greek validation is possible via QC.**

| Test | Script | Result |
|------|--------|--------|
| 1 | qc_08 v1 (Expiration 0,5) | 72 readings, all SPX weeklies (1-4 DTE), not 0DTE |
| 2 | qc_09 diagnostic | Confirmed: all contracts exp=Friday, symbol=`SPX` not `SPXW` |
| 3 | qc_10 multi-approach | Found fix: `AddIndex("SPXW")` returns true 0DTE |
| 4 | qc_08 v2 (Expiration 0,0) | NO DATA -- SPX chain has no SPXW without separate subscription |
| 5 | qc_11 v1 (Dec 15-16) | NO DATA -- QC may lack SPXW for Mon/Tue |
| 6 | qc_11 v2 (Dec 17-23) | **90 readings, all 0DTE, all Greeks=0.0** |

**Remaining path:** Compute realized Greeks from Polygon price movements (empirical delta/gamma from option price changes vs SPX changes). No external dependency needed.

### A3: Bid-ask spread -- VALIDATED

90 SPXW 0DTE readings across 5 days:

| Time bucket | Median spread | Our estimate |
|-------------|--------------|-------------|
| Open (bar 10-30) | $0.15 | 30-40 bps (~$0.20-$0.27) |
| Morning (bar 31-120) | $0.20 | 30-50 bps |
| Midday (bar 121-240) | $0.10 | 50-80 bps |
| Afternoon (bar 241-360) | $0.13 | 50-60 bps |
| Close (bar 361+) | $0.15 | 150 bps (~$1.00) |

**Verdict: Our adaptive spread model OVERESTIMATES costs for SPXW 0DTE by roughly 2x midday and 5-10x near close.** This makes labels more pessimistic than reality. Conservative direction -- acceptable but not accurate.

**Correction to prior research:** The earlier 13,600-observation study (qc_02 series, $0.40-$0.70 ATM) measured SPX Friday weeklies, not SPXW 0DTE. Those spreads don't apply to our trading product.

### A4: Greek risk assessment

**Status: UNKNOWN -- cannot validate.** The theoretical concerns stand but are unquantified:
- BS gamma diverges as T approaches 0
- BS theta acceleration may not match real behavior
- IV from close prices has bid-ask noise

These features showed signal edge in the feature scan (`atm_gamma` PF=1.176, `atm_iv` PF=1.182). Whether the edge comes from the real signal or from BS artifacts is unknown.

---

## Part B: Label Computation

### B1: How are P&L labels computed?

`build_v2_dataset.py:512-532` -- `_sim_one()` function:

For each bar where `bod >= 30` and `bod < 270` and volume >= 1:
1. Find nearest-ATM strike at signal bar
2. Get **close price** at next bar (fill bar) for both call and put
3. Simulate forward up to 30 bars with fixed stop/target:
   - Stop loss: -30% (checked first each bar)
   - Take profit: +50%
   - Timeout: 30 bars or end of day
4. Deduct spread cost from P&L
5. `label_trade = True` only when `max(call_pnl, put_pnl) > 0`

### B2: What prices are used? -- Close only

| Stage | Price used | Source |
|-------|-----------|--------|
| Entry | `call_close` / `put_close` at fill bar (entry + 1) | Wide-grid pickle |
| Hold monitoring | `call_close` / `put_close` at each subsequent bar | Wide-grid pickle |
| Exit | Same close price where barrier hit, or last valid close | Wide-grid pickle |

**No bid/ask prices used anywhere.** Entry and exit are both at close. The spread cost is modeled separately via `compute_adaptive_spread_bps()`, not derived from actual bid-ask.

**Implication:** The labeler and simulator assume you can enter and exit at the close price, then separately deduct an estimated spread. In reality, you trade at bid (sell) or ask (buy), and the close may not be between bid and ask.

### B3: Cost model -- labeler vs simulator

Both use `compute_adaptive_spread_bps()` from `features.py:271-296`.

**Inconsistency found:**

| Aspect | Labeler | Simulator |
|--------|---------|-----------|
| Exit time estimate | Fixed: `mtc - 30` bars | Actual exit bar's mtc |
| Exit VIX regime | Reuses entry bar's VIX | Uses actual exit bar's VIX |
| Stop/target price | Simulated from close prices | Same approach |
| Commission | Not included | Not included |

The labeler always assumes exit happens 30 bars later and uses entry VIX for exit spread. The simulator uses actual exit conditions. This creates a systematic cost mismatch between labels and evaluation.

**Magnitude:** Small in most cases. The spread model varies 30-150 bps by time-of-day. A 30-bar offset in minutes_remaining shifts the bucket by ~1 tier (e.g., 50 bps vs 60 bps). VIX rarely changes materially in 30 minutes. Estimated impact: ~10-20 bps difference per trade.

### B4: Are penny options included?

**Yes.** `MIN_VOLUME = 1` (line 53) is the only filter. No minimum price.

A $0.05 put with 1 contract traded qualifies. At that price:
- A $0.01 move = 20% return
- Stop at -30% = $0.015 loss ($1.50 per contract)
- Target at +50% = $0.025 gain ($2.50 per contract)

This distorts P&L labels for deep OTM options near close. The bars at the end of the day where one side has expired worthless ($0.05) and the other side is deep ITM ($40+) are included with equal weight.

**Visible in the QC data:** Dec 17 bar 360: call=$0.05, put=$46.35. Dec 22 bar 380: put=$0.05.

### B5: When both P&Ls are negative

`label_trade = False`. The bar is labeled as "don't trade." Direction is still set to the less-bad side, but the gate signal is off.

20.3% of signal bars have both call and put P&L <= 0. These are no-trade bars where the model should learn to abstain.

### B6: Is the 79.7% gate=True rate realistic?

**The architecture doc says 79.7%. The code has a validation gate that warns if > 70%:**

```python
if gate_true_rate > 0.70:
    errors.append(f"Gate=True rate {gate_true_rate:.1%} > 70% -- labels not selective enough")
```

**So the code itself flags 79.7% as a problem.** The rate is high because:

1. With stop=30% and target=50%, there's a wide neutral zone
2. Options have high realized volatility on 0DTE -- even small SPX moves create large option returns
3. If EITHER direction profits, gate=True. Two chances to win at each bar.
4. Cost deduction is small (typically 0.6-1.5%) relative to the 30%/50% barriers

**Assessment:** 79.7% is plausible given the asymmetric dual-direction setup, but it means the gate head has limited room to learn selectivity. The model sees "trade" on 4 out of 5 bars.

---

## Part C: Feature Engineering

### C1: Feature count -- mismatch in constants

| Source | Count |
|--------|-------|
| Architecture doc | 71 |
| `features.py` NUM_FEATURES | 55 |
| `build_v2_dataset.py` output | 71 (55 + 16 enriched) |
| `train.py` default | 71 |

`features.py:NUM_FEATURES = 55` is stale. The `validate_feature_shape()` function would reject the actual 71-feature dataset. Not a runtime issue (validation may not be called) but a code hygiene problem.

### C2: The 39 original features

Ported directly from v1 (`archive/v1/training/prepare.py`). Not recomputed in v2. Cover:
- Price returns (2): `ret_6`, `ret_12`
- Volume (1): `volume_ratio`
- Volatility (3): `realized_vol`, `bar_range`, `range_ratio`
- **Greeks (3): `atm_gamma`, `atm_theta_per_bar`, `charm_estimate`** (estimated, see Part A)
- **IV (3): `atm_iv`, `iv_skew`, `iv_percentile`** (estimated, see Part A)
- VIX/regime (2): `vix_regime`, `vrp`
- Market structure (5): `vwap_dist`, `poc_dist`, `va_position`, `ib_break`, etc.
- Momentum/technical (12): RSI, Bollinger, MACD, force index, etc.
- Session/time (5): `minutes_to_close`, `session_range_pct`, etc.
- Other (3): `overnight_gap`, `prev_close_dist`, etc.

### C3: The 16 enriched features

Computed in `build_v2_dataset.py:81-160` from wide-grid data. All relative/normalized:

- **Moneyness (3):** `current_moneyness_pct`, `intraday_drift_pct`, `near_atm_moneyness_pct`
- **Volume (6):** per-strike and chain-level volumes, flow ratios, zero flags
- **Spread proxy (1):** `call_hl_range_pct` (Corwin-Schultz from high-low)
- **Normalized prices (2):** `near_atm_call_price_norm`, `near_atm_put_price_norm` (option/SPX * 100)
- **Theta (1):** `theta_acceleration = 1/sqrt(minutes_to_close)`
- **Activity (1):** `near_atm_transactions`
- **Chain flow (2):** `chain_call_put_ratio`, `log_chain_volume`

### C4: Normalization -- no leakage detected

**Method:** Per-day z-score with expanding-window standard deviation.
- Mean: computed per-day (removes day fingerprints)
- Std: expanding window from all prior days (walk-forward, no look-ahead)
- Clipped to [-5, 5]
- 12 features excluded from normalization (already bounded: `minutes_to_close`, `iv_percentile`, `vix_regime`, etc.)

**Verdict: No forward-looking leakage in normalization.**

### C5: NaN handling

- Forward-fill within each day (mimics stale quotes)
- Remaining NaN at start-of-day set to 0.0
- 20.6% of bars have zero volume at nearest ATM strike (forward-filled)
- Loses distinction between "missing" and "actual zero"

### C6: 4-way temporal split -- clean

| Split | Days | Purpose |
|-------|------|---------|
| Train | ~854 | Model training |
| Val | 60 | Checkpoint selection |
| Promote | 60 | Keep/revert scoring |
| Shadow | 20 | Live-readiness eval |

No overlap. Strictly temporal (earliest = train, latest = shadow). No bar-level mixing.

---

## Summary of Findings

| Finding | Severity | Status |
|---------|----------|--------|
| Greeks/IV unvalidated for 0DTE | MEDIUM | Cannot resolve via QC. Polygon realized-Greeks path available. |
| Spread model overestimates by 2-5x | LOW | Conservative direction. Labels are more pessimistic than reality. |
| Labels use close prices, not bid/ask | LOW | Internally consistent (labeler = simulator). Not market-realistic. |
| Labeler/simulator cost model inconsistency | LOW | ~10-20 bps difference. Exit VIX and time estimated vs actual. |
| Penny options included (MIN_VOLUME=1) | MEDIUM | Deep OTM near close distorts P&L labels. No min price filter. |
| Gate=True rate 79.7% (code warns >70%) | MEDIUM | Gate head has limited selectivity signal. Driven by dual-direction + wide barriers. |
| Feature count constant stale (55 vs 71) | LOW | `validate_feature_shape()` would fail on real data. Code hygiene. |
| No commission in cost model | LOW | $1.30 RT on a $10 option = 13 bps. Small vs spread model. |
| Prior QC spread research measured wrong product | MEDIUM | 13,600 obs of SPX weeklies, not SPXW 0DTE. Findings don't apply. |

---

## Recommendations

1. **Add minimum entry price filter.** Exclude options below $0.50 or $1.00 from label computation. Penny option P&L is noise.

2. **Compute realized Greeks from Polygon.** Measure empirical delta (option price change / SPX change) and gamma (delta change over 5-10 bar windows). Compare against BS estimates. No external data needed.

3. **Align labeler and simulator cost model.** Use actual exit bar's VIX and minutes_remaining in the labeler, not the estimated 30-bar offset.

4. **Consider lowering gate threshold.** 79.7% gate=True gives the model limited room to learn selectivity. Options: raise cost estimates, add a minimum P&L threshold (e.g., `best_pnl > 0.02` instead of `> 0`), or tighten barriers.

5. **Update feature count constant.** Set `NUM_FEATURES = 71` in `features.py` or remove the stale constant.

---

## Test data and logs

| File | Contents |
|------|----------|
| `v2/research/qc_11_results_all.txt` | 90 SPXW 0DTE readings (5 days, Greeks=0.0, bid/ask valid) |
| `v2/research/qc_08_results_raw.txt` | 72 SPX weekly readings (non-0DTE, discarded) |
| `v2/docs/audit/audit_section2_logs/` | Raw QC backtest logs (days 17-23) |
| `v2/research/qc_08_greeks_comparison.py` | QC capture script (SPXW 0DTE version) |
| `v2/research/qc_11_greeks_1day.py` | Single-day SPXW capture (definitive test) |
| `v2/research/compare_greeks.py` | Local BS comparison tool (ready for realized Greeks) |
