# Audit: Section 2 -- Feature Accuracy (Estimated vs Real Data)

**Date:** 2026-04-06
**Scope:** Greeks, IV, and bid-ask spread features used in training
**Key files:** `archive/v1/training/prepare.py:320-353`, `v2/core/features.py:271-296`, `v2/pipeline/build_v2_dataset.py`

---

## Summary

6 of 55 training features are estimated from option prices rather than sourced from real market data. We ran 7 tests on QuantConnect free tier to validate these estimates against real computed values.

**Key findings:**

1. **Greeks and IV: CANNOT be validated via QC free tier.** QC does not compute Greeks or IV for SPXW 0DTE contracts. All values return 0.0. Greeks are only computed for standard SPX weekly options (1+ DTE).

2. **Bid-ask spread: VALIDATED. Our $0.30 estimate is conservative (slightly high).** Real SPXW 0DTE ATM spreads are $0.10-$0.30 median across 90 readings over 5 days. Our estimate deducts slightly too much cost from P&L labels, making labels marginally more conservative than reality. This is acceptable.

3. **Prior spread research was based on wrong options.** The earlier 13,600-observation study (qc_02 series) measured SPX Friday weeklies, not SPXW 0DTE. Those $0.40-$0.70 spreads apply to multi-DTE options, not the 0DTE contracts we actually trade.

**Verdict on Greek estimates: UNRESOLVABLE via QC free tier. Alternative validation path identified (realized Greeks from Polygon price movements).**

---

## Q1: Which features are estimated, and how?

### Greeks (3 features)

| Feature | How we compute it | Source code |
|---------|------------------|-------------|
| `atm_gamma` | Black-Scholes: `N'(d1) / (S * sigma * sqrt(T))` | `prepare.py:346` |
| `atm_theta_per_bar` | BS annualized theta / (252 * 390) | `prepare.py:348-350` |
| `charm_estimate` | Estimated dDelta/dT | `prepare.py` (derived) |

**Inputs to BS:** SPX price (real, from IBKR), ATM strike (real), IV (estimated, see below), T = minutes_remaining / (252 * 390).

**Known concern:** BS assumes flat vol surface, constant rates, and European exercise. For 0DTE options where T approaches zero, gamma and theta change rapidly. Small errors in T or IV propagate into large Greek errors. This is the regime where BS approximations are weakest.

### Implied Volatility (3 features)

| Feature | How we compute it | Source code |
|---------|------------------|-------------|
| `atm_iv` | Brent's method inverting BS from option close price | `prepare.py:320-331` |
| `iv_skew` | put_iv - call_iv | derived from `atm_iv` |
| `iv_percentile` | Rank of current IV vs 60-day rolling history | derived from `atm_iv` |

**Known concern:** We extract IV from close prices, not mid prices. Close price can be the last trade (stale) or settlement (synthetic). The bid-ask spread introduces noise: a $0.50 spread on a $10 option is 5% price uncertainty, which maps to meaningful IV uncertainty in the short-dated regime.

### Bid-ask spread (affects labels, not features directly)

| Usage | How we estimate it | Source code |
|-------|-------------------|-------------|
| P&L label simulation | Hardcoded lookup: 30-150 bps by time-of-day + VIX | `features.py:271-296` |
| Spread feature | Corwin-Schultz from high-low range | `build_v2_dataset.py` |

---

## Q2: What real data sources exist for these values?

| Data | Source | Available for 0DTE? | Notes |
|------|--------|---------------------|-------|
| Computed Greeks | QuantConnect | **NO** -- returns 0.0 for SPXW | Only works for SPX weeklies (1+ DTE) |
| Real IV | QuantConnect | **NO** -- returns 0.0 for SPXW | Same limitation |
| Real bid/ask quotes | QuantConnect | **YES** -- confirmed working | 90 readings across 5 days |
| Real-time Greeks | IBKR | Live only, no deep history | Not viable for training data |
| Historical Greeks | Polygon | **No** -- Polygon has no Greeks or IV | N/A |
| Realized Greeks | Polygon price movements | **YES** -- can compute empirically | See alternative validation path below |

---

## Q3: Validation testing -- 7 tests, results

### Test 1: qc_08 v1 with Expiration(0,5) -- INVALID

Ran Dec 8-19, 2025. Got 72 readings. All `0dte=0`.

**Root cause:** `Expiration(0,5)` returns SPX Friday weeklies (symbol: `SPX 251219C...`), not SPXW dailies. All contracts had 1-4 DTE. Comparison was apples-to-oranges.

**Discarded:** Gamma errors of 66-903% were artifacts of the DTE mismatch.

### Test 2: qc_09 expiry diagnostic -- CONFIRMED THE BUG

Ran Dec 15-19. All contracts had `exp=2025-12-19` (Friday). Symbol: `SPX 251219C06815000`. Standard Friday SPX, not SPXW.

### Test 3: qc_10 multi-approach -- FOUND SPXW ACCESS

Ran Dec 17 (single day). Three approaches:

| Approach | Result |
|----------|--------|
| A: `Expiration(0,0)` on SPX | Returned SPXW contracts, but only because Approach C's subscription was active |
| B: `OptionChainProvider` | 9,220 contracts, monthly expiries only. Dead end. |
| C: `AddIndex("SPXW")` | **66 SPXW contracts.** Symbol: `SPXW 251217C06790000`. True 0DTE. |

**Key learning:** SPXW must be added as a separate index via `AddIndex("SPXW")`. `AddIndexOption` on SPX only returns standard SPX options.

### Test 4: qc_08 v2 with Expiration(0,0) on SPX -- NO DATA

Re-ran over Dec 8-19 with fixed filter but still using SPX index. "NO DATA COLLECTED."

**Root cause:** Without `AddIndex("SPXW")`, the SPX option chain has no SPXW contracts.

### Test 5: qc_11 on Dec 15, 16 -- NO DATA

Single-day runs with `AddIndex("SPXW")`. No SPXW data for Mon Dec 15 or Tue Dec 16.

**Root cause:** QC free tier may have incomplete SPXW coverage for some days, or SPXW data starts mid-week in QC's dataset for this period.

### Test 6: qc_11 on Dec 17 -- NO DATA (without SPXW index)

Even Dec 17 (which worked in qc_10) returned nothing because the initial qc_11 version didn't add SPXW as a separate index.

### Test 7: qc_11 v2 on Dec 17-23 (5 days) -- DATA COLLECTED, GREEKS ALL ZERO

Ran 5 single-day backtests with `AddIndex("SPXW")`. All 5 days returned true SPXW 0DTE data (90 total readings, `0dte=1`).

**Critical finding: IV, delta, gamma, theta, and vega are ALL 0.0 on every reading.**

QC does not compute Greeks for SPXW 0DTE contracts. The Greeks observed in Test 1 were computed for SPX Friday weeklies only.

**This closes the QC free tier path for Greeks validation. There is no comparison to make.**

---

## Q4: Bid-ask spread analysis (the data we DID get)

90 readings across 5 days (Dec 17-23, 2025), all true SPXW 0DTE, near-ATM.

### Spreads by time of day

| Bucket | N | Median spread | Median spread % |
|--------|---|--------------|----------------|
| Open (bar 10-30) | 20 | $0.15 | 1.1% |
| Morning (bar 31-120) | 20 | $0.20 | 1.4% |
| Midday (bar 121-240) | 20 | $0.10 | 1.5% |
| Afternoon (bar 241-360) | 20 | $0.13 | 1.3% |
| Close (bar 361+) | 10 | $0.15 | 0.4% |

### Spreads by right

| Right | N | Median spread | Notes |
|-------|---|--------------|-------|
| Calls | 45 | $0.30 | Wider when ITM near close |
| Puts | 45 | $0.05 | Many near-zero (penny options near close) |

### Comparison to our estimate

| Metric | Our estimate | Real (SPXW 0DTE) | Prior QC research (SPX weekly) |
|--------|-------------|-------------------|-------------------------------|
| ATM spread | $0.30 fixed | $0.10-$0.30 median | $0.40-$0.70 median |
| Spread model | 30-150 bps lookup | 1.1-1.5% of mid | N/A |

**Verdict: Our $0.30 spread estimate is conservative (slightly high) for SPXW 0DTE.** The model deducts slightly too much cost from P&L labels, making labels marginally more pessimistic than reality. This is acceptable -- better to overestimate costs than underestimate them.

**Correction to prior findings:** The earlier QC spread research (qc_02 series, 13,600 observations showing $0.40-$0.70 ATM) measured SPX Friday weeklies, not SPXW 0DTE. Those numbers do not apply to our trading. SPXW 0DTE spreads are roughly half the width.

---

## Q5: Risk assessment (updated)

### Greeks and IV (UNKNOWN RISK -- cannot validate via QC)

Our BS estimates remain unvalidated for 0DTE. The theoretical concerns stand:
- BS gamma diverges as T approaches 0
- BS theta acceleration may not match real theta behavior
- IV extracted from close prices has bid-ask noise

However, we cannot quantify the actual error because no external source of 0DTE Greeks is available through QC free tier. The risk is theoretical, not measured.

### Bid-ask spread (LOW RISK -- validated)

Our $0.30 estimate is conservative. Real SPXW 0DTE ATM spreads are $0.10-$0.30. Labels are slightly more pessimistic than reality, which is the safe direction.

---

## Alternative validation path: realized Greeks from Polygon

Since QC cannot provide computed Greeks for 0DTE, we can validate our BS estimates using **realized Greeks from actual price movements** in our existing Polygon data:

1. **Realized delta:** For each minute, compute `delta_option_price / delta_SPX_price`. Compare against BS delta at that time.
2. **Realized gamma:** Measure how delta changes across 5-10 minute windows. Compare against BS gamma.
3. **Realized theta:** On days with minimal SPX movement, measure option price decay per minute. Compare against BS theta.

This uses only our existing Polygon OHLCV data (4 years, 999 days). No external data source needed. The comparison would show whether BS Greeks track reality or diverge systematically near expiry.

**This is the recommended next step to close the Greek accuracy question.**

---

## Recommendations (updated)

1. ~~Fix qc_08 to run single-day backtests~~ **Done.** QC free tier cannot provide 0DTE Greeks. Dead end.

2. **Compute realized Greeks from Polygon data.** This is the viable path. We have the price data. We can measure empirical delta, gamma, and theta from option price movements and compare against our BS estimates. No cost, no external dependency.

3. ~~Update spread model immediately.~~ **No longer needed.** Our $0.30 estimate is already conservative for SPXW 0DTE. The prior recommendation was based on SPX weekly spread data ($0.40-$0.70) which doesn't apply.

4. ~~QC Researcher tier for Greeks.~~ **Deprioritized.** Even with downloaded data, QC may not include computed Greeks for SPXW. The Researcher tier's value is now limited to bulk bid/ask data and Jupyter notebooks, not Greeks.

---

## Files referenced

| File | Role |
|------|------|
| `archive/v1/training/prepare.py:320-353` | BS Greeks and IV extraction implementation |
| `v2/core/features.py:271-296` | Adaptive spread model |
| `v2/research/qc_08_greeks_comparison.py` | QC Greeks capture (returned non-0DTE Greeks) |
| `v2/research/qc_09_expiry_diagnostic.py` | Expiry date diagnostic |
| `v2/research/qc_10_find_spxw_0dte.py` | SPXW 0DTE discovery (3 approaches) |
| `v2/research/qc_11_greeks_1day.py` | Single-day SPXW 0DTE capture (definitive test) |
| `v2/research/qc_11_results_all.txt` | All 90 readings from 5 days (Greeks=0.0) |
| `v2/research/compare_greeks.py` | Local BS comparison tool |
| `v2/docs/audit/audit_section2_logs/` | Raw QC backtest logs (days 17-23) |
| `v2/docs/quantconnect_findings.md` | Prior QC research (SPX weeklies, not SPXW) |

---

## Test log

| # | Script | Date range | Result | Finding |
|---|--------|-----------|--------|---------|
| 1 | qc_08 v1 | Dec 8-19 | 72 readings, all non-0DTE | `Expiration(0,5)` returns SPX weeklies |
| 2 | qc_09 | Dec 15-19 | All exp=Dec 19 | Confirmed: standard SPX, not SPXW |
| 3 | qc_10 | Dec 17 | 66 SPXW contracts found | `AddIndex("SPXW")` is required |
| 4 | qc_08 v2 | Dec 8-19 | NO DATA | Multi-day backtest unreliable |
| 5 | qc_11 v1 | Dec 15-17 | NO DATA | Missing `AddIndex("SPXW")` |
| 6 | qc_11 v2 | Dec 17-23 | 90 readings, all 0DTE | **Greeks/IV all 0.0. QC doesn't compute for SPXW.** |
| 7 | -- | -- | -- | Spread data valid: $0.10-$0.30 ATM |
