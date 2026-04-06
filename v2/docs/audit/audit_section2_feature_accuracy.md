# Audit: Section 2 -- Feature Accuracy (Estimated vs Real Data)

**Date:** 2026-04-06
**Scope:** Greeks, IV, and bid-ask spread features used in training
**Key files:** `archive/v1/training/prepare.py:320-353`, `v2/core/features.py:271-296`, `v2/pipeline/build_v2_dataset.py`

---

## Summary

6 of 55 training features are estimated from option prices rather than sourced from real market data. Real historical values exist for all 6 but are not currently used. We attempted to validate our estimates against QuantConnect's computed values but hit a data access limitation: QC free tier requires specific filter tuning to return SPXW 0DTE contracts, and multi-day extraction is unreliable.

**Verdict: Unvalidated. Cannot confirm or deny feature accuracy for 0DTE.**

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

**Known concern:** Prior QC research (13,600 observations) showed real ATM spreads are $0.40-$0.70, not the $0.30 we use. Our model is off by ~1.5-2x for ATM options. OTM spreads diverge more ($2-3 real vs our estimate). This directly corrupts labels since P&L simulation deducts spread cost.

---

## Q2: What real data sources exist for these values?

| Data | Source | Available? | Cost |
|------|--------|-----------|------|
| Computed Greeks (delta, gamma, theta, vega) | QuantConnect | Yes (free tier confirmed) | Free for backtests, $60/mo + $113 for downloads |
| Real IV per contract | QuantConnect | Yes | Same as above |
| Real bid/ask quotes | QuantConnect | Yes (confirmed in spread research) | Same as above |
| Real-time Greeks | IBKR | Live only, no deep history | Existing subscription |
| Historical Greeks | Polygon | **No** -- Polygon has no Greeks or IV | N/A |

---

## Q3: Validation attempt -- what happened?

### Test 1: qc_08 with Expiration(0,5) -- INVALID

Ran Dec 8-19, 2025. Got 72 readings. All showed `0dte=0`.

**Root cause:** `Expiration(0,5)` returns SPX Friday weeklies (symbol: `SPX 251219C...`), not SPXW dailies. Every contract had 1-4 DTE. The comparison was invalid because we computed BS Greeks assuming 0DTE time-to-expiry against options with days remaining.

**Misleading results discarded:** Gamma errors of 66-903% and theta errors of 61-998% were artifacts of the DTE mismatch, not real BS inaccuracy.

### Test 2: qc_09 expiry diagnostic -- CONFIRMED THE BUG

Ran Dec 15-19, 2025. Dumped all contract details. Every contract had `exp=2025-12-19` (Friday). Symbol format: `SPX 251219C06815000`. These are standard Friday SPX options, not SPXW.

### Test 3: qc_10 multi-approach -- FOUND THE FIX

Ran Dec 17, 2025 (single day). Three approaches tested:

| Approach | Result |
|----------|--------|
| A: `Expiration(0,0)` | **66 SPXW contracts found.** Symbol: `SPXW 251217C06790000`. True 0DTE with `dte=0`. |
| B: `OptionChainProvider` | 9,220 contracts but only monthly expiries. No SPXW. Dead end. |
| C: `AddIndex("SPXW")` | **Works.** Same 66 contracts as Approach A. |

**Conclusion:** `Expiration(0,0)` returns true SPXW 0DTE. The old `Expiration(0,5)` preferenced Friday SPX weeklies over SPXW dailies.

### Test 4: qc_08 v2 with Expiration(0,0) -- NO DATA

Re-ran with fixed filter over Dec 8-19. Got "NO DATA COLLECTED".

**Root cause (probable):** The Expiration(0,0) filter works on a single day (qc_10) but across a 10-day backtest, the option chain data may not populate on every bar, or the chain subscription has intermittent availability. The reporting mechanism (only fires on Dec 19 bar 385+) means if no samples were collected across 10 days, we get nothing.

**Status: 0DTE Greeks validation remains unachieved.** We confirmed the data exists in QC but cannot reliably extract it across multiple days via the free tier's single-Error-per-backtest constraint.

---

## Q4: What is the risk of using unvalidated estimates?

### Gamma and theta (HIGH RISK for 0DTE)

Black-Scholes Greeks have known issues near expiry:
- As T approaches 0, gamma approaches infinity for ATM options
- Theta acceleration is extreme: theta at 10 minutes remaining is ~6x theta at 6 hours
- Small errors in the time-to-expiry calculation (e.g., using trading minutes vs calendar time) produce large Greek errors
- BS assumes continuous hedging and log-normal returns -- both break down in the last hour of 0DTE

These features (`atm_gamma`, `atm_theta_per_bar`) showed signal edge in our feature scan (PF=1.176 and implicit in vol-regime features). If the estimates are systematically biased, the model is learning from a distorted signal.

### IV (MEDIUM RISK)

Brent's method from close prices introduces two error sources:
1. Close != mid price (last trade vs theoretical mid)
2. 0DTE IV surface is steep -- small price errors map to large IV swings

All three IV-derived features (`atm_iv`, `iv_skew`, `iv_percentile`) inherit this noise.

### Bid-ask spread (HIGH RISK for labels)

Our $0.30 fixed spread underestimates real ATM spreads by ~1.5-2x (QC research showed $0.40-$0.70 median). For OTM options, the gap is worse ($2-3 real).

This directly affects training labels: the triple-barrier P&L simulation deducts spread cost. Underestimating spread makes losing trades look like winners, inflating the gate=True label rate.

---

## Recommendations

1. **Fix qc_08 to run single-day backtests in a loop** (workaround for multi-day reliability). Run 10 separate single-day backtests manually and concatenate results. Tedious but viable on free tier.

2. **Alternatively, run comparison against our Polygon data directly.** We have real option OHLCV from Polygon. We can compute BS Greeks from those prices AND compare against the actual option price movements (realized gamma/theta). This doesn't need QC at all.

3. **Update spread model immediately.** QC spread research (13,600 obs) already provides enough data to replace the $0.30 estimate with regime-specific values ($0.50 ATM low-VIX, $0.70 ATM med-VIX, $0.90+ OTM). This is actionable now without any new data.

4. **If QC Researcher tier is approved ($60/mo),** download 1 year of SPXW 0DTE data with real Greeks and bid/ask. This would close all gaps definitively.

---

## Files referenced

| File | Role |
|------|------|
| `archive/v1/training/prepare.py:320-353` | BS Greeks and IV extraction implementation |
| `v2/core/features.py:271-296` | Adaptive spread model |
| `v2/research/qc_08_greeks_comparison.py` | QC Greeks capture script (fixed for 0DTE) |
| `v2/research/qc_09_expiry_diagnostic.py` | Expiry date diagnostic |
| `v2/research/qc_10_find_spxw_0dte.py` | SPXW 0DTE discovery (3 approaches) |
| `v2/research/compare_greeks.py` | Local BS vs QC comparison tool |
| `v2/research/qc_08_results_raw.txt` | Raw results from Test 1 (invalid, non-0DTE) |
| `v2/docs/quantconnect_findings.md` | Prior QC spread research (13,600 obs) |
