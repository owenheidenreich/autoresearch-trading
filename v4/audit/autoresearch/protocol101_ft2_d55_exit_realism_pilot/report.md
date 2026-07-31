# FT2-D55 Exit-Realism Pilot

## Owner memo — how much do the minute photos lie?

**Verdict: `NOT_TRUSTWORTHY_WITHOUT_HIGHRES_VALIDATION_OR_ADJUSTMENT`.**

Across 3,533,247 tradable contract-minutes, 0.07% showed no one-second BBO change. The normalized minute corpus reports `quote_age_ms == 0` on 100.00% and a null quote gap on 100.00%, so those minute columns could not reveal this staleness.

A hidden intra-minute bid dip occurred in 74.08% of contract-minutes with executable one-second coverage. The p95 minute-bid-to-worst-1s-bid gap was 9.1000 option points ($910.00 per contract).

For fill assumptions, the p95 absolute error versus the mean one-second interval price was 3.1367 points on entry asks and 3.1300 points on exit bids.

At the 90%-of-prior-bid floor stress boundary, 1.90% of eligible contract-minutes crossed and recovered before the completed-minute check; crossed paths had a p95 trigger gap of 0.4900 points ($49.00 per contract).

### Distortion by decision-time premium band

The pooled absolute-point result is dominated by expensive contracts, so the decision must also use the governed census-v4 premium bands:

| Premium band | Contract-minutes | Hidden dip | P95 adverse gap | P95 entry error | P95 exit error |
|---|---:|---:|---:|---:|---:|
| cheap_le_1 | 526,527 | 16.14% | 0.0500 | 0.0456 | 0.0450 |
| large_8_20 | 95,281 | 89.29% | 2.2000 | 1.3033 | 1.2900 |
| medium_3_8 | 81,334 | 83.45% | 1.0500 | 0.6667 | 0.6600 |
| small_1_3 | 91,666 | 77.29% | 0.4000 | 0.2683 | 0.2617 |
| very_large_20p | 2,738,439 | 84.31% | 9.6000 | 3.6593 | 3.6717 |

### What should we do?

- **Entry substrate:** Keep minute data as the entry training substrate and CBBO-1s as a validation/feature-discovery instrument. The exploratory statistics are not a trained or causally validated entry signal; a substrate change would require a separately owner-authorized experiment.
- **Exit substrate:** Use CBBO-1s to construct/calibrate exit and floor labels for the available 2025+ window, while retaining minute history for regime breadth only after applying a measured distortion model. An unadjusted minute-only exit campaign is not supported.
- **Tradeoff:** CBBO-1s is denser but begins 2025-02-20; minute data can reach 2022 and spans more regimes. Prefer a hybrid evidence design over discarding either axis.
- **Cost:** 30-session preflight `$24.115595`; conservative failed-stream billable upper bound `$2.997483`; worst-case `$27.113078` under the `$30.00` cap. Actual vendor invoice: `UNKNOWN`. Pilot ~$24.12; all available CBBO-1s history roughly $280; deferred 2022-2024 minute backfill roughly $41-$347.

## Decision checks

| Check | Pass |
|---|---:|
| one_second_coverage | yes |
| forward_fill | yes |
| hidden_adverse_excursion_rate | no |
| hidden_adverse_excursion_magnitude | no |
| entry_fill_distortion | no |
| exit_fill_distortion | no |
| hidden_floor_cross | yes |
| floor_trigger_gap | no |

## Measurement details

- Sessions: `30`
- Tradable contract-minutes: `3533247`
- Executable one-second coverage: `100.00%`
- Breach-and-recover rate: `74.08%`
- Entry minute-fill optimistic fraction: `47.83%`
- Exit minute-fill optimistic fraction: `46.22%`

Detailed strata are in `per_premium_band.csv`, `per_moneyness_band.csv`, `per_premium_moneyness_band.csv`, and `floor_slippage_by_band.csv`.

## Protective-floor limitation

The signed design defines causal completed-minute checking and an upward-only floor, but it intentionally does not yet define the forecast-to-floor equation. This pilot therefore reports a one-step response surface at floors equal to 70%, 80%, 90%, and 95% of the prior completed-minute bid. These are stress boundaries, not a fitted floor policy and not a contract change.

## Entry-signal exploratory

The one-second momentum, realized-volatility, range, quote-change, and spread-change statistics are descriptive only. No model was fit and no entry-edge claim is made. The strongest absolute Spearman correlation observed was `0.6558` for `bbo_change_count` versus `next_minute_bid_over_current_ask_return` within its reported stratum; this is hypothesis-generating, not evidence to change the entry substrate.

## Scope and route

No model was trained. No broker, protected data, runtime, launchd, promotion, paper-default, signed contract, graph, or census state was touched.

**Next: `STOP_FOR_OWNER_DECISION`.**
