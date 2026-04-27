# Pipeline Integrity Report — 🟢 GREEN

- **Started**: 2026-04-27T19:55:04.266794+00:00
- **Finished**: 2026-04-27T19:55:04.307646+00:00
- **Build ID**: `af4b8bcca4d1ff2cc21839fd85cb409cde0743bf-dirty-3a4a26d7`
- **Ingest run ID**: `d3ff8882-ec1f-41e7-a1df-866bc80951e6`

## Input

- Path: `v4/tests/fixtures/optionsdx_spx_sample.csv`
- SHA-256: `3bbde0535e7671d5f59a42a83c7923cf30ffe499ced2b082657e2ea366f77fd5`
- Normalized rows emitted: 10
- Fingerprint: `5fd6949ea6925d1551f9abbf970b9753cce1f0348261bb2a1844cf4d78fbea33`

## Determinism

✅ Re-running ingest with the same run_id produced byte-identical output.

## Sanity checks

- ✅ **bid_ask_ordering** — all rows with both bid+ask have bid <= ask
- ✅ **bid_nonnegative** — passed
- ✅ **ask_positive** — passed

## Integrity checks

- ✅ **no_duplicate_keys** — all 10 rows are unique on ['event_time', 'contract_id']
- ✅ **timestamp_monotonic_per_contract** — event_time is non-decreasing within each contract (4 contracts checked)
- ✅ **required_non_null** — all 9 required columns are non-null

## Greeks reconciliation (vs OptionsDX vendor Greeks)

- ✅ **greeks_reconciliation_optionsdx** — 10 rows reconciled within tolerance: delta max |err|=0.1035 <= 0.2, gamma max |err|=0.0118 <= 0.1, vega_per_1pct max |err|=0.8264 <= 5.0, theta_per_day max |err|=42.9092 <= 100.0, rho_per_1pct max |err|=0.0052 <= 1.0

## Leak-detector self-check

- ✅ On synthetic signal data:
  - baseline AUC = 0.809 (expect > 0.7)
  - shuffled-label AUC = 0.557 (expect < 0.6)
  - planted-leak AUC = 1.000 (expect > 0.9)

## Phase 0 exit criteria (per protocol Section 9.5)

🟢 **Pipeline-integrity report is GREEN.** All Phase-0 substrate checks pass. The substrate is ready for Phase 0.5 vendor verification — but Phase 0.5 must still gate every paid data purchase on written vendor terms.

## What this report does NOT validate

- Edge or PF on real data (Phase 2A onward; Phase 0 deliberately makes no PF claim).
- Vendor terms for OptionsDepth, Databento, IBKR (Phase 0.5).
- Fill-model calibration vs IBKR (Phase 4 / 4.5).
- Modern-regime data quality (Phase 1 once Databento 2024+ is pulled).

