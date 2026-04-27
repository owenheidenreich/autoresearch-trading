# Databento pricing-estimator query plan

**Goal**: get a concrete dollar number for the Phase-1 historical pull by running Databento's pricing estimator with the exact (symbol set, date range, schemas) the protocol requires. Replace the provisional `$200–500` placeholder.

**URL**: https://databento.com/pricing → "Estimate Cost" or batch download tooling.

---

## Query 1 — SPX 0DTE foundation

The primary Phase-1 pull. Drives every Phase-2A backtest.

| Field | Value |
|---|---|
| Dataset | `OPRA.PILLAR` |
| Symbols | SPXW (parent symbol; daily expiration) — use Databento's "parent" / "raw_symbol" mode to get all expiries; filter for 0DTE downstream |
| Schemas | `trades`, `cbbo-1m` (consolidated NBBO 1-minute), `definitions`, `statistics` |
| Date range | 2018-01-01 → present (verified with Databento that history extends from 2018-onwards in their May-2025 expansion) |
| `stype_in` | `parent` (lets you request all SPXW contracts in one batch) |
| Format | DBN or Parquet |

**Why these schemas**:
- `trades`: per-trade prints + sizes (for Lee-Ready DIY proxy and microstructure features)
- `cbbo-1m`: consolidated NBBO at 1-minute grain — matches v4 decision tempo. Heavier `tbbo` (top-of-book) is overkill at this stage.
- `definitions`: contract symbology + expiry/strike/right metadata
- `statistics`: includes daily open interest

**Why NOT**:
- `mbo` (market-by-order): tick-level book, hugely expensive, not useful at 1-min decision tempo
- `bbo-1s` second-level NBBO: overkill; revisit only if 1-min features show ambiguity
- `imbalance` / `nbbo`: not needed for v4 Phase 1

**Greeks**: NOT included; Databento doesn't sell them. v4/greeks/black_scholes.py computes them from the raw OPRA prices.

---

## Query 2 — VIX reconstruction inputs (only if reconstructing rather than buying VIX)

Skip this if you decide to buy VIX intraday history directly from CBOE/Databento. See [PHASE_0_5_VENDOR_VERIFICATION.md](../PHASE_0_5_VENDOR_VERIFICATION.md) "VIX" section.

If reconstructing:

| Field | Value |
|---|---|
| Dataset | `OPRA.PILLAR` |
| Symbols | SPX + SPXW with DTE in [23, 37] |
| Schemas | `cbbo-1m`, `definitions`, `statistics` |
| Date range | match SPX 0DTE date range |

This is the SPX option strip the VIX methodology requires (per Cboe whitepaper). The strip is interpolated to a constant 30-day horizon.

**Realistic path**: skip Query 2; buy VIX history directly. Cheaper and avoids weeks of methodology calibration.

---

## Query 3 — VIX index direct (the recommended default)

| Field | Value |
|---|---|
| Dataset | (whichever Databento dataset hosts VIX intraday — confirm with their support; possibly `XCBO.VIX` or available via CBOE direct) |
| Symbols | VIX, VVIX |
| Schemas | OHLCV-1m or trades/quotes if available |
| Date range | 2018-01-01 → present |

If Databento doesn't carry VIX intraday with the right granularity, fall back to CBOE DataShop directly.

---

## Estimator output to record

For each query, capture:

```
query_id                     # 1, 2, 3
estimator_run_at             # ISO8601
estimator_quoted_total       # USD
estimator_quoted_uncompressed_gb
estimator_quoted_billed_gb
date_range_used
schemas_used
symbols_used
notes                         # any caveats from the estimator UI
```

Save into `v4/docs/vendor_outreach/databento_estimate_YYYY-MM-DD.md`.

---

## After the estimate

If totals are close to provisional ($200–500):
- proceed with the Phase-1 pull as planned

If totals are materially higher (>$1000):
- consider trimming date range to 2022-04 (SPXW Tuesday/Thursday launch) → present. Pre-2022 data is for pretraining/stress testing only per protocol Section 5 Phase 2A regime cohorts; cheaper to skip the pre-daily era.
- re-quote and decide.

If totals are under $100:
- pull the full range. Cheap.

In all cases: write the actual quoted number into the protocol's Section 3.3 budget table, replacing the provisional estimate. Pre-purchase, the budget should be a number the vendor produced, not a number we guessed.
