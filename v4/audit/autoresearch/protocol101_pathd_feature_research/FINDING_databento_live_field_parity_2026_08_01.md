# Finding — Databento historical→live OPRA field parity (what's actually live)

**STATUS: research finding.** Establishes which fields in our historical training corpus are
actually available on Databento's LIVE OPRA feed with matching semantics — so we don't train
on features we can't decide on live (the parity discipline applied *inside* Databento). Prompted
by the owner: "same vendor" ≠ "same fields; the live feed isn't automatically the training download."

## Verdict by field

| Field(s) | Live real-time? | Basis | Use verdict |
|---|---|---|---|
| bid, ask, `bid_size`, `ask_size` (→ size_imbalance, spread, mid) | YES — consolidated CBBO/CMBP schema | our own cbbo-1s files contain `bid_sz_00`/`ask_sz_00`; live uses the same schema | **LIVE-USABLE** |
| self-computed IV/delta/gamma/etc. | YES (we compute from price+spot+strike+time) | not a vendor field | LIVE-USABLE (reproduce identically) |
| SPX underlying context | YES via ThetaData live (separate feed) | ThetaData real-time | usable (own ThetaData parity) |
| `open_interest`, `stat_open_interest` | **NO — statistics schema is daily/EOD** | Databento docs: OI is EOD summary | **BARRED as intraday feature** (prior-day static only) |
| `volume` (stat), `option_ohlcv_volume` | partial — stat-volume EOD; intraday only via trades/ohlcv | Databento docs | avoid as live feature (also ~no signal in probe) |
| last_trade/last_trade_size | YES (trades/last-sale live) | consolidated last-sale | usable if needed |

## Two decisive facts
1. **Live OPRA = consolidated schemas (`cbbo-1s`/`cmbp-1`/`cbbo-1m`); MBP-1/TBBO were retired for
   OPRA in May 2025.** Our training is `cbbo-1s`/`cbbo-1m` → same family as live. **Do not build
   on mbp-1/tbbo.** Live-adapter matching requirement: subscribe to `cbbo-1s` live, or consolidate
   `cmbp-1`→1s with the SAME rule as the historical download (else a subtle train/live gap).
2. **Open interest is daily/EOD, not real-time.**

## Impact on the entry feature research (FINDING_entry_microstructure_signal)
- **`size_imbalance` (the standout, +0.129 incremental) is LIVE-USABLE** — depends only on live
  `bid_size`/`ask_size`. The "widen the entry" direction is live-deployable.
- **`open_interest` (weak secondary, +0.089 incremental) is DISQUALIFIED as an intraday feature**
  (EOD-only). At most a prior-day static. Drop it from the widen-entry candidate list.

## Governance hook (enforce this)
The entry+exit model plan §3 already requires every feature to have a "historical/live twin" and
fail closed without one. **This inventory is the ground truth for that gate.** Enforce:
- Only fields in the LIVE-USABLE rows may be model inputs.
- EOD-only fields (OI, stat-volume) are barred from intraday decision features.
- The live adapter (transition-architecture step 8) must reproduce the exact `cbbo-1s`
  schema/sampling used in training; feature-lineage manifest records the live twin per feature.

## Honesty note
The BBO/CBBO doc page renders thin externally; the load-bearing size-field claim is confirmed
empirically from our own downloaded `cbbo-1s` parquet (contains `bid_sz_00`/`ask_sz_00`), which is
stronger than the doc. The exact live sampling/consolidation match remains a live-adapter
requirement to verify when step 8 is built + on the first live-shadow day.
