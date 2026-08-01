# Phase 0.5 Vendor Verification

**Goal**: reach a non-provisional paid-data budget without repeating v3's
failure mode. The next spend is a three-month SPXW 0DTE pilot, not a full
historical backfill and not an OptionsDepth subscription.

## Current Decision

OptionsDepth is deferred. It belongs to a later dealer-flow block only if the
SPXW quote/microstructure prototype first clears its evidence gate.

The Phase 0.5 data spend is focused on the smallest dataset that can answer one
question:

> Does quote-realistic SPXW 0DTE data with executable ask-entry / bid-exit
> labels produce a measurable improvement over v3?

The build instructions and exact download list live in
[SPXW_0DTE_DATA_DOWNLOADS.md](SPXW_0DTE_DATA_DOWNLOADS.md).

## Pilot Data Window

Use `2026-01-02` through `2026-03-31`.

Required minimum:

- Databento `OPRA.PILLAR` `definition`, parent symbol `SPXW.OPT`
- Databento `OPRA.PILLAR` `cbbo-1m`, filtered SPXW 0DTE raw symbols
- SPX context derived from SPXW put-call parity, or licensed SPX one-minute
  index bars when available
- volatility-regime context derived from same-chain 0DTE ATM IV, or licensed
  VIX one-minute index bars when available

Strongly recommended if the estimator cost is acceptable:

- Databento `OPRA.PILLAR` `ohlcv-1m` for actual option volume
- Databento `OPRA.PILLAR` `statistics` filtered to `stat_type == 9`
- 10 selected high-resolution audit days using `tcbbo` or `cbbo-1s` around
  ATM +/- `$50`

## Verification Gates

- every tradable row is `SPXW`
- every strike is `$5` aligned
- no AM-settled `SPX` contract enters the dataset
- `cbbo-1m` last-sale size is not used as volume
- open interest only comes from `statistics.stat_type == 9`
- SPX/VIX feature timestamps are `<= decision_time`
- labels change materially when using bid/ask instead of mid
- high-resolution audit confirms `cbbo-1m` is acceptable for prototype
  stop/target labeling

## Economic Rule

Do not buy the full 2022-present Databento backfill until the three-month pilot
shows evidence that it is materially better than v3 under bid/ask execution.
If `ohlcv-1m` or the high-resolution audit slice is too expensive, keep the
minimum viable dataset as definitions + `cbbo-1m` + SPX/VIX bars and remove
volume-sensitive features from the prototype.

The two-day smoke test on `2026-01-02` and `2026-01-05` clears the three-month
pilot download gate. It does not clear live deployment, full backfill, or model
promotion.

## OptionsDepth Status

OptionsDepth remains deferred to a later dealer-flow phase. Do not buy a
one-cycle subscription during Phase 0.5. Reconsider only after the Block-1
microstructure model clears its promotion packet with robust out-of-sample
performance.
