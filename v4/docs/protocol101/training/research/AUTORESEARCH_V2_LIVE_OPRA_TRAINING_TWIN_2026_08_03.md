# Autoresearch v2 Databento Live OPRA training twin

Date: 2026-08-03

Status: `FOUNDATION_BUILT_NOT_FIT_READY`

## Answer

The existing simulation and training regimen is **not identical** to the data
available to a Databento Live OPRA decision process.

The OPRA portion is structurally compatible. A bounded Monday live sample and
the owned historical corpus both decode native `cbbo-1m` as the same 15 fields,
dtypes, rtype 193, and exact minute-end `ts_recv`. The failure is the combined
decision clock: training paired option interval end `t` with the ThetaData SPX
bar stamped `t`, whose close represents the following minute and is unavailable
until `t+60s`. All 445,063 fitted rows used this unavailable context.

The former immutable model is therefore an `INVALID_EXPERIMENT`, not a live
candidate. Waiting 60 seconds and reusing the old option quote is rejected
because it preserves feature bytes by changing the trading game.

## Monday live observations

The owner-authorized capture subscribed to 492 explicitly enumerated SPXW 0DTE
raw symbols for 120 seconds, with both `cbbo-1s` and `cbbo-1m` and no model,
broker, or order path.

- 20,453 rtype-192 `cbbo-1s` rows and 914 rtype-193 `cbbo-1m` rows arrived.
- Live `cbbo-1m` gateway `ts_out-ts_recv` was 38.7 ms median and 54.5 ms p99.
- Local `cbbo-1m` receipt was 226.9 ms median and 319.5 ms p99 after `ts_recv`.
- The observed near-ATM ±25-point ladder at 5-point spacing was complete, 22/22.
- A mid-minute subscription start made the first interval non-authoritative;
  every live adapter must warm for a full minute and reset that warmup on
  reconnect.
- All 492 prior-session Databento instrument IDs changed in the live mapping.
  Stable identity is the raw OSI symbol plus the current-session mapping, never
  a prior-day instrument ID.
- A parent-symbol live definition replay found 510 current-expiry contracts,
  including 18 Monday additions that were absent from Friday's definition file.
  The live adapter must ingest the current-session definition stream before it
  constructs a candidate universe.

This capture did not prove same-session value identity with the historical API;
that requires a post-session download of 2026-08-03 and exact row comparison.

An expanded live feature-surface capture then observed CMBP-1, TCBBO, trades,
and sparse OHLCV-1m behavior. It established additional missingness, throughput,
and trade-side constraints. The full response and entry-family inventory are in
`AUTORESEARCH_V2_LIVE_FIRST_ENTRY_FEATURE_AUDIT_2026_08_03.md`.

## Shared clock law

For each exact UTC minute boundary `t`:

| Stage | Frozen rule |
|---|---|
| Feature interval | The closed minute `[t-60s,t)`. |
| Option features | Native OPRA `cbbo-1m` with `ts_recv=t`. Use the same schema historically and live; do not rebuild it from `cbbo-1s`. |
| Official SPX | ThetaData bar with `event_time=t-60s`; its represented interval ends at `t`. The bar stamped `t` is forbidden. |
| Emission | Fixed `t+L`, only if both exact source rows were actually received by the cutoff; otherwise WAIT/fail closed. |
| Entry reference | Latest fresh exact-contract OPRA `cbbo-1s` at the frozen order-arrival cutoff, with a two-second maximum age. |
| Label origin | Executable entry arrival/fill, not feature boundary `t`. A 25-minute label ends 25 minutes after arrival. |
| Identity | Raw OSI symbol plus current-session live mapping/definition. |
| Warmup | At least one complete minute before the first eligible boundary; repeat after reconnect. |

The normal difference between a market-data boundary, model emission, and order
arrival is explicit and monotone. It is not the rejected crossed-time workaround:
no source interval after `t` can enter the feature snapshot for `t`.

## Implementation

`v4/research/autoresearch_v2/live_opra_training_twin.py` implements the shared,
model-free selectors and emits a hashed source receipt binding:

- the exact closed-minute SPX row;
- the exact native CBBO-1m feature row;
- the fresh CBBO-1s executable entry row;
- feature, emission, arrival, and label-deadline clocks; and
- stable raw-symbol contract identity.

Rows arriving after the frozen cutoff, future SPX bars, missing exact minute
rows, stale entry quotes, and locked/crossed BBOs fail closed. Unit tests cover
the aligned clock, future-bar rejection, late-receipt rejection, entry-label
origin, and daily instrument-ID churn.

## Why native CBBO-1m

Direct `cbbo-1m` is the recommended feature twin because Databento publishes it
through both historical and live APIs with the same record schema. Rebuilding a
minute from `cbbo-1s` would add another aggregation and sparse-interval law and
would still require a proof that the result equals Databento's native minute.
`cbbo-1s` remains the correct execution/label substrate because it is fresher
at the action cutoff.

## Fit blockers

No distinct generation may fit until all are complete:

1. Measure live ThetaData completed-minute receipt timing and freeze `L` from
   both feeds' latency distributions. OPRA-only evidence cannot set it.
2. Bind the captured current-session definition replay into the adapter and
   prove add/modify/remove messages update the universe before decisions.
3. Post-session, download the same 2026-08-03 OPRA schemas and prove live DBN
   rows equal historical API rows for matched raw symbols and timestamps.
4. Run multi-session cold-start, reconnect, duplicate, missing-row, early-close,
   clock-skew, and numerical feature-parity tests.
5. Materialize a distinct causal development dataset and rerun causality and
   strict-serial guards. The spent holdout remains sealed and cannot confirm it.
6. Obtain a new independent confirmation epoch before any edge claim.

Before any fit, the exact feature contract must also pass
`entry_live_feature_catalog.py`. Prose `live_twin` descriptions are no longer
accepted by the autoresearch compiler.

The owner subsequently supplied Claude's assessment of the clock failure.
Claude agreed that the prior edge was invalidated, the completed-minute context
must shift causally, the 60-second crossed-time workaround is invalid, and the
causal replacement needs a fresh confirmation epoch. The live-capture scope and
expanded feature findings remain Codex evidence pending independent review.

No fit, tuning, model load, holdout access, broker, paper runtime, order path,
registry, promotion, or default change occurred.

STOP_FOR_CLAUDE_VERIFICATION
