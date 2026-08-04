# Path-D Phase 0b unblock-certification report — 2026-08-04

> **POST-CERTIFICATION PARENT CORRECTION:** This report preserves the original
> transform-receipt result. It is not the current admission authority. The
> corrected parent graph requires `entry.opra_implied_spot.v1` to depend on
> `entry.opra_cbbo1m_native.v1`; consequently the current signed ledger is
> `04b5e9584b295c970847870d32111efce1268f0b7302f3ce830970f2411923e3`
> with `8/73 ADMITTED`. The Track-B receipts remain valid, but their families
> stay barred until Track A supplies the missing OPRA arrival clock.

Status: **STOP_FOR_CLAUDE_VERIFICATION**

Ledger SHA-256: `04b5e9584b295c970847870d32111efce1268f0b7302f3ce830970f2411923e3`

In-scope result: `8/73 ADMITTED`, `65/73 BARRED`.

No model was loaded, trained, or fitted. `holdout_open_count` remained `0`. No paid data, live capture, broker, order, paper-submit, promotion, default, runtime flag, or launchd path was accessed.

## Track B — OPRA parity estimator

| Family | Features | Status | Measured availability clock | Identity tolerance | Receipts |
|---|---:|---|---:|---:|---|
| `entry.opra_implied_spot.v1` | 8 | **ADMITTED** | 7.680593 ms, measured locally for this adapter | 0.0 | shared parity-estimator implementation hash, historical/live candidate-pair identity, comparison to completed official SPX for measurement only |
| `entry.opra_implied_volatility.v1` | 6 | **ADMITTED** | 8.572375 ms, measured locally for this adapter | 1e-10 | entry.opra_implied_spot.v1 receipts, shared solver/constants hash, mutation and numerical-stability tests |
| `entry.self_computed_greeks.v1` | 7 | **ADMITTED** | 0.084262 ms, measured locally for this adapter | 1e-10 | causal spot parent receipt, shared solver/constants hash, historical/live golden-vector identity |

Official SPX was validation-only and is not accepted by the feature adapter API. Mutating it to `1e12` left every checked OPRA implied-spot result unchanged.
Across `122` fixed samples from five development sessions, median absolute residual was `2.3187` bps and maximum absolute residual was `13.1319` bps, against frozen limits of `5.0` and `35.0` bps.

## Track C — causal account state

**BARRED.** Serial replay parity and mutate-future invariance passed offline. The existing authorized 2026-08-03 paper round trip proves filled BUY/SELL actions and pre-action occupancy `0 -> 1`, but it has no post-exit account snapshot and does not record exact values for all six causal features. Historical/live ledger-transition identity therefore remains missing. Both the wrapper comparison and existing-paper audit are explicitly noncertifying; neither was substituted for the required complete observation.

Offline adapter compute p99: `0.002396 ms`; this is reported but not written as an admitted availability clock.

## Track A — multi-session live OPRA

**NOT RUN / BARRED.** The 2026-08-03 capture proves the entitlement worked under that prior session-bounded authorization, but it does not authorize a new five-session capture. Official Databento pricing says subscription plans include live streaming while grandfathered OPRA accounts may remain usage-metered; the active account plan is unknown, so zero marginal cost is not established. No Databento connection was attempted. The four Track-A families remain barred on their existing receipts.

Preflight finding: `v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/phase0b_receipts/track_a_authorization_and_cost_preflight.json` (`e332c02a6dc301c5bfe6d183c19ad7c556d04f09c978bc2c78b61c1ea99f4722`).

## Enforcement

The Phase-0 admission enforcement implementation was not modified. Missing, empty, tampered, unknown, or barred feature state still raises before model construction.

`STOP_FOR_CLAUDE_VERIFICATION`
