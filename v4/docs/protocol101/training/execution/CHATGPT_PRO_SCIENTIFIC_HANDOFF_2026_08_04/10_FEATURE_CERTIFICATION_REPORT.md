# Path-D Phase 0 feature certification report — 2026-08-04

Status: **STOP_FOR_CLAUDE_VERIFICATION**

Ledger SHA-256: `fbdf4d125adbade45defd535567f17c4534645e8c3f01dfc36204273968df55d`

No model was loaded, trained, or fitted. The protected holdout open count remained `0`. No paid data, live capture, broker, order, paper runtime, promotion, default, runtime flag, or launchd path was accessed.

| Family | Features | Certification | Availability clock | Tolerance | Evidence / blocker |
|---|---:|---|---:|---:|---|
| `entry.opra_cbbo1m_cross_section.v1` | 11 | **BARRED** | N/A | 0.0 | parent_family_not_admitted:entry.opra_cbbo1m_native.v1 |
| `entry.self_computed_greeks.v1` | 7 | **BARRED** | N/A | 0.0 | parent_family_not_admitted:entry.opra_implied_spot.v1 |
| `entry.contract_clock.v1` | 8 | **FIT_READY / ADMITTED** | 30603.667 ms measured definition replay warm-up | 0.0 | definition implementation hash, historical/live OSI geometry invariance, exchange-calendar hash |
| `entry.causal_account_state.v1` | 6 | **BARRED** | N/A | 0.0 | missing_required_receipts:historical/live ledger transition identity|serial replay parity|mutate-future invariance |
| `entry.opra_ohlcv1m_sparse.v1` | 4 | **BARRED** | N/A | 0.0 | missing_required_receipts:multi-session OHLCV receipt-latency distribution|sparse zero-fill invariance|native OHLCV versus live-trade aggregation identity |
| `entry.opra_cbbo1m_native.v1` | 11 | **BARRED** | N/A | 0.0 | same-session live-versus-Historical-API value identity; missing_required_receipts:multi-session local receipt-latency distribution|sparse-minute and freshness receipt |
| `entry.opra_implied_volatility.v1` | 6 | **BARRED** | N/A | 0.0 | parent_family_not_admitted:entry.opra_implied_spot.v1 |
| `entry.opra_implied_spot.v1` | 8 | **BARRED** | N/A | 0.0 | missing_required_receipts:shared parity-estimator implementation hash|historical/live candidate-pair identity|comparison to completed official SPX for measurement only |
| `entry.opra_cbbo1s_rolling.v1` | 12 | **BARRED** | N/A | 0.0 | missing_required_receipts:same-session live-versus-Historical-API value identity|shared rolling-window implementation hash|no-update and reconnect mutation tests |

## Exit-gate result

- OPRA-only scoped features: `73`; admitted: `8`; barred: `65`.
- `entry.contract_clock.v1` is the sole certified family. Its availability clock is measured from the owned definition capture (`capture_finished_unix_ns - capture_started_unix_ns`), not inherited from ThetaData.
- `entry.opra_cbbo1m_native.v1` remains barred because the owned evidence is one session and the contract requires a multi-session local receipt-latency distribution. The observed OPRA same-session p99 of about 319.5 ms is preserved as evidence but is not promoted into a multi-session receipt.
- Every Tier-2 family remains barred until both its own receipts and its parent admission exist.
- The 10 permanently barred controls, including `intraday_open_interest`, are included in the executable ledger so they fail with an explicit `BARRED` status.

## Enforcement

The admission check executes immediately before feature-matrix construction in the autoresearch v2 OOF fit path. Missing, empty, tampered, unknown, or barred ledger state raises before any estimator is created or fitted.

`STOP_FOR_CLAUDE_VERIFICATION`
