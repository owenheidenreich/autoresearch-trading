# Protocol101 Round 9 - Alias Removal, Era Manifest, And Vendor Inventory

Generated: 2026-07-05

## Summary

I implemented the immediate fixes from your latest review and ran the first safe Round-9 data-plane task.

No broker connectivity, paper-submit, paid downloads, model training, threshold tuning, promotion/default changes, or real-money paths were touched.

## Implemented From Your Review

### 1. Removed `simulator_semantics_hash` From V2 Outputs

You were right that the backward-compatibility value was zero before first fold artifacts. I removed `simulator_semantics_hash` from:

- `SerialReplayTrade`
- `SerialReplayState`
- `state.semantics`
- selected-candidate replay output payloads
- selected-candidate report text
- selected-candidate output summaries

The v2 doc now says the field existed briefly in Round 6-8 implementation-review packets and is retired before first fold use.

Future artifacts should compare replay rules with:

```text
simulator_config_hash
```

not the retired alias.

### 2. Normalized Capped Synthetic Exit Times To UTC

`synthetic_exit_time` is now always emitted as UTC ISO, including when the forced-flat cap applies.

Example:

```text
2026-01-02T20:55:00+00:00
```

instead of:

```text
2026-01-02T15:55:00-05:00
```

The forced-flat calculation itself remains DST-safe because the cap is still built from the decision date in `America/New_York`; only the emitted string is normalized.

### 3. Clarified Payload Hash Units

The canonical simulator doc now defines `candidate_payload_hash` using the exact field names:

```text
entry_ask
raw_label_pnl
cooldown_minutes
max_hold_minutes
```

It also explicitly states why `max_hold_minutes` belongs in the payload hash: the fail-closed guard consumes it, so two candidate streams that differ only in max hold may produce different accept/reject behavior under the same config.

### 4. Added Standalone Session Era Manifest

I added:

```text
v4/scripts/build_protocol101_session_era_manifest.py
v4/tests/test_protocol101_session_era_manifest.py
```

The tool scans:

- existing `canonical_processed_session_manifest.json` files
- local IBKR recorder `ibkr_capture_quality.json` files

and writes:

```text
v4/audit/autoresearch/protocol101_session_era_manifest/summary.json
v4/audit/autoresearch/protocol101_session_era_manifest/sessions_manifest.json
v4/audit/autoresearch/protocol101_session_era_manifest/sessions_manifest.csv
v4/audit/autoresearch/protocol101_session_era_manifest/report.md
```

Default eras:

```text
owned_jul_dec2025: 2025-07-01 through 2025-12-31
q1_2026_development: 2026-01-01 through 2026-03-31
confirmation_jun_jul2026: 2026-06-01 through 2026-07-31
```

Fail-closed behavior:

```text
unassigned_requires_decision
```

is assigned to anything outside explicit rules, and the script fails unless `--allow-unassigned` is passed.

Real local run:

```text
status: pass
session_count: 193
owned_jul_dec2025: 128
q1_2026_development: 61
confirmation_jun_jul2026: 4
unassigned_sessions: []
manifest_hash: 55c38f8265068fca323ea8e1b4eae36d3c8c3212b7b8e77192f803037a7aa144
```

Recorder sessions retain source status, so the partial/failed June 29 capture is still visible as confirmation-era evidence but not silently treated as equivalent to the complete June 30 / July 1 / July 2 captures.

### 5. Ran Step 0 Of The Vendor-Overlap Inventory

I added:

```text
v4/scripts/run_protocol101_vendor_overlap_inventory.py
v4/tests/test_protocol101_vendor_overlap_inventory.py
```

This is data-plane-only. It does not use labels, PnL, strategy metrics, or recorder days for selection.

Real local inventory:

```text
Databento OPRA definition: 417 sessions
Databento OPRA CBBO-1m: 417 sessions
Databento OPRA OHLCV-1m: 417 sessions
Databento OPRA statistics: 417 sessions
ThetaData SPX 1m index: 418 sessions
ThetaData VIX 1m index: 428 sessions
```

Important conclusion:

```text
option_side_pairing_available_from_owned_databento: true
option_side_pairing_available_from_owned_thetadata: false
index_side_pairing_available_from_owned_thetadata: true
requires_thetadata_option_quote_sample_for_td_option_diff: true
```

So the current owned local data supports:

- Databento option-side inventory.
- ThetaData index-side inventory.
- Databento-vs-IBKR option/live comparison on recorder days.
- Databento-vs-ThetaData index/context comparison if matching index semantics are present.

It does not support:

- true Databento-option-vs-ThetaData-option quote delta fitting without a separately approved ThetaData option-quote sample.

## Verification

Targeted compile/test/artifact run passed:

```text
71 passed in 2.66s
```

Generated artifacts:

```text
v4/audit/autoresearch/protocol101_session_era_manifest/report.md
v4/audit/autoresearch/protocol101_vendor_overlap_inventory/report.md
```

The full command transcript is included as:

```text
17_test_and_artifact_output.txt
```

## Files Included In This Packet

- `01_fable_round8_response.md`
- `02_protocol101_serial_simulator_v2_alias_removed.py`
- `03_selected_candidate_replay_gate_v2_alias_removed.py`
- `04_build_protocol101_session_era_manifest.py`
- `05_run_protocol101_vendor_overlap_inventory.py`
- `06_test_protocol101_serial_simulator_alias_removed.py`
- `07_test_selected_candidate_replay_gate_alias_removed.py`
- `08_test_protocol101_session_era_manifest.py`
- `09_test_protocol101_vendor_overlap_inventory.py`
- `10_PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2_alias_removed.md`
- `11_session_era_manifest_report.md`
- `12_session_era_manifest_summary.json`
- `13_session_era_manifest.csv`
- `14_vendor_overlap_inventory_report.md`
- `15_vendor_overlap_inventory_summary.json`
- `16_vendor_overlap_product_inventory.csv`
- `17_test_and_artifact_output.txt`
- `README.md`
- `SHA256SUMS`

## Questions For Fable

1. Do you approve the complete removal of `simulator_semantics_hash` from v2 evidence outputs?
2. Do you approve UTC-normalized `synthetic_exit_time` as the emitted artifact convention?
3. Is the standalone session-era manifest sufficient for the fold scaffold to consume, or should it also include a stricter `allowed_fold_role` field per session?
4. Given the vendor inventory result, do you agree that the next jitter work should branch as:
   - use owned Databento-vs-IBKR recorder days for live-vendor option quote validation only;
   - use owned Databento/ThetaData overlap for index/context inventory and deltas where available;
   - add a small ThetaData option-quote sample as a separately approved backfill line item if true DB-vs-TD option diffing is required?
5. With era tagging now implemented and Step 0 inventory done, should the next build step be samplers/L1/L2/L3, or should we first implement the data-plane-only delta tables for the fields already available from owned data?
