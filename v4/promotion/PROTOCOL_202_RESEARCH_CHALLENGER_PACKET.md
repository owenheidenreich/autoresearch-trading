# Protocol 202 Research Challenger Packet

## Status

- Protocol202 is frozen as a research challenger only.
- Protocol101 remains the paper-trading default.
- Protocol202 must not submit paper or live orders.
- Allowed use: offline comparison and future no-order shadow analysis.

## Evidence Summary

| split | Protocol202 | baseline | delta | decision |
|---|---:|---:|---:|---|
| march_2026 | $101,025 | $73,440 | $27,585 | beat |
| q1_2026 | $269,910 | $202,720 | $67,190 | beat |
| recent_2026 | $22,215 | $25,295 | -$3,080 | miss |

## Recent Gap

Protocol202 recent gap: calls and late_afternoon/midday exits gave back a small amount versus baseline.

Do not add another architecture knob unless this same pattern repeats on additional protected data or live-paper logs.

## Source Hashes

- `v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy/summary.json` sha256 `0818601639e54632bba6c74786437385cfe989ae16ac52856345554fa602fbca`
- `v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy/report.md` sha256 `cb90ecb5076a34df365877be30fe3215c90459fb83362ab7acaa5d65ee7fe6d0`
- `v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy/protocol202_model_serial_trades.csv` sha256 `c12e986d250829b3bcb90c7f18e7abcae24707a9703bedcc25cd7b83c7dc4d79`
- `v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy/protocol194_baseline_serial_trades.csv` sha256 `89cd5cd66ac68e6e564004ac8817f62db71b9fa80db58f45daf6ac768f305351`
- `v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy/threshold_sweep.csv` sha256 `f93bf1fe19a6ba716ec3eb6b0e40c724094accffe9371182c5776b43fceca83f`
- `v4/audit/autoresearch/v4_aplus_hypothesis_203_protocol202_mixed_result_attribution/summary.json` sha256 `8c75f8aa86c2b4bb7b4714f1ea8a8369e07cf86ad6f25a72619072f90ec9c5bb`
- `v4/audit/autoresearch/v4_aplus_hypothesis_203_protocol202_mixed_result_attribution/report.md` sha256 `35a850e195ac86da42df2cc774527d7476d7cfd4341f47fcb6356a7a5b818e4d`
- `v4/audit/autoresearch/v4_aplus_hypothesis_205_protocol202_recent_gap_attribution/summary.json` sha256 `d30630da3ac2b93bc6227752e71a4e8938c5c944c5daa2acdf0366a89696aa84`
- `v4/audit/autoresearch/v4_aplus_hypothesis_205_protocol202_recent_gap_attribution/report.md` sha256 `b7e0644a356f19b8f24e0e95a84162a60daf1a58b86b596eeb8342b71d8e11a4`
- `v4/audit/autoresearch/v4_aplus_hypothesis_205_protocol202_recent_gap_attribution/combo_seed_summary.csv` sha256 `a6de01ba896a010857256c12a5c898d3a6f0fa00689485d76605f66b7d9c81c4`
- `v4/audit/autoresearch/v4_aplus_hypothesis_205_protocol202_recent_gap_attribution/side_summary.csv` sha256 `2edba6b64032c9a37471087ab48f7a2b35aaa4594014b6e7d76161bba218f08c`
- `v4/audit/autoresearch/v4_aplus_hypothesis_205_protocol202_recent_gap_attribution/time_bucket_summary.csv` sha256 `d8890d66c469af25b56df73e6c6e9d52c6b321022d2fe2a73ab422056c00fafb`
