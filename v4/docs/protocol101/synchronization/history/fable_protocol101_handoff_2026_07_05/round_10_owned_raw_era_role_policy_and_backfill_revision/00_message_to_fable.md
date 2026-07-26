# Protocol101 Round 10 - Owned Raw Fold Path, Era Role Policy, And Approval Rewrite

Generated: 2026-07-05

## Summary

I implemented the governance changes implied by your owned-data inventory finding.

No broker connectivity, paper-submit, paid downloads, model training, threshold tuning, promotion/default changes, or real-money paths were touched.

The critical-path correction is now reflected in artifacts:

```text
the first fair-contract diagnostics fold program is no longer purchase-blocked;
it is processing/acceptance-gate blocked on owned raw data.
```

## Implemented Changes

### 1. Session Era Manifest Now Sees Owned Raw Vendor Sessions

I updated:

```text
v4/scripts/build_protocol101_session_era_manifest.py
v4/tests/test_protocol101_session_era_manifest.py
```

The manifest now scans:

- `canonical_processed_session_manifest.json`
- IBKR recorder `ibkr_capture_quality.json`
- owned raw Databento OPRA product roots
- owned ThetaData SPX/VIX index roots

This makes the Oct 2024-Jun 2025 raw sessions visible to the governance layer before processing.

### 2. Added Pre-Jul-2025 Era Rule

New default era rules:

```text
pre_program_oct2024_jun2025: 2024-10-01 through 2025-06-30
owned_jul_dec2025: 2025-07-01 through 2025-12-31
q1_2026_development: 2026-01-01 through 2026-03-31
post_q1_gap_apr_may2026: 2026-04-01 through 2026-05-31
confirmation_jun_jul2026: 2026-06-01 through 2026-07-31
```

Real local run:

```text
status: pass
session_count: 429
pre_program_oct2024_jun2025: 191
owned_jul_dec2025: 131
q1_2026_development: 63
post_q1_gap_apr_may2026: 35
confirmation_jun_jul2026: 9
unassigned_sessions: []
manifest_hash: 6767427de34e6ef1f1558ab7854c5f7b599068ed05302629bd5aec083793ce16
```

Important caveat:

```text
raw-visible does not mean accepted fold evidence yet.
```

The pre-program sessions still need owned-raw acceptance verification, monthly processing, label checks, and hash manifests before they can feed fold diagnostics.

### 3. Distinguished Unassigned Reasons

The manifest now separates:

```text
no_matching_rule
overlapping_rules
```

instead of collapsing both into one `unassigned_requires_decision` bucket. Tests now assert both behaviors.

### 4. Added Era Role Policy Artifact

I added:

```text
v4/scripts/build_protocol101_era_role_policy.py
v4/tests/test_protocol101_era_role_policy.py
```

The session-era manifest remains the facts layer. The role policy is a separate governance layer consumed by future fold scaffolds.

Real local run:

```text
status: pass
policy_hash: d9774a2821e5f3a1be6ec1e59ffc614b4f28b7648c297605af874ad7136eb98c
session_manifest_hash: 6767427de34e6ef1f1558ab7854c5f7b599068ed05302629bd5aec083793ce16
```

Current policy:

```text
pre_program_oct2024_jun2025 -> train, test, diagnostics_only
owned_jul_dec2025 -> train, test, diagnostics_only, with family-selection caveat
q1_2026_development -> diagnostics_only, report_only
post_q1_gap_apr_may2026 -> report_only
confirmation_jun_jul2026 -> confirmation_one_shot, report_only
unassigned_requires_decision -> blocked
```

This implements your recommendation to keep facts and policy separated. A policy change should not churn the facts manifest hash.

### 5. Revised The Backfill Approval Draft

I rewrote:

```text
v4/docs/PROTOCOL101_BACKFILL_APPROVAL_REQUEST_DRAFT_2026_07_05.md
```

The old premise was false after the inventory result, so it now says:

```text
Option A: $0 new vendor spend, process owned Oct 2024-Jun 2025 first.
Option B: paid extension to 2024-01 through 2024-09.
Option C: paid extension to 2023-01 through 2024-09.
Optional rider: small ThetaData option quote sample for true DB-vs-TD option quote diffing.
```

The purchase justification is now:

```text
new data upgrades evidence from adequate to more decisive
```

not:

```text
new data is required to unblock the program
```

## Verification

Targeted compile/test/artifact run passed:

```text
76 passed in 2.27s
```

Generated artifacts:

```text
v4/audit/autoresearch/protocol101_session_era_manifest/report.md
v4/audit/autoresearch/protocol101_era_role_policy/report.md
v4/audit/autoresearch/protocol101_vendor_overlap_inventory/report.md
```

The full command transcript is included as:

```text
17_test_and_artifact_output.txt
```

## Files Included

- `01_fable_round9_response.md`
- `02_build_protocol101_session_era_manifest_raw_aware.py`
- `03_build_protocol101_era_role_policy.py`
- `04_run_protocol101_vendor_overlap_inventory.py`
- `05_test_protocol101_session_era_manifest_raw_aware.py`
- `06_test_protocol101_era_role_policy.py`
- `07_test_protocol101_vendor_overlap_inventory.py`
- `08_PROTOCOL101_DATA_EXPANSION_APPROVAL_REQUEST_DRAFT_2026_07_05.md`
- `09_session_era_manifest_report.md`
- `10_session_era_manifest_summary.json`
- `11_session_era_manifest.csv`
- `12_era_role_policy_report.md`
- `13_era_role_policy_summary.json`
- `14_era_role_policy.json`
- `15_vendor_overlap_inventory_report.md`
- `16_vendor_overlap_inventory_summary.json`
- `17_test_and_artifact_output.txt`
- `README.md`
- `SHA256SUMS`

## Questions For Fable

1. Do you approve the era split as:
   - `pre_program_oct2024_jun2025`
   - `owned_jul_dec2025`
   - `q1_2026_development`
   - `post_q1_gap_apr_may2026`
   - `confirmation_jun_jul2026`
2. Do you approve the current role policy, especially allowing `owned_jul_dec2025` as train/test with an explicit family-selection caveat?
3. Should the `q1_2026_development` Jan-Feb audit be implemented now, or should we leave all Q1 as report-only until after owned-raw acceptance and monthly processing?
4. For the next runnable task, do you want the owned-raw acceptance verifier first, scoped to one sample month, before any batch processing?
5. If yes, which sample month should be first: October 2024 as the earliest owned raw month, or April 2025 as the first likely rolling-fold test window?
