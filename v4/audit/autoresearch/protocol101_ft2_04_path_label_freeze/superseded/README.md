# Superseded FT2-04 artifacts (pre protected-holdout repair)

These are the original FT2-04 outputs from 2026-07-28, preserved unchanged for
audit. They implemented the ORIGINAL D42 formula (census = corpus − outer-test),
which omitted the protected holdout. That leak was caught at packet verification
before any census statistic was computed, and repaired 2026-07-29 under owner
authorization (D42 amended to also exclude protected resources).

- census_sessions.pre_holdout_repair.json    — 76-session census (30 were protected-holdout sessions)
- intersection_proof.pre_holdout_repair.json — proof vs outer-test only
- receipt.pre_holdout_repair.json            — original receipt (product_contract_hash e08b02b3…101c64)
- *.pre_holdout_repair.py                     — original verifier + test (outer-test only)

The live packet files in the parent directory are the repaired versions.
Evidence is preserved, never deleted.
