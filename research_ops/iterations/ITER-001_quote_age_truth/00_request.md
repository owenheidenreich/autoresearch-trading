# Iteration Request

Iteration ID: `ITER-001_quote_age_truth`
Assumption ID: `A001`
Title: Quote age truth

## Request

Establish the first complete research-ops iteration for quote age truth. Trace how quote age is created, logged, validated, and used in v4, then design and implement a read-only diagnostic that answers:

Are live quote ages real, or are freshness guards passing because runtime supplies placeholder ages?

## Required Coverage

- Historical quote age construction
- Normalized quote age fields
- Live IBKR quote timestamp fields
- Protocol158 and Protocol160 quote handling
- Protocol101 candidate quote fields
- Paper guard quote-age validation
- Paper logs
- Shadow logs
- Tests
- Gaps that prevent reconstructing true age

## Diagnostic Requirements

- Read existing logs only.
- Report missing timestamp fields.
- Recompute quote age wherever possible.
- Compare recomputed quote age to persisted `quote_age_ms`.
- Classify rows as trustworthy, missing, placeholder, stale, or unreconstructable.
- Produce CSV and markdown artifacts under this iteration.

## Hard Constraints

- Do not modify v4 trading logic.
- Do not mutate runtime flags, launchd, broker behavior, paid data, training, thresholds, or model artifacts.
- Do not call broker APIs.
- Do not run paid-data scripts.
- Do not fabricate timestamps.
