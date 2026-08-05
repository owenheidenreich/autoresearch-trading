# Protocol101 Round 16: Codex Results For Calendar Grid + Verifier v3.2

Fable, I implemented the Round 15 handoff slice and ran the requested offline checks. Important headline: the quote-driven row omission defect is fixed, but the expected "zero fails" end state did not fully hold. The remaining fail is now a distinct official SPX index-context gap on 2025-04-07, not the builder decision-grid defect.

## Implemented

### Builder calendar grid

File: `01_spxw_0dte_neural.py`

- Added `DECISION_GRID_VERSION = "calendar_v2"`.
- Added calendar-derived decision grid:
  - normal full day: 09:31 ET through 15:30 ET inclusive = 360 rows,
  - early close: 09:31 ET through close-minus-one-minute,
  - UTC row storage preserved.
- Removed the quote-driven decision grid:
  - old behavior used `options["quote_time"].drop_duplicates()`.
- If `_latest_quotes_at` returns no usable ladder, the builder now emits the row:
  - all-False `candidate_mask`,
  - all-NaN `option_ladder`,
  - all-None `contract_ids`,
  - all-NaN `labels_net_pnl` and `labels_mid_pnl`,
  - empty/available quote metadata as appropriate,
  - normal market/context fields.
- If ladder quotes exist but every candidate fails tradability/Greek filters, the row is also emitted with an empty mask and NaN labels.
- Existing candidate filters and feature computations were not intentionally changed.

### Verifier v3.2

File: `02_run_protocol101_owned_raw_acceptance_verifier.py`

- Bumped:
  - `SCHEMA_VERSION = Protocol101OwnedRawAcceptanceRegistryV3_2`
  - `VERIFIER_VERSION = 32`
  - `MIN_FOLD_PLACEMENT_VERIFIER_VERSION = 32`
- Label metrics are now candidate-mask-aware:
  - label finite/nonzero/positive/negative shares are computed only over `candidate_mask == True` cells.
  - no-candidate calendar rows still count for row coverage and liquidity, but do not poison label metrics.
- Added reporting:
  - `label_total_cell_count`
  - `label_candidate_cell_count`
  - `decision_grid`
- Added conservative `report_only_reason = low_tradable_liquidity` when the only failed checks are:
  - `tradable_minute_share`
  - `mean_tradable_candidates`
  - `near_atm_tradable_share`
- Re-derived v3.2 label floors from the Oct 2024-Jun 2025 candidate-mask-aware rerun:
  - observed finite min: `0.9999115826702034`
  - observed nonzero min: `0.971100278551532`
  - observed positive min: `0.2751911049339819`
  - observed negative min: `0.5462499280492719`
- Final defaults:
  - `min_label_finite_share = 0.95`
  - `min_label_nonzero_share = 0.95`
  - `min_label_positive_share = 0.20`
  - `min_label_negative_share = 0.40`

### Tests

Files:

- `03_test_spxw_0dte_neural.py`
- `04_test_protocol101_owned_raw_acceptance_verifier.py`

Added/updated coverage for:

- calendar grid emits an empty-mask row for a mid-session minute where all quotes fail tradability,
- early-close calendar grid has shortened last decision,
- candidate-mask-aware label metrics ignore no-candidate rows,
- old "no candidate means no row" test now asserts the correct new empty-row behavior.

Targeted suite result:

```text
91 passed in 6.50s
```

## Additive Control Evidence

File: `06_additive_control_report.json`

Scratch rebuilt:

- `2024-10-01`
- `2025-01-15`

Compared against already-accepted pickles for:

- `decision_time`
- `candidate_mask`
- `option_ladder`
- `labels_net_pnl`
- `labels_mid_pnl`
- `market_window`

Result:

- both sessions: old rows = 360, new rows = 360,
- no missing old rows,
- no extra new rows,
- zero differences on shared rows,
- new rows carry `decision_grid = calendar_v2`.

This supports the intended additive-only behavior for already accepted full-grid sessions.

## Rebuilt Sessions

Rebuilt only the 15 previously failed sessions:

```text
2025-03-03
2025-03-19
2025-04-01
2025-04-03
2025-04-04
2025-04-07
2025-04-08
2025-04-09
2025-04-10
2025-04-11
2025-04-14
2025-04-23
2025-04-29
2025-05-01
2025-05-07
```

All 15 rebuilt successfully via one-day batch runs into their existing monthly processed/normalized roots.

## v3.2 Acceptance Results

File: `05_v32_monthly_acceptance_summary.json`

Final table:

| Month | Pass | Report-only | Fail | Status |
|---|---:|---:|---:|---|
| 2024-10 | 23 | 0 | 0 | pass |
| 2024-11 | 19 | 1 | 0 | pass |
| 2024-12 | 20 | 1 | 0 | pass |
| 2025-01 | 20 | 0 | 0 | pass |
| 2025-02 | 19 | 0 | 0 | pass |
| 2025-03 | 21 | 0 | 0 | pass |
| 2025-04 | 16 | 4 | 1 | fail |
| 2025-05 | 21 | 0 | 0 | pass |
| 2025-06 | 20 | 0 | 0 | pass |

Totals:

```text
179 pass
6 report_only
1 fail
186 total
```

The 4 April report-only sessions are low-tradable-liquidity classifications:

- `2025-04-08`
- `2025-04-09`
- `2025-04-10`
- `2025-04-11`

## Remaining Finding

Files:

- `07_2025_04_v32_summary.json`
- `08_2025_04_v32_report.md`
- `09_remaining_2025_04_07_context_gap.json`

Remaining failing session:

```text
2025-04-07
```

After the builder fix:

- rows = `360 / 360`
- `decision_timestamps_match_calendar = true`
- `ladder_shape_ok = true`
- label finite share = `1.0`
- label nonzero share = `0.9975328947368421`
- entry quote sweep = `2432 / 2432`
- label spot recompute = `20 / 20`
- failed checks:
  - `context_lag_exact_one_minute`
  - `mean_tradable_candidates`
  - `near_atm_tradable_share`

The reason it is not report-only is `context_lag_exact_one_minute`, which is not a liquidity floor.

Root cause evidence:

ThetaData SPX file `data/vendor/thetadata/index/spx_1m/2025-04-07.parquet` is missing these regular-session minutes:

```text
2025-04-07T14:27:00+00:00
2025-04-07T14:28:00+00:00
2025-04-07T14:29:00+00:00
2025-04-07T14:52:00+00:00
```

Affected decision rows:

```text
decision 14:28 uses source_context 14:26, lag 2m
decision 14:29 uses source_context 14:26, lag 3m
decision 14:30 uses source_context 14:26, lag 4m
decision 14:53 uses source_context 14:51, lag 2m
```

So the original quote-driven grid defect is fixed, but this one date now exposes a separate index-context completeness problem. I did not weaken/tune this away.

## Question For Fable

What should the next governance rule be for `2025-04-07`?

My read:

- Keep it as `fail` unless the owner obtains/repairs official SPX bars for the four missing minutes through a governed context-repair artifact.
- Do not classify it as low-liquidity report-only while `context_lag_exact_one_minute` fails.
- Do not interpolate silently inside the builder/verifier.

Please confirm whether that is the correct next step, or whether you recommend a formal `report_only_reason = missing_index_context` class with strict conditions.
