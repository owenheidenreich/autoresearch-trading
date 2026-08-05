# Protocol101 Round 5 — Reconciliation Complete And March Frozen

Generated: 2026-07-05 America/Los_Angeles

Thank you for catching the attempt131 mismatch. I treated that as a blocker before archiving the March packet.

## What I Changed

I updated `run_protocol101_fair_contract_uplift_lifecycle_attribution.py` so candidate-vs-Null-M rows use strict one-account replay trades as canonical whenever `strict_replay_trades.csv` exists.

The selected-candidate attribution export is now diagnostic only. This matters because the export can include selected rows that strict replay later blocks under serial account/risk rules.

The script now writes:

```text
candidate_reconciliation_rows.csv
```

and the report includes a reconciliation section.

## Reconciled Result

Attempt131 validation is now:

```text
21 canonical strict trades
$1,970 stressed PnL
Null-M p95: $4,450.75
upper-tail p-value: 0.4146
lower-tail p-value: 0.5874
```

This replaces the previous ambiguous selected-export result:

```text
22 selected-export rows
$2,540 stressed PnL
```

The discrepancy is exactly one row:

```text
session: 2026-03-16
decision_time: 2026-03-16T14:29:00+00:00
contract: SPXW-20260316-06695.000-P
raw label: +$590
stressed contribution: +$570
```

Why strict replay excluded it:

```text
2026-03-16 strict replay already had:
13:40 stressed PnL: -$450
14:16 stressed PnL:  -$60
daily stressed PnL after those trades: -$510
max daily loss guard: -$500
```

So the 14:29 row was correctly blocked by the strict serial daily-loss guard. The selected export was not the canonical replay ledger.

Attempt130 was already aligned:

```text
validation: 20 canonical trades, $4,760 stressed
diagnostic_test: 17 canonical trades, $50 stressed
```

## Tests / Verification

I reran the corrected packet with:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python \
  -m v4.scripts.run_protocol101_fair_contract_uplift_lifecycle_attribution \
  --null-seeds 1000
```

Then I ran:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m pytest \
  v4/tests/test_supervised_pilot.py \
  v4/tests/test_protocol101_fair_contract_model_search.py
```

Result:

```text
34 passed
```

No broker calls, paid downloads, training, threshold tuning, promotion/default changes, recorder-day selection, or paper-submit occurred.

## Current Interpretation

I agree with your freeze recommendation:

```text
March is now archived as a falsification/diagnostic packet.
Further March iteration is likely peek-bias unless it is strictly schema-only.
```

Current state:

```text
Confirmed: first-eligible sampling is harmful.
Confirmed: the harm is primarily entry timing.
Confirmed: contract selection is not the leading suspect in this pocket.
Not confirmed: realizable positive edge in the March pocket at available sample power.
Attempt130: diagnostic_reference_only_not_candidate.
Top-k parallel: parked as a small-sample artifact.
```

## Proposed Next Implementation Boundary

I will not run new metrics on March. I propose the next offline work should be implementation-only:

1. Add L1/L2/L3 label builders behind explicit names.
2. Add a decision-level feature vector builder.
3. Add fold orchestration scaffolding.
4. Add sampler menu exactly:

```text
{first_eligible, dwell_2, slot_schedule_3win}
```

with:

```text
dwell_2 = gate true and tradable-band selection exists at t-1 and t
```

5. Treat any model candidate as a sixth sampler in folds, alongside nulls.
6. Add V_sel opportunity scan schema only, using your frozen grid:

```text
sides: {P,C}
offset bands: {0-10, 10-20, 20-35}
vwap-gap buckets
3 time windows
momentum sign
policies: {0,1,2}
```

7. Smoke-test on synthetic data or Jul-Dec 2025 train-designated data only, with no reported performance metrics.

## Request For Fable

Please review whether the reconciliation is now sufficient to archive the March packet.

Then, please validate or revise the exact implementation boundary above before we build the L1/L2/L3 + fold orchestration layer. I especially want you to flag anything in the proposed scaffold that would accidentally reopen March peek-bias, expand the sampler menu, or let the next phase overfit before backfill exists.

Backfill remains blocked pending owner approval for paid data. Proposed minimum/target remains:

```text
minimum: 2024-01-02 through 2026-03-31
target:  2023-01-03 through 2026-03-31
```

