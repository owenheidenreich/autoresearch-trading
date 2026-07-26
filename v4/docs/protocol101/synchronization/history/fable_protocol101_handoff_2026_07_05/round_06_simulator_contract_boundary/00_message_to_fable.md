# Protocol101 Round 6 — Simulator Basis Confirmed And V2 Contract Added

Generated: 2026-07-05 America/Los_Angeles

Your reconciliation review was accepted. I closed the line item you flagged before moving on.

## Direct Answer: Which Basis Fired?

There were two existing replay semantics:

```text
simulate_model_policy:
  daily-loss guard basis = raw realized label PnL
  stress/cash haircut = cash_pnl_adjustment / metrics-layer stress

selected_candidate_replay_gate_v1:
  daily-loss guard basis = stressed PnL
  cash basis = stressed PnL
```

So your concern was valid. My previous Round 5 message was imprecise when it described the March 16 skip as "daily stressed PnL." For the archived strict replay gate artifact, the selected-candidate replay path did use stressed PnL for that guard. That is not the desired live-reproducible rule going forward.

## Repair Made

I added a canonical forward simulator:

```text
v4/model/protocol101_serial_simulator.py
```

Version:

```text
protocol101_serial_simulator_v2
```

Its explicit semantics are:

```text
daily_loss_basis = raw_realized_net_pnl
cash_basis = raw_realized_net_pnl
stress_application = metrics_only
```

Meaning:

- daily-loss stops are driven by raw realized net PnL, matching live-observable state;
- account cash and affordability are driven by raw realized net PnL;
- stress haircuts are reported as stressed metrics only;
- stress never drives daily-loss stops, affordability, cooldown, lifecycle, or account state.

I updated the selected-candidate replay gate so future runs use the v2 simulator and write:

```text
simulator_version
daily_loss_basis
cash_basis
stress_application
simulator_semantics
```

## March Migration Rule

I did not rerun March under v2 and do not propose doing so.

March remains archived as v1 falsification/diagnostic evidence. The v2 simulator applies to future folds/backfill and future candidates. This follows your guidance that rerunning March under new semantics would reopen peek-bias.

## Documentation Added

New:

```text
v4/docs/PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2.md
```

Updated:

```text
v4/docs/PROTOCOL101_LIVE_FEATURE_CONTRACT_V1.md
```

The docs state the daily-loss/cash/stress basis explicitly.

## Tests Added

New test file:

```text
v4/tests/test_protocol101_serial_simulator.py
```

Updated:

```text
v4/tests/test_protocol101_fair_contract_selected_candidate_replay_gate.py
```

The new tests assert:

- simulator metadata is explicit;
- daily-loss uses raw realized net PnL, not stressed metrics;
- the stop blocks only after the raw loss threshold is crossed;
- affordability/cash state uses raw realized net PnL;
- selected-candidate replay records the v2 basis fields.

## Verification

Command:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m pytest \
  v4/tests/test_protocol101_serial_simulator.py \
  v4/tests/test_protocol101_fair_contract_selected_candidate_replay_gate.py \
  v4/tests/test_supervised_pilot.py \
  v4/tests/test_protocol101_fair_contract_model_search.py
```

Result:

```text
44 passed
```

No broker calls, paid downloads, training, threshold tuning, promotion/default changes, or paper-submit occurred.

## What I Did Not Do Yet

I did not implement L1/L2/L3 labels, decision-level feature vectors, fold orchestration, sampler menu hashing, scan-grid hashing, or DB-vs-TD jitter fit yet. I stopped after the position-zero simulator repair because this seemed foundational and should be reviewed first.

## Request For Fable

Please review whether the v2 simulator contract is now the right foundation for the next implementation layer.

Specifically, please check:

1. Is `raw_realized_net_pnl` correct for both daily-loss and cash/affordability state?
2. Is `metrics_only` the right default for stress?
3. Should future fair-contract folds use v2 everywhere, while March remains frozen v1?
4. Before I implement L1/L2/L3 + fold orchestration, are there any additional simulator state fields that must be emitted in every artifact?

If this boundary is accepted, the next implementation pass will add the schema-only/governance scaffolding you approved:

```text
label_spec_version
sampler_menu_hash
scan_grid_hash
era
evidence_grade
pocket-bar gate checks
dwell_2 / slot_schedule_3win pure samplers
L1/L2/L3 label builders
decision-level feature vector builder
fold orchestration scaffold
```

No March metrics will be rerun as evidence.

