# Do Not Touch Without Approval

Last updated: 2026-05-24

This file defines hard boundaries for AI-agent work. These boundaries protect
the frozen Protocol101 control, mutable runtime state, broker behavior, paid
data, model artifacts, and validation governance.

The default allowed action is read-only cartography or verifier work. Anything
that changes a protected surface requires explicit human approval before the
work starts.

## Safe Read-Only Work

This section defines what safe read-only work means.

Safe read-only work means:

- Inspecting source files, docs, manifests, reports, and local metadata.
- Reading logs without printing credentials, account identifiers, or sensitive
  broker details.
- Producing research_ops artifacts such as cartography reports, RFCs, verifier
  reports, decision memos, dashboard updates, schemas, and local summaries.
- Running local tests that do not call broker APIs, download data, train models,
  tune thresholds, mutate runtime flags, or write model/data artifacts.
- Running filename-only secret scans that do not print secret values.

Safe read-only work does not include editing runtime files, toggling paper
order flags, launching broker sessions, downloading vendor data, training,
tuning, or scoring protected holdouts for exploration.

## Exact Runtime And Broker Boundaries

Do not modify or run these without explicit human approval:

- `v4/runtime/protocol101_paper_order_enablement.json`
- `v4/runtime/**`
- `v4/logs/**` when the action mutates, truncates, rewrites, or republishes
  raw operational evidence.
- `v4/ops/launchd/*`
- `v4/ops/ibkr/run_protocol101_paper_session.sh`
- `v4/ops/ibkr/**` when the action can alter broker startup, credentials,
  ports, sessions, paper-submit behavior, or account connectivity.
- `v4/live/ibkr_paper_executor.py`
- `v4/live/ibkr_paper_guard.py` unless the iteration explicitly targets guard
  tests and has human approval.
- Any file that can call `placeOrder`.
- Any code path that changes paper-submit default behavior.
- Any code path that changes paper-order enablement, no-order shadow behavior,
  account guard behavior, max open positions, quantity, forced-flat behavior,
  or order-limit pricing.

## Model And Artifact Boundaries

Do not modify, replace, regenerate, or commit changes to:

- Protocol101 model, scaler, manifest, threshold, or artifact files.
- Protocol051/054 surface model, scaler, manifest, threshold, or artifact files.
- Protocol066/081 lifecycle model, scaler, manifest, threshold, or artifact
  files.
- `v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/**`
- `v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/**`
- Any `model.pt`, `scaler.json`, `manifest.json`, checkpoint, tensor, or saved
  model artifact under `v4/audit/**`, `v4/models/**`, `v4/artifacts/**`, or
  generated experiment directories.
- Any script option or code path that trains or saves model artifacts.

## Data And Paid-Data Boundaries

Do not modify or run without approval:

- Paid data download scripts.
- Any script that can download paid data.
- Vendor ingestion paths such as Databento, ThetaData, OptionsDX, or equivalent
  paid data providers.
- `v4/raw/**`
- `v4/normalized/**`
- `v4/normalized_official_context/**`
- `data/processed/**`
- Generated parquet, pickle, normalized quote, option ladder, and market context
  datasets.
- Any rebuild of historical datasets used for training, replay, labels, or
  lifecycle paths.

## Training, Threshold, Promotion, And Holdout Boundaries

Do not modify or run without approval:

- Protected holdout scoring scripts.
- Any protected holdout scoring for exploration.
- Threshold selection logic.
- Protocol101 margin threshold selection.
- Protocol051 surface threshold selection.
- Lifecycle threshold selection.
- Daily-stop, sizing, or confidence-threshold retuning.
- Challenger promotion logic or promotion packets.
- Any script that trains a model.
- Any script that saves model artifacts.
- Any script that selects, promotes, or changes the operational default.

Examples of protected script categories:

- `v4/scripts/run_protocol061_sequence_lifecycle_model.py`
- `v4/scripts/run_protocol097_sequential_event_policy.py`
- `v4/scripts/run_protocol101_event_history_policy.py`
- `v4/scripts/build_lifecycle_sequence_dataset.py`
- `v4/scripts/run_protocol139_daily_stop_retune.py`
- `v4/scripts/run_protocol152_multi_contract_promotion_gauntlet.py`
- `v4/scripts/run_protocol154_multi_contract_promotion_decision.py`
- Any future `train_*`, `build_*dataset*`, `retune_*`, `promote_*`,
  `holdout_*`, or `score_*holdout*` script.

## Requires Human Approval

Human approval is required before:

- Editing any protected file or path listed above.
- Running broker, IBKR, launchd, paper-submit, account, or order scripts.
- Running paid-data, vendor-ingest, normalized-data rebuild, or dataset build
  scripts.
- Running training, threshold tuning, model-saving, challenger promotion, or
  protected holdout scoring.
- Changing runtime flags, paper-submit defaults, max contracts, account
  assumptions, lifecycle behavior, or guard behavior.

Approval must be specific to the protected surface. General approval to work on
research_ops does not authorize operational changes.

## Requires CEO Decision Memo

A CEO decision memo is required before:

- Changing `PAPER_DEFAULT_PROTOCOL101`.
- Promoting any challenger.
- Changing paper-submit default behavior.
- Changing runtime flags or launchd behavior.
- Changing broker execution behavior.
- Changing max contracts, account assumptions, affordability semantics, reserve
  semantics, or one-position constraints.
- Changing model artifacts, thresholds, training labels, validation gates, or
  protected holdout policy.
- Accepting an open P0 assumption as an operational risk.

## Requires A Separate Branch

A separate branch is required for:

- Any approved modification to `v4/`.
- Any approved broker, launchd, runtime, model, data, training, threshold,
  promotion, or holdout work.
- Any change whose rollback must be isolated from research_ops documentation.
- Any branch that could alter operational behavior if merged.

Branch work must still follow the research_ops artifact flow: cartography
report, RFC or implementation plan, implementation summary, verifier report,
decision memo, and CEO dashboard update.

## Required Exception Packet

Any exception must include:

1. Explicit human approval for the specific protected surface.
2. Iteration ID.
3. Assumption ID or RFC.
4. Cartography report.
5. Experiment RFC or implementation plan.
6. Implementation summary.
7. Verifier report.
8. Decision memo.
9. CEO dashboard update.
