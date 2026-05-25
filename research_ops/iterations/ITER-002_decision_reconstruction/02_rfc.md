# Experiment RFC

Iteration ID: `ITER-002_decision_reconstruction`
Assumption ID: `A006`
Title: Decision reconstruction

## Research Question

Can every live/paper/no-order Protocol101 decision be reconstructed from logs alone?

## Null Hypothesis

Logs are insufficient unless they contain identity, action, candidate, model, quote, risk, account, artifact, and order/fill evidence needed to reproduce the decision without hidden in-memory state.

## Required Inputs

- Existing JSONL logs under `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading`.
- No broker APIs.
- No paid data.
- No model loading.

## Required Outputs

- `decision_reconstruction_matrix.csv`
- `missing_fields_report.md`
- `log_schema_gap_report.md`
- `decision_reconstruction_summary.json`

## Formulas

For each JSONL row:

- `decision_event = event_type in {candidate_set, model_decision, risk_gate, paper_order_*, paper_entry_*, paper_exit_*, exit_decision}`
- `reconstruction_status = sufficient` only if required fields for that event type are present.
- Missing fields are counted by category.

Required categories:

- identity: timestamp, session, run id, mode, event type
- action: model action or selected action
- candidates: candidate count, sample, and full candidate features
- model: score, threshold, wait logit, candidate logits
- quote: bid/ask, quote age, raw quote timestamp
- risk: pass/fail, reason, guard inputs
- account: cash/equity/open positions
- artifacts: model/scaler/manifest references
- orders/fills: order intent, submitted order, fill/cancel status where applicable

## Columns

`decision_reconstruction_matrix.csv` includes event identity, coverage booleans, `reconstruction_status`, and `missing_fields`.

## Pass/Fail Criteria

- `logs sufficient`: all decision rows are independently reconstructable.
- `logs insufficient`: no decision rows exist, or evidence is too sparse to diagnose schema.
- `schema patch required`: decision rows exist but any required reconstruction category is absent.

## Tests Required

- Unit test that a sparse decision row is insufficient.
- CLI test that artifacts are written.
- Protected import test must continue to prove diagnostics do not import trading/broker/model/paid-data modules.

## Implementation Plan

Create `research_ops/diagnostics/decision_reconstruction.py`, add tests, run the diagnostic against existing local JSONL logs, and write artifacts under this iteration.

## Interpretation Guide

This diagnostic is a log sufficiency audit. It does not prove whether the runtime decision was correct; it asks whether an independent reviewer could reproduce it from durable evidence.

## Limitations

- It cannot recover hidden in-memory state.
- It cannot infer missing candidate features or logits.
- It cannot verify Protocol160 source inside the clean transition branch because that file is absent here.
