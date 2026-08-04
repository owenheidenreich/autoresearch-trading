# Project Sections And Hill-Climb Gates

This document defines the project sections used before any renewed model hill climbing. It is intentionally operational: each section owns a distinct part of the system, has explicit allowed mutations, and has gates that prevent skipping ahead.

## Section Map

| Section | Name | Owns | May mutate | Forbidden without approval |
|---:|---|---|---|---|
| 1 | Architecture / foundation | Project contracts, current source of truth, default/challenger roles, command safety, stage gates. | Docs, read-only audit/check scripts, tests. | Runtime paper flags, launchd defaults, model artifacts, broker paths. |
| 2 | Data acquisition / data preparation | Raw/normalized/feature/label/audit contracts, paid-data approval, provenance, schema and causal timestamp invariants. | Data contracts, safe data-audit scripts, tests. | Paid endpoint calls, protected holdout scoring, raw rewrites, unstamped feature/label datasets. |
| 3 | Model experiments / training / testing / validating | Hypotheses, training runs, threshold/model selection, strict serial replay, validation claims. | Experiment outputs and research-only artifacts. | Training before Section 1/2 gates pass, threshold tuning on exposed diagnostics, protected-holdout model selection, paper-default promotion. |
| 4 | Live paper trade | IBKR paper runtime, no-order shadow parity, paper order guards, quote freshness, fill/cancel evidence, forced-flat safety. | Paper logs and runtime diagnostics. | Broker endpoint calls, paper-submit sessions, runtime enablement flag changes, real-money trading. |
| 5 | Promotion / governance | Research freezes, default-change decisions, rollback/demotion criteria, stale-doc resolution. | Promotion packets and governance docs. | Changing operational default, promoting challengers, demoting Protocol101 paper default. |

## Required Order

1. Section 1 must say what the bot is, what the current default is, which docs are current, and which commands are safe.
2. Section 2 must prove data contracts, provenance, and paid-data safety before any new training run is considered.
3. Section 3 may only begin model hill climbing after Section 1/2 pass and the active foundation hardening gates allow it.
4. Section 4 evidence is required before promotion claims or paper-default replacement, even if Section 3 produces a better replay.
5. Section 5 is required for any default change.

## Machine-Checkable Gate

Run:

```bash
python3 -m v4.scripts.run_project_section_readiness
```

The command writes:

```text
v4/audit/autoresearch/project_section_readiness/summary.json
v4/audit/autoresearch/project_section_readiness/report.md
```

It is read-only except for its audit outputs. It does not train, tune thresholds, download paid data, call broker endpoints, submit orders, or mutate operational runtime configs.

Before any Section 3 model experiment is even proposed as runnable, run:

```bash
python3 -m v4.scripts.run_formal_validation_governance
python3 -m v4.scripts.run_protocol272_fill_model_readiness --skip-ledger
python3 -m v4.scripts.run_untouched_holdout_availability
python3 -m v4.scripts.run_live_no_order_parity_readiness
python3 -m v4.scripts.run_unified_neural_training_readiness --skip-ledger
python3 -m v4.scripts.run_section3_model_experiment_preflight
```

For an actual model run, provide a preregistered hypothesis packet and require it:

```bash
python3 -m v4.scripts.run_section3_model_experiment_preflight --hypothesis-packet path/to/packet.md --require-hypothesis
```

This command is also read-only except for its audit outputs. It does not train, score protected holdouts, tune thresholds, download data, or call brokers.

## Current Section 1/2 Definition Of Fixed

Section 1 is fixed when:

- `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` exists and identifies the current operational truth.
- `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md` exists and frames pre-change diagnostics.
- `PAPER_DEFAULT_PROTOCOL101` remains explicit in `v4/docs/NAMING_GUIDE.md`.
- Model work is governed by `v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md`, `v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md`, and `v4/docs/FOUNDATION_HARDENING_AUDIT.md`.
- Safe command boundaries are explicit.

Section 2 is fixed when:

- `v4/docs/DATA_CONTRACT.md` contains the causal time, freshness, provenance, and live-reproducibility fields.
- Data sanity and integrity checks exist in `v4/checks/`.
- Paid market-data download scripts require exact approval before paid endpoint calls.
- Local prepared-data inventory exists for read-only inspection.
- Data acquisition remains separate from model selection and protected holdout use.

## Current Hill-Climb Status

Passing Section 1/2 does **not** automatically authorize model training. Current known blockers still include:

- fill/cancel/slippage evidence,
- live/no-order full-action parity,
- decision reconstruction completeness,
- untouched holdout data availability.

The Protocol101 paper bridge now derives `quote_age_ms` from ticker timestamps and fails closed when freshness is missing. Formal validation governance now has a strategy matrix and CSCV-style proxy at `v4/audit/autoresearch/formal_validation_governance/summary.json`, but execution evidence, untouched data, and live parity remain separate gates. Those blockers belong mostly to Sections 3 and 4. They must be closed before a model can be called better than Protocol101 or before any challenger is promoted.
