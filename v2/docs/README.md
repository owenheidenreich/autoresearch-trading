# v2 Live Documentation Index

This is the index of current, accurate documentation for the v2 system. Do not read `archive/` unless the human asks for historical context.

The live documentation set is intentionally small. If a topic is not listed here, assume it is historical, generated, or out of scope for the current exact-chain reset.

## Protocol & Operations

- [program.md](../program.md) — definitive operator protocol
- [COMMANDS.md](../COMMANDS.md) — command phrases the agent should honor
- [HANDOFF.md](../HANDOFF.md) — current state snapshot and trust boundaries
- [LAYOUT.md](../LAYOUT.md) — directory map and "where to look when X fails"

## Founder Memory

- [founder_intent.md](founder_intent.md) — founder voice, standards, anti-goals, and current mission boundary
- [decision_log.md](decision_log.md) — durable project decisions that should outlive experiments
- [open_questions.md](open_questions.md) — active unresolved questions for the current phase

## System Documentation

- [current_state.md](current_state.md) — detailed system overview, end-to-end flow
- [how_training_works.md](how_training_works.md) — training loop, model architecture, loss
- [data_contract.md](data_contract.md) — dataset structure and integrity rules
- [feature_schema.md](feature_schema.md) — 47 normalized features
- [labeling.md](labeling.md) — exact-chain sidecar labels
- [contracts.md](contracts.md) — contract selection and TradeIntent schema
- [evaluator.md](evaluator.md) — scoring formula, baselines, hard gates, and baseline cache behavior

## Research Logs

- [results.tsv](../results.tsv) — official exact-chain experiment scores
- [lab_notebook.md](../lab_notebook.md) — diagnosis, hypotheses, experiment notes

## Observability

- `python3 -m v2.replay --traces` — decision trace: per-bar model decision log with oracle comparison (mandatory before keep/revert)
- `python3 -m v2.core.data_integrity` — data integrity validation (manifest, features, sidecars)

## Health Command

- `python3 -m v2.ops.status_report` — prints the current repo health, dataset fingerprint, experiment state, artifact compatibility status, and active blockers

## Historical Context

Historical documentation is archived in `archive/v2_historical/`. Only consult it when the human explicitly asks for pre-exact-chain context.
