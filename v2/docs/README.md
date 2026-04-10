# v2 Live Documentation Index

This is the index of current, accurate documentation for the v2 system. Do not read `archive/` unless the human asks for historical context.

## Protocol & Operations

- [program.md](../program.md) — definitive operator protocol
- [COMMANDS.md](../COMMANDS.md) — command phrases the agent should honor
- [HANDOFF.md](../HANDOFF.md) — current state snapshot and trust boundaries
- [LAYOUT.md](../LAYOUT.md) — directory map and "where to look when X fails"

## System Documentation

- [current_state.md](current_state.md) — detailed system overview, end-to-end flow
- [goal.md](goal.md) — mission and current phase
- [how_training_works.md](how_training_works.md) — training loop, model architecture, loss
- [data_contract.md](data_contract.md) — dataset structure and integrity rules
- [feature_schema.md](feature_schema.md) — 47 normalized features
- [labeling.md](labeling.md) — exact-chain sidecar labels
- [contracts.md](contracts.md) — contract selection and TradeIntent schema
- [evaluator.md](evaluator.md) — scoring formula, baselines, gates
- [baselines.md](baselines.md) — four replay baselines the model must beat

## Research Logs

- [results.tsv](../results.tsv) — official exact-chain experiment scores
- [lab_notebook.md](../lab_notebook.md) — diagnosis, hypotheses, experiment notes

## Historical Context

Historical documentation is archived in `archive/v2_historical/`. Only consult it when the human explicitly asks for pre-exact-chain context.
