# Documentation Index

If it's not listed here, it's archived or doesn't exist.

## What are you looking for?

- "How does the hill-climbing loop work?" --> [ART2_LOOP.md](../ART2_LOOP.md)
- "What is the current state of the system?" --> [current_state.md](current_state.md)
- "What commands can I run?" --> [COMMANDS.md](../COMMANDS.md)
- "How is a model scored?" --> [evaluator.md](evaluator.md)
- "What does the pipeline look like?" --> [PIPELINE.md](../PIPELINE.md)
- "What are the project's values?" --> [founder_intent.md](founder_intent.md)
- "What decisions have been made?" --> [decision_log.md](decision_log.md)

## Domain Knowledge

Core trading knowledge. Read these to understand WHY the system makes certain decisions.

- [0DTE Domain Knowledge](domain/0dte-domain-knowledge.md) -- Greeks, volatility, dealer mechanics, risk rules
- [Pickles Trading Knowledge](domain/pickles-trading-knowledge.md) -- Practitioner rules, VWAP framework, anti-patterns
- [Sinclair Volatility Extract](domain/sinclair-volatility-trading-extract.md) -- Realized vs implied vol, regime detection
- [Douglas Trading Zone](domain/douglas-trading-zone-extraction.md) -- Trading psychology
- [Book Knowledge Synthesis](domain/book-knowledge-synthesis.md) -- Cross-book synthesis

## Protocol and Operations

- [ART2_LOOP.md](../ART2_LOOP.md) -- canonical hill-climbing protocol (the operating loop definition)
- [COMMANDS.md](../COMMANDS.md) -- command reference
- [PIPELINE.md](../PIPELINE.md) -- one-page system overview (stages, artifacts, contracts)
- [program.md](../program.md) -- historical protocol (reference only; superseded by ART2_LOOP.md)

## Project Strategy

- [founder_intent.md](founder_intent.md) — non-negotiable standards and mission boundary
- [decision_log.md](decision_log.md) — durable decisions that outlive experiments
- [open_questions.md](open_questions.md) — active unresolved questions

## Technical Specs

- [current_state.md](current_state.md) — system overview, end-to-end flow
- [how_training_works.md](how_training_works.md) — model architecture, training loop, loss
- [data_contract.md](data_contract.md) — dataset structure and integrity rules
- [feature_schema.md](feature_schema.md) — 52 context features
- [labeling.md](labeling.md) — exact-chain sidecar labels
- [evaluator.md](evaluator.md) — scoring formula (dollar-weighted PF), baselines, hard gates

## Snapshots and Incidents

- [system_snapshot.md](system_snapshot.md) — detailed repo tree, artifact layout, file inventory (point-in-time reference)
- [incidents/](incidents/) — post-mortem analyses of significant bugs

## Research Logs

- `v2/results.tsv` — experiment outcomes (official runs only)
- `v2/lab_notebook.md` — experiment narrative and analysis
