# Documentation Index

Single source of truth for all project documentation. If it's not listed here, it's either in `archive/` (historical) or doesn't exist.

## Domain Knowledge

Core trading knowledge extracted from books and practitioner journals. Read these to understand WHY the system makes certain decisions.

- [0DTE Domain Knowledge](domain/0dte-domain-knowledge.md) — Greeks behavior, volatility concepts, dealer mechanics, risk rules, key formulas
- [Pickles Trading Knowledge](domain/pickles-trading-knowledge.md) — Practitioner journal: entry/exit rules, VWAP framework, time-of-day rules, anti-patterns
- [Sinclair Volatility Extract](domain/sinclair-volatility-trading-extract.md) — Volatility trading theory, realized vs implied, regime detection
- [Douglas Trading Zone](domain/douglas-trading-zone-extraction.md) — Trading psychology, zone theory
- [Book Knowledge Synthesis](domain/book-knowledge-synthesis.md) — Cross-book synthesis of actionable trading knowledge

## Protocol & Operations

- [program.md](../program.md) — definitive operator protocol
- [COMMANDS.md](../COMMANDS.md) — command phrases the agent should honor
- [HANDOFF.md](../HANDOFF.md) — current state snapshot and trust boundaries

## Project Strategy

- [founder_intent.md](founder_intent.md) — founder voice, standards, anti-goals, and current mission boundary
- [decision_log.md](decision_log.md) — durable project decisions that should outlive experiments
- [open_questions.md](open_questions.md) — active unresolved questions

## System Documentation (Supervised Approach)

- [current_state.md](current_state.md) — detailed system overview, end-to-end flow
- [how_training_works.md](how_training_works.md) — training loop, model architecture, loss
- [data_contract.md](data_contract.md) — dataset structure and integrity rules
- [feature_schema.md](feature_schema.md) — 47 normalized features
- [labeling.md](labeling.md) — exact-chain sidecar labels
- [contracts.md](contracts.md) — contract selection and TradeIntent schema
- [evaluator.md](evaluator.md) — scoring formula, baselines, hard gates
- [harness-rebuild.md](harness-rebuild.md) — harness rebuild guide

## RL Approach Documentation

- [rl_overview.md](rl_overview.md) — v3 pure-RL project overview and architecture

## Research Logs

- Supervised: `supervised/results.tsv`, `supervised/lab_notebook.md`
- RL: `rl/results.tsv`, `rl/lab_notebook.md`

## Historical Context

Historical documentation is archived in `archive/`. Only consult when you need pre-current context.
