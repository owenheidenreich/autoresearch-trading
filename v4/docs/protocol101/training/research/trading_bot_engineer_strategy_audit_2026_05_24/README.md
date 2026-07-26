# Trading Bot Engineer Strategy Audit - 2026-05-24

This folder is a research handoff for an outside trading bot engineer or ML engineer.

It does not train a model, retune thresholds, score untouched data, call a broker, buy data, or change `PAPER_DEFAULT_PROTOCOL101`.

The goal is to stop asking "which neural tweak beats Protocol101?" and start asking the harder strategy questions:

- What kind of trade is Protocol101 actually taking?
- Why did trader-belief Protocol101 survive while many broader ML challengers stayed research-only?
- Where is Protocol101 genuinely weak?
- Which questions should be answered before the next model is trained?

## Recommended Read Order

1. `00_EXECUTIVE_BRIEF.md` - bottom line and what changed in framing.
2. `01_PROTOCOL_HISTORY_MAP.md` - family-level review of the major protocol lines.
3. `02_PROTOCOL101_ANATOMY_AND_TRADES.md` - what Protocol101 actually does and what its trades look like.
4. `03_CHALLENGERS_AND_FAILURE_PATTERNS.md` - why higher-PnL challengers did not replace Protocol101.
5. `04_PROTOCOL101_WEAK_POINTS_AND_UNASKED_QUESTIONS.md` - the key trader-level unknowns.
6. `05_RECOMMENDED_DIAGNOSTIC_AGENDA.md` - the next work before model training.
7. `06_EVIDENCE_INDEX.md` - source artifacts and paths.
8. `07_OUTSIDE_ENGINEER_BRIEF.md` - concise brief and questions for an external specialist.

## Supporting Tables

- `protocol_family_scorecard.csv` - compact lineage map.
- `selected_protocol_inventory.csv` - individual protocol references for the main evidence trail.
- `protocol101_trade_profile_snapshot.csv` - extracted Protocol101 trade behavior.
- `diagnostic_question_backlog.csv` - prioritized questions and evidence needed.

## Current Decision

Protocol101 remains the paper default.

Several research challengers beat Protocol101 on exposed diagnostic splits, but none should replace it yet. The blockers are not just "model quality." They are strategy definition, timing/fill realism, live candidate parity, validation exposure, and unclear trader objectives around scalping, runners, continuation, and slot opportunity cost.
