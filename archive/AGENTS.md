# Agent Directives

## First Steps

1. Read `HANDOFF.md` — current state and trust boundaries.
2. Read `docs/founder_intent.md` — founder voice, standards, anti-goals.
3. Read `program.md` — the definitive protocol.
4. Read `COMMANDS.md` — what the human can ask you to do.

## Hard Rules

- **Default mutable surface:** `supervised/train.py` and `supervised/core/policy.py`.
- **Expanded surface is allowed only when the active protocol/hypothesis requires it.** For the current side-collapse reset this includes `supervised/core/metrics.py`, `supervised/replay.py`, `supervised/core/data_integrity.py`, `supervised/ops/pre_run_gate.py`, and the live docs that must stay in sync.
- **One hypothesis per experiment.** Coordinated edits are allowed when they are inseparable parts of the same hypothesis.
- **Promotion is score-gated.** Score = `min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. Must also beat all four baselines. Traces may justify continuing a hypothesis family, but not promoting it.
- **Direction mix is diagnostic, not a hard score gate.** Always report call count, put count, minority share, and direction balance.
- **Every experiment trains from scratch.** No warm-starting.
- **Log everything** in `supervised/results.tsv` and `supervised/lab_notebook.md`.
- **All training runs on Akash H100 GPU, never locally.**
- **Run harness eval before GPU spend.**
- **Do not read `archive/` as default context.** Only consult it when the human explicitly asks for historical context.

## Live Documentation

See `docs/README.md` for the full index of current, accurate documentation.
