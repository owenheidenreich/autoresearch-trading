# Agent Directives

## First Steps

1. Read `v2/HANDOFF.md` — current state and trust boundaries.
2. Read `v2/docs/founder_intent.md` — founder voice, standards, anti-goals.
3. Read `v2/program.md` — the definitive protocol.
4. Read `v2/COMMANDS.md` — what the human can ask you to do.

## Hard Rules

- **Two mutable files only:** `v2/train.py` and `v2/core/policy.py`. Everything else is the immutable evaluation harness.
- **One change per experiment.** Small, testable hypotheses.
- **Keep/revert based on score only.** Score = `min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. Must also beat all four baselines.
- **Every experiment trains from scratch.** No warm-starting.
- **Log everything** in `v2/results.tsv` and `v2/lab_notebook.md`.
- **All training runs on Akash H100 GPU, never locally.**
- **Run harness eval before GPU spend.**
- **Do not read `archive/` as default context.** Only consult it when the human explicitly asks for historical context.

## Live Documentation

See `v2/docs/README.md` for the full index of current, accurate documentation.
