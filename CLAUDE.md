# Agent Directives

## First Steps

1. Read `v2/program.md` -- the definitive protocol.
2. Read `v2/COMMANDS.md` -- what the human can ask you to do.
3. When the human says a command (e.g. "begin experiment loop"), execute it per those files.

## v2 Autoresearch Protocol (Karpathy's method)

This project follows Karpathy's autoresearch design (github.com/karpathy/autoresearch).

**Core rules:**
1. **Two mutable files only:** `v2/train.py` (model/loss) and `v2/core/policy.py` (trading params). Everything else is the immutable evaluation harness.
2. **One change per experiment.** Small, testable hypotheses. Not shotgun changes.
3. **Keep/revert based on score only.** Score = `min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. Model must also beat all three baselines.
4. **Session limits:** 50 experiments, 6 hours, 8 no-improve streak, 3hr plateau, 3 crashes. Stop when any limit fires.
5. **Every experiment needs a hypothesis.** Write it BEFORE GPU spend.
6. **When stuck (3+ reverts):** Stop. Read trade-level replay data. Form a hypothesis about WHY. Then try structural changes.
7. **Never warm-start from an incompatible architecture.** If you change the model shape, fresh start.
8. **Log everything** in `v2/results.tsv` and `v2/lab_notebook.md`.

The experiment runner (`python v2/ops/run_experiment.py --id exp_NNN`) handles all plumbing: train, replay, baselines, artifacts. You handle research decisions.

## Code Quality

- Before reporting a task complete, run `python -m py_compile <file>` on every changed file.
- Commit every time a change is made.
- Before editing a file, re-read it. After editing, verify the change applied. The Edit tool fails silently on stale context.
- One change per experiment. Do not bundle unrelated changes.

## Context Management

- After 10+ messages, re-read any file before editing. Do not trust memory of file contents.
- For tasks touching >5 files, launch parallel sub-agents.
- File reads are capped at 2,000 lines. Use offset/limit for larger files.

## User Decisions

The user must agree on definition of every step of the Loop in ART2.
