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

**All training runs on Akash H100 GPU, never locally.** The dev machine is a MacBook. Local use is for: editing code, committing, running replay/evaluation, reading results. The experiment runner (`v2/ops/run_experiment.py`) runs on the GPU machine.

## Experiment Loop Execution

When told "begin experiment loop" or "run the full experiment loop":

1. `./v2/ops/deploy.sh boot` -- boot GPU
2. `./v2/ops/deploy.sh start` -- upload code + data, install deps, verify CUDA
3. `./v2/ops/deploy.sh run` -- start inner_loop.py on GPU (NOT a custom script)
4. `./v2/ops/deploy.sh sync` -- auto-sync results to local (or `./v2/ops/deploy.sh status`)

**Hard rules:**
- NEVER write custom loop scripts. Use `inner_loop.py` via `deploy.sh run`.
- NEVER bypass `deploy.sh` with raw `sshpass` commands. If `deploy.sh` has a bug, fix `deploy.sh`.
- NEVER delete `model.pt` unless executing a "fresh start" command.
- NEVER shut down the GPU lease until the loop finishes or the user says to stop.
- Each experiment in the autoresearch loop requires a hypothesis and a code change BEFORE training. Re-running identical code is not an experiment.
- If `deploy.sh start` fails, fix the issue in `deploy.sh`, don't work around it.

## Code Quality

- Before reporting a task complete, run `python -m py_compile <file>` on every changed file.
- Commit every time a change is made.
- Before editing a file, re-read it. After editing, verify the change applied. The Edit tool fails silently on stale context.
- One change per experiment. Do not bundle unrelated changes.

## Context Management

- After 10+ messages, re-read any file before editing. Do not trust memory of file contents.
- For tasks touching >5 files, launch parallel sub-agents.
- File reads are capped at 2,000 lines. Use offset/limit for larger files.

