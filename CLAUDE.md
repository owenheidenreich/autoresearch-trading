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

**Claude IS the experiment loop.** Like Karpathy's autoresearch, the AI agent drives every iteration: form hypothesis, edit code, train, read score, keep/revert, repeat.

### One-time setup:
1. `./v2/ops/deploy.sh boot` -- boot GPU
2. `./v2/ops/deploy.sh start` -- upload code + data, install deps, verify CUDA

### Per-experiment cycle (Claude drives this):
1. Read last experiment results (results.tsv, artifacts, trade-level data)
2. Form a hypothesis. Write it in lab_notebook.md.
3. Edit `v2/train.py` and/or `v2/core/policy.py` locally. One change per experiment.
4. `git commit` the change.
5. `./v2/ops/deploy.sh push` -- upload changed files to GPU
6. `./v2/ops/deploy.sh experiment exp_NNN` -- run single experiment (~5 min, blocking)
7. `./v2/ops/deploy.sh pull` -- download results + artifacts
8. Read the score. If improved AND beats all baselines: **KEEP**. Otherwise: **REVERT** (`git checkout v2/train.py v2/core/policy.py`).
9. Log to results.tsv.
10. Check session limits. If any limit hit, stop. Otherwise go to step 1.

### Hard rules:
- NEVER fire off inner_loop.py and walk away. It re-trains identical code with no mutations. That is not autoresearch.
- NEVER bypass `deploy.sh` with raw `sshpass` commands. If `deploy.sh` has a bug, fix `deploy.sh`.
- NEVER delete `model.pt` unless executing a "fresh start" command.
- NEVER shut down the GPU lease until the session ends or the user says to stop.
- Each experiment requires a hypothesis and a code change BEFORE training. Re-running identical code is not an experiment.
- If `deploy.sh` fails, fix the issue in `deploy.sh`, don't work around it.

## Code Quality

- Before reporting a task complete, run `python -m py_compile <file>` on every changed file.
- Commit every time a change is made.
- Before editing a file, re-read it. After editing, verify the change applied. The Edit tool fails silently on stale context.
- One change per experiment. Do not bundle unrelated changes.

## Context Management

- After 10+ messages, re-read any file before editing. Do not trust memory of file contents.
- For tasks touching >5 files, launch parallel sub-agents.
- File reads are capped at 2,000 lines. Use offset/limit for larger files.

