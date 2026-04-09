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
3. **Keep/revert based on score only.** Score = `min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. Model must also beat all four baselines.
4. **Session limits:** 20 experiments, 10 hours, 6 no-improve streak, 4hr plateau, 3 crashes. Stop when any limit fires.
5. **Every experiment needs a hypothesis.** Write it BEFORE GPU spend.
6. **When stuck (3+ reverts):** Stop. Read trade-level replay data. Form a hypothesis about WHY. Then try structural changes.
7. **Every experiment trains from scratch.** There is no warm-starting. train.py always creates a fresh model with random weights. Architecture changes are always safe.
8. **Log everything** in `v2/results.tsv` and `v2/lab_notebook.md`.

**All training runs on Akash H100 GPU, never locally.** The dev machine is a MacBook. Local use is for: editing code, committing, running replay/evaluation, reading results. The experiment runner (`v2/ops/run_experiment.py`) runs on the GPU machine.

## Experiment Loop Execution

**Claude IS the experiment loop.** Same as Karpathy's autoresearch: edit code, train, read score, keep or revert, repeat. The only difference is training runs on a remote GPU.

### One-time setup:
1. `./v2/ops/deploy.sh boot` -- boot GPU
2. `./v2/ops/deploy.sh start` -- upload code + data, install deps

### The loop:
```
LOOP FOREVER:
  1. Edit v2/train.py and/or v2/core/policy.py
  2. git commit
  3. ./v2/ops/deploy.sh run_one exp_NNN
  4. Read the score from stdout
  5. If score improved AND beats all baselines: KEEP (branch advances)
  6. If not: REVERT (git checkout HEAD~1 -- v2/train.py v2/core/policy.py)
  7. Go to 1
```

`run_one` uploads code to GPU, trains from scratch, replays, scores, downloads model.pt. Every experiment trains from random init -- there is no warm-starting, no model state to manage between experiments.

### Keep/revert state:
- **Code state**: git. Keep = committed. Revert = `git checkout`.
- **Model output**: `model.pt` is downloaded after each experiment. It's the output, never an input.
- **Score**: Claude reads from `run_one` stdout. Best score tracked in conversation.

### Hard rules:
- NEVER run inner_loop.py. Claude IS the loop.
- NEVER bypass `deploy.sh` with raw `sshpass` commands.
- Each experiment requires a hypothesis and a code change BEFORE training.
- If stuck (3+ reverts): stop, analyze trade-level data, form a real hypothesis.
- **After EVERY experiment**, regenerate all three visual artifacts:
  1. `python -m v2.plot_trades --model v2/model.pt` (generates `v2/output/trades.html` + `v2/output/equity.html`)
  2. `python v2/plot_progress.py` (generates `v2/output/progress.png`)
  These are how the human monitors what the model is doing. Never skip this step.

## Code Quality

- Before reporting a task complete, run `python -m py_compile <file>` on every changed file.
- Commit every time a change is made.
- Before editing a file, re-read it. After editing, verify the change applied. The Edit tool fails silently on stale context.
- One change per experiment. Do not bundle unrelated changes.

## Context Management

- After 10+ messages, re-read any file before editing. Do not trust memory of file contents.
- For tasks touching >5 files, launch parallel sub-agents.
- File reads are capped at 2,000 lines. Use offset/limit for larger files.

