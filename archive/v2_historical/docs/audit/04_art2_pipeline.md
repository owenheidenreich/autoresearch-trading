# Section 4: ART2 Pipeline

## Scope
The Autonomous Research and Trading (ART2) pipeline as a whole: how the system creates models through autonomous experimentation, the keep/revert loop, session management, artifact tracking, and what can be improved in the neural network research sequence. This section treats ART2 as a meta-system that orchestrates Market Data, Training, and Validation into a coherent research loop.

## Critical Files

| File | Role | Lines | Mutable? |
|------|------|------:|----------|
| `v2/program.md` | Definitive ART2 protocol. Compute setup, data specs, mutable vs immutable files, scoring formula, architecture facts, experiment loop rules, session limits. | doc | No (protocol) |
| `v2/COMMANDS.md` | Human-facing command reference. Maps verbal commands to protocol actions. | doc | No (protocol) |
| `CLAUDE.md` | Agent directives. Experiment loop execution rules, code quality, context management. The bridge between protocol and agent behavior. | doc | No (protocol) |
| `v2/ops/inner_loop.py` | Experiment orchestrator. Manages the cycle: run experiment -> read score -> keep/revert -> enforce session limits. SessionState tracking (experiment count, best score, streaks, plateau). | 392 | No (infra) |
| `v2/ops/run_experiment.py` | Single experiment execution. Trains model, replays, scores, compares to baselines. The atomic unit of one ART2 iteration. | 210 | No (infra) |
| `v2/ops/artifact.py` | Artifact bundles. Saves model.pt, policy.json, manifest.json (fingerprints, git SHA, score, timestamp), train.py.snapshot per experiment. Self-describing packages. | 211 | No (infra) |
| `v2/ops/model_manage.py` | Keep/revert mechanics. `keep`: candidate -> model.pt + model_best.pt. `revert`: discard candidate. | 51 | No (infra) |
| `v2/ops/deploy.sh` | GPU lifecycle. boot/start/run_one/ssh/status/download/stop for Akash H100 deployment. Handles code upload, remote execution, result download. | 921 | No (infra) |
| `v2/ops/monitor.py` | Web dashboard. GPU health, session state, experiment history, score charts, live log tail, source code display. Self-restarts on file changes. | 1265 | No (infra) |
| `v2/ops/lease_check.py` | GPU lease management. Auto-funds Akash lease if < 1hr remaining. | 160 | No (infra) |
| `v2/results.tsv` | Experiment results log. Experiment ID, score, status (keep/revert), description. Append-only. | data | Output |
| `v2/lab_notebook.md` | Experiment narrative. Hypotheses, observations, signal scans, phase summaries. Human-readable research log. | doc | Output |
| `v2/HANDOFF.md` | Session transfer notes. Current best, config snapshot, recent changes, ranked hypotheses. | doc | Output |

## The ART2 Loop

```
PROTOCOL (program.md + CLAUDE.md)
    |
    v
Claude (the agent) IS the experiment loop
    |
    |  1. Form hypothesis
    |  2. Edit train.py and/or policy.py (the only mutable files)
    |  3. git commit
    |
    v
deploy.sh run_one exp_NNN
    |  uploads: v2/ to GPU
    |  runs: run_experiment.py on H100
    |  downloads: model_candidate.pt
    |  prints: score
    |
    v
Score evaluation
    |  if score > best AND beats all 4 baselines:
    |    model_manage.py keep
    |    artifact.py save (bundle model + policy + manifest)
    |    update results.tsv, lab_notebook.md
    |    best_score = score
    |  else:
    |    model_manage.py revert
    |    git checkout HEAD~1 -- v2/train.py v2/core/policy.py
    |    log revert in results.tsv
    |
    v
Post-experiment (ALWAYS):
    |  plot_trades.py -> trades.html + equity.html
    |  plot_progress.py -> progress.png
    |  analyze_losses.py (if applicable)
    |
    v
Session limit check
    |  50 experiments max
    |  6 hours max
    |  8 consecutive reverts -> stop
    |  3 hours without improvement -> stop
    |  3 consecutive crashes -> stop
    |
    v
Loop back to step 1, or stop
```

## Key Interfaces

**Inputs to ART2:**
- `data.pt` (from Market Data) -- fixed dataset for the session
- `model.pt` / `model_best.pt` -- current best model (output becomes input for comparison)
- Human direction (hypotheses, constraints, "begin experiment loop")

**Outputs from ART2:**
- Sequence of experiment artifacts in `v2/artifacts/exp_NNN/`
- `results.tsv` -- cumulative experiment log
- `lab_notebook.md` -- research narrative
- Best `model.pt` -- the promoted model
- Visual artifacts: `trades.html`, `equity.html`, `progress.png`

**State files:**
- `.best_score` -- current session best
- `.inner_loop_state.json` -- session state (experiment count, streaks, plateau tracking)
- `.baseline_cache.json` -- cached baseline scores (avoid recomputation)

## Artifact Bundle Structure

```
v2/artifacts/exp_NNN/
    manifest.json    -- fingerprints, git SHA, score, timestamp, hyperparams, train metrics
    policy.json      -- DecisionPolicy snapshot
    model.pt         -- trained weights
    train.py.snapshot -- exact code that produced this model
```

## How Sections Interconnect Within ART2

```
                    +----------------+
                    |   CLAUDE.md    |
                    |  program.md    |  <-- Protocol layer (rules)
                    |  COMMANDS.md   |
                    +-------+--------+
                            |
                            v
                    +-------+--------+
                    | Claude (agent) |  <-- Decision layer (hypotheses, edits)
                    +-------+--------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
      +-------+--------+         +-------+--------+
      |   train.py     |         |   policy.py    |  <-- Research surface (mutable)
      +-------+--------+         +-------+--------+
              |                           |
              +-------------+-------------+
                            |
                            v
                    +-------+--------+
                    |  deploy.sh     |
                    |  run_one       |  <-- Execution layer (GPU)
                    +-------+--------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
      +-------+--------+         +-------+--------+
      |   train.py     |         |   replay.py    |  <-- Compute layer (H100)
      |  (on GPU)      |         |  (on GPU)      |
      +-------+--------+         +-------+--------+
              |                           |
              v                           v
      model_candidate.pt              score
              |                           |
              +-------------+-------------+
                            |
                            v
                    +-------+--------+
                    | keep / revert  |  <-- Decision gate
                    | model_manage   |
                    | artifact.py    |
                    +-------+--------+
                            |
                            v
                    +-------+--------+
                    | results.tsv    |
                    | lab_notebook   |  <-- Record layer
                    | progress.png   |
                    +----------------+
```

## Dependencies on Other Sections

| Section | Dependency |
|---------|------------|
| Market Data | `data.pt` is a fixed input for the entire session |
| Training Runs | `train.py` and `policy.py` are the mutable research surface; `run_experiment.py` wraps training |
| Validation | `replay.py` and scoring determine keep/revert; baselines set the promotion bar |
| IBKR Paper Trading | The best `model.pt` produced by ART2 is the artifact deployed to paper trading |

## Audit Surface Area

- Loop integrity: does keep/revert correctly advance or roll back code and model state?
- Session limits: are all 5 limits (experiments, time, streak, plateau, crashes) enforced?
- Artifact completeness: does every experiment produce a self-describing bundle?
- Baseline fairness: must the model beat all 4 baselines -- are they appropriate?
- One-change-per-experiment: is this discipline enforced or advisory?
- Hypothesis-before-GPU: does the protocol prevent blind hyperparameter fishing?
- From-scratch training: is warm-starting prevented? Are there any code paths that load prior weights?
- Revert correctness: after revert, is the code state identical to before the experiment?
- Score tracking: is results.tsv append-only and consistent with artifact manifests?
- Neural network principles: is the Transformer architecture well-suited for time-series option data? Are there architectural improvements (attention patterns, positional encoding, feature interactions) that could improve learning?

---

## Audit Questions -- Direct Improvements

**1. Session limits disagree across three source-of-truth documents.**
`inner_loop.py:34-38` defines: 50 experiments, 6 hours, 8 no-improve, 3hr plateau, 3 crashes. `program.md:155-161` defines: 20 experiments, 10 hours, 6 no-improve, 4hr plateau, 3 crashes. `CLAUDE.md:17` matches program.md. Three documents, two different sets of numbers. When Claude drives the loop manually (reading `program.md`), limits differ from `inner_loop.py` autonomous mode. Which set is canonical?

**2. `results.tsv` has duplicate experiment IDs, breaking reproducibility.**
Lines 17-20 of `results.tsv` show `exp_001` through `exp_004` appearing a second time. The file is append-only with no session separator. `inner_loop.py:347` resets IDs to 1 each session. When Claude drives the loop, IDs are manually chosen. Neither guarantees cross-session uniqueness. Post-hoc analysis of the research trajectory is unreliable.

**3. `run_experiment.py` is dead code -- `deploy.sh` calls `run_experiment_wf.py` instead.**
`deploy.sh:823` runs `python3 -m v2.ops.run_experiment_wf`, but `program.md` and `CLAUDE.md` reference `run_experiment.py` as canonical. `inner_loop.py:273` calls `run_experiment` (not `run_experiment_wf`). If anyone runs `inner_loop.py`, they get single-split evaluation instead of walk-forward CV -- a silent correctness bug producing different scores for the same model.

**4. `model_manage.py` does not verify model integrity before promoting.**
`model_manage.py:22-30` does raw `shutil.copy2` with no fingerprint check. Meanwhile, `artifact.py:44-51` has `_file_fingerprint()` and `inner_loop.py:200-211` validates fingerprints on revert. The keep path -- the one that matters most -- skips validation entirely. A corrupted download from Akash would be silently promoted to `model_best.pt`.

**5. Artifact bundles saved for every experiment, including reverts.**
`run_experiment_wf.py:69-88` calls `save_artifact()` unconditionally before the keep/revert decision. Reverted experiments accumulate artifacts indefinitely. On a long research campaign, `get_best_artifact()` scanning becomes misleading and disk-heavy.

**6. `_run_sync` downloads `model.pt` from remote, not `model_candidate.pt` -- race condition risk.**
`deploy.sh:729,743` downloads the remote's `model.pt` (last fold's model, overwritten each fold) to local `model_candidate.pt`. If sync happens mid-fold, it could download a partially-written or intermediate-fold model file. No lock or completion marker exists. Safe in `run_one` (SSH blocks) but `_run_sync` polls asynchronously.

## Audit Questions -- Deeper Planning

**7. Sortino caps at 6.0, making the score a pure PDR optimizer past early convergence.**
`program.md:78`: `score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. Results show Sortino consistently hits 6.0. Once capped, `score = 6.0 * PDR`, and the entire loop optimizes exclusively for fewer losing days. This drives extreme selectivity (77 trades over 60 days in exp_046) rather than higher risk-adjusted returns. A live 0DTE bot trading 1.28 times/day vs 5 times/day -- the score cannot distinguish these strategies.

**8. "One change per experiment" discipline is entirely advisory -- nothing enforces it.**
Neither `inner_loop.py` nor `run_experiment_wf.py` checks the git diff between experiments. Claude can bundle multiple changes. When a multi-change experiment reverts, you learn nothing about which change was responsible, wasting GPU time. Consider a git-diff check or at minimum a manifest field recording changed lines.

**9. The research surface (train.py + policy.py) may be too narrow for the problem.**
The protocol locks everything except `train.py` and `policy.py`. Lab notebook documents multiple cases where the bottleneck was in locked files: feature pipeline issues (sessions 4-5), label design (Phase 3), simulator bugs (Phase 4 audit found 3 critical bugs), cost model errors (data audit). Each required breaking the "immutable harness" rule. The question is whether the current harness is finally correct, or whether more hidden bugs exist that the research loop cannot reach.

**10. Walk-forward CV uses fixed fold boundaries, not rolling windows -- regime-dependent scoring.**
`program.md:113-119` shows 5 folds with expanding training. Fold 0 trains on days 0-673 and tests on 674-733. Fold 4 trains on days 0-913 and tests on 914-973. The production model comes from fold 4 (most data) but aggregate score averages all 5 folds equally. A change that helps fold 4 (current regime) but hurts fold 0 (old regime) gets penalized equally. For deployment into tomorrow's market, fold 4 is the only one that matters.

**11. The loop has no explore/exploit mechanism -- pure hill-climbing with monotonic score gating.**
`inner_loop.py:147`: `score > state.best_score` is strict improvement. No mechanism for lateral moves, simulated annealing, or population-based exploration. After exp_046 hit 5.667, 5 experiments failed to improve. Experiments 068-069 attempted a fundamentally different approach (MFE-based training) and were immediately rejected. The loop structurally cannot explore paradigm shifts that require temporary regression.

**12. Artifact manifest does not capture the full experiment context for reproducibility.**
`artifact.py:92-105` stores git SHA, score, fingerprints, hyperparams, val_loss. Missing: active session limits, which fold produced the model, per-fold scores, the hypothesis, baseline scores it was compared against, random seed. Without per-fold scores, you cannot tell if an aggregate score came from consistent folds or one outlier.

**13. No guard against training/inference feature mismatch across model versions.**
The model checkpoint does not store a feature schema or feature names. If `compute_features.py` changes feature selections (as happened in sessions 4-5 when features went 71 -> 58 -> 47), old artifacts become silently incompatible. `artifact.py:load_artifact()` reconstructs from hyperparams but does not check current data's feature count against what the model was trained on.

## Related Documentation

- `v2/program.md` -- the definitive protocol
- `v2/COMMANDS.md` -- human-facing command reference
- `CLAUDE.md` -- agent directives
- `v2/HANDOFF.md` -- session transfer notes
- `v2/lab_notebook.md` -- experiment narrative
