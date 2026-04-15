# ART2 Operating Loop

The canonical definition of the hill-climbing loop for the SPX 0DTE research system.

This document defines the exact closed-loop system for turning a hypothesis into a candidate, a candidate into evidence, and evidence into a keep/revert decision.

## Precedence

When documents disagree, this precedence applies:

1. **Code** (`metrics.py`, `config.py`, `simulator.py`, etc.) -- ground truth
2. **This document** (`ART2_LOOP.md`) -- operating loop definition
3. `current_state.md` -- current system state
4. `evaluator.md` -- scoring and evaluation rules
5. `COMMANDS.md` -- command reference
6. `PIPELINE.md` -- system map
7. `program.md` -- historical protocol (reference only)
8. Everything else -- reference/history

If this document contradicts the code, the code wins. If `program.md` contradicts this document, this document wins.

---

## Stage 0 -- Preconditions

Before any experiment run is allowed:

- [ ] `python3 -m v2.ops.health` passes (config, data, model, smoke)
- [ ] Config fingerprint matches across data.pt and model.pt
- [ ] Dataset fingerprint is current (no rebuild pending)
- [ ] Evaluator fingerprint (`score_config_fingerprint()`) has not changed since last promoted model
- [ ] No unresolved BLOCKER items in observability gaps
- [ ] `python3 -m v2.ops.pre_run_gate --data v2/data.pt` passes

If any precondition fails, fix it before running. Do not work around.

---

## Stage 1 -- Data Authority

### Canonical artifacts

| Artifact | Path | Identity |
|---|---|---|
| Dataset manifest | `v2/data.pt` | `dataset_fingerprint` in metadata |
| Per-day sidecars | `v2/data_sidecars/{date}.pt` | `chain_sidecar_digest` in manifest |
| Build report | `v2/build_reports/{fp}.json` | keyed by dataset fingerprint |
| Raw market cache | `~/.cache/autoresearch-trading/data/` | `download_manifest.json` |

### What is a data change

Any of these invalidate the current dataset:
- Raw data source change (new download, corrected pkl)
- Feature computation change (`compute_features.py`, `core/features.py`)
- Label computation change (`build_v2_dataset.py` labeling logic)
- Schema change (`chain_data.py` contract features, `config.py` dimensions)
- Sidecar rebuild (new or modified per-day sidecars)

### What must happen after a data change

1. Rebuild dataset: `python -m v2.pipeline.build_v2_dataset`
2. Dataset fingerprint changes automatically
3. All existing model checkpoints are invalidated (trained on old fingerprint)
4. All existing eval reports are invalidated
5. Retrain from scratch before any new evaluation

---

## Stage 2 -- Training Stack

### Current stack

```
supervised (train.py)
  --> sequential BC (train_seq.py)
    --> trajectory collection (collect_trajectories.py)
      --> AWAC (train_awac.py)
```

### What each step emits

| Step | Checkpoint | Log | Fingerprints |
|---|---|---|---|
| Supervised | `models/model.pt` | `runs/{run_id}/training_log.jsonl` | config_fp, dataset_fp, score_fp |
| Sequential BC | `models/seq_agent.pt` | `runs/{run_id}/seq_training_log.jsonl` | config_fp, dataset_fp, encoder_source |
| Trajectory | `trajectories/fold{N}_{label}.pt` | manifest.json alongside | source_checkpoint sha256, config_fp |
| AWAC | `models/seq_agent_awac_fold{N}.pt` | `runs/{run_id}/awac_training_log.jsonl` | config_fp, dataset_fp, trajectory_sha256 |

### Training success vs training completion

A training run **completed** means it ran to the end without crashing. The `training_log.jsonl` has `status: "completed"` in its summary line.

A training run **succeeded** means:
- It completed
- Val loss converged (best epoch is not in the first 20% of training)
- No sub-loss exploded (all components finite)
- Gradient norms stayed bounded
- The resulting checkpoint passes health check validation

Completion is necessary but not sufficient. A completed run with early-best convergence or exploding sub-losses is informative but not promotable.

### Rules

- All training runs on Akash H100 GPU, never locally
- Every experiment trains from scratch (no warm-starting)
- One hypothesis per experiment
- Default mutable surface: `v2/train.py` and `v2/core/policy.py`
- Encoder load failure is a hard error (not a warning)

---

## Stage 3 -- Validation Loop

### How a candidate is judged

1. **Replay**: `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote`
2. **Traces**: `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
3. **Eval report**: saved to `v2/runs/{run_id}/eval_report.json` (stored trades for re-scoring)
4. **Replay diagnostics**: saved to `v2/runs/{run_id}/replay_diagnostics.json` (per-fold breakdown, fill failures, worst trades)
5. **Baselines**: model score compared against all 4 baselines
6. **Plots**: `python3 -m v2.plot_trades` and `python3 v2/plot_progress.py`

### Score formula (v4.0 -- from `metrics.py`)

```
score = (0.5 * sortino_term + 0.5 * pf_term) * positive_day_rate * dd_mult

where:
  sortino_term = min(daily_sortino, 10.0)
  pf_term      = min(profit_factor, 4.0)    # dollar-weighted PF
  dd_mult      = 1.0 if dd <= 12%, linear decay to 0.0 at 25%
```

Score config fingerprint: `score_config_fingerprint()` in `v2/core/metrics.py`.

### Hard gates

| Condition | Score | Gate failure reason |
|---|---|---|
| Total trades < 30 | -1.0 | `too_few_trades` |
| Traded days < 15 | -0.5 | `too_few_traded_days` |
| Max account drawdown > 25% | -0.2 | `excessive_drawdown` |

No other hard gates. Direction balance is diagnostic only (not a gate).

### Canonical vs diagnostic metrics

**Canonical** (used for scoring and promotion):
- Dollar-weighted profit factor
- Daily Sortino ratio
- Positive day rate
- Max account drawdown
- Net dollar P&L

**Diagnostic only** (useful for analysis, not for gating):
- Percentage-weighted profit factor
- Win rate
- Direction balance (call/put split)
- Trades per day
- Exit reason breakdown

### 4 baselines

1. **Random**: random entry bar, random contract
2. **ATM-Always**: enter ATM at first eligible bar every day
3. **Simple-Rules**: rule-based entry using simple thresholds
4. **ATM-Trailing**: ATM entry with trailing stop

A candidate must beat all 4 baselines to be promotable.

---

## Stage 4 -- Promotion Loop

### What is required to keep a candidate

All of the following must be true:

1. Score > 0 (no gate failures)
2. Beats all 4 baselines
3. `training_log.jsonl` exists and has `status: "completed"`
4. `eval_report.json` exists and is valid JSON
5. `replay_diagnostics.json` exists, shares report_id with eval_report
6. All fingerprints match across artifacts (config, dataset, evaluator)
7. Human review of traces and trade analysis

### Keep/revert decision

```
python3 -m v2.ops.model_manage keep    # promotes candidate to model.pt
python3 -m v2.ops.model_manage revert  # discards candidate
```

The decision is informed by traces, not score alone. A high-scoring model with degenerate behavior (all one direction, all one exit type, single-day concentration) should be reverted.

### After keep

1. Regenerate visualizations (`plot_trades`, `plot_progress`)
2. Analyze trades from `v2/output/trades.csv`
3. Update `v2/lab_notebook.md` with full entry
4. Update `v2/results.tsv` with official result
5. Update `v2/docs/current_state.md` snapshot
6. Commit everything in one clean commit
7. Form next hypothesis from trace analysis

### After revert

1. Update `v2/lab_notebook.md` with entry explaining why
2. Update `v2/results.tsv` with revert result
3. Commit
4. Diagnose: use failure packet to understand what went wrong

---

## Stage 5 -- Paper-Trading Readiness Gate

Before a promoted model is eligible for IBKR paper trading:

- [ ] Canonical evaluation artifacts generated from current evaluator fingerprint
- [ ] Promotion packet complete (all required files present and valid)
- [ ] Failure packet workflow tested (at least one bad run triaged via triage_index.json)
- [ ] Run identity/fingerprints intact across full chain (config -> data -> model -> eval -> promotion)
- [ ] Replay outputs reproducible from stored artifacts (re-score eval_report.json trades, get same result)
- [ ] No BLOCKER observability gaps remain open
- [ ] At least one full promotion cycle completed with artifact gate enforcement
- [ ] Kill-switch design ready
- [ ] Trade intent contract stable (`TradeIntent` schema not changing)

---

## What Makes a Run Promotable vs Only Informative

Three tiers:

### Tier 1: Informative

Produced evidence, useful for learning, but NOT eligible for keep.

Examples:
- Screening run (1-fold, not official)
- Run with stale evaluator fingerprint
- Run missing required diagnostics artifacts
- Run with incomplete training (crashed, timed out)
- Run where best epoch was in first 20% (convergence flag: early_best)

Informative runs are still logged in `lab_notebook.md`. They teach you something. They do not earn promotion.

### Tier 2: Promotable

Passes all gates, beats all baselines, artifact packet complete, fingerprints valid, human approves.

Requirements:
- Official 5-fold run completed
- Score > 0 on all folds (or aggregate)
- Beats all 4 baselines
- Promotion packet complete (see below)
- No fingerprint mismatches
- Human reviewed traces and trade analysis
- Behavior is not degenerate (not all one side, not all one exit type)

### Tier 3: Paper-trade eligible

Promotable + all Stage 5 readiness gates satisfied.

The distinction matters because most runs are informative. A run that teaches something is not the same as a run that earns promotion. A promoted model is not automatically safe to paper trade.

---

## When to Stop and Investigate Instead of Hill Climb

Mandatory pause conditions -- do NOT start next experiment if any are true:

- Evaluator fingerprint changed since last promoted model
- Dataset fingerprint changed since last training run
- Key docs disagree on score formula, gates, or training stack
- Promotion packet from last kept run is incomplete
- Replay diagnostics missing for current model
- Known BLOCKER in observability gaps unresolved
- Health checks failing (`python3 -m v2.ops.health` reports errors)
- Build report shows NaN rate >20% on any feature
- 6 consecutive no-improve experiments (session limit)

These are not suggestions. They are hard stops. Autonomous iteration is only safe when the foundation is verified.

---

## Rerun Matrix

What changes force what reruns:

| Change | Must rerun |
|---|---|
| Raw data (new download, corrected pkl) | Rebuild dataset + retrain all |
| Feature computation (`compute_features.py`) | Rebuild dataset + retrain all |
| Label computation (`build_v2_dataset.py`) | Rebuild dataset + retrain all |
| Schema change (`config.py` dimensions) | Rebuild dataset + retrain all |
| Model architecture (`train.py`) | Retrain from scratch |
| Policy change (`policy.py`) | Replay only (no retrain needed for replay-only policy changes) |
| Evaluator change (`metrics.py` scoring) | Re-score existing eval_reports; re-run baselines |
| Simulator change (`simulator.py`) | Replay + re-run baselines |
| Cost model change (spread, commission) | Replay + re-run baselines |

---

## Failure Packet (required for every reverted or failed run)

When a run is reverted or fails, these files must be available for triage:

```
v2/runs/{run_id}/
  triage_index.json            -- single entry point: verdict + pointers to all evidence
  training_log.jsonl           -- per-epoch learning curve with lifecycle status
  replay_diagnostics.json      -- per-fold scores, worst trades, gate rejections
  eval_report.json             -- stored trades for re-scoring

v2/build_reports/{dataset_fp}.json  -- NaN audit + feature stats from build time

v2/artifacts/{exp_id}/
  manifest.json                -- identity + all fingerprints
  model.pt                     -- checkpoint
  policy.json                  -- exit rules
  train.py.snapshot            -- exact code that produced the model
```

**5-minute triage sequence:**
1. Open `triage_index.json` -- verdict, gate failure, pointers to all evidence
2. `training_log.jsonl` -- Did val_loss converge? Which epoch was best? Sub-loss explosion? Gradient blow-up? Did it complete or crash?
3. `replay_diagnostics.json` -- Which folds failed? Gate rejection rate? Worst 5 trades?
4. `build_reports/{dataset_fp}.json` -- NaN rates spike? Label distribution shift?
5. `manifest.json` -- Fingerprints match expectations? Evaluator current?

## Promotion Packet (required for every kept run)

When a run passes all gates and is kept:

```
v2/runs/{run_id}/
  triage_index.json            -- verdict: "promoted", all pointers
  training_log.jsonl           -- proves convergence (status: "completed")
  eval_report.json             -- stored trades for re-scoring
  replay_diagnostics.json      -- fold breakdown, best/worst days

v2/artifacts/{exp_id}/
  manifest.json                -- all fingerprints, score, hyperparams
  model.pt                     -- checkpoint
  policy.json                  -- exit rules
  train.py.snapshot            -- exact training code
  policy.py.snapshot           -- exact policy code
```

**Promotion review checklist:**
1. All fold scores positive and within 1 std of mean
2. No gate failures, beats all 4 baselines
3. `training_log.jsonl` shows clean convergence (best epoch not in first 20%)
4. `replay_diagnostics.json` shows balanced exit reasons, direction balance > 0.3
5. Fill failure count is 0 or negligible
6. Evaluator fingerprint matches current `score_config_fingerprint()`
