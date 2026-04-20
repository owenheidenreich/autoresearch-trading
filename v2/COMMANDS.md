# v2 Commands

When the human says one of these, do it. Read [ART2_LOOP.md](ART2_LOOP.md) for the full protocol.

All training runs are remote on the Akash H100. Local commands are for replay, analysis, plotting, data rebuilds, commits, and docs.

## Research

Harness-integrity repair (2026-04-17) replaced the old `run_screen` / `run_one`
flow with a scope-separated pipeline. CV selects configs; a separate final-train
step produces the only promotable artifact. See
`/Users/gduby/.claude/plans/delightful-yawning-tiger.md` for the rationale.

- `screen latest exp_NNN`
  Single-fold parity debug (matches fold 4 of full CV). No artifact, no results.tsv entry:
  `./v2/ops/deploy.sh run_screen_latest exp_NNN`

- `screen mini exp_NNN`
  Regime triage across early/mid/late windows (folds 0, 2, 4). No artifact:
  `./v2/ops/deploy.sh run_screen_mini exp_NNN`

- `audit direction b labels`
  Check sparse gate-label density before spending GPU:
  `python3 -m v2.analysis.gate_label_audit --mode sparse_high_conviction --screen-mode mini`

- `screen mini direction b`
  Run the current abstention-first screen with the Direction B bundle:
  `TRAIN_ENV="TRAIN_FEATURE_SET=full79 LINEAR_SCORE_HEADS=0 GATE_ARCH=decoupled_mlp OPP_LABEL=sparse_high_conviction GATE_POSITIVE_PNL=0.30 GATE_POSITIVE_MAX_WINNERS=2 GATE_TARGET_MODE=binary SEL_TARGET_MODE=soft_pnl SOFT_TEMP=0.40 POLICY_GATE_THRESHOLD_MODE=quantile POLICY_GATE_TARGET_PASS_RATE=0.10 CKPT_SELECTION_MODE=val_replay" ./v2/ops/deploy.sh run_screen_mini exp_next_b1`

- `run cv experiment exp_NNN`
  Full 5-fold CV. Emits a `CV_EVAL` artifact — NOT a deployable model:
  `./v2/ops/deploy.sh run_cv exp_NNN`

- `run final train exp_NNN`
  Produce a `FINAL_TRAIN` artifact from the chosen CV config. This is the only
  path that produces a model `model_manage keep` will promote:
  `./v2/ops/deploy.sh run_final_train exp_NNN`

- `begin experiment loop`
  Use the full loop from `v2/ART2_LOOP.md`: form one hypothesis, edit the
  approved mutable surface, commit, screen (latest or mini) first, then if the
  screen justifies it run full CV. If CV passes all gates, run final_train,
  inspect the candidate locally, then keep or revert.

## Evaluation

- `evaluate model`
  Run:
  `python3 -m v2.replay --data v2/data.pt --mask promote`

- `evaluate promoted artifact`
  Run:
  `python3 -m v2.replay --data v2/data.pt --mask promote`

- `evaluate on shadow`
  Run:
  `python3 -m v2.replay --data v2/data.pt --mask shadow`

- `analyze trades`
  Run:
  `python3 -m v2.analysis.analyze_losses`

- `audit dataset` or `audit anomalies`
  Run the separate raw/sidecar anomaly track with trace overlap:
  `.venv/bin/python3 -m v2.core.data_integrity --data v2/data.pt --raw-audit --sidecar-audit --trace-path v2/artifacts/replay_traces.csv`

- `audit side bias`
  Run the standard side-bias audit against the executable snapshot labels and a model checkpoint:
  `.venv/bin/python3 -m v2.core.data_integrity --data v2/data.pt --side-bias-audit --model v2/models/model_candidate.pt`

- `harness eval`
  Run:
  `python3 -m v2.analysis.harness_eval --data v2/data.pt`

## Decision Traces

- `trace model` or `decision trace`
  Generate per-bar decision trace for the candidate model (mandatory before keep/revert):
  `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
  Output: `v2/artifacts/replay_traces.csv` + printed summary with gate accuracy, selection accuracy, P&L gap, failure modes.

- `trace artifact` or `trace exp_NNN`
  Trace a specific artifact:
  `python3 -m v2.replay --artifact v2/artifacts/exp_NNN --data v2/data.pt --mask promote --traces`

## Data Integrity

- `validate data` or `check data`
  Run the full data integrity pipeline (manifest, features, sidecars):
  `python3 -m v2.core.data_integrity --data v2/data.pt`

## Data

- `download full chain`
  Run:
  `python3 -m v2.pipeline.download_full_chain`

- `rebuild dataset`
  Rebuild the canonical exact-chain dataset from raw caches:
  1. `python3 -m v2.pipeline.build_v2_dataset --output v2/data.pt --sidecar-dir v2/data_sidecars`
  2. `python3 -m v2.analysis.harness_eval --data v2/data.pt --build`
  3. `python3 -m v2.analysis.harness_eval --data v2/data.pt`

## GPU

- `boot gpu`
  Run:
  `./v2/ops/deploy.sh boot`

- `start gpu`
  Run:
  `./v2/ops/deploy.sh start`
  Uploads code, data.pt, and training-stripped sidecars (replay-only fields removed + float16 downcast + zstd compression, ~518MB vs ~3.5GB full). Full sidecars remain local for replay/analysis.

- `stop gpu`
  Run:
  `./v2/ops/deploy.sh stop`

- `gpu status`
  Run:
  `./v2/ops/deploy.sh status`

## Post-Run Checklists

These are **mandatory**. Do not report results to the user until every step is complete.

### After every screening run (`run_screen_latest` or `run_screen_mini`)

1. Review output: per-fold scores, stability, gate failures, pooled PF/DD/trades
2. Log to `v2/lab_notebook.md`: hypothesis tested, result table, decision (promote to CV / kill / continue family)
3. Commit the lab notebook update
4. If promoting to CV: `./v2/ops/deploy.sh run_cv exp_NNN`

### After every CV run (`run_cv`)

The `run_cv` command emits a `CV_EVAL` artifact. It is NOT deployable.

1. **Inspect the CV_EVAL artifact** at `v2/artifacts/exp_NNN/`:
   - `cv_report.json` — full scope-separated results
   - `folds/<window_id>/model.pt` — per-fold debug checkpoints
2. **Confirm all folds passed hard gates:** `any_gate_failure` must be false.
3. **Log to `v2/lab_notebook.md`:** stability score, pooled metrics, per-fold distribution, decision (final_train / kill / iterate).
4. **If promoting:** `./v2/ops/deploy.sh run_final_train exp_NNN` — trains the deployable model on the full pre-shadow span with a named internal validation slice.

### After every final-train run (`run_final_train`)

1. **Validate model locally:**
   `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote`
   Confirm numbers are consistent with the source CV's pooled metrics.

2. **Run promote trace:**
   `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`

3. **Keep or revert** (informed by trace, not score alone):
   `python3 -m v2.ops.model_manage keep` or `python3 -m v2.ops.model_manage revert`

   `keep` refuses anything that isn't tagged `artifact_kind=FINAL_TRAIN`. CV_EVAL
   artifacts cannot be promoted by design — a fold checkpoint must never become
   the production model.

4. **Regenerate visualizations:**
   `python3 -m v2.plot_trades` → trades.html, equity.html, trades.csv
   `python3 v2/plot_progress.py` → progress.png

5. **Analyze trades:** read `v2/output/trades.csv` and report:
   - Exit reason breakdown (stop-loss, take-profit, trailing, max-hold)
   - Direction split (calls/puts, WR per side)
   - Bar-of-day profitability
   - Average P&L per trade

6. **Update lab notebook** (`v2/lab_notebook.md`):
   Full entry with hypothesis, results table, trace summary, keep/revert decision, next direction.

7. **If promoting, update live docs:**
   - `v2/docs/current_state.md` (snapshot section)

8. **Commit everything** in one clean commit: code, artifacts, models, docs, lab notebook.

9. **Form next hypothesis** from trace analysis (see ART2_LOOP.md).

**Present to user in one message:** results table, trace comparison vs baseline, trade analysis summary, what docs were updated, proposed next hypothesis.

## Health Checks

- `health check` or `check health`
  Full pipeline integrity check (config + data + model + smoke test):
  `python3 -m v2.ops.health`

- `quick health` or `health quick`
  Fast check (config + data + model only, < 5 seconds):
  `python3 -m v2.ops.health quick`

## Monitoring

- `project status`
  Run:
  `python3 -m v2.ops.status_report`

- `plot progress`
  Run:
  `python3 v2/plot_progress.py`

- `plot trades`
  Run:
  `python3 -m v2.plot_trades`

## Analysis Utilities

These are **diagnostic tools**, not canonical pipeline steps. Use for investigation.

**Canonical (used by pipeline/gates):**
- `harness_eval.py` — regression test harness (called by pre_run_gate)
- `policy_sweep.py` — parameter sweep over DecisionPolicy
- `frontier_study.py` — agent frontier analysis (imported by autoresearch.py)

**Manual diagnostics (run when investigating):**
- `fold_diagnosis.py` — per-fold drawdown and metric breakdown
- `flip_day_study.py` — side-flip forensics for sequential agent
- `analyze_losses.py` — losing-day deep dive
- `behavioral_report.py` — trade behavior summary (pure numpy, no v2 imports)
- `grid_study.py` — hyperparameter grid search (subprocess-based)



## Autogenerated harness reference

<!-- AUTOGEN:START harness_commands -->

### CV pipeline (regenerated from code)

**Screening modes** (see `v2.core.walkforward.SCREENING_MODES`):

- `full` — all canonical folds (official cross-validation)
- `latest` — latest fold only (matches fold n-1 of full CV)
- `mini` — folds 0, 2, 4 (early, mid, late regime triage)

**Deploy commands:**

- `./v2/ops/deploy.sh run_screen_latest exp_NNN` — fold 4 only, no artifact
- `./v2/ops/deploy.sh run_screen_mini   exp_NNN` — folds 0, 2, 4, no artifact
- `./v2/ops/deploy.sh run_cv            exp_NNN` — full 5-fold, emits CV_EVAL artifact
- `./v2/ops/deploy.sh run_final_train  SRC_ID`   — train deployable model from chosen CV

**Artifact kinds** (see `v2.core.artifact_kind.ArtifactKind`):

- `cv_eval` — not deployable
- `fold_checkpoint` — not deployable
- `final_train` — DEPLOYABLE

Only `final_train` artifacts are accepted by `model_manage keep`.

**results.tsv schema** (`cv_report_v2`):

| column | meaning |
|---|---|
| `experiment` | experiment id |
| `screening_mode` | full / latest / mini / legacy |
| `status` | revert (default) or keep (after model_manage.keep) |
| `stability_score` | mean fold score across evaluated folds |
| `pooled_pf` | pooled profit factor across all trades from all folds |
| `pooled_dd` | pooled max account drawdown |
| `pooled_trades` | pooled total trades |
| `pooled_traded_days` | pooled days with ≥1 trade |
| `any_gate_failure` | true if any fold gate-failed |
| `per_fold_scores` | list of fold scores |
| `description` | human-readable note |

<!-- AUTOGEN:END harness_commands -->
