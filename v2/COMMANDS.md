# v2 Commands

When the human says one of these, do it. Read [program.md](program.md) for the full protocol.

All training runs are remote on the Akash H100. Local commands are for replay, analysis, plotting, data rebuilds, commits, and docs.

## Research

- `run screening experiment exp_NNN`
  Screen first (1-fold, no artifacts):
  `./v2/ops/deploy.sh run_screen exp_NNN`

- `run official experiment exp_NNN`
  Official 5-fold scored run:
  `./v2/ops/deploy.sh run_one exp_NNN`

- `begin experiment loop`
  Use the full loop from `v2/program.md`: form one hypothesis, edit the approved mutable surface for that hypothesis, commit, screen first, then if screening passes or justifies a same-family follow-up run official, then keep or revert.

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

- `stop gpu`
  Run:
  `./v2/ops/deploy.sh stop`

- `gpu status`
  Run:
  `./v2/ops/deploy.sh status`

## Post-Run Checklists

These are **mandatory**. Do not report results to the user until every step is complete.

### After every screening run (`run_screen`)

1. Review output: direction balance, gate failure, score, trade count, WR, PF, DD
2. Log to `v2/lab_notebook.md`: hypothesis tested, result table, decision (promote to official / kill / continue family)
3. Commit the lab notebook update
4. If promoting to official: `./v2/ops/deploy.sh run_one exp_NNN`

### After every official run (`run_one`)

1. **Validate model locally:**
   `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote`
   Confirm numbers match remote output.

2. **Run promote trace:**
   `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
   Compare trace diagnostics (gate acc, selection acc, delta gap, direction) vs current best.

3. **Keep or revert** (informed by trace, not score alone):
   `python3 -m v2.ops.model_manage keep` or `python3 -m v2.ops.model_manage revert`

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
   - `v2/HANDOFF.md` (current live code, research position, key findings)
   - `v2/program.md` (current status section)
   - `v2/docs/current_state.md` (snapshot section)

8. **Commit everything** in one clean commit: code, artifacts, models, docs, lab notebook.

9. **Form next hypothesis** from trace analysis (see program.md Trace-Informed Hypothesis Formation).

**Present to user in one message:** results table, trace comparison vs baseline, trade analysis summary, what docs were updated, proposed next hypothesis.

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
