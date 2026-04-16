# Pipeline Overview

One-page map of the SPX 0DTE research system. For detailed docs see [docs/README.md](docs/README.md).

## Stage map

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  1. DATA ACQUISITION                                           │
 │     python -m v2.pipeline.download_full_chain                  │
 │     Output: raw pickle files                                   │
 ├─────────────────────────────────────────────────────────────────┤
 │  2. DATASET BUILD                                              │
 │     python -m v2.pipeline.build_v2_dataset                     │
 │     Output: v2/data.pt (52 features) + v2/data_sidecars/ (22) │
 ├─────────────────────────────────────────────────────────────────┤
 │  3. TRAINING (Akash H100 GPU)                                  │
 │     deploy.sh start strips sidecars for upload (training-only  │
 │     fields, ~880MB vs ~3.5GB full). See v2/ops/strip_sidecars. │
 │     Supervised:  python -m v2.train                            │
 │     Seq BC:      python -m v2.train_seq                        │
 │     AWAC RL:     python -m v2.train_awac                       │
 │     Output: v2/models/model.pt                                 │
 ├─────────────────────────────────────────────────────────────────┤
 │  4. VALIDATION                                                 │
 │     python -m v2.replay --model v2/models/model.pt --mask promote
 │     Output: ReplayMetrics, eval_report.json, trades, traces    │
 ├─────────────────────────────────────────────────────────────────┤
 │  5. PROMOTION                                                  │
 │     python -m v2.ops.model_manage keep exp_NNN                 │
 │     Gates: dollar PF>1, DD<25%, beats baselines, human review  │
 │     Output: v2/artifacts/exp_NNN/ bundle                       │
 ├─────────────────────────────────────────────────────────────────┤
 │  6. PAPER TRADING (deferred)                                   │
 │     TradeIntent -> IBKR API                                    │
 │     Requires: validated checkpoint, kill switches, shadow mode │
 └─────────────────────────────────────────────────────────────────┘
```

## Source-of-truth files

| What | Where |
|------|-------|
| Dataset | `v2/data.pt` + `v2/data_sidecars/` |
| Current model | `v2/models/model.pt` |
| Experiment results | `v2/results.tsv` |
| Experiment narrative | `v2/lab_notebook.md` |
| Shared constants | `v2/core/config.py` (RuntimeConfig) |
| Trade contract | `v2/core/schema.py` (TradeIntent, SimulatedTrade) |
| Risk policy | `v2/core/policy.py` (DecisionPolicy) |
| Feature definitions | `v2/core/features.py` (52 FEATURE_NAMES) |
| Score formula | `v2/core/metrics.py` (dollar-weighted PF) |
| Eval artifacts | `v2/output/eval_report.json` |

## Artifact flow

```
GPU training run
  -> v2/models/model_candidate.pt
  -> replay + trace review
  -> keep/revert decision
  -> v2/artifacts/exp_NNN/  (manifest.json, model.pt, policy.json, eval_report.json, snapshots)
  -> promoted model becomes v2/models/model.pt
  -> results.tsv + lab_notebook.md updated
  -> trades.html + equity.html regenerated
```

## Health checks

```bash
python -m v2.ops.health quick    # config + data + model (< 5s)
python -m v2.ops.health          # full check including 1-day smoke test
python -m v2.ops.health model    # just check model compatibility
```

## Key contracts between stages

| Boundary | Contract | Validated by |
|----------|----------|-------------|
| Data -> Training | `data.pt["metadata"]` must match RuntimeConfig | `train.py` startup assertion |
| Training -> Replay | Checkpoint `hyperparams` must match RuntimeConfig and dataset fingerprint | `replay.py` main() warning |
| Replay -> Promotion | `eval_report.json` stores evaluator fingerprint + raw trades | Health check detects stale evaluator |
| Any change | RuntimeConfig fingerprint changes -> downstream artifacts flagged stale | `python -m v2.ops.health` |
