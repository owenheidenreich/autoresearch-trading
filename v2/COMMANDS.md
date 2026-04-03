# v2 Command Reference

When the human says one of these commands, do exactly what is described. No improvisation. Read `v2/program.md` for full protocol context.

---

## Research Loop

### `begin experiment loop`
Start the autonomous ART² research cycle. Read `v2/program.md`, create a branch `autoresearch/v2-<tag>`, run baseline, then loop: mutate train.py or policy.py, commit, run experiment, keep/revert, log. Stop when session limit fires.

### `run experiment <id>`
```bash
python v2/ops/run_experiment.py --id <id> > run.log 2>&1
grep "^score:" run.log
```
Trains model, replays on promote_mask, compares baselines, saves artifact. One command, one score.

### `keep`
Score improved and beats all baselines. Branch advances. Update results.tsv with status=keep.

### `revert`
Score did not improve. Undo mutations:
```bash
git checkout v2/train.py v2/core/policy.py
```
Log to results.tsv with status=discard.

### `check session status`
```bash
python v2/ops/monitor.py
```
Shows experiment count, best score, streaks, time remaining, fingerprints.

### `watch session`
```bash
python v2/ops/monitor.py --watch
```
Auto-refresh dashboard every 30 seconds.

---

## Evaluation

### `evaluate model`
```bash
python -m v2.replay --model v2/model.pt --mask promote
```
Replay current model on promote_mask (60 held-out days). Shows score, sortino, P&L, baselines.

### `evaluate on shadow`
```bash
python -m v2.replay --model v2/model.pt --mask shadow
```
Evaluate on shadow_mask (20 days). Only for live-readiness checks. Never used for promotion.

### `evaluate on val`
```bash
python -m v2.replay --model v2/model.pt --mask val
```
Evaluate on val_mask (60 checkpoint-selection days). Diagnostic only.

### `compute baselines`
```bash
python -m v2.replay --baselines --mask promote
```
Compute Random, ATM-Always, Simple-Rules baselines on promote_mask. No model needed.

### `evaluate with gate <threshold>`
```bash
python -m v2.replay --model v2/model.pt --mask promote --gate 0.7
```
Override gate threshold to test selectivity sensitivity.

---

## Data Pipeline

### `rebuild dataset`
```bash
python -m v2.pipeline.build_dataset --tier 3
```
Rebuild v2/data.pt with Tier 3 oracle labels and 4-way split. Takes 30-60 min on CPU. Destroys existing data.pt.

### `rebuild dataset tier 1`
```bash
python -m v2.pipeline.build_dataset --tier 1
```
Fast rebuild (~5 sec) with fixed-risk labels. For quick smoke tests only.

### `inspect dataset`
```python
import torch
d = torch.load('v2/data.pt', map_location='cpu', weights_only=False)
print('Keys:', list(d.keys()))
print('Metadata:', d.get('metadata', {}))
print('X shape:', d['X'].shape)
print('Train bars:', d['train_mask'].sum().item())
print('Val bars:', d['val_mask'].sum().item())
print('Promote bars:', d['promote_mask'].sum().item())
print('Shadow bars:', d['shadow_mask'].sum().item())
```

---

## Training

### `train model`
```bash
python -m v2.train
```
Train on train_mask, checkpoint on val_mask. Saves to v2/model.pt. Uses defaults from train.py.

### `train with custom hyperparams`
```bash
TRAIN_LR=5e-4 TRAIN_EPOCHS=50 TRAIN_BATCH_SIZE=2048 python -m v2.train
```
All hyperparams tunable via env vars: TRAIN_LR, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_D_MODEL, TRAIN_DEPTH, TRAIN_DROPOUT, TIME_BUDGET, WEIGHT_GATE, WEIGHT_DIR, WEIGHT_STRIKE, WEIGHT_RISK, WEIGHT_GATE_POS.

---

## Sweeps

### `run gpu sweep`
```bash
python -m v2.ops.gpu_sweep
```
Grid search across multiple configs (gate weight, LR, architecture, dropout). Ranks by score.

### `run fast sweep`
```bash
python -m v2.ops.fast_sweep
```
Inline sweep on GPU. Lower overhead than gpu_sweep.

---

## Session Management

### `fresh start`
Reset all experiment state. Use after structural changes (new data, new architecture).
```bash
rm -f v2/model.pt v2/.best_score v2/.inner_loop_state.json v2/results.tsv
```

### `init session`
Initialize session state for a new experiment run:
```python
from v2.ops.inner_loop import init_session
state = init_session()
```

### `check best score`
```bash
cat v2/.best_score
```

### `view results`
```bash
cat v2/results.tsv
```

---

## Artifacts

### `list artifacts`
```bash
ls v2/artifacts/
```

### `inspect artifact <id>`
```bash
cat v2/artifacts/<id>/manifest.json
```
Shows git SHA, fingerprints, score, hyperparams, timestamp.

### `load artifact <id>`
```python
from v2.ops.artifact import load_artifact
bundle = load_artifact('v2/artifacts/<id>', current_dataset_fingerprint='...')
model = bundle['model']
policy = bundle['policy']
```
Hard-fails on fingerprint mismatch. Use for reproducible evaluation.

---

## Diagnostics

### `analyze trades`
After running replay, inspect trade-level data:
```python
from v2.replay import load_model, replay_validation
import torch
data = torch.load('v2/data.pt', map_location='cpu', weights_only=False)
model = load_model('v2/model.pt')
metrics, trades = replay_validation(model, data, mask_key='promote_mask')
# trades is a list of SimulatedTrade objects
for t in trades[:10]:
    print(f"{t.trade_date} {t.intent.right} pnl={t.net_pnl_pct:.1%} exit={t.exit_reason} held={t.bars_held}bars")
```

### `check fingerprints`
```python
from v2.core.metrics import score_config_fingerprint
from v2.core.policy import DEFAULT_POLICY
import torch
data = torch.load('v2/data.pt', map_location='cpu', weights_only=False)
print(f"Score: {score_config_fingerprint()}")
print(f"Policy: {DEFAULT_POLICY.fingerprint()}")
print(f"Dataset: {data['metadata']['fingerprint']}")
```

### `verify determinism`
Run replay twice and verify identical scores:
```python
from v2.replay import load_model, replay_validation
import torch
data = torch.load('v2/data.pt', map_location='cpu', weights_only=False)
model = load_model('v2/model.pt')
m1, _ = replay_validation(model, data)
m2, _ = replay_validation(model, data)
assert m1.score == m2.score, f"Non-deterministic: {m1.score} vs {m2.score}"
print(f"Deterministic: score={m1.score}")
```

---

## Live Trading (Phase 5 -- stubs, not yet implemented)

### `begin shadow session`
Run model on live market data, generate intents, but place no orders. Verify intent stream matches what replay would produce. Requires v2/live/ modules to be built.

### `begin paper session`
Run model on live market data with real IBKR paper orders. Full RTH 9:30-16:00 ET. Requires v2/live/ modules and IBKR paper account.

### `kill switch`
Emergency stop all live trading activity immediately. Close all positions.

---

## Quick Reference

| Command | What happens |
|---------|-------------|
| `begin experiment loop` | Start autonomous research cycle |
| `run experiment exp_001` | Train + replay + score + save artifact |
| `keep` / `revert` | Accept or reject the last experiment |
| `evaluate model` | Replay on promote_mask with baselines |
| `rebuild dataset` | Tier 3 labels + 4-way split |
| `train model` | Train on train_mask, checkpoint on val_mask |
| `fresh start` | Reset all experiment state |
| `check session status` | Monitor dashboard |
| `analyze trades` | Inspect trade-level replay data |
