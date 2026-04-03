# ART² v2 Program

You are an autonomous researcher improving a trading model that trades SPX 0DTE long options. The model learns to emit TradeIntents (trade/no-trade, direction, strike, stop, target, hold). Replay simulates those intents against historical data. Score measures the equity curve. Read this file, then start experimenting.

For command syntax, see `v2/COMMANDS.md`. For detailed specs, see `v2/docs/`.

## Compute

**All training runs on Akash H100 GPU.** Never run training locally -- the dev machine is a MacBook. Local use is limited to: editing code, committing, reading results, and running replay/evaluation (which is CPU-friendly).

Akash workflow:
1. Boot GPU: `./v2/ops/deploy.sh boot`
2. Upload code + data: `./v2/ops/deploy.sh start`
3. Training runs on the remote GPU
4. Download results: `./v2/ops/deploy.sh sync`
5. Evaluate locally: `python -m v2.replay --model v2/model.pt --mask promote`
6. Shut down GPU: `./v2/ops/deploy.sh stop`

The experiment runner (`v2/ops/run_experiment.py`) is designed to run ON the GPU machine, not locally.

## Setup

1. **Create a branch**: `git checkout -b autoresearch/v2-<tag>` from current main.
2. **Read the in-scope files**:
   - This file (`v2/program.md`) -- your instructions.
   - `v2/train.py` -- the model and training loop. You modify this.
   - `v2/core/policy.py` -- the trading policy. You can modify this too.
   - `v2/lab_notebook.md` -- experiment log.
3. **Verify data**: `v2/data.pt` must exist and be Tier 3 (check metadata).
4. **Boot Akash GPU** and establish baseline by running `run_experiment.py --id baseline` on the GPU without changing any code.
5. **Record baseline** in `v2/results.tsv`.

## What You CAN Modify

Two files only:

- **`v2/train.py`** -- Model architecture, loss function, hyperparameters, optimizer, batch construction, head design. Everything about how the model learns.
- **`v2/core/policy.py`** -- Gate threshold, risk output ranges, cooldown bars, time blocks, order style, exit policy. Everything about how model outputs become trading decisions.

## What You CANNOT Modify

Everything else. These are the immutable evaluation harness:

- `v2/core/simulator.py` -- how trades play out
- `v2/core/metrics.py` -- how score is computed
- `v2/replay.py` -- how model outputs become trades and get evaluated
- `v2/core/labels.py` -- how oracle labels are generated
- `v2/core/schema.py` -- TradeIntent and SimulatedTrade contracts
- `v2/data.pt` -- the dataset
- `v2/ops/run_experiment.py` -- the experiment runner
- `v2/ops/inner_loop.py` -- session limits and keep/revert logic

## The Goal

**Get the highest score.** The score is:

```
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

Where:
- `daily_sortino` = Sortino ratio from daily dollar returns on a $10K equity curve
- `positive_day_rate` = fraction of traded days that were profitable
- `dd_mult` = 1.0 when max drawdown <= 8%, linear decay to 0.0 at 20%

Hard gates (score goes negative if any fail):
- Minimum 30 trades
- Minimum 15 traded days
- At least 15% minority direction (must trade both calls and puts)
- Max account drawdown <= 20%

Model must also beat all three baselines:

- **Random**: 2% chance to trade per bar, random candidate, fixed stop=30%/target=50%/hold=120. Averaged over 20 seeds.
- **ATM-Always**: Buy 1 ATM call at bar 30 every day. Fixed stop=30%/target=50%/hold=120.
- **Simple-Rules**: Buy call on +momentum (>0.5%), put on -momentum. ATM, stop=25%/target=40%/hold=60. 10-bar cooldown.

## Running an Experiment

On the Akash GPU:
```bash
python v2/ops/run_experiment.py --id exp_NNN > run.log 2>&1
grep "^score:" run.log
```

The script trains the model, replays on `promote_mask` (60 held-out days the model never trained on), compares against baselines, and saves an artifact bundle.

## Output Format

```
---
score:                2.345678
daily_sortino:        3.12
positive_day_rate:    0.72
max_account_drawdown: 0.06
net_pnl_dollars:      4230.50
total_trades:         187
traded_days:          48
profit_factor:        2.31
win_rate:             0.58
beats_random:         true
beats_atm:            true
beats_rules:          true
training_seconds:     301.2
status:               complete
```

## Logging Results

Log to `v2/results.tsv` (tab-separated):

```
experiment	score	status	description
baseline	-1.000000	keep	initial baseline
exp_001	0.500000	keep	increased LR to 5e-4
exp_002	0.300000	discard	switched to GeLU (worse)
exp_003	0.000000	crash	OOM on batch_size=4096
```

Do NOT commit results.tsv. Leave it untracked.

## The Experiment Loop

LOOP:

1. Look at git state and last experiment results.
2. Decide what to try. Write your hypothesis.
3. Edit `v2/train.py` and/or `v2/core/policy.py`.
4. `git commit` your changes.
5. Upload to Akash and run the experiment on GPU.
6. Download results. Read score.
7. If crashed: read the log, try to fix. If unfixable, log as crash, move on.
8. Log results to `v2/results.tsv`.
9. If score improved AND beats all baselines: **KEEP**. Branch advances.
10. If score equal or worse: **REVERT**. `git checkout v2/train.py v2/core/policy.py`
11. Check session limits (see below). If any limit hit, stop.
12. Go to step 1.

## Session Limits

These are hard ceilings enforced by the orchestrator:

| Limit | Threshold | Action |
|-------|-----------|--------|
| Experiments | 50 max | Stop session |
| Time | 6 hours | Stop session |
| No-improve streak | 8 consecutive reverts | Stop, rethink approach |
| Plateau | 3 hours without improvement | Stop session |
| Crash storm | 3 consecutive crashes | Stop, fix infrastructure |

When stopped: log findings to `v2/lab_notebook.md`, summarize what worked, propose next directions, wait for human review.

## When You're Stuck

If you hit 3+ consecutive reverts:

1. **Stop trying random things.**
2. Read the trade-level replay data. Look at which trades lost money and why.
3. Form a hypothesis about WHY the model is failing.
4. Try structural changes, not just hyperparameter tweaks.

## Key Architecture Facts

- **Input**: (batch, 60, 39) -- 60 bars of 39 normalized features
- **Output**: TradeIntent fields (gate, direction, strike_offset, stop, target, hold, confidence)
- **Oracle labels**: Tier 3 -- searched 6 stops x 5 targets x 5 holds x all strikes x call+put
- **Evaluation**: Replay on promote_mask (60 days model never saw during training)
- **Score**: Account curve health (Sortino * consistency * drawdown guard)
- **Equity**: $10K starting, $100 SPX multiplier, 1 contract max
- **Training**: Akash H100 GPU. 5-minute time budget per experiment. Never local.

## Data Split

| Split | Days | Purpose |
|-------|------|---------|
| train_mask | 859 | Model training |
| val_mask | 60 | Checkpoint selection (best val_loss) |
| promote_mask | 60 | Keep/revert scoring (model never sees this) |
| shadow_mask | 20 | Live-readiness eval only (never used for promotion) |
