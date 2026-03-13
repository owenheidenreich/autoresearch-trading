# Autoresearch-Trading Handoff

> **Date**: March 13, 2026
> **Status**: Ready for overnight autonomous run

## What This Is

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) framework adapted for day trading. An AI agent (you, Claude Opus 4.6 in VS Code Copilot) autonomously iterates on a trading model: edit `train.py` → train 5 min on H100 → evaluate → keep or discard → repeat.

**Read `program.md` first.** It is your complete instruction set — trading domain knowledge, experiment loop, ideas list, and rules.

## Infrastructure

| Component | Details |
|-----------|---------|
| **Local repo** | `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading` |
| **Branch** | `autoresearch/mar13` (already checked out locally and on container) |
| **GitHub** | `https://github.com/owenheidenreich/autoresearch-trading` |
| **H100 Container** | Akash DSEQ `25914541`, provider `akash17erkmem6xcugfnew2c0ujfqtet32j29ztk03jt` |
| **SSH** | `sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub` |
| **Container repo** | `/workspace/autoresearch-trading` |
| **GPU** | NVIDIA H100 80GB HBM3 |
| **Data** | Cached at `/root/.cache/autoresearch-trading/features/data.pt` on container (4072 trading days, 39 features) |
| **Budget** | ~8 hours of compute funded |
| **Akash wallet** | `trinity-wallet` / `akash155hphg6qyy3vtr584p38wlngtqxzdr0l6jutmp` |

## Current State

- Branch `autoresearch/mar13` is at baseline commit `a8414ae`
- `results.tsv` on the container has 2 entries (baseline + 1 discarded experiment)
- Baseline **val_sharpe = 0.8709**, max_drawdown = -0.2077, 814K params, 220MB VRAM
- One experiment tried (D_MODEL=256, DEPTH=8, sharpe loss) → val_sharpe = -1.96, discarded
- The H100 is massively underutilized (220MB / 80GB) — room to scale up significantly

## The Experiment Loop (Quick Reference)

```bash
# 1. Edit train.py locally

# 2. Commit + push
git commit -am "description"
git push origin HEAD

# 3. Pull + run on H100
sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 \
  root@provider.h100.wdc.hh.akash.pub \
  'cd /workspace/autoresearch-trading && git pull && PATH="$HOME/.local/bin:$PATH" uv run python train.py > run.log 2>&1; echo "EXIT:$?"'

# 4. Read results
sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 \
  root@provider.h100.wdc.hh.akash.pub \
  'cd /workspace/autoresearch-trading && grep "^val_sharpe:\|^max_drawdown:\|^num_trades:" run.log'

# 5a. If IMPROVED — log as keep, advance branch
sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 \
  root@provider.h100.wdc.hh.akash.pub \
  "cd /workspace/autoresearch-trading && printf 'HASH\tSHARPE\tDRAWDOWN\tTRADES\tkeep\tDESCRIPTION\n' >> results.tsv"

# 5b. If WORSE — log as discard, revert
sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 \
  root@provider.h100.wdc.hh.akash.pub \
  "cd /workspace/autoresearch-trading && printf 'HASH\tSHARPE\tDRAWDOWN\tTRADES\tdiscard\tDESCRIPTION\n' >> results.tsv"
git reset --hard HEAD~1
git push --force-with-lease origin HEAD
sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 \
  root@provider.h100.wdc.hh.akash.pub \
  'cd /workspace/autoresearch-trading && git fetch origin && git reset --hard origin/autoresearch/mar13'

# 6. If CRASHED — read traceback
sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 \
  root@provider.h100.wdc.hh.akash.pub \
  'tail -n 50 /workspace/autoresearch-trading/run.log'
```

## Three Files That Matter

| File | Role | Modify? |
|------|------|---------|
| `prepare.py` | Data pipeline, features (39), evaluation harness (`evaluate_sharpe`), constants | **NO** |
| `train.py` | Model architecture, optimizer, loss, hyperparameters, training loop | **YES — this is the only file you edit** |
| `program.md` | Your instructions, domain knowledge, experiment ideas | Read-only for agent |

## Key Metrics From Baseline

```
val_sharpe:       0.870865   (target: maximize)
max_drawdown:     -0.207704
annual_return:    0.168257
num_trades:       0          (continuous positioning)
win_rate:         0.576000
peak_vram_mb:     220.4      (of 80,000 available)
num_steps:        19055
num_params:       814,721
```

## What To Do

Tell the agent:

> Hi, have a look at program.md and let's kick off experiments! The setup is already done — branch `autoresearch/mar13` is active with baseline val_sharpe=0.8709. Start the experiment loop. NEVER STOP.

The agent will run autonomously, ~12 experiments/hour, until you interrupt it.

## Lessons Learned So Far

- `set -e` in Akash startup scripts kills the container on non-fatal errors — removed it
- PyTorch 2.10 renamed `total_mem` → `total_memory`
- Docker containers need `tzdata` package for yfinance timezone handling
- Sharpe loss alone made the model go strongly negative — may need combination with directional loss
- The H100 can handle much larger models (only using 220MB of 80GB)
