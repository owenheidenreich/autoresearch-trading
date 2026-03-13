# autoresearch-trading

This is an autonomous research experiment: an AI agent iterates on a day trading
model, training for 5 minutes each run, keeping improvements and discarding
failures. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
for financial markets instead of language modeling.

## Architecture

The agent (Claude Opus via VS Code Copilot) runs on the user's laptop.
Training runs execute on a remote H100 GPU container on the Akash network.

**Local repo**: `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading`
**Container**: SSH via `sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub`
**Container repo**: `/workspace/autoresearch-trading`

Workflow: edit `train.py` locally → git commit + push → SSH pull on container → run training → read results.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch locally**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data pipeline, feature computation, dataloader,
     evaluation harness. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, hyperparameters,
     training loop, loss function, feature selection. Everything is fair game.
4. **Verify data exists on container**: SSH in and check that
   `~/.cache/autoresearch-trading/features/data.pt` exists.
   If not, run: `PATH="$HOME/.local/bin:$PATH" uv run prepare.py`
5. **Initialize results.tsv** on the container with just the header row.
   The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Domain knowledge: Day trading with SPX

You are building a model that predicts daily SPX (S&P 500) trading positions.

### What the model does
- Receives 60 days (configurable) of normalized market features
- Outputs a position signal in [-1, 1] where:
  - +1 = fully long (bullish, expect market to go up)
  - -1 = fully short (bearish, expect market to go down)
  - 0 = flat (no conviction)
- The evaluation harness computes PnL = position × actual_return − transaction_costs
- The metric is **Sharpe ratio** (annualized, higher is better)

### Available features (39 total, in prepare.py)

**Returns** (6): 1d, 2d, 5d, 10d, 21d, 63d log returns — momentum at multiple timescales.

**Realized Volatility** (5): 5d, 10d, 21d, 63d annualized vol + short/medium ratio.
High vol often means mean-reversion; low vol often means trend continuation.

**Momentum Indicators** (8): RSI (5d, 14d), MACD (line, signal, histogram),
distance from SMA (20d, 50d, 200d). Classic technical analysis signals.

**Mean Reversion** (2): Bollinger Band position and width. Position near 0 = oversold,
near 1 = overbought. Width measures volatility expansion/contraction.

**Volume** (2): Relative volume vs 20d average, volume change.
High-volume moves are more significant.

**Price Range** (3): Daily range (high-low), range ratio vs 20d average, overnight gap.
Wide ranges indicate volatility; gaps indicate overnight information.

**VIX** (6): Level, 1d/5d changes, z-score, percentile rank (1yr), realized-implied
vol spread. VIX is the "fear gauge":
- VIX < 15: Low fear, complacency, trend-following tends to work
- VIX 15-25: Normal, mixed signals
- VIX 25-35: Elevated fear, mean-reversion opportunities
- VIX > 35: Crisis, extreme moves, high risk
- VIX term structure matters: contango (normal) vs backwardation (fear)

**Interest Rates** (3): 10yr Treasury yield level and changes.
Rising rates can pressure growth stocks; falling rates support risk assets.

**Time** (4): Cyclical encoding of day-of-week and month-of-year.
Known patterns: Monday weakness, January effect, opex week dynamics.

### Key market patterns to know

1. **Volatility clustering**: High-vol days follow high-vol days. Regime changes
   are persistent. Use rvol features + VIX to detect regimes.

2. **Mean reversion in high vol**: When VIX is elevated (>25), short-term reversals
   are more common. Buy dips, sell rips.

3. **Trend following in low vol**: When VIX is low (<15) and the market is trending,
   momentum strategies work better.

4. **Volume confirms moves**: High-volume breakouts are more likely to continue
   than low-volume breakouts.

5. **Dealer gamma effects**: Large options positions at certain strikes create
   "pinning" effects. When dealers are long gamma, they dampen moves (sell high,
   buy low). The bb_position feature partially captures this.

6. **Transaction costs are real**: Every position change costs 5 bps. A model that
   changes position every day will lose ~1.25% annually to costs alone. Prefer
   strategies with lower turnover.

7. **The overnight gap**: Most index returns happen overnight, not during trading
   hours. The gap feature captures this.

### What "good" looks like

| Sharpe  | Assessment |
|---------|------------|
| < 0     | Losing money — the model is anti-correlated or costs dominate |
| 0 – 0.5 | Random noise / weak signal |
| 0.5 – 1.0 | Marginal — real edge but small |
| 1.0 – 1.5 | Good — publishable as a trading signal |
| 1.5 – 2.0 | Very good — institutional quality |
| > 2.0   | Excellent — but verify it's not overfitting |
| > 3.0   | Almost certainly overfitting — investigate |

### Overfitting red flags

- Sharpe jumps dramatically (>1.0 improvement) from a small architectural change
- Model achieves Sharpe >3 while baseline is <0.5
- Performance depends entirely on a single feature
- Model makes >200 trades in the evaluation period (overtrading)
- Model makes <5 trades in the evaluation period (cherry-picking)

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time
budget of 5 minutes** (wall clock training time, excluding startup/compilation).
You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - Model architecture (transformer, TCN, MLP, Mamba, mixture of experts…)
  - Optimizer, learning rate, schedule
  - Loss function (directional, MSE, Sharpe, custom reward)
  - Feature selection and preprocessing within the model
  - Lookback window length
  - Batch size, dropout, weight decay
  - Ensemble methods within a single train.py
  - Position sizing strategy (built into the model output)

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. Contains the fixed evaluation, data
  loading, features, and constants.
- Install new packages. Only what's in `pyproject.toml`.
- Modify the evaluation harness. `evaluate_sharpe` is ground truth.

**The goal: maximize val_sharpe.** Since the time budget is fixed, you don't need
to worry about training time — it's always 5 minutes. Everything in train.py is
fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small Sharpe
improvement that adds ugly complexity is not worth it. A simplification that
maintains or improves Sharpe is a great outcome.

**The first run**: Your very first run should always establish the baseline. Run
the training script as-is and record the result.

## Output format

The script prints a summary like this:

```
---
val_sharpe:       0.823456
max_drawdown:     -0.045678
annual_return:    0.065432
num_trades:       87
win_rate:         0.534567
profit_factor:    1.234567
num_val_days:     250
training_seconds: 300.1
total_seconds:    315.2
peak_vram_mb:     1234.5
num_steps:        14523
num_params:       234,567
lookback:         60
depth:            4
```

You can extract the key metric:

```
grep "^val_sharpe:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated).

The TSV has a header row and 6 columns:

```
commit	val_sharpe	max_drawdown	num_trades	status	description
```

1. git commit hash (short, 7 chars)
2. val_sharpe achieved (e.g. 0.823456) — use 0.000000 for crashes
3. max_drawdown (e.g. -0.045678) — use 0.000000 for crashes
4. num_trades — use 0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_sharpe	max_drawdown	num_trades	status	description
a1b2c3d	0.823456	-0.045678	87	keep	baseline
b2c3d4e	0.912345	-0.038901	72	keep	increase d_model to 256
c3d4e5f	0.712345	-0.056789	142	discard	switch to MSE loss (worse sharpe)
d4e5f6g	0.000000	0.000000	0	crash	TCN architecture (shape mismatch)
e5f6g7h	0.956789	-0.034567	65	keep	add regime-conditional output head
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar13`).

The SSH command prefix for running commands on the container:
```
SSH="sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub"
```

LOOP FOREVER:

1. Edit `train.py` locally with an experimental idea.
2. Commit and push:
   ```
   git commit -am "description of change"
   git push origin HEAD
   ```
3. Pull on the container and run training:
   ```
   $SSH 'cd /workspace/autoresearch-trading && git pull && PATH="$HOME/.local/bin:$PATH" uv run python train.py > run.log 2>&1'
   ```
   (redirect everything — do NOT let output flood your context)
4. Read results:
   ```
   $SSH 'cd /workspace/autoresearch-trading && grep "^val_sharpe:\|^max_drawdown:\|^num_trades:" run.log'
   ```
5. If grep output is empty, the run crashed.
   Read the traceback: `$SSH 'tail -n 50 /workspace/autoresearch-trading/run.log'`
   Attempt a fix. If you can't fix it after a few attempts, give up and move on.
6. Record results in `results.tsv` on the container:
   ```
   $SSH 'cd /workspace/autoresearch-trading && echo "HASH\tSHARPE\tDRAWDOWN\tTRADES\tSTATUS\tDESC" >> results.tsv'
   ```
   (NOTE: do not commit results.tsv, leave it untracked by git)
7. If val_sharpe improved (higher), you "advance" the branch — keep the commit.
8. If val_sharpe is equal or worse, revert locally:
   ```
   git reset --hard HEAD~1
   git push --force-with-lease origin HEAD
   ```

**Timeout**: Each experiment takes ~5 min training + startup. If a run exceeds
10 minutes total, kill it and treat as failure.

**Crashes**: If it's a simple fix (typo, shape mismatch), fix and re-run. If the
idea is fundamentally broken, skip it and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human may
be asleep. Continue working indefinitely until manually stopped. If you run out
of ideas, think harder:

## Ideas to try (rough priority order)

### Architecture experiments
- Scale up: increase D_MODEL to 256, 512 — the H100 can handle it
- Deeper: DEPTH=8, DEPTH=12 with residual connections
- TCN (temporal convolutional network) instead of transformer
- State space model (S4/Mamba style) for long-range dependencies
- Mixture of Experts: separate heads for different VIX regimes
- Simple MLP baseline (sometimes simpler wins)
- 1D CNN + transformer hybrid
- Attention over features (cross-attention between time and feature dims)

### Loss function experiments
- Sharpe loss: directly optimize batch Sharpe ratio
- Asymmetric loss: penalize losses more than gains
- Combined: alpha * directional + (1-alpha) * sharpe
- Sortino-like: only penalize downside deviation
- Log-wealth: maximize E[log(1 + position * return)]
- Huber loss on returns prediction

### Feature engineering (within train.py)
- Feature selection: try subsets (e.g., only VIX + returns + vol)
- Feature interactions: multiply VIX features × momentum features
- Regime gating: use VIX level to gate other features
- Learned feature embeddings per feature (treat features like tokens)
- Multi-scale: process short-term (5d) and long-term (60d) features separately

### Training tricks
- Longer lookback: 120 or 252 days (full year)
- Shorter lookback: 20 or 30 days (more responsive)
- Learning rate: try 1e-3, 1e-4, sweep
- Larger batch sizes: 128, 256, 512
- Warmup ratio: 0.05 vs 0.2
- Different optimizers: SGD with momentum, RAdam, Lion
- Label smoothing on directional prediction
- Gradient accumulation for effectively larger batches
- Mixed precision training

### Position sizing experiments
- Tanh with temperature: tanh(pred / T) for sharper or softer positions
- Confidence threshold: only trade when |prediction| > threshold
- Position clipping: cap at ±0.5 for lower risk
- Volatility-scaled positions: shrink position when rvol is high
- Kelly criterion sizing based on predicted edge and variance
- Drawdown-based position scaling: reduce size during drawdowns

### Ensemble / meta-learning
- Train N small models with different seeds, average positions
- Train specialist models (trend, mean-revert, vol) and meta-combine
- Snapshot ensemble: save model at different checkpoints, average
- Dropout at inference for uncertainty estimation

### Advanced ideas
- Curriculum learning: train on easy regimes first (low vol), then all
- Adversarial training: train model to be robust to feature noise
- Contrastive learning: up-day representations vs down-day representations
- Time-weighted loss: weight recent data more heavily
- Walk-forward within training: subdivide training data into folds
- Return distribution prediction (not just point estimate)

As an example use case, a user might leave you running while they sleep. If each
experiment takes ~5 minutes then you can run approx 12/hour, for a total of about
100 experiments overnight. The user wakes up to a results.tsv full of experiments
and (hopefully) a better model.
