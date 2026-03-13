# autoresearch-trading (intraday edition)

Autonomous research experiment: an AI agent iterates on an **intraday SPY options
trading model**, training for 5 minutes each run, keeping improvements and
discarding failures. The end goal: a model that can trade $500 autonomously on
SPY 0DTE/1DTE options during regular trading hours.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Architecture

The agent (Claude Opus via VS Code Copilot) runs on the user's laptop.
Training runs execute on a remote H100 GPU container on the Akash network.

**Local repo**: `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading`
**Container**: SSH via `sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub`
**Container repo**: `/workspace/autoresearch-trading`

Workflow: edit `train.py` locally -> git commit + push -> SSH pull on container -> run training -> read results.

## Setup

1. **Agree on a run tag**: based on today's date (e.g. `mar13-intraday`).
2. **Create the branch locally**: `git checkout -b autoresearch/<tag>`.
3. **Read the in-scope files**:
   - `prepare.py` -- fixed data pipeline (Polygon.io), 71 features, evaluation harness. **Do not modify.**
   - `train.py` -- the file you modify. Everything is fair game.
4. **Set Polygon API key on container**:
   ```
   $SSH 'echo "export POLYGON_API_KEY=YOUR_KEY" >> ~/.bashrc'
   ```
5. **Run data prep on container**:
   ```
   $SSH 'cd /workspace/autoresearch-trading && PATH="$HOME/.local/bin:$PATH" POLYGON_API_KEY=KEY uv run prepare.py'
   ```
   This downloads SPY/VIX/QQQ/SPX 5-min bars + daily context from Polygon.io.
   Takes 5-15 minutes first time (rate limits).
6. **Initialize results.tsv** on the container.
7. **Confirm and go**.

## Data source: Polygon.io

We use **Polygon.io** ($29-99/mo) for all market data:
- SPY 5-min bars (primary instrument)
- VIX 5-min bars (fear gauge intraday)
- QQQ 5-min bars (NQ proxy, risk-on/risk-off)
- SPX index 5-min bars (ES proxy)
- Market internals (TICK, TRIN, A/D where available, synthesized otherwise)
- VIX futures proxy (VIXY ETF)
- Daily bars for SPY, VIX, TNX going back to 2010 (context features)

API key: set `POLYGON_API_KEY` environment variable.

## Domain knowledge: Intraday SPY options trading

### Trader context

17+ years experience. Primarily delta futures and SPX options (long/short & spreads).
Trades ES/NQ/RTY/VIX/CL/SPX/SPY/QQQ, mostly intraday. Goal: 1% portfolio/week.
Style: disciplined, level-to-level, risk-aware, clearly defined entries and exits.

**End goal**: Give a $500 account to an AI model. It trades SPY 0DTE/1DTE options
intraday based on the model's hourly position signals. The position signal maps to:
- **+1**: Buy ATM SPY calls (max conviction bullish)
- **-1**: Buy ATM SPY puts (max conviction bearish)
- **0**: Flat (no position, preserve capital)
- **Fractional**: Scale position size proportionally

### What the model does
- Receives 36 hourly bars (6 trading days) of 71 normalized features
- Outputs a position signal in [-1, 1] every hour during RTH
- Decision points: 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET
- Target: next-hour log return (not next-day)
- Evaluation: hourly PnL = position * return - transaction_costs
- Metric: **Sharpe ratio** (annualized from hourly, higher is better)

### Available features (71 total, in prepare.py)

**Intraday Price** (12): 5m/15m/30m/1h/2h/4h returns, VWAP distance (session VWAP
with +/-1 std bands), bar range, bar range ratio, bar volume ratio.

**Session Structure** (8): Initial Balance high/low distance and width (first 30min),
overnight high/low distance, session range %, session position [0,1], gap from
previous close.

**Intraday Momentum** (10): RSI(14) and RSI(5) on 5m bars, MACD (line/signal/hist)
on 5m, EMA ribbon (8/21/34 on simulated 15m), EMA alignment signal (+1/-1/0),
CCI(20) on 5m, ROC(12) = 1-hour rate of change.

**Market Internals** (8): TICK level and MA distance, TRIN level and extreme flags,
A/D line slope, A/D volume ratio, put/call ratio (placeholder), internals composite.

**Multi-Instrument** (8): ES-SPY basis, NQ/ES ratio change (risk-on/off), VIX level
intraday, VIX session change, VIX term spread (contango/backwardation), VIX term
ratio, TNX session change, ES-NQ correlation.

**Daily Context** (20): Carried forward from daily bars -- returns (1d/5d/21d),
realized vol (5d/21d + ratio), RSI(14), SMA distances (20/50/200), Bollinger
position and width, VIX (level/zscore/percentile/rv-iv spread), TNX (level/change),
volume ratio, gap.

**Time Encoding** (5): Cyclical time-of-day, day-of-week, minutes since open.

### Key intraday patterns

1. **Initial Balance (IB)**: First 30 minutes set the tone. IB range < 0.5% typically
   means range day (mean-revert). IB range > 1% means trend day (momentum). The model
   has `ib_width`, `ib_high_dist`, `ib_low_dist` to detect this.

2. **VWAP is the magnet**: Price tends to revert to VWAP in range days, and VWAP acts
   as support/resistance in trend days. `vwap_dist` is the most important single feature.

3. **Session position**: Where price sits within the session range predicts near-term
   reversion. Position near 0 (session low) or 1 (session high) in a range day = fade.

4. **EMA ribbon alignment**: When 8 > 21 > 34 EMA (or inverted), trend is strong.
   When mixed, market is choppy -- better to be flat (0 position).

5. **Market internals confirm or deny**: TICK > +500 with bullish price = strong.
   TICK diverging from price = warning. TRIN < 0.5 = extreme bullish breadth.
   TRIN > 2.0 = extreme bearish breadth (potential reversal).

6. **VIX intraday**: Rising VIX during session = fear increasing = downside potential.
   Falling VIX = complacency = trend continuation or upside.

7. **Time-of-day effects**:
   - 10:00-10:30: Initial reversal zone (the "10am fake-out")
   - 11:30-13:30: Lunch hour, low volume, choppy (be flat or small)
   - 14:00-15:30: Power hour, volume returns, strong moves
   - 15:30-16:00: Last 30min, close positions

8. **NQ/ES divergence**: When QQQ leads SPY higher or lower, the move is tech-driven.
   When they diverge, it signals rotation (less conviction).

9. **Overnight gap behavior**: Large gaps (>0.5%) tend to fill same day.
   Small gaps tend to signal direction continuation.

10. **Transaction costs are higher for options**: We use 8 bps (vs 5 for equities)
    to account for bid-ask on options. Model should not flip positions every hour.

### Risk management for $500 account

- **Position sizing**: $500 = 1-2 ATM SPY options at any time. Signal magnitude
  determines whether we buy 1 or 2 contracts, or stay flat.
- **Max loss per trade**: ~$100 (20% of account). Stop loss built into option decay.
- **Daily loss limit**: ~$150 (30% of account). Model should go flat after losses.
- **Prefer flat**: In a $500 account, preservation matters more than aggression.
  A model that's flat 40-50% of the time is good.
- **0DTE theta decay**: Options lose value fast. The model needs to have conviction
  within 1 hour or exit. This is why we predict next-hour, not next-day.

### What "good" looks like (intraday)

| Sharpe   | Assessment |
|----------|------------|
| < 0      | Losing money |
| 0 - 0.5  | Noise |
| 0.5 - 1.0 | Marginal edge |
| 1.0 - 1.5 | Good -- tradeable signal with $500 account |
| 1.5 - 2.0 | Very good -- consistent profits, scale up |
| > 2.0    | Excellent -- but verify not overfitting |
| > 3.0    | Almost certainly overfitting |

### Overfitting red flags (intraday)

- Sharpe > 3 while baseline < 0.5
- Model trades on every single bar (no flat time)
- Model makes > 500 trades (overtrading with 8bps costs)
- Model makes < 10 trades (cherry-picking)
- Same position all day every day (not actually trading)
- Performance collapses outside of one specific time window

## Experimentation

Each experiment runs on a single GPU. Fixed **5-minute time budget**.
Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` **only**. Everything is fair game:
  - Architecture (transformer, TCN, MLP, Mamba, MoE...)
  - Optimizer, LR, schedule
  - Loss function
  - Feature selection/preprocessing within the model
  - Lookback window (default: 36 hourly bars)
  - Position sizing strategy

**What you CANNOT do:**
- Modify `prepare.py` (read-only)
- Install new packages
- Modify the evaluation harness

**Goal: maximize val_sharpe.**

## Output format

```
---
val_sharpe:       0.823456
max_drawdown:     -0.045678
annual_return:    0.065432
num_trades:       187
win_rate:         0.534567
profit_factor:    1.234567
num_val_bars:     1500
num_val_days:     250
training_seconds: 300.1
total_seconds:    315.2
peak_vram_mb:     1234.5
num_steps:        14523
num_params:       234,567
lookback:         36
depth:            4
```

Extract: `grep "^val_sharpe:" run.log`

## Logging results

`results.tsv` on the container (tab-separated):

```
commit	val_sharpe	max_drawdown	num_trades	status	description
```

## The experiment loop

SSH prefix:
```
SSH="sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub"
```

LOOP FOREVER:

1. Edit `train.py` locally.
2. Commit and push:
   ```
   git commit -am "description of change"
   git push origin HEAD
   ```
3. Pull and run on container:
   ```
   $SSH 'cd /workspace/autoresearch-trading && git pull && PATH="$HOME/.local/bin:$PATH" uv run python train.py > run.log 2>&1'
   ```
4. Read results:
   ```
   $SSH 'cd /workspace/autoresearch-trading && grep "^val_sharpe:\|^max_drawdown:\|^num_trades:" run.log'
   ```
5. If empty = crash. Read: `$SSH 'tail -n 50 /workspace/autoresearch-trading/run.log'`
6. Log to results.tsv.
7. If improved -> keep. If worse -> revert:
   ```
   git reset --hard HEAD~1
   git push --force-with-lease origin HEAD
   ```

**NEVER STOP.**

## Ideas to try (priority order for intraday)

### Architecture experiments
- **Multi-scale temporal**: Process recent bars (last 6 = today) separately from
  historical bars (last 30 = prior week) with two branches, then combine
- **Time-aware attention**: Weight attention by time-of-day (power hour matters more)
- **Feature-group attention**: Cross-attention between feature groups (internals ×
  price × session structure = confluence)
- Scale up: D_MODEL=256, DEPTH=8 (H100 has plenty of headroom)
- TCN for local patterns + transformer for global
- Mamba/SSM for efficient long-range
- Mixture of Experts: separate heads for range day vs trend day
- Simple GRU (sometimes simpler wins for short sequences)

### Loss function experiments
- **Combined**: 0.7×directional + 0.3×sharpe (SHARPE_ALPHA=0.3)
- Asymmetric loss: penalize losses 2× more than gains (options lose fast)
- Sortino-like: only penalize downside
- Log-wealth: maximize E[log(1 + position × return)]
- Time-weighted: recent training data weighted more
- Huber loss on return prediction

### Feature engineering (in train.py)
- **Feature selection**: Only use intraday + session + internals (drop daily context)
- **Regime gating**: Use ib_width to gate other features (range vs trend day)
- **Confluence detector**: Count how many feature groups agree on direction
- Feature interactions: VIX × momentum, VWAP × session_position
- Learned feature embeddings (treat each feature as a token)
- Feature dropout: randomly zero 20% of features (regularization)

### Lookback experiments
- Shorter: LOOKBACK=12 (2 trading days, more responsive)
- Longer: LOOKBACK=48 (8 trading days, more context)
- LOOKBACK=6 (today only -- pure intraday, no history)
- LOOKBACK=72 (12 days, maximum daily context)

### Position sizing ("Size Matters")
- **Confidence threshold**: Flat when |pred| < 0.2 (force selectivity)
- Position clipping: cap at ±0.5 (conservative, like the real trader)
- Tanh temperature: tanh(pred / T) for sharper/softer
- Volatility scaling: shrink position when intraday vol is high
- Time-of-day scaling: reduce position during lunch (11:30-13:30)
- Drawdown scaling: reduce size after consecutive losses
- Three-state output: dedicated flat zone (dead band)
- Smooth positions: penalize large jumps (|pos_t - pos_{t-1}|)

### Training tricks
- Larger batches: 128, 256, 512
- Learning rate sweep: 1e-3, 1e-4, 5e-4
- Warmup ratio: 0.05 vs 0.2
- RAdam, Lion optimizer
- Gradient accumulation (effective batch=256)
- Mixed precision (bfloat16)
- Label smoothing on sign(return)

### Ensemble / meta
- Train N small models, average positions
- Snapshot ensemble from different checkpoints
- Dropout at inference for uncertainty
- Specialist models (trend/mean-revert/vol) + meta-combine

### Advanced
- **OTF detector**: If recent bars are HH/HL or LL/LH, trend is strong
- **Regime classifier**: Auxiliary head predicts range/trend day, routes to specialist
- **Anti-lunch filter**: Reduce weight on 11:30-13:30 predictions
- Curriculum: Train on clear trend/range days first, then all
- Time-weighted loss: Weight recent training years more heavily
- Adversarial: Add noise to features during training
- Return distribution: Predict mean + variance, use for Kelly sizing

## Mapping model output to live options trades

When the model is eventually deployed with real money:

| Signal range | Action | Size ($500 account) |
|-------------|--------|-------------------|
| pred > +0.5 | Buy ATM SPY call (0DTE/1DTE) | 2 contracts |
| +0.2 < pred < +0.5 | Buy ATM SPY call | 1 contract |
| -0.2 < pred < +0.2 | **FLAT** (no position) | 0 |
| -0.5 < pred < -0.2 | Buy ATM SPY put | 1 contract |
| pred < -0.5 | Buy ATM SPY put (0DTE/1DTE) | 2 contracts |

Close all positions by 15:45 ET. Never hold 0DTE overnight.