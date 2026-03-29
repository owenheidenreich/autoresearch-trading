# Inner Loop Architecture

How the training pipeline works end-to-end: what happens on the GPU, how experiments run, and how models get promoted. This document covers the SETUP → TRAIN → TEARDOWN phases of the ART² lifecycle.

---

## 1. The Big Picture

The inner loop is **purely mechanical** — it has no intelligence. It doesn't know why a model is good or bad. It just runs whatever code it's given, grades the result, and keeps or throws away the model.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         THE TWO LOOPS                                    │
│                                                                          │
│  OUTER LOOP (Opus — the brain, runs between cycles)                      │
│  ═══════════════════════════════════════════════                          │
│  Reads metrics → understands trading behavior → modifies train.py        │
│  Runs once per cycle (~1-2 hours)                                        │
│                                                                          │
│  INNER LOOP (mechanical — the gym, runs during training)                 │
│  ═════════════════════════════════════════════════════                    │
│  Takes train.py as-is → uploads to GPU → trains → scores → keep/revert  │
│  Runs many experiments per cycle (~10-20, each ~6 min)                   │
│  No AI involved. No strategic decisions. Just: run, measure, judge.      │
│                                                                          │
│  Cycle N:  [inner loop trains 10-20 experiments]                         │
│                          ↓                                               │
│            [Opus reads results, modifies train.py]                       │
│                          ↓                                               │
│  Cycle N+1: [inner loop trains the MODIFIED train.py]                    │
│                          ↓                                               │
│            [Opus reads results, modifies train.py]                       │
│                          ...                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### What's on the GPU

Three things get uploaded to the Akash H100:

| File | What it is | Analogy |
|------|-----------|---------|
| `train.py` | Model architecture + training loop (the code) | The lesson plan |
| `best_model.pt` | Current best model weights (warm start) | The student's brain |
| `data.pt` | ~383,000 one-minute bars with 37 features each | The textbook |

---

## 2. What Happens Inside the GPU (Step by Step)

When `inner_loop.py` runs `python train.py` on the H100, this is what happens for ~5 minutes:

### Step 1: Load the Brain

train.py loads `best_model.pt` — the current best model weights. Think of it as a student who already has some knowledge. We're continuing their education, not starting from zero (unless it's a fresh start).

### Step 2: Show It Market Data, Make It Trade

This is the core of training. The loop picks random days from the data and **simulates a full trading day, bar by bar:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SIMULATED TRADING DAY (one of 16 days per batch)                        │
│                                                                          │
│  Bar 0 (9:30 AM):                                                        │
│    Show model 120 bars of history (37 features each)                     │
│    Model outputs from ALL FOUR HEADS simultaneously:                     │
│      Gate head:      "Should I trade?"     → 72% yes                    │
│      Direction head: "Which option?"       → CALL_ATM (45%)             │
│      Value head:     "P&L remaining?"      → +0.15                      │
│      Risk head:      "How much to risk?"   → stop 35%, size 0.3         │
│    Gate says TRADE → we "enter" a CALL_ATM position                     │
│                                                                          │
│  Bar 1 (9:31 AM):                                                        │
│    Model sees updated features + position state:                         │
│      "you're holding, 1 bar in, +2% unrealized P&L"                     │
│    Gate head: 85% yes (keep holding)                                     │
│    Value head: +0.12 remaining                                           │
│    → Keep holding                                                        │
│                                                                          │
│  Bar 2 (9:32 AM):                                                        │
│    Market moved against us: -5% unrealized                               │
│    Gate head: 40% yes (wants out)                                        │
│    → Model exit triggered! Record the -5% loss.                          │
│                                                                          │
│  Bar 3 (9:33 AM):                                                        │
│    Back to flat. 5-bar cooldown after the stop.                          │
│    Gate says TRADE → blocked by cooldown.                                │
│                                                                          │
│  ...                                                                     │
│                                                                          │
│  Bar 389 (4:00 PM):                                                      │
│    Day ends. Flatten anything still open.                                │
│    Record all trades and P&L for this simulated day.                     │
└──────────────────────────────────────────────────────────────────────────┘
```

This happens for **16 random days simultaneously** (a batch). The model makes decisions on real historical data — and we know the right answers because we can see what every option actually did.

### Step 3: Grade It (the Loss Function)

After each batch, we compute **how wrong the model was** across 6 dimensions:

| Loss Term | Weight | What it measures | Plain English |
|-----------|--------|-----------------|---------------|
| **Gate** | 0.5 | Did you trade when profitable? Hold when not? | "Did you say yes at the right time?" |
| **Direction** | 2.5 | Did you pick the best option out of 6? | "Did you pick the right horse?" |
| **P&L alignment** | 0.5 | Did your overall prediction point toward profit? | "Was your gut right?" |
| **Confidence** | 0.10 | Were you sure on winners, unsure on losers? | "Did you know what you knew?" |
| **Value** | 0.3 | While holding, did you predict remaining upside? | "Did you know when to leave?" |
| **Risk** | 0.2 | Did you set the right stop distance and size? | "Did you manage your risk?" |

The "right answers" come from **hindsight** — we know which options were profitable because it's historical data. Direction loss is weighted heaviest (2.5) because picking the right strike matters most.

### Step 4: Adjust the Brain

Standard neural network backpropagation. The loss tells us which neurons were wrong and by how much. We nudge all four heads' weights slightly in the right direction. This is one training "step."

### Step 5: Repeat for 5 Minutes

Steps 2-4 repeat until TIME_BUDGET (default 300 seconds) runs out. Each step processes 16 days. In 5 minutes, the model sees hundreds of simulated trading days.

**85% of batches are day-sequential** — walking through days bar-by-bar, carrying position state (this is how the value and risk heads learn). **15% are random** — scattered bars with no position context (this prevents the model from only learning "what to do when holding" and forgetting "what to do when flat").

### Step 6: Final Exam (Validation)

Training stops. Now the model gets tested on data it **never trained on** (the last 30% of dates). Same bar-by-bar simulation, same rules — but no weight updates. Just measuring. This produces ~40 metrics including profit factor, Sharpe, win rate, drawdown, and exit breakdown.

### Step 7: Score and Judge

Back on the local machine, `inner_loop.py` receives the metrics and computes a single number:

```
score = profit_factor × sharpe × frequency_multiplier × penalties × bonuses
```

If the score beats the previous best AND no anomalies are detected → **KEEP** (new best model). Otherwise → **REVERT** (throw it away, restore old code).

The inner loop doesn't know *why* a model is good or bad. It doesn't know that "high stop rate in the afternoon" means "theta is eating the positions." It just sees numbers go up or down. **That's why Opus exists** — to read those numbers, understand the trading behavior behind them, consult domain knowledge, and make intelligent changes to train.py.

---

## 3. Experiment Flow (Technical)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         OPUS (Strategist)                                │
│                                                                          │
│  Reads: history.jsonl, metrics, trade diagnostics, lab_notebook.md       │
│  Decides: what hypothesis to test next                                   │
│  Calls: inner_loop.py experiment --mutation FILE --summary "hypothesis"  │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     inner_loop.py (Mechanical Layer)                      │
│                                                                          │
│  1. VALIDATE  ─── syntax check + safety check (run_loop.py)             │
│  2. BACKUP    ─── save train.py to artifacts/exp-N/train_before.py      │
│  3. APPLY     ─── write mutation to train.py (skip if baseline)         │
│  4. UPLOAD    ─── SCP train.py + best_model.pt → Akash H100            │
│  5. TRAIN     ─── SSH: TIME_BUDGET=300 python train.py                  │
│  6. DOWNLOAD  ─── SCP model_candidate.pt ← Akash (temp file)           │
│  7. PARSE     ─── extract METRICS_JSON from stdout                      │
│  8. SCORE     ─── anomaly detection + score comparison                  │
│  9. KEEP/REVERT                                                          │
│       KEEP:   model_candidate.pt → best_model.pt                        │
│               train.py → best_train.py                                   │
│               score → .best_score                                        │
│       REVERT: restore train.py from backup, best_model.pt unchanged     │
│  10. RECORD   ─── experiments.v2.jsonl + history.jsonl + status.json    │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                          Opus reads result,
                          diagnoses behavior,
                          loops to next experiment
```

---

## 4. What Goes In, What Comes Out

### Experiment Inputs

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUTS TO EACH EXPERIMENT                                               │
│                                                                         │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │
│  │ train.py (mutated) │  │ best_model.pt      │  │ data.pt          │  │
│  │ ─ Model arch       │  │ ─ Warm-start       │  │ ─ (batch, 37)    │  │
│  │ ─ Loss function    │  │   weights           │  │   features       │  │
│  │ ─ Hyperparams      │  │ ─ 4-head ckpt      │  │ ─ Labels/targets │  │
│  │ ─ Training loop    │  │ ─ Or absent for     │  │ ─ Price arrays   │  │
│  │                    │  │   fresh start       │  │ ─ Day boundaries │  │
│  └────────────────────┘  └────────────────────┘  └──────────────────┘  │
│                                                                         │
│  Also uploaded: run_loop.py, prepare.py (for evaluate_trades import)    │
│  Environment: TIME_BUDGET=300, optional TRAIN_* overrides (PBT mode)   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Experiment Outputs

```
┌─────────────────────────────────────────────────────────────────────────┐
│ OUTPUTS FROM EACH EXPERIMENT                                            │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ METRICS_JSON (printed to stdout, parsed by inner_loop.py)      │    │
│  │                                                                 │    │
│  │  score              ─── composite score (PF × Sharpe × freq)   │    │
│  │  profit_factor      ─── gross profit / gross loss               │    │
│  │  trades_per_day     ─── average daily trade count               │    │
│  │  trade_sharpe       ─── Sharpe ratio of trade returns           │    │
│  │  win_rate           ─── % of trades with positive P&L           │    │
│  │  avg_winner/loser   ─── mean P&L for winners/losers             │    │
│  │  num_trades         ─── total trades on validation set          │    │
│  │  stop_loss_rate     ─── % exits via stop-loss                   │    │
│  │  model_exit_count   ─── exits via gate head NO_TRADE            │    │
│  │  max_drawdown       ─── worst peak-to-trough drawdown           │    │
│  │  hit_ruin           ─── true if equity < 25% of starting cap    │    │
│  │  num_trade_dates    ─── # unique dates with trades              │    │
│  │  direction_collapse ─── % of trades in dominant direction       │    │
│  │  short_hold_pct     ─── % of trades held ≤ 1 bar               │    │
│  │  chunk_details      ─── per-chunk PF/WR breakdown               │    │
│  │  training_seconds   ─── wall clock of training                  │    │
│  │  peak_vram_mb       ─── GPU memory high water mark              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐    │
│  │ model_candidate.pt   │  │ Trade Diagnostics (stdout)           │    │
│  │ ─ 4-head state_dict  │  │ ─ Best/worst 5 trades                │    │
│  │ ─ has_value_head=T   │  │ ─ Time-of-day P&L breakdown          │    │
│  │ ─ has_risk_head=T    │  │ ─ Direction (CALL/PUT) breakdown      │    │
│  │ ─ position_dim=7     │  │ ─ Exit reason distribution            │    │
│  │ ─ account_dim=4      │  │ ─ Stop-loss alignment info            │    │
│  └──────────────────────┘  └──────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Four-Head Model Architecture (Training Context)

```
                    TRAINING FORWARD PASS
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  INPUT: (batch, lookback=120, 37 features)                               │
│         ─ From data.pt: normalized 1-min bars                            │
│         ─ Batches are either day-sequential (85%) or random (15%)        │
│                                                                          │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────┐                        │
│  │  SHARED TRANSFORMER BACKBONE                  │                        │
│  │  input_proj: Linear(37 → 64) + GELU + Drop   │                        │
│  │  + LayerNorm + learned pos_embed (1,120,64)   │                        │
│  │  TransformerEncoder(depth=3, heads=4, ff=192)  │                        │
│  │  Pre-LN, causal mask, GELU, dropout=0.3       │                        │
│  │  Output: last bar → (batch, 64)               │                        │
│  └──────┬───────────────────────────────────────┘                        │
│         │                                                                │
│         │  + POSITION STATE (7 dims, built during day-sequential sim)    │
│         │    [0] in_trade      [1] bars_held       [2] unrealized_pnl    │
│         │    [3] acct_health   [4] loss_streak      [5] best_pnl         │
│         │    [6] bars_since_high                                         │
│         │                                                                │
│         │  + ACCOUNT STATE (4 dims, risk head only)                      │
│         │    [0] growth_ratio  [1] log_size  [2] daily_pnl  [3] win_rate │
│         │                                                                │
│    ┌────┴──────────────┬───────────────┬───────────────────────┐        │
│    ▼                   ▼               ▼                       ▼        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ GATE HEAD│  │ DIR HEAD │  │ VALUE HEAD   │  │ RISK HEAD          │  │
│  │ (2 out)  │  │ (6 out)  │  │ (1 out)      │  │ (3 out)            │  │
│  ├──────────┤  ├──────────┤  ├──────────────┤  ├────────────────────┤  │
│  │ Uses:    │  │ Uses:    │  │ Uses:        │  │ Uses:              │  │
│  │ backbone │  │ backbone │  │ backbone     │  │ backbone           │  │
│  │ + pos    │  │ only     │  │ + pos state  │  │ + pos state        │  │
│  │ state    │  │          │  │              │  │ + account state    │  │
│  ├──────────┤  ├──────────┤  ├──────────────┤  ├────────────────────┤  │
│  │NO_TRADE  │  │CALL_ATM  │  │remaining_pnl │  │stop_pct [.15-.60] │  │
│  │TRADE     │  │CALL_OTM5 │  │(scalar, MSE) │  │size_frac [0-1]    │  │
│  │          │  │CALL_OTM10│  │              │  │conviction [-1,+1] │  │
│  │          │  │PUT_ATM   │  │              │  │                    │  │
│  │          │  │PUT_OTM5  │  │              │  │                    │  │
│  │          │  │PUT_OTM10 │  │              │  │                    │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └──────────┬─────────┘  │
│       │             │               │                      │            │
│       ▼             ▼               ▼                      ▼            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        LOSS COMPUTATION                          │   │
│  │  (see Section 4 for full breakdown)                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Checkpoint Contract

Every saved model includes these metadata keys for compatibility verification:

```python
{
    'model_state_dict': state_dict,
    'position_state_dim': 7,    # MUST match TradingModel.POSITION_STATE_DIM
    'account_state_dim': 4,     # MUST match TradingModel.ACCOUNT_STATE_DIM
    'has_value_head': True,     # MUST be True (v6 requirement)
    'has_risk_head': True,      # MUST be True (v6 requirement)
    'num_features': 37,         # v3 feature set
    'lookback': 120,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 3,
}
```

**Version gate:** train.py and replay.py reject checkpoints missing `has_value_head`, `has_risk_head`, or with `position_state_dim < 7`. Incompatible checkpoints → training from scratch.

---

## 6. Training Regiment

### Epoch Structure

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      TRAINING TIMELINE                                    │
│                                                                          │
│  TIME_BUDGET = 300s (default, tunable per experiment)                    │
│                                                                          │
│  ┌──────┐  ┌─────────────────────────────────────────────┐  ┌────────┐  │
│  │WARMUP│  │              MAIN TRAINING                   │  │COOLDOWN│  │
│  │ 15%  │  │  Step loop until time_budget exhausted       │  │  30%   │  │
│  │of LR │  │  Each step: one batch → forward → loss →     │  │of LR   │  │
│  │sched │  │  backward → clip grad (1.0) → optimizer step │  │sched   │  │
│  └──────┘  └─────────────────────────────────────────────┘  └────────┘  │
│                                                                          │
│  Optimizer: AdamW (lr=2.5e-4, weight_decay=0.08)                        │
│  Scheduler: linear warmup → constant → linear cooldown                   │
│  Gradient clipping: max_norm=1.0                                         │
│  Batch size: 512 (random) or 16 days (day-sequential)                   │
│                                                                          │
│  progress = elapsed_time / TIME_BUDGET                                   │
│  Training stops when progress >= 1.0 AND step > 5                       │
│                                                                          │
│  If WARM_FREEZE_RATIO > 0:                                               │
│    Transformer + input layers FROZEN until progress >= freeze_ratio      │
│    Only heads train on early steps (protects backbone)                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Warm-Start Logic

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      CHECKPOINT LOADING                                   │
│                                                                          │
│  best_model.pt exists on Akash?                                          │
│     │                                                                    │
│     ├── YES → Load checkpoint                                            │
│     │    │                                                               │
│     │    ├── has_value_head=True AND has_risk_head=True                  │
│     │    │   AND position_state_dim >= 7?                                │
│     │    │    │                                                          │
│     │    │    ├── YES → load_state_dict(strict=False)                    │
│     │    │    │         Missing keys get random init                     │
│     │    │    │         ✓ Warm start from previous best                  │
│     │    │    │                                                          │
│     │    │    └── NO  → WARNING: "Checkpoint incompatible"               │
│     │    │              Skip loading, train from scratch                  │
│     │    │                                                               │
│     │    └── (shape mismatch also causes skip)                           │
│     │                                                                    │
│     └── NO  → Random initialization with domain-knowledge biases:       │
│               Gate: pro-trade (+0.3 TRADE, -0.3 NO_TRADE)               │
│               Direction: ATM favored (+0.20), OTM penalized             │
│               Risk: conservative sizing, neutral conviction              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Batching Strategy

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     BATCH SELECTION (each step)                           │
│                                                                          │
│  random() < DAY_SEQ_RATIO (0.85)?                                        │
│     │                                                                    │
│     ├── YES (85%) ── DAY-SEQUENTIAL BATCH ──────────────────────────┐   │
│     │   Sample 16 random training days                               │   │
│     │   For each day, walk bar-by-bar chronologically:               │   │
│     │                                                                │   │
│     │   bar 0 ─── bar 1 ─── bar 2 ─── ... ─── bar 389              │   │
│     │     │         │         │                    │                 │   │
│     │   reset    carry      carry              end of               │   │
│     │   pos      pos        pos                day                  │   │
│     │   state    state      state                                   │   │
│     │                                                                │   │
│     │   Position state simulated: entries, exits, stops tracked      │   │
│     │   Account state updated: balance, win rate, loss streak        │   │
│     │   Value head loss computed (only while holding)                │   │
│     │   Risk head loss computed (only while holding)                 │   │
│     │   Gate exit labels injected (when holding + should exit)       │   │
│     │                                                                │   │
│     │   OUTPUT: position-aware loss with full trade simulation       │   │
│     └────────────────────────────────────────────────────────────────┘   │
│     │                                                                    │
│     └── NO (15%) ── RANDOM BATCH ───────────────────────────────────┐   │
│         Sample 512 random bars from training set                     │   │
│         Position state = zeros (no trade context)                    │   │
│         Only gate + direction loss computed                          │   │
│         OUTPUT: entry-signal loss only (no exit/value/risk)          │   │
│         Purpose: prevents gate from over-specializing to trade       │   │
│         context — must also work on "cold" bars                      │   │
│         └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Position State Simulation (Day-Sequential Mode)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                POSITION STATE MACHINE (per day, per bar)                  │
│                                                                          │
│  Each bar, the model sees: features(120,37) + position_state(7)          │
│  The simulation decides entry/exit based on model predictions:           │
│                                                                          │
│                     ┌──────────────┐                                     │
│               ┌────▶│   FLAT       │◀──── Day start (reset)              │
│               │     │ is_holding=0 │                                     │
│               │     └──────┬───────┘                                     │
│               │            │                                             │
│               │     Gate predicts TRADE                                  │
│               │     AND not in cooldown                                  │
│               │     AND bar >= NO_TRADE_BEFORE_BAR                      │
│               │            │                                             │
│               │            ▼                                             │
│               │     ┌──────────────┐                                     │
│               │     │   HOLDING    │                                     │
│               │     │ is_holding=1 │                                     │
│               │     │ bars_held++  │                                     │
│               │     │ track P&L    │                                     │
│               │     └──────┬───────┘                                     │
│               │            │                                             │
│               │     Exit check (priority order):                         │
│               │     1. Stop-loss: unrealized_pnl < -dynamic_stop         │
│               │     2. Model exit: gate predicts NO_TRADE                │
│               │     3. Max hold: bars_held > 60                          │
│               │            │                                             │
│               │            ▼                                             │
│               │     ┌──────────────┐                                     │
│               │     │ EXIT         │                                     │
│               │     │ Record P&L   │                                     │
│               └─────│ Update acct  │                                     │
│                     │ 5-bar cool-  │                                     │
│                     │ down if stop │                                     │
│                     └──────────────┘                                     │
│                                                                          │
│  While holding, value head trains on remaining P&L:                      │
│    target = clamp(best_pnl_across_6_options, -1.0, 1.0)                 │
│                                                                          │
│  While holding, risk head trains on:                                     │
│    stop_pct target = optimal stop distance (from realized data)          │
│    size_frac target = account-aware position size                        │
│    conviction target = tanh(realized_pnl × 3.0)                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Loss Computation (Full Breakdown)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      MULTI-COMPONENT LOSS                                │
│                                                                          │
│  ┌─── GATE LOSS (weight: 0.5) ──────────────────────────────────────┐   │
│  │  Target: TRADE if best_pnl > 0, NO_TRADE otherwise               │   │
│  │  Override: exit-labeled bars → NO_TRADE (when holding)            │   │
│  │  Loss: F.cross_entropy(gate_logits, target)                       │   │
│  │  Weighted by: time-of-day learned weights × sample importance     │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── DIRECTION LOSS (weight: 2.5) ─────────────────────────────────┐   │
│  │  Only on TRADE bars (gate_target == 1)                            │   │
│  │  Soft targets: P&L-weighted distribution over 6 options           │   │
│  │    shifted_pnl = pnl - min(pnl) + eps, then normalize to [0,1]   │   │
│  │  Loss: cross-entropy with soft targets + 0.20 entropy bonus       │   │
│  │  Time-weighted: 1.5× in morning, 1.0× at close                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── PNL ALIGNMENT (weight: 0.5) ──────────────────────────────────┐   │
│  │  pnl_signal = P(trade) × Σ(P(dir_i) × pnl_i)                    │   │
│  │  Loss: -mean(pnl_signal)  ← negative = bonus for profitable dirs │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── CONFIDENCE CALIBRATION (weight: 0.10) ────────────────────────┐   │
│  │  Only on TRADE bars                                               │   │
│  │  outcome = tanh(pnl × 5.0)  ← +1 for winners, -1 for losers     │   │
│  │  Loss: -mean(outcome × log(trade_confidence))                     │   │
│  │  Effect: high confidence rewarded on winners, penalized on losers │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── VALUE HEAD LOSS (weight: 0.3, day-seq only) ──────────────────┐   │
│  │  Only while holding (is_holding > 0.5)                            │   │
│  │  Target: best P&L across all 6 options, clamped [-1.0, 1.0]      │   │
│  │  Loss: MSE(value_pred, target)                                    │   │
│  │  Purpose: teaches model how much P&L remains in current trade     │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── RISK HEAD LOSS (weight: 0.2, day-seq only) ───────────────────┐   │
│  │  Only while holding                                               │   │
│  │  Three sub-components (weights 0.5 / 0.3 / 0.2):                 │   │
│  │    stop_loss:  MSE vs optimal stop distance                       │   │
│  │    size_loss:  MSE vs account-aware target size                   │   │
│  │    conv_loss:  MSE vs tanh(realized_pnl × 3.0)                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─── REGULARIZATION (optional, via env vars) ──────────────────────┐   │
│  │  Gate entropy:     maximize entropy to prevent gate collapse      │   │
│  │  Temporal smooth:  penalize large gate changes between bars       │   │
│  │  Time-of-day weights: 390 learned params (one per bar of day)     │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  TOTAL = 0.5×gate + 2.5×dir + 0.5×pnl + 0.10×conf                     │
│        + 0.3×value + 0.2×risk + regularization                          │
│                                                                          │
│  All weights tunable via TRAIN_* env vars (and PBT)                     │
│  Score config is LOCKED — cannot be modified by agent                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Dynamic Stop-Loss in Training

```
  stop_level = DYNAMIC_STOP_BASE × iv_factor × vix_factor

  iv_factor  = 1.0 + clamp(atm_iv, 0) × 0.15     ← higher IV → wider stop
  vix_factor = 1.0 + clamp(vix_regime, 0) × 0.10  ← higher VIX → wider stop

  Effective range: [0.15, 0.60]  (DYNAMIC_STOP_MIN to DYNAMIC_STOP_MAX)

  Training uses pre-computed stopped P&L at 3 levels (tight/med/wide)
  and selects per-bar based on the formula above.
```

---

## 8. Validation After Training

```
┌──────────────────────────────────────────────────────────────────────────┐
│           VALIDATION (runs after TIME_BUDGET exhausted)                   │
│                                                                          │
│  evaluate_trades(model, val_data)                                        │
│    ─ Bar-by-bar simulation on validation set (30% of data)               │
│    ─ Same position-state machine as training                             │
│    ─ Same dynamic stop formula                                           │
│    ─ Computes: PF, Sharpe, win rate, trades/day, exit breakdown          │
│                                                                          │
│  evaluate_sharpe(model, val_data)                                        │
│    ─ Equity curve construction                                           │
│    ─ Computes: equity Sharpe, max DD, Calmar, Sortino                   │
│                                                                          │
│  Score = PF × trade_sharpe × freq_mult × penalties × bonuses            │
│                                                                          │
│  Output: METRICS_JSON line to stdout + trade diagnostics                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Scoring & KEEP/REVERT Decision

The scoring formula is the single most important piece of the inner loop. It determines whether an experiment's model is "better" than the current best. The formula lives in `evaluate_trades()` in [prepare.py](../../training/prepare.py) and is configured by the `_score_config` dict in [best_train.py](../../training/best_train.py). The score formula is **LOCKED** — neither the inner loop agent nor Opus may change `_score_config` values or the scoring logic.

### How the Score Is Computed

The score has three layers: a **base score**, **penalties**, and **bonuses**. Each builds on the previous.

#### Layer 1: Base Score (prepare.py:3577-3598)

```
  base_score = profit_factor × trade_sharpe × freq_mult
```

Each component:

| Component | Formula | What it measures |
|-----------|---------|-----------------|
| `profit_factor` | gross_wins / gross_losses | Does the model make more than it loses? |
| `trade_sharpe` | (mean_pnl / std_pnl) × sqrt(trades_per_year) | Is profit consistent or lucky? |
| `freq_mult` | Bell curve around `freq_center` (default 2.5 tpd) | Is trade frequency in the sweet spot? |

**Frequency multiplier detail** — the model must trade in a band, not too much, not too little:

```
  freq_center = 2.5 trades/day    freq_width = 2.5
  freq_lo = max(0.5, center - width) = 0.5
  freq_hi = center + width = 5.0

  ┌──────────────────────────────────────────────────────────────────┐
  │ Trades/day:  0    0.5    1.0    2.5    5.0    7.0    10.0       │
  │              │     │      │      │      │      │       │        │
  │ Score:      -10   ramp ──────── full ────── decay ── floor(0.1) │
  │                                                                  │
  │  < 0.5 tpd   → hard -10.0 (model barely trades)                │
  │  0.5 to 0.5  → linear ramp from -5.0 up to raw_score           │
  │  0.5 to 5.0  → full score (freq_mult ≤ 1.0)                    │
  │  > 5.0       → quadratic decay: (freq_hi / tpd)²               │
  │              floor at 0.1 (never fully zeroed)                   │
  └──────────────────────────────────────────────────────────────────┘
```

#### Layer 2: Penalties (prepare.py:3600-3655)

Applied **multiplicatively** to the base score. Each penalty has a threshold — no effect below it, increasing punishment above it. All penalties only apply when `score > 0` (don't punish an already-negative score).

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ Penalty              Threshold   Formula            Floor       │
  │ ─────────────────────────────────────────────────────────────── │
  │ Consecutive losses   > 3 in row  × max(0.2, 1 - 0.15×excess)  0.2  │
  │ Short holds          > 30%       × max(0.7, 1 - (rate-0.30))  0.7  │
  │ Stop-loss rate       > 30%       × max(0.5, 1 - (rate-0.30))  0.5  │
  │ Max drawdown         > 10%       × max(0.3, 1 - 0.5×|DD|)     0.3  │
  │ Avg risk fraction    > 30%       × max(0.3, 1 - 0.5×excess)   0.3  │
  └──────────────────────────────────────────────────────────────────┘

  Example: 5 consecutive losses with threshold 3
    excess = 5 - 3 = 2
    penalty = max(0.2, 1.0 - 0.15 × 2) = max(0.2, 0.70) = 0.70
    score *= 0.70  (30% reduction)
```

**Why these penalties exist:** Without them, the model could achieve a high PF by taking one lucky trade, or a high Sharpe by rapid-fire scalping that hits stops 50% of the time. The penalties encode trading wisdom: don't overtrade, don't hold too briefly, don't rely on stops as an exit strategy, don't risk too much per trade.

#### Layer 3: Bonuses (prepare.py:3626-3647)

Also multiplicative. These reward good trading behavior beyond what PF/Sharpe already capture.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ Bonus                Config value   Formula                      │
  │ ─────────────────────────────────────────────────────────────── │
  │ Win rate             0.0 (off)      1 + bonus × (wr - 0.40)/0.6 │
  │ R:R ratio            0.3            1 + 0.3 × (rr - 1.0)/2.0    │
  │ Hold time quality    0.0 (off)      bell curve at 30 bars        │
  └──────────────────────────────────────────────────────────────────┘

  With current _score_config, only R:R bonus is active (0.3).
  A model where avg_winner = 2× avg_loser gets:
    rr_ratio = 2.0, bonus = 1 + 0.3 × (2.0 - 1.0)/2.0 = 1.15
    score *= 1.15  (15% boost)
```

#### Layer 4: Ruin Check (prepare.py:3675-3694)

The final gate. If the equity curve ever drops below 25% of starting capital (the "ruin threshold"), the score is crushed.

```
  ruin_threshold = 0.25 (25% of starting capital remaining)

  If min_equity < $10,000 × 0.25 = $2,500:
    ruin_severity = 1.0 - (min_equity_frac / ruin_threshold)
    ruin_mult = max(0.0, 1.0 - severity)

    Near-total ruin (mult < 0.05) → score capped at -5.0
    Partial ruin → score *= ruin_mult
```

### The _score_config Dictionary (best_train.py:875-888)

These values are **read-only** and locked by tests + safety validation:

```python
_score_config = {
    'win_rate_bonus': 0.0,        # Off — PF already rewards winning
    'rr_bonus': 0.3,              # Active — reward avg_win > avg_loss
    'drawdown_penalty': 0.5,      # Active — punish deep drawdowns
    'hold_bonus': 0.0,            # Off — don't bias hold duration
    'freq_center': 2.5,           # Sweet spot: 2.5 trades/day
    'freq_width': 2.5,            # Band: 0.5 to 5.0 trades/day
    'consec_loss_threshold': 3,   # Allow 3 losses in a row before penalty
    'short_hold_threshold': 0.30, # 30%+ 1-bar holds triggers penalty
    'stop_rate_threshold': 0.30,  # 30%+ stop-outs triggers penalty
    'ruin_penalty': 1.0,          # Full ruin penalty active
    'ruin_threshold': 0.25,       # Ruin at 75% capital loss
    'risk_fraction_penalty': 0.5, # Penalize over-sizing positions
}
```

### Complete Score Pipeline (Visual)

```
  profit_factor ──┐
                   ├── × ── base_score
  trade_sharpe ───┘        │
                            ├── × freq_mult ── adjusted_score
  trades_per_day ──────────┘        │
                                     │
  consec_losses ─── penalty? ───────×
  short_hold_pct ── penalty? ───────×
  stop_loss_rate ── penalty? ───────×
  max_drawdown ──── penalty? ───────×
  avg_risk_frac ─── penalty? ───────×
                                     │
  win_rate ───────── bonus? ────────×
  rr_ratio ───────── bonus? ────────×
  avg_hold_bars ──── bonus? ────────×
                                     │
  min_equity ─────── ruin? ─────────× ──── final_score
                                              │
                                              ▼
                           compare to .best_score
                           + check anomaly flags
                                  │
                            KEEP or REVERT
```

### Anomaly Detection

Before comparing scores, the system checks for anomalies — conditions where the score number can't be trusted regardless of how high it is. Anomaly detection runs in `run_loop.py` via `detect_anomalies()`.

```
  CRITICAL (blocks KEEP regardless of score):
    ─ metric_inconsistent    win_rate outside [0,1], PF<0, etc.
    ─ trades_per_day_extreme outside [0.25, 20.0]
    ─ cost_realism_low       coverage < 0.20
    ─ entry_quality_too_low  avg < 0.15
    ─ low_quality_rate_high  rate > 0.65
    ─ single_date_specialist ≤1 trade date with ≥3 trades

  NON-CRITICAL (logged but don't block):
    ─ do_nothing_zero        gate never says NO_TRADE
    ─ exit_pct_extreme       98%+ of actions are exits
    ─ actionable_bar_rate_low < 3% of bars get trades
    ─ high_cost_entry_rate   > 65% entries have bad cost
```

**Why anomaly detection exists:** A model that only trades on one specific date (single_date_specialist) might get PF=10 by memorizing that day. A model with 0% cost realism means it traded on bars with no spread data — we can't trust its P&L. These flags catch degenerate strategies that score well numerically.

### Decision Logic

```
  keep = (score > best_score) AND (no critical anomaly flags)

  KEEP:
    model_candidate.pt  ──promote──▶  best_model.pt
    train.py            ──snapshot──▶  best_train.py
    score               ──write────▶  .best_score
    Record to results/promoted/history.jsonl

  REVERT:
    train.py            ──restore──▶  from artifacts/exp-N/train_before.py
    best_model.pt       ──unchanged── (still matches best_train.py)
    .best_score         ──unchanged──
```

---

## 10. Key Invariant

**`best_model.pt`, `best_train.py`, and `.best_score` are always in sync.**

- Model is downloaded to a **temp file** (`artifacts/exp-N/model_candidate.pt`) first
- Only promoted to `best_model.pt` on KEEP
- On REVERT, `best_model.pt` is unchanged — still matches `best_train.py`
- `best_model.pt` is uploaded to Akash before each experiment for warm start
- This invariant means any experiment can crash/timeout without corrupting state

---

## 11. Experiment Lifecycle (Detailed)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    SINGLE EXPERIMENT TIMELINE                             │
│                                                                          │
│  t=0s     Opus calls: inner_loop.py experiment --mutation FILE           │
│           │                                                              │
│  t=1s     VALIDATE: AST parse + safety check (run_loop.py)              │
│           ├── Syntax errors? → ABORT, return error JSON                  │
│           ├── Safety violation? → ABORT, return error JSON               │
│           │   (score lock, arch lock, feature lock, dangerous patterns)  │
│           │                                                              │
│  t=2s     BACKUP: train.py → artifacts/exp-N/train_before.py            │
│           APPLY: write mutation to train.py (or skip if baseline)        │
│           │                                                              │
│  t=5s     UPLOAD: SCP train.py → Akash                                  │
│           SCP best_model.pt → Akash (for warm start)                    │
│           │                                                              │
│  t=15s    TRAIN: SSH → TIME_BUDGET=300 python train.py                  │
│           │  ┌─────────────────────────────────────────┐                │
│           │  │ Warm-start checkpoint load (or fresh)    │                │
│           │  │ Hybrid batch loop:                       │                │
│           │  │   85% day-sequential (position sim)      │                │
│           │  │   15% random (cold-bar generalization)   │                │
│           │  │ Until TIME_BUDGET exhausted               │                │
│           │  │ Validation: evaluate_trades on val set    │                │
│           │  │ Print METRICS_JSON + trade diagnostics    │                │
│           │  └─────────────────────────────────────────┘                │
│           │                                                              │
│  t=320s   DOWNLOAD: SCP model_candidate.pt ← Akash                     │
│           │                                                              │
│  t=325s   PARSE: extract METRICS_JSON from stdout                       │
│           SCORE: anomaly detection + score comparison                    │
│           │                                                              │
│  t=326s   KEEP or REVERT (see Section 7)                                │
│           │                                                              │
│  t=327s   RECORD: append to experiments.v2.jsonl + history.jsonl        │
│           UPDATE: status.json (best_score, accept_rate, phase=idle)     │
│           │                                                              │
│  t=328s   Return result JSON to Opus                                    │
│           Opus reads, diagnoses, plans next experiment                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 12. File Layout

```
tools/
  inner_loop.py          ─── Mechanical layer (SSH, validation, scoring, keep/revert)

training/
  train.py               ─── Active training code (mutated per experiment)
  best_train.py          ─── Snapshot of train.py that produced best_model.pt
  best_model.pt          ─── Current best model checkpoint
  .best_score            ─── Float: score of best_model.pt
  .inner_loop_state.json ─── Run state (experiment_id, best_score, kept/total counts)
  run_loop.py            ─── Utility library: validation, anomaly detection, metric parsing
  program.md             ─── Contract injected into Sonnet agent's prompt
  prepare.py             ─── evaluate_trades(), evaluate_sharpe(), score formula

results/
  current_run.txt        ─── Points to active run directory name
  run-YYYY-MM-DD-HHMMSS/
    status.json          ─── Monitor.py-compatible status (phase, scores, counts)
    experiments.v2.jsonl ─── Full experiment records (monitor.py compatible)
    history.jsonl        ─── Same records, read by Opus for strategic decisions
    loop.log             ─── Inner loop log output
    train_baseline.py    ─── Snapshot of train.py at run init
    artifacts/
      exp-1/
        train_before.py  ─── train.py before mutation
        train_candidate.py ─ Proposed mutation (if not baseline)
        model_candidate.pt ─ Downloaded model (temp, promoted on KEEP)
        metrics.json     ─── Parsed training metrics
        train_output.log ─── Full stdout/stderr from Akash
      exp-2/
        ...
```

---

## 13. SSH/SCP Transport

All communication with Akash GPU happens via SSH/SCP with `sshpass`:

```
Local Machine                          Akash H100 GPU
─────────────                          ──────────────
train.py           ──SCP upload──▶     /root/autoresearch-trading/training/train.py
best_model.pt      ──SCP upload──▶     /root/autoresearch-trading/training/best_model.pt

                   ──SSH──▶            TIME_BUDGET=300 python train.py

model_candidate.pt ◀──SCP download──  /root/autoresearch-trading/training/best_model.pt
```

- Connection params from `.deploy-state` file (SSH_HOST, SSH_PORT), written by `deploy.sh boot`
- 3 retries on SCP failure with 3s backoff
- SSH timeout = TIME_BUDGET + 240s (4 min buffer)
- SSH keepalive: `ServerAliveInterval=30`, `ServerAliveCountMax=3`

---

## 14. PBT (Population-Based Training)

An alternative to Opus-driven sequential experiments. Instead of Opus forming a hypothesis and testing one change at a time, PBT runs a **population** of competing configurations simultaneously, then evolves the best ones forward.

### Key Concepts

- **Generation** — One round of the tournament. Every member in the population trains and gets scored. Then the population evolves based on results. Generation 0 is the initial population; generation 1 is the first evolved population; and so on.
- **Member** — One configuration of hyperparameters within a generation. Each member trains the same model architecture with different settings (learning rate, loss weights, dropout, etc.). Member 0 is always special (see below).
- **Population** — The full set of members in a generation. Default size is 6. All members train sequentially on the same GPU (one at a time), using the same `train.py` and starting from the same `best_model.pt` checkpoint.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│ GENERATION 0 (initial population)                                   │
│                                                                     │
│  Member 0: BASELINE — exact default hyperparameters from train.py   │
│  Member 1: defaults + random perturbation (0.5x to 2x on log scale)│
│  Member 2: defaults + random perturbation                           │
│  Member 3: defaults + random perturbation                           │
│  Member 4: defaults + random perturbation                           │
│  Member 5: defaults + random perturbation                           │
│  [Member 6+: specialists, if --specialists flag used]               │
│                                                                     │
│  Each member trains → gets scored → results recorded                │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼  Rank all members by score
                                    │
┌───────────────────────────────────┴─────────────────────────────────┐
│ EVOLUTION (select_next_generation)                                   │
│                                                                     │
│  1. ELITE — Member 0 of next gen = exact copy of best member       │
│             (no mutation, preserves the winning config)              │
│                                                                     │
│  2. EXPLOIT — Members 1 to N/2: clone a parent from TOP 25%,       │
│              apply STANDARD mutation (small perturbations)           │
│              "Refine what works"                                     │
│                                                                     │
│  3. EXPLORE — Members N/2+1 to N-1: clone a parent from TOP 50%,  │
│              apply STRONG mutation (large perturbations)             │
│              "Search for something better"                           │
│                                                                     │
│  Anti-stagnation: if no improvement for 2+ generations,             │
│  replace 2 members with FULLY RANDOM configs                        │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GENERATION 1 (evolved population)                                   │
│                                                                     │
│  Member 0: ELITE (best from gen 0, unchanged)                       │
│  Member 1: EXPLOIT (clone of top-25% parent, small mutation)        │
│  Member 2: EXPLOIT (clone of top-25% parent, small mutation)        │
│  Member 3: EXPLORE (clone of top-50% parent, large mutation)        │
│  Member 4: EXPLORE (clone of top-50% parent, large mutation)        │
│  Member 5: EXPLORE (clone of top-50% parent, large mutation)        │
│                                                                     │
│  Each member trains → gets scored → evolve again...                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Concrete Example (6 members, 3 generations)

```
Generation 0:
  Member 0 (baseline):  LR=3e-4, DROPOUT=0.15, GATE_W=0.5  → score 2.1
  Member 1 (perturbed): LR=5e-4, DROPOUT=0.22, GATE_W=0.8  → score 3.4  ← best
  Member 2 (perturbed): LR=1e-4, DROPOUT=0.08, GATE_W=0.3  → score 1.8
  Member 3 (perturbed): LR=7e-4, DROPOUT=0.30, GATE_W=0.6  → score -2.0
  Member 4 (perturbed): LR=2e-4, DROPOUT=0.12, GATE_W=1.1  → score 2.8
  Member 5 (perturbed): LR=4e-4, DROPOUT=0.18, GATE_W=0.4  → score 0.5

  Ranked: [1, 4, 0, 2, 5, 3]  (top-25% = member 1, top-50% = members 1,4,0)

Generation 1:
  Member 0 (ELITE):   exact copy of gen0-member1 (score 3.4 winner)
  Member 1 (EXPLOIT): gen0-member1 + small tweak → LR=4.8e-4, DROPOUT=0.20
  Member 2 (EXPLOIT): gen0-member1 + small tweak → LR=5.3e-4, GATE_W=0.9
  Member 3 (EXPLORE): gen0-member4 + big tweak   → LR=1e-4,   DROPOUT=0.35
  Member 4 (EXPLORE): gen0-member1 + big tweak   → LR=8e-4,   GATE_W=0.2
  Member 5 (EXPLORE): gen0-member0 + big tweak   → LR=6e-4,   DROPOUT=0.05

Generation 2:
  Same process — evolve from gen 1 results...
```

### What Each Member Actually Does

Each member doesn't get its own copy of `train.py`. Instead, hyperparameters are passed as **environment variable overrides**. The member's config dict is converted to env vars (e.g., `LR=0.0005`, `GATE_W=0.8`), and `train.py` reads them via `_env_float("LR", default=3e-4)`. This means every member runs the exact same code but with different tuning.

### Parameter Space

```
  Tier 1 — Loss Weights:    GATE_W, DIR_W, PNL_W, EXIT_W, CONF_W, VALUE_W, RISK_W
  Tier 2 — Optimizer:       LR, WEIGHT_DECAY, DROPOUT, WARMUP_RATIO, COOLDOWN_RATIO, GRAD_CLIP
  Tier 3 — Regularization:  RECENT_BOOST, DAY_DIVERSITY, GATE_ENTROPY, TEMPORAL_SMOOTH,
                             FREEZE_RATIO, DAY_SEQ_RATIO, label smoothing
```

**Excluded from PBT:** D_MODEL, DEPTH, N_HEADS (would break warm-start checkpoint shapes), BATCH_SIZE (GPU memory constraint).

**Focus modes:** `--focus loss_weights` (tier 1 only), `--focus regularization` (tiers 2+3), `--focus all` (default).

### Specialists (Optional)

With `--specialists`, additional members are appended for regime-specific configs (morning, midday, afternoon, high-volatility). These start from baseline and are evolved separately from generalists.

### State & Resumability

State is saved to `training/.pbt_state.json` after every member completes. If a PBT run is interrupted (GPU crash, timeout), `pbt-run` resumes from the exact member where it stopped — no work is lost.

### Commands

```
pbt-init --population 6 --generations 3 --focus all    # Create initial population
pbt-run                                                  # Train all members, evolve
pbt-status                                               # Show progress
```

---

## 15. Monitor Integration

`inner_loop.py` writes two files that `monitor.py` (localhost:8420) reads:

1. **`status.json`** — current phase (idle/validating/training/scoring), experiment count, best score, accept rate, timestamps
2. **`experiments.v2.jsonl`** — append-only log of all experiment results with full metrics

---

## 16. Loop Detection

Tracks consecutive identical revert reasons. If the same reason appears 3+ times in a row, logs `LOOP DETECTED` warning. This prevents the agent from repeating the same failing approach without adapting.

---

## 17. deploy.sh Integration

`inner_loop.py` depends on `deploy.sh` for Akash lifecycle:

| Command | Purpose |
|---------|---------|
| `deploy.sh boot` | Create Akash lease, write `.deploy-state` (SSH_HOST, SSH_PORT) |
| `deploy.sh start` | Upload workspace, install deps, start watchdog |
| `deploy.sh status` | Check GPU health, disk, running processes |
| `deploy.sh stop -y` | Download results, close lease, clean up |

`inner_loop.py` reads `.deploy-state` for SSH connection params. It does NOT call `deploy.sh` directly — Opus manages the infrastructure lifecycle separately.
