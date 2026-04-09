# How Training Works

A plain-language explanation of what happens when you run a training experiment.

## Two Loops, Not One

There are two nested loops in this system. Understanding which one you're looking at avoids a lot of confusion.

```
OUTER LOOP (autoresearch / ART2)           <-- Karpathy-style, code changes
  for each experiment (up to 50):
    1. Form hypothesis, edit train.py or policy.py
    2. Commit the change
    3. Upload to Akash GPU
    4. --- INNER LOOP (PyTorch training) -- <-- epochs, weight updates
    |   for each epoch (up to 30):
    |     pass through all training data
    |     update model weights
    |     check validation loss
    |     save best checkpoint
    |   ---------------------------------- <-- produces one model.pt
    5. Replay model on 60 held-out days
    6. Compute score
    7. Score improved AND beats baselines? KEEP. Otherwise? REVERT.
    8. Check session limits, continue or stop
```

The **inner loop** (epochs) produces one trained model. It is standard machine learning. This is what `train.py` does.

The **outer loop** (experiments) is the Karpathy autoresearch part. This is where the AI researcher (Claude) mutates the code, trains, evaluates, and keeps or reverts. Claude drives this loop using `deploy.sh run_one` which uploads code to the GPU, runs `run_experiment.py`, and downloads the result.

One experiment = one full training run (up to 30 epochs) + replay + scoring. A session can run up to 50 experiments.

### What changed from v1 to v2

The two-loop structure is the same in both versions. What changed:

| Aspect | v1 | v2 |
|--------|----|----|
| **Scoring** | Prediction accuracy (did the model predict correctly?) | Equity curve health (Sortino ratio * win rate * drawdown guard) |
| **Mutable files** | Just `train.py` | `train.py` (model/loss) + `policy.py` (trading parameters) |
| **Evaluation** | Compare predictions to labels | Full replay simulation: execute trades on held-out days, track P&L on a $10K account |
| **Artifacts** | Loose checkpoint files | Bundled snapshots (code + model + results per experiment) |
| **Session limits** | Informal | Enforced: 20 experiments, 10hr, 6 no-improve streak, 4hr plateau, 3 consecutive crashes |
| **Data split** | Train/val | Walk-forward CV: 5 folds, 300 total test days across diverse market regimes |
| **Model output** | Direction prediction | Full TradeIntent: gate, direction, strike offset, stop, target, hold, confidence |

The inner loop (epochs) did not change meaningfully between v1 and v2. It is still "pass data through model, compare to labels, nudge weights, repeat."

## The Big Picture

You have ~1000 days of historical SPX market data. For each 1-minute bar, an oracle has already figured out the best possible trade (or no trade) by brute-force searching every combination of direction, strike, stop-loss, target, and hold time.

Training = teaching the model to match those oracle decisions, using only the market data visible at that moment (no future peeking).

## What the Model Does

The model is a small transformer. It takes the last 30 bars of 47 features (28 price/market + 11 option/Greeks + 8 volume/flow) and predicts P&L for both directions:

| Output | What it means |
|--------|--------------|
| call_pnl | Predicted P&L if buying an ATM call here |
| put_pnl | Predicted P&L if buying an ATM put here |
| risk | Stop-loss %, profit target %, max hold fraction |

Trading decisions are derived at inference time: gate = max(call_pnl, put_pnl) > threshold, direction = argmax(call_pnl, put_pnl).

## What an Epoch Is

An **epoch** = one complete pass through all training samples.

Your training data has ~859 days of bars. Each epoch:

1. **Shuffle** the data into batches of 2048 samples.
2. For each batch:
   - Feed 2048 windows (each 30 bars x 47 features) into the model.
   - Model predicts call_pnl and put_pnl for all 2048 windows.
   - Compare predictions to actual P&L labels. Compute a **loss** (a number that measures how wrong the model is).
   - Run **backpropagation**: calculus that figures out which of the model's ~170K weights to nudge, and by how much, to reduce the loss.
   - **Update the weights** by a tiny amount in the direction that reduces the loss.
3. After all batches: run the **validation set** (60 separate days the model did not train on) to check whether the model is learning general patterns or just memorizing.

The default is 30 epochs with a 5-minute time budget (whichever comes first).

Each epoch, the model sees the same data in a different random order. Its weights are slightly better each time. Like re-reading a textbook where each pass builds understanding.

## The Loss Function

The loss is P&L regression with direction-asymmetric penalties:

| Sub-loss | What it measures | Weight |
|----------|-----------------|--------|
| call_pnl | How close is predicted call P&L to actual? (asymmetric Huber, 4x penalty for false optimism) | 1.0 |
| put_pnl | How close is predicted put P&L to actual? (asymmetric Huber, 6x penalty for false optimism) | 1.0 |
| risk | Are stop/target/hold close to label values? (Huber loss) | 0.5 |

The asymmetric Huber loss penalizes optimistic errors (model predicted profit, actual was loss) more than pessimistic errors. Puts get stricter penalties (6x vs 4x for calls) because put predictions are noisier. Sample weighting: bars with larger |P&L| get more weight (1 + 2*|max_pnl|).

Lower loss = model's P&L predictions are closer to actual outcomes.

## What the Log Output Means

```
Epoch  12 | train_loss=0.4321 | val_loss=0.5123 | gate_acc=0.82 | dir_acc=0.61
```

- **train_loss**: average loss on training data (goes down as the model learns)
- **val_loss**: average loss on 60 held-out validation days (the number that matters)
- **gate_acc**: fraction of bars where trade/no-trade was correct
- **dir_acc**: fraction of trade bars where call/put was correct
- **lr**: learning rate (starts at 3e-4, decays via cosine schedule)

The checkpoint with the lowest val_loss gets saved as `model.pt`.

## After Training

The saved model is replayed on the test fold (60 held-out days per fold, 5 folds in walk-forward CV). Replay feeds real market data through the model bar-by-bar, executes the trades it decides to make, and simulates P&L on a $10K account. The resulting equity curve produces the **score**:

```
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```

This score determines whether the experiment is kept or reverted.

## Where Training Runs

Training runs on an Akash H100 GPU, never locally. The local MacBook is only used for editing code, committing, and reading results. The GPU machine runs `v2/ops/run_experiment.py`, which calls `train.py` then runs replay and scoring.

## Key Hyperparameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| EPOCHS | 30 | Max passes through training data |
| BATCH_SIZE | 2048 | Samples per weight update |
| LR | 5e-4 | How big each weight nudge is |
| LOOKBACK | 30 | Bars of history the model sees |
| D_MODEL | 64 | Width of the transformer's internal representation |
| DEPTH | 3 | Number of transformer layers |
| TIME_BUDGET | 300s | Hard time cap on training |
| PNL_W | 1.0 | Weight for P&L regression loss |
| RISK_W | 0.5 | Weight for risk parameter loss |
