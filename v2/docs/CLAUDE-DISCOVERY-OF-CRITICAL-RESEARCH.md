# Deep Hypothesis: Teaching the Model to Actually Trade

> **SUPERSEDED (2026-04-08):** The two core problems identified here (P&L capped at 49.3%, risk labels all constant) were solved by the Tier 3 label rebuild, not by the MFE approach proposed below. Current data.pt uses Tier 3 grid-search labels: P&L up to 1.19, variable risk params (stop std=0.10, target std=0.32, hold std=72.1). MFE-based training was attempted in exp_068-069 and failed (WR dropped 77% to 47%, trades/day jumped 2.5 to 5.6). The model reverted to P&L training on Tier 3 labels (exp_070). This doc is retained as historical research context.

## Context

Score is 5.52 (walk-forward, 5 folds). 67 experiments. Hyperparameter tweaks exhausted. The user's insight: the model is scalping with heavy risk management, not trading. Pickles categorizes trades as big wins, small wins, small losses, big losses. Our model can't even distinguish these categories.

## The Root Cause: Crushed Labels

**Finding 1: P&L labels are hard-capped at 49.3%.**
```
metadata: fixed_stop=0.3, fixed_target=0.5, fixed_hold=30
Max call P&L: 0.493, Max put P&L: 0.494
ZERO bars with P&L > 50%
```
The label generation simulates with target=0.50, so no trade can ever show > ~50% gain. The model has NEVER seen a big win in training.

**Finding 2: Risk labels are ALL IDENTICAL.**
```
stop_pct:    mean=0.300, std=0.000 (CONSTANT)
target_pct:  mean=0.500, std=0.000 (CONSTANT)
max_hold:    mean=30.0,  std=0.0   (CONSTANT)
```
RISK_W=0.5 wastes 33% of the loss signal training the risk head to output a constant. The model cannot learn adaptive risk management because the labels contain zero information.

**Finding 3: The actual trades go much bigger.**
Raw option prices are in data.pt. Uncapped MFE from forward-looking price data:
```
Window 30 bars:  10% of bars have MFE > 100%, max 12x
Window 120 bars: 39% of bars have MFE > 100%, max 45x
Window 240 bars: 57% of bars have MFE > 100%, max 67x
```
The model is trained to predict "will this go up 50%?" when the real question is "is this a 50% scalp or a 1,200% runner?"

**Finding 4: Trailing stops compress the upside.**
Current trailing tiers lock at +80% once unrealized hits +120%. With a 50% target, TAKE_PROFIT fires before the trade even reaches the first meaningful trailing tier. Every trade is a scalp by design.

## The Hypothesis

**The model needs to predict TRADE QUALITY (how far can this go?), not just direction.** By computing MFE from raw option prices inside train.py, we can teach the model Pickles' taxonomy:

- **Runner bars** (MFE > 100%): Set wide target (150-300%), long hold. Trailing stops lock in 50-80% on reversals, but let parabolic moves run.
- **Scalp bars** (MFE 20-100%): Set tight target (30-50%), short hold. Quick TAKE_PROFIT.
- **Skip bars** (MFE < 20%): Gate says no trade.

This changes the model from "can I reliably capture 50%?" to "how big can this trade be, and how should I manage it?"

---

## Implementation Plan

### Step 1: Compute MFE Labels in train.py

data.pt contains `atm_call_prices` and `atm_put_prices` (386K bars of raw option prices). We can compute uncapped MFE and MAE directly:

```python
# In TradeDataset.__init__ or as a preprocessing step in train():
def compute_mfe_labels(atm_prices, bar_of_day, dates, window=120):
    """Compute max favorable excursion from raw option prices."""
    N = len(atm_prices)
    mfe = torch.zeros(N)
    mae = torch.zeros(N)
    for i in range(N - window):
        if dates[i] != dates[i + window]:  # same day only
            continue
        entry = atm_prices[i]
        if entry <= 0.5 or torch.isnan(entry):
            continue
        fwd = atm_prices[i+1:i+window+1]
        valid_fwd = fwd[~torch.isnan(fwd) & (fwd > 0)]
        if len(valid_fwd) < 5:
            continue
        mfe[i] = (valid_fwd.max() / entry) - 1.0
        mae[i] = (valid_fwd.min() / entry) - 1.0
    return mfe, mae
```

Compute for both call and put ATM prices. Store as `mfe_call`, `mfe_put`, `mae_call`, `mae_put`. This runs once at training start (vectorizable with rolling max).

**Files modified:** `v2/train.py` only.

### Step 2: Replace P&L Prediction with MFE Prediction

Current model predicts `call_pnl` and `put_pnl` (capped at 49.3%). Replace with:

```python
# New prediction heads
self.call_mfe_head = nn.Sequential(
    nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
    nn.Linear(d // 2, 1),
)
self.put_mfe_head = nn.Sequential(
    nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
    nn.Linear(d // 2, 1),
)
```

MFE labels have much richer signal: continuous from 0 to 67x, with meaningful variance. The model learns to predict the POTENTIAL of each bar, not a capped outcome.

**Gate logic changes:**
```python
# Old: gate = max(pred_call_pnl, pred_put_pnl) > 0 (trade if any direction profitable)
# New: gate = max(pred_call_mfe, pred_put_mfe) > threshold (trade if enough upside potential)
```

**Direction logic unchanged:** direction = argmax(pred_call_mfe, pred_put_mfe).

### Step 3: MFE-Conditioned Risk Head

Replace constant risk targets with MFE-derived targets. The risk head should see the model's own MFE prediction:

```python
# Risk head input includes P&L context
pnl_context = torch.cat([call_mfe.detach(), put_mfe.detach()], dim=-1)  # (B, 2)
risk_input = torch.cat([film_risk_out, pnl_context], dim=-1)  # (B, d+2)
self.risk_head = nn.Sequential(
    nn.Linear(d + 2, d // 2), nn.GELU(), nn.Dropout(dr),
    nn.Linear(d // 2, 3),
)
```

Risk targets derived from MFE:
```python
# Target: fraction of predicted MFE (let the trade capture most of its potential)
target_label = mfe_best * 0.7  # aim for 70% of maximum favorable excursion
# Hold: longer for higher MFE (more time to develop)
hold_label = torch.clamp(mfe_best * 100, 30, 350) / hold_hi  # scale by potential
# Stop: wider for higher MFE (give it room, but proportional)
stop_label = torch.clamp(mae_best.abs() * 0.5, 0.15, 0.50)  # half of MAE as stop
```

### Step 4: Loss Function Changes

**Kill the old P&L regression. Add MFE regression:**

```python
# MFE loss: Huber regression on uncapped MFE
# Log-transform to handle the heavy tail (MFE ranges 0 to 67x)
pred_call_mfe = outputs['call_mfe'].squeeze(-1)[valid]
true_call_mfe = torch.log1p(mfe_call[valid])  # log(1 + MFE) to compress range
call_mfe_loss = F.huber_loss(pred_call_mfe, true_call_mfe, delta=1.0, reduction='none')
```

**Runner-weighted sampling:**
```python
# Pickles' insight: optimize for finding big wins
# Current: 1 + 2*|max_pnl| (symmetric, caps at ~1.98)
# Proposed: exponential weighting on positive MFE
best_mfe = torch.max(mfe_call[valid], mfe_put[valid])
sample_weight = 1.0 + 5.0 * torch.clamp(best_mfe, 0, 3.0)
# A +300% MFE bar gets 16x weight vs a 0% bar
# This teaches the model that finding runners is THE priority
```

**Risk loss with MFE-derived targets (not constants):**
```python
# Now the risk targets have actual variance
risk_loss = F.huber_loss(risk_out, mfe_derived_risk_target, delta=0.5)
```

### Step 5: Policy Changes

In `v2/core/policy.py`:

```python
# Old: target_range = (0.15, 1.65)  -- never used beyond 0.50 anyway
# New: target_range = (0.15, 5.00)  -- let the model express "this could 5x"
target_range: tuple[float, float] = (0.15, 5.00)

# Old: max_hold_range = (10, 250)
# New: max_hold_range = (10, 350)  -- 350 bars ≈ 5.8 hours, almost full day
max_hold_range: tuple[int, int] = (10, 350)
```

Keep `exit_policy = "TRAILING"`. Here's why:
- With target=5.00 and TRAILING, the trade runs until either:
  - Hits target (5x return) -- TAKE_PROFIT
  - Reaches trailing tier and reverses -- TRAILING_STOP locks in 25-80%
  - Hits stop loss -- controlled loss
- Trailing stops PROTECT the big wins from giving everything back
- But with wide targets, the TAKE_PROFIT ceiling is no longer the constraint
- Best of both worlds: protection on reversals, room to run on trends

### Step 6: MFE Computation Window

**Key design decision:** What forward window to use for MFE computation?

Arguments for each:
- **30 bars (current hold):** Conservative. Matches current model's scope. But the current model is the problem.
- **120 bars (2 hours):** Good balance. 39% of bars show MFE > 100%. Captures most of the upside without requiring full-day holds.
- **240 bars (4 hours):** Aggressive. 57% show MFE > 100%. But requires very long holds that theta decay might erode.

**Recommendation: Start with 120 bars.** This gives the model 2 hours of forward information to learn from. The hold_hi in policy should be 350 to let the model hold for the MFE to develop. The trailing stops provide protection during the hold.

**Why not 30?** Because 30 bars only shows 10% of runners. The model would still learn to scalp. 120 bars reveals the trade's TRUE potential.

---

## Architecture Enhancements (from original research, still relevant)

### Step 7: CLS Token for Better Temporal Aggregation

Replace last-token pooling with a learned [CLS] token. This helps the model attend to the MOST RELEVANT bars in the lookback window (e.g., the IB break bar, the volume spike bar, the VWAP cross).

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
# In forward:
cls = self.cls_token.expand(B, -1, -1)
h = torch.cat([cls, h], dim=1)  # (B, T+1, d)
# Use h[:, 0, :] as representation (attends to all positions)
```

### Step 8: Multi-Scale Input (Elder's Triple Screen)

Feed both 1-min (30 bars) and 5-min (30 bars = 150 min history) timescales:

```python
# Downsample fine features to create coarse
coarse = window[:, ::5, :]  # every 5th bar -> 6 bars of 5-min data
# Or reshape: average groups of 5 consecutive bars
coarse = window.reshape(B, -1, 5, F).mean(dim=2)  # (B, 6, F)
```

This directly addresses the known weakness: "regime encoder reads only last bar, no intraday trend context." Multi-scale input gives 150 minutes of trend context. Critical for detecting runner conditions (trend days show up on the 5-min scale).

---

## Feature Pipeline Rebuild (Phase B, separate from experiment loop)

After proving MFE-based training works in the experiment loop, rebuild data.pt with richer features:

1. **Raw VIX level** (not z-scored) -- regime classifier for risk calibration
2. **GEX from Polygon OI** -- dealer positioning, trend vs mean-reversion regime
3. **Straddle momentum** (30/60-bar ATM straddle returns) -- Oxford paper's strongest signal
4. **IV surface shape** (skew, butterfly from 82-strike wide grid)
5. **Volume-price divergence count** (Coulling/Wyckoff temporal signal)

---

## Execution Order

### Session 8: MFE-Based Training (Priority 1)

| Exp | Change | What We Learn |
|-----|--------|--------------|
| 068 | Compute MFE labels in train.py (120-bar window). Train call_mfe/put_mfe heads. Runner-weighted sampling. Keep current risk head (RISK_W=0). | Does MFE prediction improve direction accuracy? |
| 069 | Add MFE-conditioned risk head (risk sees predicted MFE). Widen target_range to 5.00, hold to 350. | Does dynamic risk improve score? |
| 070 | Log-transform MFE targets to handle heavy tail. Adjust Huber delta. | Better regression on skewed distribution? |
| 071 | CLS token replacing last-token pooling. | Better temporal pattern detection? |
| 072 | Multi-scale input (1-min + 5-min). | Better trend/regime detection? Runner identification? |

### Session 9: Refinement

Based on Session 8 results:
- Tune MFE window (30 vs 120 vs 240)
- Tune runner weighting scale
- Tune MFE-to-target conversion
- Direction-dependent MFE thresholds (puts decay faster)
- Test STOP_TP_TIME on high-confidence runner predictions

### Phase B: Feature Pipeline Rebuild

After Session 8-9 results stabilize:
1. Rebuild data.pt with raw VIX, GEX, straddle momentum, IV surface
2. Rebuild labels with Tier 3 grid search (variable risk params)
3. Re-run experiment loop with richer data

---

## Why This Is Different From Prior Approaches

| Aspect | Old (Current) | New (MFE-Based) |
|--------|--------------|-----------------|
| Training target | P&L capped at 49.3% | Uncapped MFE (0 to 67x) |
| Risk labels | ALL constant (0.30, 0.50, 30) | MFE-derived (variable) |
| Trade quality | Binary (profitable/not) | Continuous (scalp to runner) |
| Risk head | Learns nothing (constant target) | Learns adaptive risk from MFE |
| Target range | 0.15-1.65 (never > 0.50 used) | 0.15-5.00 (model decides) |
| Sample weighting | Symmetric 1+2*\|pnl\| | Asymmetric, runners 16x weighted |
| What model learns | "Is this bar profitable at 50% target?" | "How big can this trade be?" |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| MFE prediction is harder than capped P&L (noisier) | Log-transform to compress range. Huber loss handles outliers. |
| Model over-predicts MFE (always says "runner") | Asymmetric loss: heavily penalize predicting runner when actual MFE is low |
| Wide targets reduce trade count (gate too selective) | Monitor trade count per fold. Can adjust gate threshold in policy.py |
| Long holds increase exposure to theta decay | Direction-dependent hold caps. MFE-conditioned hold (not blind extension) |
| MFE computation adds training overhead | Precompute once, vectorize with rolling max. ~30s overhead for 386K bars |
| Trailing stops still cap upside at 80% | True for non-parabolic moves. But 80% lock >> current 50% target. Progress. |

## Verification Plan

For each experiment:
1. `python -m py_compile v2/train.py` -- syntax check
2. `./v2/ops/deploy.sh run_one exp_NNN` -- train 5 WF folds (~25 min)
3. Compare: mean score, per-fold scores, fold std, trade count, direction split
4. **NEW: Check MFE calibration** -- do bars where model predicts high MFE actually have high realized MFE?
5. **NEW: Check trade P&L distribution** -- do we see big wins emerging (not just 50% scalps)?
6. KEEP if: mean score > 5.523 AND trade profile shows emerging Pickles taxonomy (big/small wins + controlled losses)
7. Post-experiment: plot_trades, plot_progress, analyze_losses
8. Log to results.tsv and lab_notebook.md

## What NOT to Try

- More hyperparameter tweaks (exhausted)
- Gate threshold changes without new information
- Loss function variants without new targets
- Anything that doesn't change the fundamental "capped at 50%" problem

---

## Critical Files

| File | Role | Changes |
|------|------|---------|
| [v2/train.py](v2/train.py) | Model + training | MFE computation, new heads, new loss, new weighting |
| [v2/core/policy.py](v2/core/policy.py) | Trading policy | Widen target_range, extend max_hold_range |
| [v2/core/simulator.py](v2/core/simulator.py) | Trade execution | IMMUTABLE (trailing tiers fixed) |
| [v2/replay.py](v2/replay.py) | Evaluation | IMMUTABLE (model_to_intent uses policy ranges) |
| [v2/core/labels.py](v2/core/labels.py) | Label generation | IMMUTABLE (explains why current labels are constant) |
