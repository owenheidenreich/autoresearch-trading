# Critical Investigation: Profit Factor Metric Disconnect

**Date:** 2026-04-15
**Severity:** Critical — all experiment conclusions based on PF are unreliable
**Status:** Diagnosed, fix identified, not yet applied

## The Problem

The equity curve (`equity.html`) shows the best AWAC-RL agent losing **81.5% of a $10,000 portfolio** across 300 out-of-sample days (761 trades, final equity $1,852). Yet the reported profit factor across the same trades is **PF 0.974** — implying near-breakeven performance.

These two numbers cannot both be correct. Investigation reveals **the PF metric is wrong**.

## Root Cause: Three Compounding Bugs

### Bug 1: Profit Factor computed on unweighted percentage returns

**File:** `v2/core/metrics.py`, lines 106-124

```python
pnls = [t.net_pnl_pct for t in trades]      # per-trade fractions (e.g., -0.09)
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]
m.gross_profit = sum(wins)                    # sum of positive fractions
m.gross_loss = abs(sum(losses))               # sum of loss fractions
m.profit_factor = m.gross_profit / m.gross_loss  # fraction / fraction
```

This treats every trade as equal regardless of dollar size. A 50% gain on a $1.25 option ($62.50) is weighted the same as a 50% loss on a $35.40 option ($1,770). The actual dollar impact differs by **28x**, but the PF counts them identically.

**Impact:** PF 0.974 (percentage-weighted) vs PF 0.830 (dollar-weighted). The agent loses $8,148 across 300 days while appearing near-breakeven.

### Bug 2: `intent.qty` is always 0 in sequential replay trades

**File:** `v2/replay.py`, line 493-505

The sequential replay creates SimulatedTrade objects with a bare `TradeIntent(trade=True, bar_index=..., decision_day=...)`. The `qty` field defaults to 0 in the TradeIntent dataclass.

This breaks two downstream calculations in `compute_metrics()`:
- **`dollar_weighted_profit_factor`** (line 179): multiplies by `intent.qty`, gets 0 for every trade → DWPF = 0.000
- **`max_account_drawdown`** (line 219): dollar_pnl = `net_pnl_pct * entry_price * 100 * qty` = 0 for every trade → drawdown always 0%

**Impact:** The dollar-weighted PF and account drawdown metrics are computed but are always zero. The hard gate `max_account_drawdown > 0.25` (line 297) never triggers because DD is always reported as 0%.

### Bug 3: Score formula uses the broken PF

**File:** `v2/core/metrics.py`, lines 304-315

```python
pf_term = min(metrics.profit_factor, 4.0)   # uses unweighted percentage PF
score = (0.5 * sortino_term + 0.5 * pf_term) * pdr * dd_mult
```

The promotion score is 50% driven by the broken PF. With PF 0.974 instead of the real 0.830, and DD_mult=1.0 instead of ~0.0 (real DD is >25%), every score computed this session was inflated.

## Per-Fold Reality Check

| Fold | Pct PF (reported) | Dollar PF (real) | Net $ (real) | Pct PF said... | Dollar PF says... |
|------|-------------------|-----------------|-------------|----------------|-------------------|
| 0 | 1.189 | 0.896 | -$937 | "Profitable!" | Losing money |
| 1 | 0.718 | 0.563 | -$5,404 | "Losing badly" | Even worse |
| 2 | 0.948 | 1.004 | +$26 | "Nearly even" | Barely breakeven |
| 3 | 0.805 | 0.793 | -$2,096 | "Losing" | Confirmed losing |
| 4 | 1.232 | 1.028 | +$264 | "Solidly profitable!" | Barely breakeven |
| **Overall** | **0.974** | **0.830** | **-$8,148** | "Almost there!" | **Losing badly** |

**Fold 0, which we spent hours celebrating as PF 1.189, is losing $937 in dollar terms.**
**Fold 4, reported as PF 1.232, makes only $264 — nearly breakeven.**

## What This Means for Past Research

### Conclusions that remain valid:
- **Training stability improvements are real.** AWAC produces smoother training curves than REINFORCE. This is an engineering win regardless of PF.
- **Behavioral structure is real.** The agent demonstrates thesis persistence, regime-aware side selection, and disciplined participation. These are genuine learned behaviors.
- **Encoder unfreezing was correctly identified as negative.** The direction of that result (worse than frozen) holds under any metric.

### Conclusions that are now unreliable:
- **"PF improved from 0.963 to 0.974"** — Both numbers are from the broken metric. The real dollar PFs may tell a different story about which model is better.
- **"The training method was the binding constraint"** — This was based on fold-0 PF 1.189, which is actually dollar-PF 0.896 (losing money).
- **All promotion gate decisions** — The DD gate never fired because DD was always 0%. Models that should have been rejected were evaluated as passing.
- **All fold-level comparisons** — Percentage PF can rank folds differently than dollar PF.

### Whether we need to retrain:
**No.** The model architecture and training pipeline are not broken. The bug is in the **evaluation metric**, not the training objective. The AWAC training loop uses the environment's reward (which correctly computes dollar P&L via `_apply_trade_result`), so the model was trained on the right signal. The problem is that the metrics we used to judge the output do not accurately reflect dollar performance.

However, the model is clearly not profitable. Whether that's because:
(a) PF 0.83 in dollars is close enough to improve with better features/training, or
(b) the model is fundamentally losing money and needs a different approach

...requires re-evaluating all past results with the corrected metric before deciding.

## Files That Need To Change

### Fix 1: `v2/core/metrics.py` — Use dollar-weighted PF as the primary metric

**Lines 106-124:** Replace percentage-based PF with dollar-weighted PF as the primary `profit_factor`.

```python
# Current (BROKEN):
pnls = [t.net_pnl_pct for t in trades]
m.gross_profit = sum(p for p in pnls if p > 0)
m.gross_loss = abs(sum(p for p in pnls if p <= 0))
m.profit_factor = m.gross_profit / m.gross_loss

# Fixed:
qty_default = 1  # assume 1 contract when intent.qty is 0
dollar_pnls = [
    t.net_pnl_pct * t.entry_price * contract_multiplier * max(t.intent.qty, qty_default)
    for t in trades
]
m.gross_profit = sum(p for p in dollar_pnls if p > 0)
m.gross_loss = abs(sum(p for p in dollar_pnls if p < 0))
m.net_pnl = sum(dollar_pnls)
m.profit_factor = m.gross_profit / m.gross_loss if m.gross_loss > 0 else (10.0 if m.gross_profit > 0 else 0.0)
```

### Fix 2: `v2/core/metrics.py` — Fix account drawdown with qty fallback

**Line 219:** Use `max(t.intent.qty, 1)` instead of `t.intent.qty` so trades with qty=0 still get counted.

```python
# Current (BROKEN when qty=0):
dollar_pnl = t.net_pnl_pct * t.entry_price * contract_multiplier * t.intent.qty

# Fixed:
dollar_pnl = t.net_pnl_pct * t.entry_price * contract_multiplier * max(t.intent.qty, 1)
```

### Fix 3: `v2/replay.py` — Set qty=1 in sequential replay trades

**Line 494:** When creating SimulatedTrade from sequential replay, set `qty=1` on the TradeIntent.

```python
# Current:
trade = SimulatedTrade(
    intent=TradeIntent(trade=True, bar_index=info.bar, decision_day=day),
    ...
)

# Fixed:
trade = SimulatedTrade(
    intent=TradeIntent(trade=True, bar_index=info.bar, decision_day=day, qty=1),
    ...
)
```

### Fix 4: `v2/plot_trades.py` — Use max(qty, 1) in equity computation

**Line 292:** Same qty fix for plotting.

```python
# Current:
pnl_dollar = t.net_pnl_pct * t.entry_price * contract_multiplier

# Confirmed correct (uses entry_price directly, not qty) but should be explicit:
pnl_dollar = t.net_pnl_pct * t.entry_price * contract_multiplier * max(t.intent.qty, 1)
```

### Optional but recommended: Keep percentage PF as a secondary metric

Rename the current percentage PF to `pct_profit_factor` for diagnostic use, but make the primary `profit_factor` always dollar-weighted.

## Verification After Fix

After applying all fixes:
1. Re-run the 5-fold AWAC evaluation
2. Verify `profit_factor` matches the dollar-weighted value
3. Verify `max_account_drawdown` is non-zero and realistic
4. Verify the DD hard gate (>25%) correctly rejects losing models
5. Regenerate equity.html and confirm it matches the metrics
6. Re-evaluate whether any prior experiment result changes direction under corrected metrics

## Impact on Project Timeline

**Research time lost:** The AWAC improvements, encoder unfreezing, and session memory experiments used the broken metric for evaluation. Their percentage-PF conclusions need to be re-checked against dollar-PF. The behavioral findings (thesis persistence, regime-aware side selection, training stability) remain valid.

**Compute cost:** GPU time was spent training models that were evaluated against a broken metric. The models themselves are not necessarily bad — they were trained on correct rewards — but we don't know if the "improvements" we measured are real in dollar terms.

**Next step:** Fix the metric, re-evaluate all checkpoints with corrected metrics, then decide whether the current model family is dollar-viable or needs fundamental changes.
