# v2 Lab Notebook

## Experiment 0: Initial Assessment (2026-04-03)

**Hypothesis:** A 163K-param causal transformer trained on tier-1 oracle labels
can learn selective, profitable trade entry decisions.

**Setup:**
- Dataset: 387,990 bars, 999 days, tier-1 oracle labels (35.4% trade rate)
- Model: 64d, 3 layers, 4 heads, 5 output heads (gate/direction/strike/risk/confidence)
- Training: 1 epoch (CPU, hit 120s budget), batch=1024, lr=3e-4

**Baselines (30 val days):**

| Baseline | PF | WR | TPD | Score |
|----------|-----|-----|-----|-------|
| Random | 0.797 | 34% | 2.3 | 0.0 |
| ATM-Always | 1.345 | 45% | 1.0 | 0.0 |
| Simple-Rules | 0.937 | 40% | 7.8 | -0.06 |

**Result (1 epoch model):**

| Metric | Value |
|--------|-------|
| PF | 45.2 |
| WR | 85.5% |
| TPD | 36.5 |
| Score | 0.0 |

**Diagnosis:**
- PF=45 is an artifact: oracle labels guarantee profitability by construction.
  The model just needs to say "trade" at any labeled bar and it wins.
- 36 TPD is way too many trades. Gate threshold 0.5 too low.
- Score = 0 because frequency penalty kills it (center=1.5, width=2.5).
- Gate accuracy 73% after 1 epoch means model hasn't learned selectivity.

**Key Insight:** The bottleneck is not "can the model find profitable trades"
(oracle labels make that easy) but "can it learn WHEN to be selective."
The score formula penalizes overtrading. The model needs to learn that
trading less with higher conviction beats trading more.

**Next:**
- Train more epochs for convergence
- Test higher gate thresholds (0.7, 0.8, 0.9) during replay
- Investigate class imbalance in gate labels (35% trade vs 65% no-trade)
- Consider weighting gate loss to penalize false positives more

---

## Experiment 1: Extended Training (2026-04-03)

**Hypothesis:** Training for 15 epochs with lr=5e-4 will improve gate selectivity
and reduce overtrading.

**Setup:** 15 epochs, batch=2048, lr=5e-4, CPU

**Result:** (pending)
