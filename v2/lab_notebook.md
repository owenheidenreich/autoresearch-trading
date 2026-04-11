# v2 Lab Notebook — Exact-Chain Era

Pre-exact-chain history archived to `archive/v2_historical/logs/lab_notebook_pre_exact_chain.md`.

## Exact-Chain Diagnosis

The v4 exact-chain rebuild replaced the old ATM/OTM ladder with per-bar executable contract scoring.

Key dataset properties:
- Dataset fingerprint: `46f2d184e186496f`
- 986 unique trading days with full-chain sidecars
- Exact-chain trade rows average ~38 executable contracts per bar
- Top-vs-second contract margin: ~0.024 median (very tight)
- Best-call vs best-put margin: ~0.703 median (side is strongly learnable)

Conclusion: **side is learnable; hard one-hot contract selection is too sharp.** The model needs to learn side first, then soft within-side ranking, before attempting hard contract selection.

## Official Experiment Log

| Exp | Score | Folds | Status | Hypothesis |
|-----|-------|-------|--------|------------|
| 074 | 0.000 | [0,0,0,0,0] | revert | exact-chain baseline attempt — zero trades all folds |
| 075 | 0.000 | [0,0,0,0,0] | revert | decouple gate from selection (three independent losses) — still zero trades |
| 076 | -0.040 | [0,0,0,-0.2,0] | revert | gate_threshold=-100 (disable absolute threshold) — 1 fold fires |
| 077 | -0.300 | [-0.3,-0.3,-0.3,-0.3,-0.3] | revert | SEL_W=0 drop selection CE, rely on PNL regression — negative all folds |
| 078 | -0.740 | [-1,-0.5,-1,-1,-0.2] | revert | PNL_W=5.0 stronger P&L regression — more trades but negative edge |

All five experiments failed. The model fires too few trades (gate collapse) or fires trades with negative edge (contract selection noise).

## Screening Rejects (2026-04-10)

### Side CE approach (exp_079 through exp_079h)

| Screen | Change | Side Loss | Dir Balance | Result |
|--------|--------|-----------|-------------|--------|
| 079 | side CE from max(call/put scores) | 0.6498 frozen | 0C/78P | side CE zero gradient via max() |
| 079b | SEL_W=3.0, PNL_W=1.0 | 0.6498 frozen | 132C/13P | reweight didn't help, still frozen |
| 079c | logsumexp instead of max | 0.6851→0.6785 | 0C/126P | tiny movement, not enough |
| 079d | PNL_W=0 diagnostic | 0.6758→0.6574 | 642C/4P | PnL interference confirmed but not root cause |
| 079e | learned side_bias scalar | 0.6727 frozen | 13C/0P | PnL regression pins global bias to 0 |
| 079f | decouple base_scores from side_bias | 0.6684 frozen | 13C/0P | still pinned: scalar can't learn ~50/50 target |
| 079g | context-dependent side_head(d→1) | 0.67→4.96(overfit) | **8C/7P balanced!** | side CAN learn but massively overfits |
| 079h | dropout(0.3) + SEL_W=1.0 | 0.66→2.10(overfit) | 6C/153P | slower overfit, still best_epoch=1 |

**Key findings from side CE exploration:**
1. `torch.max()` gives gradient to only 1 contract per side — useless for 38-class space
2. `logsumexp()` helps marginally but PnL regression dominates
3. A global scalar side_bias can't learn because oracle side is ~50/50 (gradients cancel)
4. A context-dependent side_head CAN learn side but overfits immediately (96→1 linear memorizes)
5. The side signal in the context embedding is real but too noisy for a single linear layer to generalize

### Soft KL selection (exp_079i) — MOST PROMISING

| Screen | Change | Sel Loss | Dir Balance | Result |
|--------|--------|----------|-------------|--------|
| 079i | KL(softmax(pnl/0.05), softmax(scores)) | 2.02→1.72 | **16C/54P** | best_epoch=11, beats ATM+Trailing |

**This is the breakthrough approach.** Soft KL selection:
- Actually learns across epochs (best_epoch=11, not 1)
- Selection loss decreases on validation (2.02 → 1.72)
- Achieves partial direction balance (23% calls)
- Beats 2 of 4 baselines (first time in exact-chain era)
- Still fails on drawdown (102.9%) and WR (27.1%)
- Temperature 0.05 may be too sharp — try 0.1 or 0.2

## Screening Session 2 (2026-04-10, exp_080–089)

### Root cause diagnosis
The single `score_head` output per contract is trained by three conflicting losses: PNL regression (pushes scores toward realized P&L, mostly negative), soft KL selection (ranks contracts), and gate BCE (decides trade/no-trade). PNL regression dominates and causes direction collapse. Supervised gate labels are 80/20 skewed toward trade, making "always trade" the BCE-optimal strategy.

### Screening results

| Screen | Change | Dir Balance | WR | Trades | DD | Key Finding |
|--------|--------|-------------|-----|--------|-----|-------------|
| 080 | PNL_W=0, TEMP=0.20 | **179C/244P (42%)** | **36.2%** | 423 | 104% | **Best config.** Removing PNL regression fixed direction |
| 081 | + NO_TRADE_W=3.0 | 62C/143P (30%) | 32.2% | 205 | 103% | Gate oscillated, unstable |
| 082 | + PNL_W=0.1 | 25C/197P (11%) | 28.8% | 222 | 102% | Even 0.1 PNL_W reimposed direction collapse |
| 083 | + side_head (d//4, drop=0.3) | 568C/7P (1%) | 31.3% | 575 | 101% | Side head corrupted shared context encoder |
| 084 | GATE_W=3.0 | 86C/277P (24%) | 33.9% | 363 | 101% | No effect on trade rate (still 93.7%) |
| 085 | gate_threshold=0.0 | 228C/132P (63%) | 32.2% | 360 | 102% | Filtered good trades, not bad ones |
| 086 | side-masked KL targets | 316C/84P (79%) | 30.5% | 400 | 103% | Lost side signal, call-biased |
| 087 | SOFT_TEMP=1.0 | 250C/106P (70%) | 31.7% | 356 | 100% | Too soft, direction worsened |
| 088 | context*contract interaction | 191C/28P (87%) | 31.1% | 219 | 100% | dir_acc improved (0.47→0.53) but call-biased |
| 089 | L2 score regularization | — | — | — | — | Not yet run |

### Key findings from session 2

1. **PNL regression at ANY weight causes direction collapse** (exp_082 confirmed at PNL_W=0.1)
2. **Gate label skew is 80/20 trade/no-trade** among supervised rows — "always trade" is BCE-optimal regardless of GATE_W
3. **Auxiliary heads corrupt shared context** (exp_083 side_head destroyed direction)
4. **KL temperature has diminishing returns** — 0.20 is best; 0.05 too sharp, 1.0 too soft
5. **Score magnitudes drift with PNL_W=0** — KL is scale-invariant, so scores grow unbounded, gate can't learn stable threshold
6. **Multiplicative interaction improved dir_acc trending** (exp_088: 0.47→0.53) but created systematic call bias

### Remaining blockers (exp_080 baseline)
- **Trade rate 93.7%**: gate always says trade (80/20 label skew)
- **WR 36.2%**: model picks wrong contracts (dir_acc ~0.49, random on side)
- **DD 104%**: consequence of overtrading + wrong contracts

## Diagnostic Investigation (2026-04-10)

Before running more experiments, ran two local diagnostics to determine if the problem is the model, the evaluation, or the data.

### Diagnostic 1: Oracle Replay (perfect contract selection)

| Window | Score | WR | DD | C/P | Trades |
|--------|-------|-----|-----|-----|--------|
| Test (60d) | **6.000** | 100% | 0% | 267/268 | 535 |
| Train (846d) | **6.000** | 100% | 0% | 3746/3718 | 7464 |

**Verdict: Evaluation gates are achievable.** Perfect contract selection scores 6.0 (max) on both windows. 100% WR, zero drawdown, balanced direction. The problem is purely in model training, not the evaluation harness.

### Diagnostic 2: Direction Signal (logistic regression)

| Features | Train acc | Test acc | Baseline |
|----------|-----------|----------|----------|
| Current bar (47 features) | 61.1% | **60.9%** | 52.6% |
| Window mean/std (lb=30, 141 features) | 62.6% | 60.2% | 52.6% |
| Window mean/std (lb=60, 141 features) | 62.4% | 60.5% | 52.6% |
| Window mean/std (lb=90, 141 features) | 62.4% | 60.9% | 52.6% |

**Verdict: Direction signal EXISTS and is LEARNABLE.** A simple logistic regression achieves 60.9% test accuracy on call/put prediction, 8 points above baseline. Longer lookback windows don't help — the signal is in the current bar's features, not in temporal patterns.

Top predictive features for direction:
1. Feature 46 (put_call_txn_ratio): strongest predictor
2. Feature 27 (trend_5min): directional momentum
3. Feature 9 (gamma): options market structure
4. Feature 16 (rsi_7): momentum oscillator

### Implications

1. **The evaluation is fine** — oracle achieves perfect score, so gates are not too tight
2. **Direction is learnable** — 61% accuracy from simple logistic regression means a transformer should do better
3. **Lookback doesn't matter** — current-bar features already capture the direction signal (no benefit from 60 or 90 bars)
4. **The bottleneck is the model architecture** — specifically how direction learning interacts with contract scoring in the single score_head

### Current hypotheses (updated)

The direction signal is real (61% learnable) and the evaluation is achievable (oracle=6.0). The model's architecture prevents it from exploiting the signal because:

1. PNL regression dominates and overrides direction learning
2. Without PNL regression, scores drift and the gate can't calibrate
3. The KL loss doesn't explicitly teach direction — it teaches ranking which may not propagate direction gradient effectively

**H-score-reg**: L2 regularization on scores (SCORE_REG_W=0.01) to prevent magnitude drift and stabilize gate — queued as exp_089, not yet run.

**H-interaction-normed**: Multiplicative interaction (exp_088) showed dir_acc improvement — retry with layer normalization on the interaction to prevent call bias.

**H-two-stage**: Decompose selection into explicit side classification + within-side KL ranking. Requires careful architecture to avoid exp_083's context corruption.

### Abandoned approaches
- Side CE (exp_079-079h): fundamentally limited
- PNL regression at any weight: causes direction collapse
- Gate reweighting (NO_TRADE_W): destabilizes training
- Gate threshold tuning: filters randomly, not by quality
- Side-masked KL: loses side signal
- Auxiliary side head: corrupts context
