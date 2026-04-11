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

## Current Hypotheses (Recovery Plan, Updated)

### H1: Tune soft KL selection (exp_079j+)
The soft KL approach works — it teaches both side AND within-side ranking simultaneously. Next steps:
- Try higher temperature (0.1, 0.2) to soften the target distribution and reduce winner-take-all
- Try SEL_W=2.0 or 3.0 to emphasize selection over PnL regression
- If direction balance improves, gate calibration may solve the drawdown issue

### H2: Gate BCE reweighting (exp_080, conditional)
Only after soft KL selection produces positive scores. `no_trade_weight = 3.0` to reduce overtrading.

### Abandoned: Side CE approach
The side CE (exp_079-079h) is fundamentally limited: it requires either a global bias (can't work for ~50/50 targets) or a learned head (overfits immediately). Soft KL selection supersedes it by teaching side and ranking together through a single, well-conditioned loss.
