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

## Screening Rejects

(None yet — screening mode not yet implemented.)

## Current Hypotheses (Recovery Plan)

### H1: Side CE from contract_scores (exp_079)
Replace hard contract selection CE with a 2-class side CE derived from `contract_scores`. Compute `best_call_score` and `best_put_score` from the per-contract scores, then optimize 2-logit side CE on trade rows with both sides present. Remove the old hard `selection_loss`. Keep PnL regression and gate loss unchanged.

**Rationale:** Best-call vs best-put margin is 0.703 — side is clearly learnable. Hard one-hot selection across ~38 contracts has too many classes for the loss to provide useful gradient.

### H2: Soft within-side ranking (exp_080)
After side CE works, restrict to the oracle side on trade rows. Build target distribution as `softmax(oracle_side_pnl / 0.05)`. Optimize KL divergence between oracle-side target probs and model side-contract logits. Keep side CE from exp_079.

**Rationale:** Within a side, contracts are close in quality (0.024 median margin). Soft ranking via KL gives gradient to nearby contracts rather than hard one-hot CE.

### H3: Gate BCE reweighting (exp_081, conditional)
Only if overtrading remains after exp_080. Supervised rows only. `no_trade_weight = 3.0` via manual sample weighting on `gate_target == 0`.

**Rationale:** Gate calibration is downstream of selection quality. Fix selection first.
