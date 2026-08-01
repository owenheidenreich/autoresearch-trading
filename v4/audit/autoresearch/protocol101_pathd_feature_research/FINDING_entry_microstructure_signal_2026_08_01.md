# Directional finding — do the re-admitted microstructure features help ENTRY?

**STATUS: Tier-S research probe. Model-free. Training-only (fit-history sessions 1-14 sampled;
NO holdout touched). NOT a model, NOT feasibility, NO edge claimed.** Ran while Codex builds
the signed-17 baseline, to see whether widening the entry feature set is a promising next
direction (the legitimate version of "use what the old model had," now that Path D re-admits
microstructure the parity mask removed).

## What was tested
On 14 sampled fit-history sessions / 57 hourly decision-minutes / 406 band-filtered ($3-8)
candidate contracts: the **within-minute rank correlation** (Spearman, averaged over minutes)
of each candidate field vs its **forward-30-minute upside** (causal max-mid excursion). Then
the **incremental** signal after residualizing on `abs_moneyness` (the dominant signed-17
geometry driver). Model-free; no fitting.

## Result

| Field (re-admitted microstructure = not in signed-17) | Raw within-min ρ | Incremental over moneyness | Robustness |
|---|--:|--:|---|
| **size_imbalance** (bid_sz vs ask_sz) | **+0.167** | **+0.129** | t≈**2.59**, 58% of minutes positive |
| ask_size | −0.126 | −0.083 | t≈−2.21 |
| open_interest | +0.038 | +0.089 | weak |
| bid_size | +0.087 | +0.077 | t≈1.4 (n.s.) |
| depth_total | −0.026 | +0.071 | weak |
| spread / spread_frac / volume | ≈0 | ≈0 | negligible |
| (control) abs_moneyness — in signed-17 | −0.117 | — | t≈−1.79 |

**Order-book size imbalance is the standout** — the strongest single candidate-ranking signal,
**stronger than the moneyness geometry the signed-17 relies on**, statistically significant
(t≈2.6 across 57 clustered minutes), and it **survives controlling for moneyness** (+0.129
incremental) → genuinely additive to the signed-17. Open interest is a weak secondary. Spread,
volume, quote-age carry ~nothing here.

Notably, `bid_size`/`ask_size` are exactly the fields the **parity-driven microstructure mask
removed** — so that mask was discarding a real (if moderate) entry signal that Path D
legitimately re-admits (same-vendor). That's the concrete support for the owner's hypothesis.

## Honest caveats
- Directional probe only: 14 sampled sessions, 57 minutes, 406 rows, one label (fwd-30min MFE),
  model-free rank correlation, NO holdout, NO model.
- Magnitude is **moderate**, not huge (ρ≈0.13-0.17). Per-minute rhos are noisy (58% positive).
- Couldn't control for delta/gamma (null in the base normalized view — greeks live in the
  derived view), so the incremental control is moneyness-only.
- **"Rank signal exists" ≠ "tradeable edge after fees/OOF/holdout."** Order-book imbalance
  signals often decay fast and can be eaten by spread/fees on 0DTE. This is a promising
  hypothesis to TEST properly, not proven edge.

## Recommendation (a direction, not an action)
If Codex's signed-17 baseline underwhelms — or as a follow-up regardless — **test a widened
entry = signed-17 + order-book imbalance (and possibly OI)** under the SAME fair Path-D
firewall + guards (out-of-sample, matched-random, action-conditioned calibration, one-shot
holdout). That is the disciplined version of "use the extra features." Do NOT add it to the
current run (it would change two things at once and break decision #1's clean baseline).

Artifact: `entry_feature_probe.json` (this directory).
