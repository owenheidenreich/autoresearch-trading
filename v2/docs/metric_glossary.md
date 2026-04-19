# Metric Glossary

One-sentence "means" and "does not mean" for every metric this research system
reports. The purpose is to prevent score inflation by storytelling: a metric is
only as useful as the claim it supports, and the claim is bounded by what the
metric does *not* prove.

If you add a metric, add an entry here before you use it in a conclusion. If a
reported result leans on a metric that has no entry here, that reported result
is not yet evidence.

Precedence: if a code-level docstring disagrees with this glossary, the code
wins and this file is updated to match. Flag the drift.

## CPU audit metrics (signal layer)

### AUC (area under ROC)

- **Means:** ranking quality of a binary target across bars. Higher = the
  candidate score discriminates positive from negative bars on the held-out
  split.
- **Does not mean:** tradability, calibration, or post-cost profitability. An
  AUC of 0.62 is only a statement that a rank order exists, not that its top
  decile is worth trading.

### Top-decile precision

- **Means:** positive-label rate among the highest-scoring 10% of val bars.
- **Does not mean:** that those bars are net-profitable after costs. Positive
  in our label does not imply positive after spread + commission.

### Top-decile precision lift

- **Means:** `top_decile_precision / val_base_rate`. A multiplicative signal
  strength over chance. ≥ 1.5× is the current research-tier bar.
- **Does not mean:** tradability. A 3× lift on a 5% base rate still yields a
  thin 15% precision operating point.

### Realized mean `slice_best_pnl` at pass rate K

- **Means:** if we trade the top-K% of bars by score, the mean best in-slice
  raw PnL before costs.
- **Does not mean:** realized after-cost PnL, since the ranker — not this
  metric — chooses which in-slice contract to trade. Upper bound on what the
  gate's decision could deliver if the ranker were perfect.

### Threshold sweep monotonicity

- **Means:** does the realized-mean curve fall as the pass rate rises? A
  monotone sweep (one inversion allowed at the widest rate) is an operational
  stability signal — the score is rank-useful across pass rates, not just at
  a single lucky operating point.
- **Does not mean:** economic profitability. A monotone curve can still sit
  entirely below the cost floor.

## Training metrics

### Val loss convergence (best epoch not in first 20%)

- **Means:** the model found meaningful structure past the initialization
  regime. A late best epoch is a necessary (not sufficient) sign of
  learning.
- **Does not mean:** the learned structure generalizes economically. An
  "abstain everything" checkpoint can have the lowest val loss and still
  be worthless.

### Gradient norm bounded, sub-losses finite

- **Means:** training did not blow up.
- **Does not mean:** it learned anything useful.

## Replay / scoring metrics (v4.0, dollar-weighted)

### Score = 0.5 · sortino_term + 0.5 · pf_term, scaled by positive_day_rate · dd_mult

- **Means:** the evaluator's single composite number. Canonical, used for
  promotion.
- **Does not mean:** any individual component. A good score can hide a bad
  component via the mean + clip.

### Profit factor (dollar-weighted)

- **Means:** sum(winning $PnL) / abs(sum(losing $PnL)) across all trades,
  clipped at 4.0 inside the score.
- **Does not mean:** risk-adjusted return. A PF of 1.5 on one monster trade
  and many small losses is not equivalent to a PF of 1.5 from steady trades.

### Daily Sortino ratio

- **Means:** mean daily dollar return / std of negative daily dollar
  returns. Clipped at 10.0 inside the score.
- **Does not mean:** drawdown safety. Sortino punishes daily negative
  variance, not peak-to-trough dollar decline.

### Gate pass rate

- **Means:** fraction of eligible bars the model chose to trade.
- **Does not mean:** economic engagement. A high pass rate on a noisy score
  is still noise; a low pass rate on an accurate score is discipline, not
  cowardice.

### Direction balance

- **Means:** diagnostic only. Absolute deviation of call/put share from 0.5.
- **Does not mean:** a score gate. Per [v2/ART2_LOOP.md](../ART2_LOOP.md), this
  was deliberately demoted from a gate to a diagnostic.

## Governance metrics

### Provenance-comparable

- **Means:** two results share dataset fingerprint, sidecar schema, feature
  set, label mode, cost model, split recipe, and (optionally) git commit.
  See [v2/core/provenance.py](../core/provenance.py).
- **Does not mean:** they are equally good. Comparability is a pre-condition
  for any comparison, not a quality statement.

### Research-tier

- **Means:** this label / gate is a diagnostic bridge target, not yet a
  deployable one. `OPP_LABEL` prefixed `research_*`; `research_tier=True` in
  provenance; `model_manage keep` refuses.
- **Does not mean:** the label is worthless. It means "learnable ≠
  tradable" is enforced mechanically rather than remembered.
