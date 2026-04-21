# Layer-2 Shared-Encoder Diagnostics — 2026-04-21

## Verdict

**Exp A (detach-side) is the new Layer-2 neural baseline.** It clears the
§2 bar from the plan by wide margins and replaces the separate-MLP neural
path. The fixed-quantile tree baseline remains viable as a second
reference, but the neural path now has a variant that beats it on both
PF and DD.

| Architecture | Trades | PF | DD | TPD | Mean PnL | Call% | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Post-A1 teacher baseline | 298 | 0.961 | 65.4% | 0.993 | — | — | — |
| Separate-MLP (prior neural) | 296 | 0.899 | 196.3% | 0.993 | — | — | catastrophic |
| Tree fixed-q hybrid (reference) | 287 | **1.122** | **56.2%** | 0.957 | +68.8 | 21.3% | prior best |
| Shared encoder (plain) | 299 | 1.153 | 75.9% | 0.997 | +83.8 | 15.4% | PF ✅, DD ❌ |
| **A — detach-side** | 275 | **1.455** | **36.9%** | 0.917 | **+225.4** | 15.3% | **PASS** |
| B — side-dropout 0.30 | 299 | 1.137 | 66.7% | 0.997 | +75.4 | 14.7% | PF ✅, DD ❌ |
| C — wrong-side-α 2.0 | 281 | 0.832 | 122.9% | 0.937 | −97.3 | 11.7% | regression |
| D — A+B combined | 277 | 1.451 | 36.1% | 0.923 | +226.6 | 14.4% | PASS (≈ A) |

Bar from [layer2_shared_encoder_2026_04_21.md](layer2_shared_encoder_2026_04_21.md) §2:
PF ≥ 1.122, DD ≤ 56.2%, TPD ∈ [0.8, 1.1].

## Why A works

The prior doc's leading hypothesis was:
> "Tree models use two independent HistGBM regressors (one per target).
> ... The shared-encoder trunk forces both heads through the same
> representation. The entry head benefits; but the side head now 'reads
> through' the entry head's representation; its error pattern on bad
> bars is correlated with the entry head's optimism on the same bars."

A direct test of this hypothesis is **Exp A (`--detach-side`)**: in the
forward pass, `h_side = h.detach()`, so the side head's backward path
cannot update trunk weights. The trunk is optimised by the entry loss
alone; the side head becomes a linear probe on entry-optimised features.

Result: the hypothesis is supported quantitatively.

- **Entry top-decile mean: 1958 → 3008** (+54%). The trunk, freed from
  the side head's compromise, now identifies valuable bars much better
  than the plain shared encoder.
- **PF: 1.153 → 1.455** (+26%). The better bar selection + cleaner
  side signal compound.
- **DD: 75.9% → 36.9%** (−39 pp, a massive drop). Correlated errors
  between heads were the dominant cause of drawdown gaps.
- **Mean PnL per trade: 83.8 → 225.4** (+169%). Per-trade edge
  jumped because the encoder is no longer compromising between two
  loss signals.
- **Trade count: 299 → 275** (modest −8%). A is slightly more
  selective; still well within the plan's TPD band.

## Why B and C do less / fail

**B (side-dropout 0.30)** adds extra regularisation specifically on the
side head's path but does NOT cut the gradient flow into the trunk. The
trunk still receives side-loss signal, just in a noisier form. PF barely
moves (1.153 → 1.137) and DD improves only partially (75.9 → 66.7).
The problem wasn't too much side-head capacity — it was the side head's
gradient steering the trunk representation. Regularising the head output
doesn't address that.

**C (wrong-side-α 2.0)** penalised samples where prediction and target
disagreed in sign. The side head became more conservative, but in doing
so it distorted the learned representation. Mean PnL per trade went
**negative** (−97.3), PF collapsed to 0.832, DD ballooned to 122.9%.

The likely mechanism: at α=2.0, the side head over-weights minimising
wrong-side errors relative to magnitude errors. The trunk (still
entangled with the side head in the non-detach config) accommodates by
pulling the shared representation away from features that predict
bar-value toward features that predict direction-with-certainty. Bars
where the model was uncertain about direction got pushed toward
smaller-magnitude predictions, and the within-day ranking degraded. The
direction head was "more right" on hard calls but the bar-selection
policy was worse.

**D (A + B)** is tied with A within noise. Once the side head's gradient
is detached (A), adding extra side-head dropout (B) accomplishes nothing
meaningful — you can't further regularise a head that's already
decoupled from the thing you cared about regularising (the trunk).
Keep A alone; don't add B for complexity's sake.

## Slice breakdown (abstention + side_error)

From `replay_report.json` for each run:

| Run | abstention n | abstention $ | side_error n | side_error $ |
|---|---:|---:|---:|---:|
| Tree baseline | 10 | +$1,553 | 3 | −$1,009 |
| Shared enc plain | 6 | +$2,743 | 5 | −$1,077 |
| **A — detach** | **6** | **+$1,714** | **3** | **−$1,059** |
| D — A+B | 6 | +$1,714 | **1** | −$1,104 |

A enters abstention less often than the tree (6 vs 10) but each entry is
still more profitable than tree's ($1,714 vs $1,553). Side-error
exposure matches the tree (3 trades). D edges A on side-error (1 vs 3
trades) at the cost of $45 deeper per-trade loss; that's the kind of
noise that doesn't warrant adding B.

## Shipped artifact

**Architecture decision:** Layer-2 shared encoder with `--detach-side`
is the new neural baseline.

Reproduce:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach \
  --entry-target entry_value_rank \
  --side-target  time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 --side-quantile 0.10 \
  --score-mode product \
  --detach-side \
  --device cpu

.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
```

Code surface:
- `Layer2SharedEncoder` in [v3/layer2/neural.py](../layer2/neural.py)
  now supports three additive diagnostic levers:
  `detach_side` (bool), `side_dropout` (Optional[float]),
  `wrong_side_alpha` (float, default 1.0).
- `train_neural.py` exposes each via CLI:
  `--detach-side`, `--side-dropout`, `--wrong-side-alpha`.
- Defaults preserve the plain shared-encoder behavior (all three off /
  at 1.0), so prior runs reproduce bit-for-bit without flags.

## What this changes in the layer hierarchy

The tree baseline ([layer2_entry_side_2026_04_21.md](layer2_entry_side_2026_04_21.md))
is no longer the sole promotion target. There are now two Layer-2
variants meeting the plan's bar, with different strengths:

| | Tree fixed-q hybrid | Shared encoder + detach-side |
|---|---|---|
| PF | 1.122 | **1.455** |
| DD | **56.2%** | 36.9% (better) |
| Trades | 287 | 275 |
| TPD | 0.957 | 0.917 |
| Why it works | Independent per-target splits decorrelate errors | Trunk is entry-only; side is linear probe |
| Risk | Harder to extend (HGB capacity is fixed) | Depends on trunk not overfitting entry signal |

Neither is strictly dominated. The shared encoder wins on both PF and
DD but makes fewer trades; the tree takes slightly more trades with
slightly weaker per-trade edge. Both are deployable CPU baselines for
now.

## What this does NOT do

- **No GPU run.** The plan said "CPU must clear the tree bar before
  GPU," and A clears that by a wide margin — GPU is now justified but
  is a separate decision. No `deploy.sh` invocation here.
- **No new features.** W2a feature set is still frozen.
- **No hyperparameter sweep.** A was a single boolean flip. The win was
  architectural, not tuning. The other levers were tested once each at
  their single starting value from the prior doc.
- **C is not retried at lower α.** That's a hyperparameter sweep the
  plan forbids; if wrong-side-α is revisited later it should come from
  a new hypothesis, not a backoff.

## Plan audit

- [x] Three diagnoses implemented as additive CLI flags; defaults
      preserve the plain shared-encoder behavior
- [x] Each run: full 5-fold CPU + replay
- [x] A — cleared §2 bar (PF 1.455 ≥ 1.122, DD 36.9% ≤ 56.2%,
      TPD 0.917 ∈ [0.8, 1.1])
- [x] B — partial; falls short of DD bar
- [x] C — regression; hypothesis rejected
- [x] D (A+B) — cleared, ≈ A within noise
- [x] This research doc written
- [ ] Commit — pending
