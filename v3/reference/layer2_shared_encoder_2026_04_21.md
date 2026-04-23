# Layer-2 Shared-Encoder Multitask — 2026-04-21

## Verdict

**Partial pass. Not promoted.** The shared-encoder multitask model beats
both the post-A1 teacher baseline and the previous separate-head MLP on
PF, and matches the tree baseline's trade coverage. It **fails** the
drawdown bar (DD 75.9% vs tree 56.2%) set by the plan. Per plan §6 this
is a hard stop — no GPU run, no features added, no threshold tuning.

| System | Trades | PF | DD | Trades/day | Mean PnL | Call% |
|---|---:|---:|---:|---:|---:|---:|
| Post-A1 teacher baseline | 298 | 0.961 | 65.4% | 0.993 | — | — |
| Layer-2 separate-MLP (prior neural) | 296 | 0.899 | 196.3% | 0.993 | — | — |
| Layer-2 tree (fixed-quantile hybrid, current baseline) | 287 | **1.122** | **56.2%** | 0.957 | +68.8 | 21.3% |
| **Layer-2 shared encoder (this run)** | 299 | **1.153** ✅ | **75.9%** ❌ | 0.997 ✅ | +83.8 | 15.4% |

Pass/fail against the plan's §2 bar:
- PF ≥ 1.122 ? **1.153 ✅**
- DD ≤ 56.2% ? **75.9% ❌**
- Trades/day ∈ [0.8, 1.1] ? **0.997 ✅**

## What changed (code)

Shipped in `c1e7546`-equivalent surface (new commit for this work):

- [v3/layer2/neural.py](../layer2/neural.py) — new:
  - `Layer2SharedEncoder` (trunk + 2 heads; depth defaults to 2)
  - `SharedEncoderPredictor` (per-head pickle-friendly wrapper; two
    instances per fold back the same encoder state, so `replay.py`
    loads them with the same interface as the separate-head MLP and
    the tree baseline)
  - `train_multitask` (single training loop; per-sample per-head
    validity masks; zero-weight invalid rows contribute nothing to
    that head's loss)
- [v3/layer2/train_neural.py](../layer2/train_neural.py) — rewired:
  - Default `--depth 2` (was 3). One linear head per task instead of a
    second trunk-depth head, so the param count stays comparable to
    the separate-head configuration.
  - New CLI flags `--w-entry`, `--w-side` (default 1.0 each; equal
    weighting is the honest starting point per plan §3b).
  - One `train_multitask` call replaces the two `train_regressor`
    calls. Manifest + audit record `architecture: "shared_encoder"`.

`common.py` and `replay.py` are unchanged. Same per-fold artifact shape
(`entry_model.pkl`, `side_model.pkl`, `calibration.json`,
`training_info.json`) so the replay harness works without modification.

## What the numbers say

### The shared encoder IS a real improvement over separate MLPs

- PF 0.899 → 1.153 (+28.2%)
- DD 196.3% → 75.9% (−120.4 pp)
- From catastrophic fail to beating the teacher baseline on PF

This validates the §1 hypothesis: trees implicitly share feature splits
across outputs, and separate MLPs were relearning that structure per head
and overfitting the smaller per-head signal. A shared representation
does recover meaningful multitask capacity.

### The shared encoder matches (slightly exceeds) the tree baseline on PF

- Tree PF 1.122 → Shared-encoder PF 1.153
- Mean PnL: 68.8 → 83.8 (per-trade edge is actually higher)
- Top-1/day mean entry_value_raw: 1988.7 → 2025.0 (marginally better
  bar selection)

### But the shared encoder fails the drawdown bar

- Tree DD 56.2% → Shared-encoder DD 75.9% (+19.7 pp)
- Gross profit 182.0k → 189.2k (+7.2k)
- Gross loss  162.3k → 164.2k (+1.9k)
- Call bias 21.3% → **15.4%** (more put-heavy)

Higher mean PnL with a wider DD gap means the shared encoder is picking
trades with more extreme outcomes on both sides. The per-trade edge
went up but per-trade variance went up more. The tree's implicit
per-head split diversity was evidently carrying a risk-control signal
that the shared representation is washing out.

### Slice breakdown is consistent with that reading

| Slice | Tree trades | Tree mean PnL | Shared trades | Shared mean PnL |
|---|---:|---:|---:|---:|
| abstention | 10 | +$1,553 | **6** | **+$2,743** |
| side_error | 3 | −$1,009 | **5** | **−$1,077** |

- Abstention: fewer trades, but each is 76% more profitable on average.
  The shared encoder is MORE selective here and the selections are
  better. Good sign.
- Side_error: MORE trades, each slightly worse. The shared encoder is
  entering the failure mode more often, and slightly more expensively.
  Bad sign — and consistent with the DD story.

## Working hypothesis for the DD gap

The tree baseline uses two independent HistGBM regressors (one per
target). Each tree's splits are chosen to minimize that head's loss,
with no shared representation pressure. The emergent per-head split
patterns differ, which gives the replay policy uncorrelated
error signals across the two heads — decorrelated errors smooth DD.

The shared-encoder trunk forces both heads through the same
representation. The entry head benefits (higher mean PnL, higher PF).
But the side head now "reads through" the entry head's representation;
its error pattern on bad bars is correlated with the entry head's
optimism on the same bars. When both heads agree that a bar is
attractive AND the side call is wrong, the loss is larger than it
would be with independent splits.

This is the leading suspect. It is not a tested finding.

## What this does NOT warrant

- **Hyperparameter sweep.** Plan §7: "If it fails §2, the signal is
  that the architecture itself (or the loss shape) is wrong — not that
  a different H / depth / dropout would save it." The shared-encoder
  *hypothesis* is partially validated (PF works, multitask helps); the
  *risk-control* side is what's off. That's a different question than
  H / depth / dropout.
- **GPU training.** Plan §5 is explicit: CPU must clear the tree bar
  before GPU. It did not. No `deploy.sh`.
- **Adding features.** W2a is frozen per the plan.

## What a reasonable next architecture would try (design notes, not plan)

Not proposed here, just honest record of where diagnosis would lead if
the user wants to continue past this stop:

1. **Detach the side head's gradient from the trunk** (or use a small
   independent side trunk in addition to the shared one). If the
   entry-head representation correlates errors as suspected, decoupling
   the side head should recover the tree's decorrelation while
   preserving the shared-representation gain for the entry head.
2. **Per-head dropout asymmetry.** Higher dropout on the side head
   might reduce its dependence on entry-correlated features.
3. **Risk-aware loss.** Current loss is weighted Huber on both heads.
   A PnL-aware penalty on the side head (e.g., asymmetric weighting
   that penalises wrong-side more than small wrong-magnitude) would
   target the DD directly rather than mean prediction error.

These are hypotheses. Each would be a separate experiment with its own
§2 bar. Not in this plan's scope.

## Reproducing this run

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_shared_enc_fixedq_60_10 \
  --entry-target entry_value_rank \
  --side-target  time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 --side-quantile 0.10 \
  --score-mode product \
  --device cpu

.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_shared_enc_fixedq_60_10
```

Artifacts:
- `v3/artifacts/layer2_shared_enc_fixedq_60_10/` — full 5-fold run
  (gitignored locally; `audit.json`, `replay_report.json`,
  `layer2_trades.csv`, `teacher_baseline_trades.csv` are the review
  surface)
- `v3/artifacts/layer2_shared_enc_fixedq_60_10_smoke/` — fold-4 smoke
  (PF 1.498, DD 21.8% on fold 4 alone; a single-fold view is not a
  promotion signal — documented for completeness)

## Plan audit

Verification checklist (plan §8):

- [x] `python -m py_compile v3/layer2/train_neural.py` passes
- [x] Fold-4 smoke run completes without NaN / divergence
- [x] Full 5-fold CPU run produces `oof_predictions.pkl` + per-fold
      manifests
- [x] `replay.py` reads the new manifests unchanged and produces
      `replay_report.json`
- [ ] `replay_report.json` clears the §2 bar — **FAILED on DD**
- [x] Per-fold diagnostics (multitask val loss, per-head val loss)
      logged in each fold's `training_info.json`
- [x] This research doc written (honest non-promotion framing)
- [ ] One commit on `codex/v3-orc-xsp-10k` — pending
