# Bar-Quality Signal Audit — Hypothesis Pivot (2026-04-18)

## Why this doc exists

Evaluates the state of Codex's bar-quality branch (see `handoff_bar_quality_branch_2026-04-18.md`), runs a CPU diagnostic that answers whether the bar-quality target is predictable from current features, and records the pivot recommendation so the next session does not spend GPU on a rerun that cannot work.

## What Codex built (summary)

Continuous `[0,1]` bar-quality target (`0.55·pnl + 0.25·path + 0.20·sparsity`) at `v2/train.py:668-736`; gate regresses to a margin around `BAR_QUALITY_PASS_THRESHOLD=0.83`. Legitimate `LINEAR_SCORE_HEADS` bug fix (both heads now linear). New `POLICY_GATE_MIN_THRESHOLD` knob on the policy to cap quantile calibration from pushing the gate below the learned no-trade boundary. The target design is sound and the audit confirms `~10.1%` positive rate as intended.

## What the partial GPU readout revealed

`exp_next_c1_screen_mini` fold-0 partial: `val_replay` selected epoch 1 with `PF=0.000 / score=0.0000` as best across 15 epochs; held-out replay forced trading via quantile calibration at threshold `−0.3155` and got `PF=0.559 / DD=87.7% / 288 trades / 4.8 TPD`. Codex's read: the quantile threshold went negative — policy-inference mismatch — and proposed `POLICY_GATE_MIN_THRESHOLD=0.0` before rerunning.

The red flag: every epoch lost to "abstain everything" on val_replay. That is consistent with the model correctly concluding it cannot discriminate — **not** a calibration bug.

## CPU diagnostic: is bar-quality learnable at all?

New script `v2/analysis/bar_quality_signal_audit.py`. Fits class-balanced `LogisticRegression` and a 1-hidden-layer MLP (64 units) on `(X, y)` where `y = bar_quality >= 0.83` and `X` is one of:

- `current` — last-bar features, 79 dims
- `pooled` — lookback mean+std, 158 dims
- `flat` — flattened 30-bar lookback, 2370 dims

Train/val split is the same one `CKPT_SELECTION_MODE=val_replay` uses (fold train days minus val days; val days for held-out AUC).

### Results (mini-screen, folds 0 / 2 / 4)

| Representation | Fold 0 LR / MLP | Fold 2 LR / MLP | Fold 4 LR / MLP |
|---|---|---|---|
| `current` (79)    | 0.553 / 0.526 | 0.535 / 0.522 | 0.525 / 0.523 |
| `pooled` (158)    | 0.544 / 0.525 | 0.536 / 0.539 | 0.532 / 0.523 |
| `flat` (2370)     | 0.528 / 0.508 | 0.523 / 0.505 | 0.505 / 0.509 |

Top-decile precision (matches `POLICY_GATE_TARGET_PASS_RATE=0.10`) sits at the base rate (~0.09–0.16) across every combination — **zero lift** over random.

### Control: coarser label

`pooled` at `BAR_QUALITY_PASS_THRESHOLD=0.50` (base rate ≈ 48%): AUC 0.48–0.50 on all folds. The strictness of the quality cut is not the problem.

## Hypothesis of record

**The binding constraint is features × target alignment, not gate architecture.** The 79-feature context is nearly orthogonal to the bar-quality target the gate is being asked to learn. Every gate/scorer/label iteration since exp_171 was doomed by this, not by its specific design choice.

Corollary: the `val_replay` → epoch-1 / `PF=0.000` selection is Bayes-optimal under the learnable signal level — the gate should abstain. `POLICY_GATE_MIN_THRESHOLD=0.0` is a real fix for the inference mismatch but not a fix for the underlying signal floor. Do **not** run `exp_next_c1_floor0`.

## Next branches (ranked)

1. **Redefine bar-quality around something the current features *do* predict.** Candidate labels to audit next: session-structure conditional (e.g., `opening_gap > X AND vwap_reclaim_state ∈ S`), low-VIX × directional-momentum bars, IV-curvature regime shifts. Reuse the same audit script with a new label function — if LR AUC ≥ 0.60 on ≥ 2/3 folds, that is the gate target.
2. **Enrich the feature set with forward-looking edge.** Order-flow imbalance, options-flow proxies, ES-tape momentum at the gate-decision bar, GEX proxies. This is a larger undertaking (sidecar rebuild, likely paid data) and should be gated on (1) falsifying — i.e., no learnable rewording of bar-quality exists under the current 79 features.
3. **Regime-condition the target.** Train per-VIX-bucket gates with separate calibration. Cheap to try; the audit script already bucketed VIX in `gate_label_audit.py` and can be extended.

What is **not** a next branch:

- Another gate / scorer architecture tweak on the 79-feature × 0.83-threshold target. The signal floor is the ceiling.
- Another `val_replay` selection-criterion rework. The criterion reported exactly what was happening.

## Deferred defensive patch (pick up when signal is found)

Add a `CKPT_MIN_TRADES` floor to `_replay_selection_key` / the val-replay checkpoint loop in `v2/train.py` (around line 1594) so a future run with genuine signal cannot silently pick a 0-trade checkpoint on a tiebreak. Not urgent today — this failure mode only fires when the signal floor is unreachable, and we are pivoting off that condition.

## How to re-run the audit

```bash
# Default: mini-screen folds (0, 2, 4), current-bar features
python3 -m v2.analysis.bar_quality_signal_audit --screen-mode mini

# Strengthen: flattened 30-bar lookback (slow; ~7 min)
python3 -m v2.analysis.bar_quality_signal_audit --screen-mode mini --representation flat

# Try a coarser label without code change:
BAR_QUALITY_PASS_THRESHOLD=0.50 python3 -m v2.analysis.bar_quality_signal_audit --screen-mode mini --representation pooled
```

Exit code `0` iff signal clears AUC≥0.60 AND top-decile precision≥0.25 on ≥ half the selected folds. Use `--strict` to require every fold.
