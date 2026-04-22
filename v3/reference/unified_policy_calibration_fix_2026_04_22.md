# Unified Policy Calibration + Loss Fix — 2026-04-22

## Purpose

Pick up where the first unified-policy smoke left off. The smoke reported
`46` trades, `PF 0.464`, `DD 50.4%`, `trade_share 0.767`, and a calibrated
`decision_margin` of `-0.083`. This note documents the audit of that result,
the four trader-level fixes applied, and the dev-tier rolling re-run.

## Audit Findings

### 1. Calibrator picked a *negative* margin

The previous calibrator objective was lexicographic `(in_band, mean_pnl,
pf, trades)`. On the validation window no margin produced an in-band trade
share, so the selector collapsed to "max mean PnL" and chose `-0.08295`.

That means **trade on bars where the model's own flat score is higher than
its best contract score**. An OOF sweep of the original smoke predictions
showed why this was catastrophic:

| margin | trades | share | PF | sum PnL |
|---|---|---|---|---|
| -0.083 (as calibrated) | 46 | 0.767 | **0.464** | -$12,099 |
|  0.000 | 20 | 0.333 | 1.005 | +$27 |
|  0.010 | 17 | 0.283 | **1.453** | +$1,870 |
|  0.020 | 13 | 0.217 | 1.545 | +$1,823 |

At `m ≥ +0.01` the same model's output produces `PF 1.453`. The model is
not the bottleneck; the calibrator objective is.

### 2. Flat column was *down-weighted* in regression

`reg_weight_train[:, 0] = 0.5` was a mirror image of the expected rule.
The flat anchor has a known, exact, always-correct label (`$0`). Letting
the predicted flat score drift noisily is how `decision_margin =
best_nonflat_score - flat_score` ends up with a noisy denominator and a
negative calibrated cutoff.

### 3. Ranking hinge margin of `0.05` is ~`$5` of PnL

`arcsinh⁻¹(0.05) × 100 ≈ $5`. A 0DTE trader cannot tell apart two strikes
at the $5 level — neither should the ranker. `0.20` corresponds to `~$20`
of PnL, which is a meaningful strike-pick signal.

### 4. Ranking loss only had a single "best-vs-rest" anchor

That trains the model to push the single best tradeable action above
everything else. It does not train the model to distinguish *losing*
contracts from flat. A listwise-style bidirectional pressure is cheap to
add and exactly matches the downstream trade rule:

- winning contracts (`utility > +$10`) must outrank flat by the hinge margin
- losing contracts (`utility < -$10`) must rank *below* flat by the hinge margin

## Changes

Implemented in this pass:

- **Calibrator**: floor the `decision_margin` grid at `0.0`. Prefer
  pf-qualified + in-band candidates first, then pf-qualified alone, then
  enough-sample, then raw PF. Require at least `max(6, 10% of days)` trades
  before a PF estimate counts as reliable. File:
  [train_unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_unified_policy.py).
- **Regression weight on flat**: `0.5 → 1.5` to anchor the flat prediction
  near its exact `$0` target. File:
  [unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/unified_policy.py).
- **Ranking hinge margin**: `0.05 → 0.20`.
- **New `_flat_ranking_loss`** term: pushes winning contracts above flat
  and flat above losing contracts with the same hinge margin. Added to
  both training and validation loops at weight `w_ranking`.
- **Loss weights rebalanced**: `w_regression 1.0 → 0.5`, `w_ranking 0.75
  → 1.0`. Ranking is now primary; regression carries the scale signal.
- **Smoke budget**: `max_epochs 4 → 8`, `patience 2 → 3`. The previous
  smoke stopped before ranking had time to separate flat from contracts.

## Smoke Re-Run (latest rolling window, `window_12`)

Command:

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --run-dir v3/artifacts/layer2_unified_policy_smoke_fixed \
  --tier smoke \
  --device cpu
```

| metric | pre-fix smoke | post-fix smoke |
|---|---|---|
| trades | 46 | 39 |
| trade_share | 0.767 | **0.650** (in-band) |
| PF | 0.464 | **0.909** |
| max DD % | 50.4 | **15.1** |
| mean PnL / trade | -$263 | -$34.5 |
| calibrated margin | -0.08295 | **+0.30489** |
| clean winner rate | 0.24 | 0.41 |
| fast loser rate | 0.33 | 0.26 |
| beats V1 same-bar rate | 0.41 | 0.54 |

Still below the `1.132` aggregate baseline on this single window, but
the single-window comparison is structurally unfair — the `1.132` figure
is a 13-window rolling aggregate, and `window_12` (Nov 2025–Feb 2026 OOS)
appears to be a regime where 95% of the model's ranked picks are calls.

## Dev Rolling Re-Run (13 windows, 1 seed, CPU)

Command:

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --run-dir v3/artifacts/layer2_unified_policy_dev_v1 \
  --tier dev \
  --device cpu
```

Runtime: `526s`. Aggregate across `780` OOS days:

| metric | value |
|---|---|
| trades | `410` |
| trade_share | `0.526` (**in-band**) |
| PF | `1.066` |
| max DD % | `48.8` |
| mean PnL / trade | `+$24.6` |
| clean winner rate | `0.37` |
| fast loser rate | `0.21` |
| shakeout winner rate | `0.08` |
| beats V1 same-bar rate | `0.554` |

Per-window PF:

| W | margin | trades | share | PF | DD | mean |
|---|---|---|---|---|---|---|
| 0 | 0.215 | 45 | 0.75 | 1.404 | 20.2% | +$125 |
| 1 | 0.462 | 52 | 0.87 | 1.404 | 22.4% | +$131 |
| 2 | 0.459 | 4 | 0.07 | ∞ | 0.0% | +$797 |
| 3 | 0.356 | 33 | 0.55 | 1.122 | 8.3% | +$37 |
| 4 | 0.398 | 1 | 0.02 | ∞ | 0.0% | +$57 |
| 5 | 0.374 | 19 | 0.32 | 1.488 | 14.6% | +$170 |
| 6 | 0.308 | 14 | 0.23 | 2.670 | 9.2% | +$515 |
| **7** | **0.061** | **59** | **0.98** | **0.317** | **79.8%** | **-$338** |
| 8 | 0.273 | 32 | 0.53 | 2.317 | 6.9% | +$369 |
| 9 | 0.409 | 8 | 0.13 | 1.411 | 9.2% | +$173 |
| **10** | **0.222** | **56** | **0.93** | **0.515** | **49.3%** | **-$203** |
| **11** | **0.000** | **59** | **0.98** | **1.078** | **24.6%** | **+$34** |
| 12 | 0.379 | 28 | 0.47 | 0.895 | 14.0% | -$39 |

Slippage stress (all 410 trades):

| round-trip | PF | DD | mean |
|---|---|---|---|
| $0 | 1.066 | 48.8% | +$24.6 |
| $10 | 1.039 | 54.0% | +$14.6 |
| $25 | 0.999 | 62.4% | -$0.4 |

## Gate Status

| gate | threshold | result |
|---|---|---|
| W1 PF vs baseline | `PF > 1.132` | **FAIL** (`1.066`) |
| W1 DD vs Layer-2.5 | `DD ≤ 21.4%` OR `+0.15 PF` | FAIL |
| W1 trade share | `0.25 ≤ share ≤ 0.70` | **PASS** (`0.526`) |
| Patience fast-loser | `≤ 0.209` | FAIL (`0.2098`, ~tie) |

## Interpretation

The redesign is now inside the baseline's neighborhood, not collapsed beside
it:

- aggregate PF `0.464 → 1.066` (×`2.30`), only `6%` below the honest
  `V0 + time-stop` baseline of `1.132`
- fast-loser rate `0.33 → 0.21`, inside a hair of the `0.209` patience gate
- trade-share gate cleared at `0.526`

The remaining PF gap comes from three single-window outliers where the
calibrator settled for a low margin because no in-band PF-qualified
candidate existed on that window's validation set:

- W07: margin `0.061`, val PF < 1, OOS `PF 0.317`, DD `79.8%`
- W10: margin `0.222`, OOS `PF 0.515`, DD `49.3%`
- W11: margin `0.000`, OOS `PF 1.078` (saved by luck, share `0.98`)

Removing these three windows from the aggregate (conceptually, not
honestly) would put PF well above baseline. But honestly accepting them
means the **calibrator's fallback behavior** is now the binding
constraint, not the model's ranking.

## What This Means in Plan Terms

The plan's section 1 promotion contract says:

> Promotion requires the same result shape on all 3 promotion seeds. No
> seed may fail below PF 1.0.

A seed here does not fail below `PF 1.0` overall. It *does* include two
sub-`1.0` windows (`W07` `0.317` and `W10` `0.515`), so a seed-by-seed
analysis at the window level will still flag those. Deciding whether
that counts as a seed-level fail is a pre-promotion governance question.

## Next Work

### Option A — GPU 3-seed promotion now

The plan's own contract calls for this. At `PF 1.066` single-seed CPU,
seed averaging is the legitimate way to see whether the aggregate clears
`1.132`. Cost is one proper GPU lease.

### Option B — Harden the calibrator's fallback

If no candidate on validation is `pf_qualified`, fall back to a fixed
*high* margin (e.g. the `80th` percentile of positive validation margins)
rather than the lowest grid point. This trades some trades for lower
exposure in bad-signal windows.

### Option C — Side-balance regularization

Call share across `13` windows is still high (e.g. W03 = `33/33` all
calls, W05 = `19/19` all calls). That is probably real regime bias, not
a model pathology — OOS windows that span Nov 2025 – Feb 2026 are in a
trending-up regime. Adding an explicit side-balance term risks trading
against the real market direction. Do not ship this without OOS
falsification of "always-call".

Recommendation: run option A first. The calibrator fallback (option B)
is cheap but shouldn't be merged in front of a promotion seed run that
will expose whether it was actually the bottleneck.

## What This Does Not Prove

- this is CPU-budget training (8 epochs)
- still 1 seed, not the 3-seed promotion requirement
- side bias is still present (95% call picks) — likely training-regime
  bias; if aggregate rolling performance is strong, the next lever is a
  side-balance regularizer or explicit side-wise contrastive loss, not a
  regression weight change
- Layer-3 outer loop composition is still not wired into this path

## Next Work

- run one full 3-seed promotion-tier rolling run on GPU
- if rolling PF clears the `1.132` baseline, wire up the two-stage outer
  loop (Layer-2 entry → Layer-3 exit → retrain entry on composed utility)
- if rolling PF does not clear, pick one of:
  - side-balance regularization on the ranking loss
  - clean-entry probability as an inference-side gate rather than margin-only
  - longer training budget with more sequence context
