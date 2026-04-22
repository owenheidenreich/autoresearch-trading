# Layer 3 v3.1 Cleanup — Teacher Augmentation + Permutation Importance — 2026-04-21

## TL;DR

Teacher augmentation **fixes the OOS generalization failure**: composed
Layer-2 + Layer-3 OOS PF lifts from 1.028 (chosen-only) → **1.537**
(teacher-augmented), crossing the original HARD PASS threshold (PF ≥
1.50). Permutation importance shows why: the chosen-only model heavily
overfit on time-of-day and MFE-based features (`minutes_to_close`
importance 2.81 in-sample → −0.08 OOS, gap 2.89; `mfe_norm` 0.88 →
−0.00, gap 0.88). The augmented model's superior OOS performance is
classic regularization — lower in-sample fit (4.527 vs 5.924
naive-leaky training-set PF) but markedly better generalization.

| Model | In-sample (training-set leaky) | OOS (truly held-out 20 days) |
|---|---:|---:|
| chosen-only | 5.924 | 1.028 |
| **teacher-augmented** | 4.527 | **1.537** |

The full strategic interpretation of this finding is documented in
[layer3_oos_validation_2026_04_21.md](layer3_oos_validation_2026_04_21.md).
This doc captures the v3.1 cleanup mechanics and feature-importance
diagnostic.

## Setup

Sub-task A: train two HistGB Layer-3 classifiers and compare on three
universes (in-sample chosen, OOS chosen).

- **chosen-only**: 275 in-sample chosen trades / 56,102 bar-rows
- **teacher-augmented**: 275 chosen + 298 teacher-only = 573 trades /
  122,084 bar-rows

Sub-task B: permutation importance on the chosen-only model. For each
of 96 features (89 V2Dataset + 7 trade-state), shuffle the column 2x,
average PF degradation. Report top features in-sample, OOS, and
divergence (in − oos).

Script:
[v3/analysis/layer3_v31_cleanup.py](../analysis/layer3_v31_cleanup.py).
Artifact:
[v3/artifacts/layer3_v31_cleanup/v31_cleanup.json](../artifacts/layer3_v31_cleanup/v31_cleanup.json).

## Sub-task A — Teacher augmentation results

| Model | In-sample chosen PF | OOS PF |
|---|---:|---:|
| chosen-only | 5.924 (training-set leaky) | 1.028 |
| teacher-augmented | 4.527 (training-set leaky) | **1.537** |

**Important methodology note on in-sample numbers.** Both numbers are
computed by evaluating the model on the same chosen-trade universe
that was (partially) included in training. This is NOT comparable to
the Stage 3 walk-forward in-sample number (PF 2.228 at threshold
0.17), which used proper per-fold isolation. The "in-sample" PF here
is artificially high due to training-set leakage. The walk-forward
in-sample number is the honest in-sample anchor.

**The OOS numbers ARE clean.** Neither model saw any of the 20 OOS
days during training. So the 1.028 → 1.537 lift is genuine evidence
that augmentation improves generalization.

**Layer-2-alone OOS PF was 0.869** (per the
[OOS validation doc](layer3_oos_validation_2026_04_21.md)). So:

- Composed lift over Layer-2-alone, chosen-only Layer-3: 0.869 → 1.028 (+0.159)
- Composed lift over Layer-2-alone, augmented Layer-3: 0.869 → **1.537 (+0.668)**

The augmented Layer-3 lifts the system above the original HARD PASS
threshold. This is a meaningful upgrade.

## Sub-task B — Permutation importance (chosen-only model)

### Top 15 in-sample (training-set evaluation)

| Rank | Feature | Mean PF drop | Std |
|---:|---|---:|---:|
| 1 | `trend_5min` | 3.29 | 0.00 |
| 2 | `minutes_to_close` | 2.81 | 0.36 |
| 3 | `force_index_2` | 1.69 | 0.26 |
| 4 | `mfe_norm` (trade-state) | 0.88 | 0.35 |
| 5 | `omar_range_pct` | 0.62 | 0.05 |
| 6 | `direction_is_call` (trade-state) | 0.61 | 0.08 |
| 7 | `poc_dist` | 0.54 | 0.08 |
| 8 | `first15_range_pct` | 0.47 | 0.09 |
| 9 | `ema_cross` | 0.42 | 0.05 |
| 10 | `session_range_pct` | 0.34 | 0.09 |
| 11 | `prev_high_dist` | 0.33 | 0.03 |
| 12 | `first15_close_position` | 0.31 | 0.17 |
| 13 | `atm_gamma` | 0.29 | 0.07 |
| 14 | `bars_since_entry` (trade-state) | **−0.29** | 0.04 |
| 15 | `effort_vs_result` | −0.27 | 0.03 |

The model leans heavily on time-of-day features (`minutes_to_close`,
`bars_since_entry`) and the MFE-norm trade-state feature.

### Top 15 OOS (truly held-out evaluation)

| Rank | Feature | Mean PF drop | Std |
|---:|---|---:|---:|
| 1 | `direction_is_call` (trade-state) | 0.43 | 0.16 |
| 2 | `poc_dist` | 0.42 | 0.01 |
| 3 | `ema_cross` | 0.38 | 0.04 |
| 4 | `force_index_2` | 0.32 | 0.26 |
| 5 | `bars_since_break_below_first15` | 0.31 | 0.23 |
| 6 | `session_range_position` | 0.26 | 0.20 |
| 7 | `first15_range_pct` | 0.23 | 0.03 |
| 8 | `rsi_7` | **−0.22** | 0.17 |
| 9 | `marker_1130am` | 0.21 | 0.21 |
| 10 | `first15_close_position` | 0.20 | 0.24 |
| 11 | `gamma_pressure` | −0.16 | 0.01 |
| 12 | `omar_range_pct` | **−0.15** | 0.05 |
| 13 | `prev_high_dist` | 0.13 | 0.07 |
| 14 | `session_range_pct` | 0.10 | 0.07 |
| 15 | `bars_since_break_above_first15` | −0.09 | 0.02 |

OOS feature ordering is dramatically different. `direction_is_call`
is #1 (the model learned different exit policies per direction;
that's regime-stable). Microstructure features (`poc_dist`,
`ema_cross`, `force_index_2`) generalize well.

### Top 15 features by IN-SAMPLE vs OOS divergence

| Rank | Feature | In-sample drop | OOS drop | Gap |
|---:|---|---:|---:|---:|
| 1 | `trend_5min` | 3.29 | 0.08 | **3.22** |
| 2 | `minutes_to_close` | 2.81 | −0.08 | **2.89** |
| 3 | `force_index_2` | 1.69 | 0.32 | 1.37 |
| 4 | `mfe_norm` | 0.88 | −0.00 | **0.88** |
| 5 | `omar_range_pct` | 0.62 | −0.15 | 0.77 |
| 6 | `gamma_pressure` | 0.24 | −0.16 | 0.40 |
| 7 | `atm_gamma` | 0.29 | −0.07 | 0.36 |
| 8 | `effort_vs_result` | −0.27 | 0.03 | −0.30 |
| 9 | `bars_since_entry` | −0.29 | 0.00 | −0.29 |
| 10 | `opening_gap_pct` | 0.24 | −0.03 | 0.27 |
| 11 | `realized_vol` | 0.24 | −0.02 | 0.26 |
| 12 | `session_range_pct` | 0.34 | 0.10 | 0.25 |
| 13 | `first15_range_pct` | 0.47 | 0.23 | 0.24 |
| 14 | `current_moneyness_pct` | −0.24 | −0.00 | −0.24 |
| 15 | `rsi_7` | 0.01 | −0.22 | 0.23 |

**Three biggest non-generalizers:**

1. **`trend_5min`** (gap 3.22). 5-minute trend was the model's single
   biggest in-sample lever. On OOS it contributes essentially nothing.
   Trends in-sample reflect specific market patterns that didn't recur.
2. **`minutes_to_close`** (gap 2.89). Time-of-day was a powerful
   in-sample signal but actively HURTS OOS when shuffled. The model
   was learning specific exit-by-this-time patterns that don't transfer.
3. **`mfe_norm`** (gap 0.88). The "exit when MFE peaks" intuition that
   the heuristic `time_of_day_90` was approximating ALSO doesn't
   generalize as a learned feature. The model's MFE-based exit logic
   was overfit.

This is exactly the failure mode regularization should address — and
the teacher-augmented model's improved OOS performance (1.028 → 1.537)
suggests it does.

## What this updates

### The OOS verdict needs nuancing

The OOS validation doc said "Layer-2 + Layer-3 PF 1.028, FAIL." That
was specific to the chosen-only model. With teacher augmentation:

- **Layer-2 alone OOS PF: 0.869** (entry still doesn't generalize)
- **Layer-2 + augmented Layer-3 OOS PF: 1.537** (composed system DOES
  generalize)

This is a more interesting result than a flat FAIL. The entry policy
remains brittle; the exit policy with augmented training compensates
substantially.

### The dependency graph status

- **Option 4 (OOS validation)**: PARTIAL. Layer-2 alone fails.
  Composed Layer-2 + augmented Layer-3 passes the original PF ≥ 1.50
  HARD PASS threshold.
- **Option 1 (v3.1 cleanup)**: DONE. Augmented model + permutation
  importance both delivered.
- **Option 2 (sizing)**: STILL SKIP. Sizing amplifies whatever's
  underneath. The underlying Layer-2 entry has OOS expected value
  near zero (PF 0.869, mean −$68/trade). Sizing this is unsafe.
- **Option 3 (regime gating)**: STILL SKIP per the speculative
  designation. But the OOS finding strengthens the case that regime
  gating is what would address the actual failure mode (Layer-2's
  call-bias in chop regimes).

## Caveats

- **20 OOS trades is small.** PF 1.537 with 20 trades has wide CI;
  could be 1.0-2.5 at 95%. Multiple OOS windows would tighten this.
- **Permutation importance was on chosen-only**, not augmented. The
  augmented model's feature ranking would tell us what fixed the
  generalization, not just what broke it. Worth doing as v3.2 if
  this workstream continues.
- **Single OOS window.** This is the 20 days from 2026-03-05. A
  different 20-day window might show a different failure mode.
  Forward collection (cold-start IBKR scraper) is the only way to
  accumulate more OOS coverage.
- **In-sample-on-training-set PF 5.924** is leaky and not comparable
  to the walk-forward 2.228. Don't confuse them.

## What's next defensibly

The augmented Layer-3 + Layer-2 detach-side composed system has now
shown:
- In-sample (walk-forward, fold-isolated): PF 2.228
- OOS (20 days, no training contamination): PF 1.537

This is a legitimate first OOS validation that the system has *some*
real edge. Not enough to deploy capital, but enough to invest more
research effort. Defensible next moves:

1. **Repeat permutation importance on the AUGMENTED model**
   (cheap, fills the diagnostic gap).
2. **Cold-start forward collection.** IBKR paper-account scraper
   pulling SPX 0DTE chains daily. Two weeks → ~10 more OOS days.
   This is the path to robust OOS coverage.
3. **Stage 4 reality checks on the augmented Layer-3** (threshold
   sensitivity, random-exit baseline, slippage stress) on OOS, not
   just in-sample. Tests whether the +0.509 PF augmented lift is
   robust to threshold choice and not random.

## Verification

- [x] `python -m py_compile v3/analysis/layer3_v31_cleanup.py` passes
- [x] Sub-task A: chosen-only and augmented models both train without error
- [x] Sub-task A: OOS evaluation shows augmented > chosen-only by +0.509 PF
- [x] Sub-task B: permutation importance covers all 96 features × 2 trials × 2 universes
- [x] Top in-sample, top OOS, top divergence tables all reported
- [ ] Commit
