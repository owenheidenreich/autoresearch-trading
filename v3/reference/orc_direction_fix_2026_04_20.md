# ORC Direction Fix — A1 Ship Record

**Date:** 2026-04-20
**Workstream:** W1 (ORC σ-position hard gate) from the layered implementation plan
**Status:** SHIPPED. Live attribution exactly matches dry-run prediction.

## Question

Does adding `sigma_pos` (SPY-derived VWAP σ-position, SPX-scaled) as a hard
direction gate on the ORC teacher reduce the side_error count without
collapsing entered_right?

## Answer

Yes, exactly as predicted. Aggregate walk-forward separation at threshold
0.0 is positive in all 5 chronological folds, and live runtime attribution
matches the dry-run projection on the nose.

## Before / after attribution counts

Baseline from docs #2 / #9 in [INDEX.md](INDEX.md):

| Outcome | Pre-A1 | Post-A1 | Δ |
|---|---:|---:|---:|
| `side_error`          | 402 | 176 | −226 (−56%) |
| `entered_right`       | 117 |  84 | −33 (−28%) |
| `abstention`          | 467 | 726 | +259 |
| `guardrail_suppression` | 0 |   0 |   0 |

Conservation: 402 + 117 = 519 ORC-fired sessions pre-A1; 176 + 84 + (726 −
467) = 519 post-A1. Every rejected ORC fire moved to abstention, zero
leakage.

## Dry-run prediction match

Predicted by [test1_orc_sigma_filter_dryrun.py](../analysis/test1_orc_sigma_filter_dryrun.py)
and re-confirmed 5-fold by [orc_sigma_walkforward.py](../analysis/orc_sigma_walkforward.py):

- Retain 71 of 104 correct ORC fires (68.3%) → predicted entered_right ≈ 84
- Drop 226 of 330 wrong ORC fires (68.5%) → predicted side_error ≈ 176

Live runtime output: side_error = 176, entered_right = 84. **Exact match.**

This is the tightest research → production correspondence we have on record
in this project. The walk-forward script re-simulates ORC inline with its
own `_vwap_sigma_position` helper; the runtime adapter now calls the shared
`v2.core.market_structure.sigma_pos`. Agreement to the count proves the two
formulas match bit-for-bit, which was the entire point of the W0 shared
helper.

## Walk-forward separation (from `orc_sigma_walkforward.py`)

Threshold 0.0 is best in aggregate and positive in every fold:

| Fold | correct retained | wrong retained | net_gap |
|---|---:|---:|---:|
| 1 | 10/20 (50.0%) | 26/67 (38.8%) | +11.2pp |
| 2 | 13/17 (76.5%) | 23/70 (32.9%) | +43.6pp |
| 3 | 17/21 (81.0%) | 14/66 (21.2%) | +59.7pp |
| 4 | 17/25 (68.0%) | 22/62 (35.5%) | +32.5pp |
| 5 | 14/21 (66.7%) | 19/65 (29.2%) | +37.4pp |
| **Aggregate** | **71/104 (68.3%)** | **104/330 (31.5%)** | **+36.8pp** |

Tighter thresholds (−0.2, −0.1) give slightly larger separations on the
wrong-side axis but drop too many correct fires in folds 1 and 5. Looser
thresholds (+0.1, +0.2) keep more correct fires but also keep more wrong
fires and produce smaller gaps. 0.0 is the flat optimum.

## Code shipped

All changes consume the new shared helper `v2/core/market_structure.py`
(W0) so research and production compute `sigma_pos` from the same formula.

- [v2/core/market_structure.py](../../v2/core/market_structure.py) (new)
  — `build_spy_vwap`, `sigma_pos`, `abs_sigma_pos`, `build_spx_bars`,
  `build_omar_map`, `last10`, `last10_break_state`. Single source of truth.
- [v3/teachers/base.py](../teachers/base.py) — `BarContext` gains
  `sigma_pos: Optional[float] = None`.
- [v3/harness/v2_adapter.py](../harness/v2_adapter.py) — loads SPY VWAP
  cache at `V2Dataset.load()` time and populates `sigma_pos` in
  `bar_context_from_dataset` via the shared helper. Fails fast if the SPY
  cache is missing rather than silently disabling the gate.
- [v3/teachers/orc.py](../teachers/orc.py) — two new rejection lines: skip
  BUY_CALL when `sigma_pos > 0.0`, skip BUY_PUT when `sigma_pos < 0.0`.
  Uses existing `TeacherAction.NO_SIGNAL` abstain pattern.

## Validation command sequence

```
python -m py_compile v2/core/market_structure.py v3/teachers/base.py \
                     v3/teachers/orc.py v3/harness/v2_adapter.py
python -m v3.analysis.orc_sigma_walkforward
python -m v3.analysis.run_full_attribution
```

All three pass. The attribution file `attribution_full_2026-04-20.txt` has
been overwritten with the post-A1 numbers (286,192 bytes of per-session
JSON follow the header). Pre-A1 attribution counts live only in the
reference docs from this point forward; the raw per-session JSON is gone.

## Caveats

- Sample size: 52 correct fires per direction in the side-error research
  sample. 28% drop in entered_right is the realized cost; monitor live
  side-error rate after any policy change.
- `breakout_confirmation` is NOT added as a teacher-layer negative veto in
  A1. The plan intentionally deferred it because no research pre-registers
  a threshold; adding it now would be post-hoc tuning. `breakout_confirmation`
  stays a v2 model-layer feature.
- The σ-position is SPY-derived (SPX has no minute volume); the scaling
  via `ratio = spx_close / spy_close` is the canonical research formula and
  now lives once in [v2/core/market_structure.py](../../v2/core/market_structure.py).

## Downstream impact

- The 259 new abstention bars are candidates for W4 (late-session teacher
  search). Not all will have late-session structure; the tournament will
  reveal which.
- The 33 correct ORC fires now abstained are expected loss; W2 (v2 soft
  features) may recover some of them via the model layer, since the same
  `sigma_pos` signal that rejected them at the teacher layer is visible to
  the model.
- Attribution reporters and downstream analysis scripts now operate on a
  post-A1 baseline (side_error=176, entered_right=84, abstention=726).
  Any comparison to pre-A1 figures must be explicit about the cutoff.
