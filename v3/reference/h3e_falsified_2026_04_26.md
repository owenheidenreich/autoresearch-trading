---
date: 2026-04-26
parent: regime_diagnostic_2026_04_26.md
status: FALSIFIED — H3e (target reshape + 4 chosen-contract Greeks) regressed all four gates on seed 42; reverted via `git revert`
---

# H3e Falsified: Profitability-Conditioned Target + Greeks Features

## Setup

H3e bundled two changes (per
[~/.claude/plans/codex-carried-out-an-zesty-axolotl.md](../../../.claude/plans/codex-carried-out-an-zesty-axolotl.md)):

1. **Profitability-conditioned target.** `target = int(current_pnl >= suffix_max[i] AND current_pnl > 0)` —
   eliminates 67% loss-bar dilution that plagued the original target.
2. **Per-bar chosen-contract Greeks in trade_state.** Added `iv_current_t`,
   `delta_abs_t`, `theta_to_premium_t`, `gamma_dollar_norm_t` from sidecar
   columns [7, 8, 19, 21] — 14-feature variant.

Hypothesis: spread-minimization (cross-cell PF range narrows from 1.30 baseline
/ 1.76 H3a → ≤ 1.00 H3e) by giving the oracle Greeks-aware exit signal AND
removing target dilution.

## Result on seed 42 (376 OOS trades, hybrid_live scoring)

```
   variant   agg_pf   spread    floor
  baseline    2.195    0.995    1.405
       h3a    2.308    1.596    1.606
       h3e    1.591    1.446    0.838
```

| Gate | Target | H3e | Verdict |
|---|---|---|---|
| PRIMARY: spread ≤ 1.00 | narrows | 1.446 | **FAIL** (no narrowing) |
| SECONDARY: floor ≥ 1.40 | lifts | 0.838 | **FAIL** (floor crashed) |
| TERTIARY: agg ≥ 1.831 | preserved | 1.591 | **FAIL** (-0.603 PF vs baseline) |
| DISCIPLINE: per-seed delta ≥ -0.10 | | -0.603 | **FAIL** |

Bootstrap CI on delta-PF: **[-1.10, -0.20]**, p[d ≤ 0] = 0.997.
Statistically significant regression.

## Per-cell pattern

```
trend  vol     n     base    h3a    h3e   d_h3e_vs_base
bear   high   32    2.40   1.97   1.93     -0.47
chop   low    47    1.41   1.61   1.87     +0.46  (improvement on baseline-weak cell)
chop   mid    44    2.16   2.65   1.51     -0.64
chop   high   43    2.35   3.20   2.29     -0.06
bull   low   143    2.38   2.64   1.46     -0.92  (large regression)
bull   mid    45    1.83   1.61   0.84     -0.99  (largest regression)
```

H3e *did* lift the cell baseline was weakest in (chop × low, +0.46). But it
crashed all the strong cells. Floor barely moved; ceiling collapsed; spread
roughly the same.

## Mechanism (most likely cause)

The plan flagged this exact risk: *"the current target's 'exit on the way
down for losers' signal currently saves ~$800 on a -$1000 trade by exiting
at -$200. With H3e, the model never trains on that signal — losers default
to time-stop, taking the full premium loss. This could net-negative
aggregate PF if losers dominate."*

The data confirms: **the loser-defense signal mattered more than the small-
winner discrimination signal**. By stripping target=1 from all loss bars,
the model lost its ability to exit losers early. Net: aggregate PF dropped
by 27.5%, statistically significant.

The Greeks features alone (theta_to_premium, gamma_dollar_norm, etc.) didn't
compensate. Possibly they need more training signal density to be useful;
possibly they're partially redundant with state_features' atm_* aggregates.

## Discipline action taken

Per the plan's decision tree, reverted via `git revert`:
- C1 (target reshape) → reverted in `1778af8`
- C2b (Greeks plumbing into trade_state) → reverted in `ff09899`

Kept as harmless infrastructure for any future Greeks experiment:
- C2a: `_ContractPath` extension with Greek arrays (path objects now carry
  per-bar ives/deltas/theta_to_premiums/gamma_dollars; no behavior change
  since no caller uses them after the revert)
- C2c: extended look-ahead audit (29-bar synthetic test for path Greeks
  causality)

Net code change vs research/h3a-features pre-H3e: zero behavioral change;
+1 dataclass field set on `_ContractPath` and +1 audit function.

## What this rules out and what's next

**Ruled out:**
- "Profitability-conditioned target alone solves small-winner failure" —
  falsified. The loser-defense signal removal is more harmful than the
  signal sharpening is helpful.
- "Adding chosen-contract Greeks narrows the regime spread on the
  HistGB+target architecture" — falsified at this scale (376 trades seed
  42). Could be tested with a different target, but the bundled experiment
  failed.

**Remaining options on the table (none auto-executed):**

1. **H3e-clean (per the plan)**: drop H3a's 3 trajectory features, re-test
   target+Greeks. Isolates whether H3a's regime-volatile features were
   interfering. ~50 min CPU. But honest assessment: the bull-cell
   collapses (-0.92, -0.99) suggest the failure is in the target reshape,
   not H3a interference. H3e-clean is unlikely to recover.

2. **H3e-greeks-only**: keep H3a + Greeks, drop the target reshape. Tests
   whether Greeks features help on top of H3a's regime-conditional
   baseline. If puts back H3a's per-side asymmetry but adds spread
   stability, it's a potential keeper.

3. **Revert to baseline (drop H3a too)**: baseline has the smallest spread
   (1.30 vs H3a 1.76 vs H3e 1.45) and decent aggregate (1.881). It might
   already be the regime-stable champion, and chasing structural changes
   has been net-negative.

4. **Sequence model (H3c)**: deeper structural change. The plan's deferred
   path. Larger lift potential but more risk.

5. **Inference-time regime gating**: pragmatic — at deployment, observe
   regime from past data and pick (baseline | H3a | future variants) per
   regime. No retraining; no new features. Gets the strengths of each
   without the cross-cell contamination.

## Evidence files

```
v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42_h3e.npz  (kept for reproducibility)
v3/artifacts/research/h3e_seed42_eval.json                          (gate results)
scripts/research_eval_h3e_oracle.py                                  (eval script)
scripts/look_ahead_audit_h3.py                                       (audit, includes H3e path test)
```

## Branch state

```
research/h3e-features:
  1778af8 Revert "H3e C1: profitability-conditioned target"
  ff09899 Revert "H3e C2b: plumb per-bar chosen-contract Greeks into trade_state"
  a33ea3a H3e C2c: extend look-ahead audit (KEPT — useful infra)
  fac9bea H3e C2b: plumb Greeks (REVERTED above)
  a437aba H3e C2a: extend _ContractPath (KEPT — passive scaffolding)
  245523b H3e C1: target reshape (REVERTED above)
  369ad4e small-winner failure diagnostic
  + H3a chain from research/h3a-features
```

## Cost summary

- ~46 min CPU: seed-42 build (single-process)
- ~5 min: eval + revert + writeup
- $0 GPU
- Total: ~50 min CPU. Well under the 16h plan budget; falsification was cheap.
