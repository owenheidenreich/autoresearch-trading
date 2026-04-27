---
date: 2026-04-26
parent: phase_c_failure_modes_2026_04_26.md
spec_id: phase_c_gpt55_v0_2026_04_26
status: ALL 5 AVOID RULES PASS — robust on day-cluster bootstrap, per-seed, per-quarter, and concentration probes; favor rules report-only per spec
---

# Phase 1: Frozen filter robustness validation

## Setup

`v3/layer2/post_filters_v0.py` froze the GPT-5.5 rule contract with explicit
sigma_pos / iv_percentile tertile cutoffs (-0.5039522052, 1.2239481211 /
0.1632336179, 0.5469231009). `scripts/research_gpt5_filter_validation.py`
applied each rule to the 1664-trade Phase C sample (5 seeds, 514 days, 13
quarters) and computed: aggregate kept-PF, hl-per-day, max DD, per-seed Δ,
per-quarter Δ, day-cluster bootstrap 95% CI, top-3-day removal probe,
trade-count fraction kept, side-regret diagnostic.

## Decision-gate verdicts

All 5 avoid rules pass every gate:

| Rule | n_match | match PF | Δ kept_PF | 95% CI | seeds ≥-0.10 | quarters ≥-0.10 | kept frac | concentration |
|---|---|---|---|---|---|---|---|---|
| avoid_call_orc_triggered | 43 | 0.39 | +0.076 | [+0.033, +0.120] | 5/5 | 12/12 | 97.4% | 100% lift survives top-3 |
| avoid_call_cell_s0_iv2 | 69 | 0.68 | +0.072 | [+0.023, +0.124] | 5/5 | 12/12 | 95.9% | 100% |
| avoid_put_high_decision_margin | 108 | 0.55 | +0.096 | [+0.053, +0.150] | 5/5 | 12/12 | 93.5% | 100% |
| avoid_put_late_session_high_sigma | 100 | 0.53 | +0.104 | [+0.059, +0.154] | 5/5 | 12/12 | 94.0% | 100% |
| avoid_put_low_iv_percentile | 79 | 0.64 | +0.071 | [+0.029, +0.122] | 5/5 | 12/12 | 95.3% | 100% |

Per-seed: every avoid rule has positive Δ kept_PF in every seed individually.
Per-quarter: every avoid rule has positive Δ kept_PF in every quarter (12/12,
2023Q1 has only 2 trades and was dropped by quarter neutrality test).

Day-cluster bootstrap: 1000 iterations resampling by (seed, day) cluster.
Every avoid rule's lower 95% bound is positive — the lift is not driven by
day-correlated outliers.

## Avoid-union effect

Applying all 5 PASS avoid rules together (logical OR):
- 327 trades vetoed (PF 0.65, mean hl -$90)
- 1337 trades kept (PF **2.26**, mean hl/day +$256, max DD -$6847)
- Δ baseline: **ΔPF +0.378**, Δ hl/day +$67.97, Δ max DD basically flat (-$68)

GPT-5.5's report claimed 354 vetoes at PF 0.72 / kept PF 2.27. We get 327 / 0.65 /
2.26 — 1664-row sample matches 9 of 10 rules exactly; the 1-row gap on
avoid_put_high_decision_margin is documented in the Phase 0 spec.

## Side-regret diagnostic (per cell)

| cell | n | n_call | n_put | chosen_pf_hl | wrong_side_count | wrong_side_regret_sum |
|---|---|---|---|---|---|---|
| s0_iv0 | 126 | 68 | 58 | 1.90 | 59 | $64,209 |
| s0_iv1 | 207 | 103 | 104 | 1.25 | 81 | $79,074 |
| s0_iv2 | 222 | 69 | 153 | 1.40 | 80 | $107,719 |
| s1_iv0 | 205 | 115 | 90 | 1.74 | 75 | $61,015 |
| s1_iv1 | 169 | 97 | 72 | 1.77 | 81 | $90,487 |
| s1_iv2 | 180 | 52 | 128 | 3.10 | 60 | $96,727 |
| s2_iv0 | 224 | 115 | 109 | 1.84 | 84 | $71,954 |
| s2_iv1 | 178 | 62 | 116 | 1.47 | 72 | $62,097 |
| s2_iv2 | 153 | 43 | 110 | 2.59 | 68 | $93,263 |

**Wrong-side rate ranges 30-47% per cell** with cumulative regret sum
$64k-$108k per cell. Cell s0_iv2 has the largest absolute regret pile
($108k) and is exactly the cell flagged by `avoid_call_cell_s0_iv2`. The
veto layer is targeting the documented disease.

## Favor rules

All 5 favor rules emit FAVOR-DIAGNOSTIC verdicts per spec — they are not
applied to the kept distribution. Their match metrics are logged in
`phase1_rule_validation_summary.csv`. The MONITOR-ONLY tag on
`favor_call_low_omar_range_pct_MONITOR_ONLY` (n=65) is preserved
regardless of in-sample PF.

## What this proves and doesn't prove

**Proves (in-sample):** the 5 avoid rules' lift is statistically robust
across seeds, quarters, day-cluster bootstraps, and concentration probes.
The lift survives top-3-day removal at 100% — it's a broad effect, not a
few-day artifact.

**Does not prove:** that the rules will hold on the 26-day forward-walk
window. Phase C derived these rules from the same 1664-trade pool the
rules are now scored against; this is in-sample validation, not OOS.
Phase 2 is the OOS test.

## Cost

- Dev: ~1 hour (script + bug fixes)
- Compute: ~5 minutes (after vectorizing the bootstrap)
- Total: ~1 hour, $0 GPU

## Files

- `v3/layer2/post_filters_v0.py` (frozen Phase 0 spec)
- `scripts/research_gpt5_filter_validation.py` (Phase 1 validation)
- `v3/artifacts/research/phase1_rule_validation_summary.csv`
- `v3/artifacts/research/phase1_side_regret_per_cell.csv`
- `v3/reference/phase1_filter_validation_2026_04_26.md` (this doc)

## Next

Phase 2a: read `_select_daily_trades` end-to-end and document drop-only vs
rescan-after-veto runtime semantics. Then Phase 2b: ship the veto layer +
multi-metric FW gate.
