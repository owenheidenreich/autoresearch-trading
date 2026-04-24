# Candidate-L3 Oracle Retrain: Seed-43/44 Floor Regression Localized — 2026-04-23

## Context

Previous cycle (Codex) produced the candidate-trained-L3 oracle retrain:
composed PFs `2.854 / 1.635 / 1.691`, mean `2.060`, min `1.635`, mean DD
`12.8%`, trades `982`. High-mean challenger but failed the min-PF floor
(champion min `1.786`).

Codex's suggested next moves were:
1. localize seed-43/44 losing windows
2. test a mixture target between chosen-trained-L3 and candidate-trained-L3
   oracle surfaces

This note covers step 1.

## Per-window comparison

Diffed calibrated composed windows between the current promoted champion
(per-seed chosen-L3 oracle + chosen-trained Layer-3) and the candidate-L3
oracle retrain (candidate-L3 oracle + candidate-trained Layer-3).

### Seed 43 (PF 1.802 → 1.635, −0.167)

| W | champion n | champion $ | candL3 n | candL3 $ | Δn | Δ$ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 47 | +13323 | 47 | +10204 | +0 | −3119 |
| 3 | 32 | +5940 | 30 | +3184 | −2 | −2755 |
| 7 | 8 | +90 | 29 | +10178 | +21 | **+10087** |
| 8 | 12 | +763 | 56 | −2098 | +44 | −2861 |
| 9 | 11 | +4992 | 10 | +2725 | −1 | −2266 |
| 11 | 17 | +378 | 49 | +6106 | +32 | **+5728** |
| **12** | **26** | **+5881** | **14** | **−1154** | **−12** | **−7036** |
| total | 241 | +32100 | 348 | +35098 | **+107** | **+2998** |

Net PnL is up `+$2998`. PF is down. Explanation: candidate-L3 oracle
admits `+107` additional trades. They are roughly break-even in aggregate
but thicken the denominator (gross_loss) more than the numerator
(gross_profit). Top loser is **W12** (champion `+$5881` with 26 winners →
candL3 `−$1154` with 14 trades — candL3 exits *earlier* or picks
different contracts that stop out).

Seed-43 regression is **diluted-PF via extra-day admission**, not a
concentrated losing cluster.

### Seed 44 (PF 2.264 → 1.691, −0.573)

| W | champion n | champion $ | candL3 n | candL3 $ | Δn | Δ$ |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 33 | +5616 | 20 | +4084 | −13 | −1532 |
| 4 | 6 | +4173 | 17 | +1898 | +11 | −2274 |
| **5** | **16** | **+499** | **40** | **−6594** | **+24** | **−7093** |
| 6 | 18 | +2621 | 53 | +9760 | +35 | +7138 |
| 7 | 27 | +1723 | 31 | +12671 | +4 | **+10948** |
| **8** | **38** | **+13866** | **12** | **+2047** | **−26** | **−11818** |
| 11 | 23 | +5785 | 25 | +3533 | +2 | −2252 |
| total | 333 | +50400 | 340 | +48552 | +7 | **−1848** |

**Bimodal failure**:

- **W8**: champion caught 38 winners worth `+$13,866`; candL3 only
  enters 12 trades and captures `+$2,047`. **candL3 MISSES winners
  champion catches.**
- **W5**: champion only enters 16 trades in a dangerous regime; candL3
  admits 24 extras and they lose `−$7,093`. **candL3 ADMITS losers
  champion avoids.**

These are **opposite failure modes in the same seed**. A single
magnitude-style correction (e.g. tighter global decision margin) cannot
fix both — it would tighten W5 (good) while further starving W8
(worse). This matches the earlier global-margin-offset falsification
from cycle 2329a65.

## Implication for the mixture-target hypothesis

Codex's second suggestion was a mixture target between chosen-trained-L3
and candidate-trained-L3 oracles.

Chosen-L3 oracle teaches "pick contracts whose champion-exit PnL is
best" — heavily call-biased, conservative. Candidate-L3 oracle teaches
"pick contracts whose candidate-trained-exit PnL is best" — side-
diverse, admits more entries.

A 50/50 blend should:

- **Soften W5 overtrading on seed 44** (candidate signal's "admit
  more" pressure is diluted by chosen signal's more selective pattern)
- **Partially restore W8 coverage on seed 44** (chosen signal's "these
  contracts are winners" pattern is preserved)
- **Reduce W12 churn on seed 43** (chosen signal's conservative
  selection is preserved)
- **Preserve some side diversity** (candidate signal still contributes)

The diagnostic does NOT guarantee the mixture works. It says:

- failure is not uniform — different windows fail for different reasons
- the two oracles encode different priors about when to trade
- a mixture is the only narrow rescue candidate that isn't "sweep a
  global scalar" (already falsified)

## Next cycle

Build a 50/50 mixture oracle `simulated_l3_oracle_seed{S}_mix50_fp.npz`
for each seed. Retrain entries against that mixture; compose with the
current champion's chosen-trained Layer-3 robust `0.90`; compare against
both the champion and the candidate-L3 oracle retrain.

Stop conditions for the next cycle (per post-cycle-12 addendum
principles):
- if mixture min PF regresses below champion's `1.786` → stop, not a
  promotion path
- if mixture mean PF falls below champion's `1.950` with no DD win →
  stop, worse on every axis
- if mixture mean DD rises above `12%` without a PF win → stop, worse
  on the DD axis where candidate-L3 was supposed to help

## Artifacts

- `v3/artifacts/layer3_unified_cpu_simL3_candidateL3_seed{42,43,44}/`
- `v3/artifacts/layer3_unified_cpu_simL3_{objfix,perseed}_seed{42,43,44}/`
- diagnostic script: run inline in this cycle (no new module yet)
