# Phase 4b — the lift is ordering, not resolution: outcome A

**Date:** 2026-08-19 · **Job:** 46 lifecycle-training · **Declaration:**
[`PHASE_4B_DECLARATION_V1.json`](../../work/lifecycle-training/PHASE_4B_DECLARATION_V1.json)
(`6f404d28…`) · **Receipt:** `phase_4b_receipt.json` · **Specified by:**
[`PHASE_4A_ADJUDICATION_2026_08_19.md`](PHASE_4A_ADJUDICATION_2026_08_19.md) §5

## 1. The verdict

**A — ordering information exists and is material.** The fit proceeds as designed, with the
volatility channel documented as conditioning and the D1–D8 suite unchanged.

| | |
|---|---|
| Precondition | Phase 4a headline reproduced **exactly** at +8.649074789891253pp |
| Lift | **+8.65pp** = **resolution +0.28pp** + **ordering +8.37pp** |
| P(resolved) | 0.9422 population → 0.9506 selected |
| P(gain first \| resolved) | **0.3259 population → 0.4140 selected** |
| Ordering null (120 draws) | mean +0.07pp, **97.5th percentile +3.79pp**, max +4.55pp |
| Draws at or above observed | **0 of 120** |
| D2 matched control | **+7.67pp**, agrees with D1 in direction, 585 strata |

**The lift is 97% ordering.** Selection barely moves resolution (+0.84pp of headroom used); it moves
the gain-first share by **+8.8 points**.

## 2. The result refutes two stated positions, including this session's own

**Fable's quantitative account (adjudication §3) does not hold.** It proposed that the hit rate is
dominated by P(the path resolves), which magnitude raises mechanically, and estimated a no-skill
gain-first share tending toward 30/80 = 37.5%. Measured: the population gain-first share is
**32.59%**, and resolution is not a lever at all.

**The reason is a hard ceiling, not a statistical argument.** **94.2% of candidate paths already
resolve** inside the 60-minute horizon — 0DTE options move far enough in percentage terms to touch
+50% or −30% almost always. Verified independently on a separate 102-session sample of 330,021
labelled actions: **P(resolved) = 0.9476, P(gain-first | resolved) = 0.3258.** So even a selector
that chose *only* resolving paths — perfect resolution selection, unattainable — would raise
precision from 30.87% to at most 32.58%, a ceiling of **+1.71pp**. The observed lift is +8.65pp.
Resolution mechanics cannot produce it, and no amount of magnitude skill changes that.

**This session's prior (referral §8.10) was also wrong.** It leaned toward the lift being a
rediscovery of already-priced magnitude, on the grounds that `realised_vol_15m` was the one
sign-stable coefficient. D3 measures that channel directly: **`realised_vol_15m` alone is worth
−1.07pp**, and the full feature set with it removed still delivers **+6.80pp**. The stable
coefficient is conditioning, not the source of the lift — which is what the adjudication's outcome-A
language anticipated and this session did not.

## 3. D2 — the matched control

Matching within session on entry-ask decile × `realised_vol_15m` quintile removes exactly the two
channels a magnitude story would use: leverage and volatility. Across 585 strata containing both a
selected and an unselected action, selected precision is **39.06%** against matched **31.39%** —
**+7.67pp surviving**, in the same direction as D1. Most of the ordering component survives the
removal of price and volatility as explanations.

## 4. D3 — the corrected ablation, with the tie-break repaired

Randomized tie-breaking, 20 seeded draws averaged. Diagnostic only; no verdict weight.

| Group | Alone (4b, corrected) | Alone (4a, contaminated) | Without |
|---|---|---|---|
| Contract price | +6.49pp | +6.49pp | +6.95pp |
| **Chain internals** | **+1.97pp** | +1.86pp | +6.03pp |
| Ordering fields | +1.70pp | +1.86pp | +10.04pp |
| Per-contract chain | +0.16pp | +0.16pp | +6.18pp |
| Tape | **−0.16pp** | **−9.41pp** | −0.76pp |
| Clock | **−0.28pp** | **+3.40pp** | +8.19pp |
| `realised_vol_15m` alone | **−1.07pp** | — | — |
| Full set minus `realised_vol_15m` | **+6.80pp** | — | — |

The two rows that moved are exactly the two the correction predicted. Tape and clock are
minute-common: they cannot rank contracts, so their Phase-4a "alone" figures were measuring an
arbitrary stored-order tie-break (ascending `contract_id`, the deepest-OTM puts). With the tie-break
randomized, **tape goes from −9.41pp to −0.16pp and clock from +3.40pp to −0.28pp** — both to
approximately nothing, which is what a group that cannot rank contracts should score. Chain
internals and the contract-varying groups are unchanged, as expected.

## 5. The open question this creates

**The ordering component has a strong chronological gradient, and nothing here explains it.**

| Fold | Lift | Resolution | Ordering | P(gain first \| resolved) |
|---|---|---|---|---|
| 0 | +1.70pp | +0.96pp | **+0.74pp** | 0.338 → 0.346 |
| 1 | +9.82pp | +0.23pp | +9.60pp | 0.314 → 0.417 |
| 2 | +9.84pp | +0.44pp | +9.40pp | 0.328 → 0.426 |
| 3 | +13.22pp | −0.49pp | **+13.71pp** | 0.323 → 0.470 |

By calendar year: ordering **+2.44pp (2022)**, +9.10pp (2023), **+18.74pp (2024)**.

The adjudication asked which factor vanishes in fold 0. The answer is **ordering** — it is nearly
absent in the earliest window and grows monotonically. The population gain-first share is flat
across folds (0.314–0.338), so this is not a change in what the label pays; it is a change in how
well the features rank. Whether that is a real regime effect, a consequence of the expanding-window
folds giving later folds more training data, or something else, **this run does not separate.** It
bears directly on a fit whose training prefix ends 2024-01-29.

## 6. Provenance and constraints

Preconditions both passed: the declaration's implementation hashes matched the files on disk, and
the Phase-4a headline reproduced bit-for-bit. Population: 1,043,873 scored actions, 324 sessions,
648 selections, training prefix only. **No dollar column was read.** No score block was touched. The
alpha ledger is charged for the count with its accuracy columns null, as in Phase 4a.

`feature_information_preflight.py` was not edited — the published Phase-4a declaration pins it — so
the seeded tie-break lives in a new module and the default selection path is the Phase-4a function
itself, asserted by test.

## 7. What this does not establish

The label is a binary path-order outcome on quoted mids. **Nothing here is economics.** A +8.37pp
ordering component says the features rank contracts by which barrier arrives first; it says nothing
about dollars, and the executable round trip is charged only at the simulator. Ledger rows 332 and
338 remain undisturbed: they concern expected gross P&L of buying premium, which this phase did not
measure and could not have.
