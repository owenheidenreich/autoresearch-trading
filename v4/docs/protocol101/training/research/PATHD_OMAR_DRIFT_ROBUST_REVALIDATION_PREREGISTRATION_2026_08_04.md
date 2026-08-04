# Path-D omar Drift-Robust Re-Validation — PRE-REGISTRATION (2026-08-04)

**Status: FROZEN on commit. DIAGNOSTIC-ONLY — this is the THIRD analysis of the same 213
development sessions, with the hypothesis chosen after seeing the previous two results. No outcome
here is confirmatory. The protected holdout stays closed (`holdout_open_count` = 0); the only
confirmation path for anything in this programme is fresh live-paper observation.**

Authorized by the owner's directive to review and execute
[`FABLE_PATH_TO_A_TRAINED_MODEL_BRIEF_2026_08_04.md`](FABLE_PATH_TO_A_TRAINED_MODEL_BRIEF_2026_08_04.md)
(2026-08-04). Zero cost: owned data only, no training, no broker contact, no purchase, no runtime
mutation.

---

## 1. Why this study exists, and why it comes before everything else

The Option 0 re-screen left **one survivor of sixty**: `omar_clipped_neg3_pos3`, excess IC
+0.052/+0.063/+0.078 at 15/30/60 m against a drift-preserving wild bootstrap, and both nulls
agreeing (`PATHD_SPX_EXCESS_SKILL_RESCREEN_RESULTS_2026_08_04.md`).

The same day, the causal-sizing correction (`pathd_omar_causal_sizing.py`, commit `78df5eb3`)
**demonstrated** that the primary null is mis-specified for this exact feature: the wild bootstrap
imposes the session's mean drift at every minute, so a surrogate path climbs smoothly and "high
omar" implies "still drifting up". At fixed omar bins the pooled *surrogate* forward-move profile
runs monotonically **−5.43 → +4.51** SPX points while the *real* profile is flat.

**The defect's direction is anti-conservative for the survivor.** The spurious drift channel adds
variance to the surrogate's omar that carries no negative covariance with the target, *diluting*
the bounded-path reversion artifact in the surrogate baseline. A diluted (less negative) surrogate
baseline makes `real − surrogate` *more* reversion-signed. The omar excess (+0.05…+0.08) is a ~15%
residual on a raw IC of −0.52 that is ~85% artifact — a modest mis-calibration of the artifact
baseline could account for all of it.

The secondary (block) null does not share the drift defect — iid block draws make future drift
independent of the past — and it accepts omar even more strongly (+0.062…+0.107, p 0.0000). But the
block null was itself proven mis-specified in the same run for two other members
(`session_range_bps`, `spx_vwap_gap_*`), because moving 30-minute blocks destroys the placement of
the intraday volatility profile. Agreement between two nulls with different known defects is not
validation.

**Therefore, before any causal-normalization study, any ES work, and any training question: does
the omar excess survive nulls that repair the drift defect while keeping the volatility profile?**
If it does not, the last survivor of the sixty closes, and with it the directional programme on
this feature set (owner decision key → Option C). Running a causal-capture study first would risk
building on an artifact.

## 2. Hypothesis (stated before the run)

**H1:** the within-session reversion excess of `omar_clipped_neg3_pos3` (negative-signed excess IC
against forward 15/30/60-minute moves) survives drift-repaired martingale nulls.

**H0:** the excess is an artifact of the wild bootstrap's constant-per-minute drift and will not
exceed drift-repaired nulls.

My prior, honestly stated: **uncertain, leaning fragile.** The wild defect inflates the excess; the
block null's independent acceptance argues the other way; the two do not cancel.

## 3. Corpus, clock, features — all frozen, identical to the Option 0 re-screen

- Same 213-usable-of-215 development sessions (`development_sessions`, dev corpus only; holdout
  untouched). Same 10:00–15:00 ET minute boundary grid, t−60s staleness cap, same
  `_selection_map` / `_features_from_path` path (kernel parity asserted per session, tolerance
  1e-6, abort on failure).
- Feature under test: `omar_clipped_neg3_pos3` only — the running-range construction
  `(close − open) / (running max − running min)`, causal by construction.
- Targets: forward 15/30/60-minute close-to-close moves on the same selection maps.
- Raw-IC cross-run parity check: the real-path omar ICs must reproduce the Option 0 receipt values
  to 1e-9 per member-horizon (mean across sessions), else abort.

## 4. The two repaired nulls (primary contribution of this study)

Both are true martingales — the property whose absence broke the previous nulls — and they split
the volatility-profile question between them: N1 preserves the placement of every `|r_t|` exactly,
N2 preserves the marginal `|r|` distribution while destroying placement.

**N1 — zero-drift wild (martingale wild).** `r*_t = s_t · r_t` with iid Rademacher signs on the raw
log returns — no `mu` term. Preserves `|r_t|` at every timestamp exactly; destroys drift entirely.
Directionally **conservative** for the reversion claim: real sessions carry drift that dilutes the
real path's bounded-path artifact, while the surrogate's artifact is undiluted, biasing
`real − surrogate` toward the momentum side.

**N2 — permuted-time wild.** `r*_t = s_t · r_{π(t)}` with a uniform within-session permutation
`π` and iid Rademacher signs. A true martingale that preserves the marginal absolute-return
distribution while destroying the *placement* of the intraday volatility profile — the
complementary weakness to N1, which preserves placement exactly. This mirrors the wild-vs-block
complementarity of the Option 0 run, but with both nulls now martingales.

*Design note, recorded before freezing.* The owner-flagged prescription "resample the session
drift; do not impose it at every minute" was attempted first as N2 and **failed its own
known-answer gate twice** in the declared smoke check:

1. Block means drawn iid from the session's raw block-mean set → expected future increments equal
   the session's mean drift → not a martingale. Pooled surrogate spread **10.9 points** (defective
   reference wild: 11.5).
2. Block means drawn from the **centered** set → every increment mean-zero, but a 30-minute
   *persistent* drift burst is itself short-horizon momentum: at time t the rest of the current
   block shares its drawn drift with the recent past. Pooled surrogate spread **5.5 points**.

The second catch is a finding in its own right: **a "drift-burst-preserving martingale" is
self-contradictory at horizons that overlap the burst length.** The real data's flat fixed-bin
profile says the market carries no such persistence at 15–60 m; a valid H0 null cannot carry it
either. N2 was therefore replaced with the permuted-time wild above before freezing. Both catches
are evidence the known-answer gate detects exactly the defect class it was built for.

**Reference (no gate):** the original drift-preserving wild bootstrap, re-run unchanged, so the
delta attributable to the repair is visible in one table.

## 5. Known-answer gates — each null must first prove it fixed the defect

The defect was demonstrated on the pooled fixed-bin statistic at 60 m (fixed omar bin edges
−1.0…+1.0 step 0.2, as in `pathd_omar_causal_sizing.py`). The defective wild produced a monotone
surrogate profile spanning **9.94 points** (−5.43 → +4.51). For each repaired null, over the same
pooled accumulation across all sessions and surrogates:

- **KA-1:** |surrogate pooled mean(top bin) − surrogate pooled mean(bottom bin)| ≤ **2.0 SPX
  points**.
- **KA-2:** |Spearman rho of surrogate pooled bin means vs bin index| ≤ **0.6** (defective wild
  ≈ +1.0).

A null failing its known-answer gate is **void** — its IC comparison cannot be interpreted, and it
must be reported as `NULL_REPAIR_FAILED` for that null. If both nulls are void, the study is
inconclusive and stops there.

## 6. Test statistics, family, thresholds — frozen

- Per session, per null: excess IC = real IC − mean of **200** surrogate ICs (same estimator as
  Option 0). Negative control per null: surrogate #1 vs mean of surrogates #2–200; both nulls'
  control members are tested as **one combined 12-member maxT family** so the overall
  false-`INVALID` rate is held at ALPHA.
- Primary family per null: **6 members** (omar × {15, 30, 60} m × {+, −} sign), tested separately
  per null — survival requires passing both, which is stricter than any combined correction. The
  family is small because the hypothesis is now specific; this is disclosed as the third look at
  these sessions.
- Session-blocked maxT with **20,000** permutations, seed **7203**; surrogate seed derivation
  `sha256("90211|<null>|<session>")`. ALPHA **0.05** one-sided on maxT p. Expanding-fold
  sign-stability ≥ **4/5** (same folds as Option 0).
- Survival requires the **negative (reversion) sign** — the sign observed in Option 0 — at the
  member level.

## 7. Verdict key — frozen, with stop rules

| Outcome | Verdict | Consequence |
|---|---|---|
| omar (reversion sign) survives maxT + folds at 60 m, and 15/30 m do not contradict in sign, under **both** N1 and N2; controls clean; KA gates pass | `OMAR_EXCESS_ROBUST_TO_DRIFT_NULLS` | Authorizes *designing* the causal-capture study (Phase B, its own pre-registration). Still no training, no purchase, no holdout. |
| omar fails maxT or folds under **either** valid repaired null | `OMAR_NULL_FRAGILE_CLOSED` | The last survivor of the sixty closes. Do-not-retest ledger entry. Owner decision key resolves to **Option C**: stand down the directional programme on this feature set; keep the execution plane warm. |
| A repaired null fails its KA gate | `NULL_REPAIR_FAILED` (per null) | That null is void. If both void: study inconclusive; report and stop; no verdict on omar is claimed. |
| Negative control breaches (maxT p < ALPHA on any control member) | `INVALID` | Machinery defect; no claim about omar either way. |

No threshold, member, gate, or seed may be changed after this file is committed. The runner
refuses to execute twice (output directory existence).

## 8. What this study can and cannot say

It **can** close the omar survivor as null-fragile, or establish that the association is robust to
the specific defect that broke the sizing. It **cannot** confirm tradability, effect size, ES
transfer, or anything about money — the effect size remains `NOT_IDENTIFIED` regardless of outcome,
and any survivor remains hypothesis-generating only, on triple-visited data.

## 9. Execution

```bash
PYTHONPATH=. python -m v4.research.pathd_omar_drift_robust_revalidation
```

Output: `v4/audit/autoresearch/pathd_omar_drift_robust_revalidation_2026_08_04/`
(`receipt.json` self-sealed, `results.csv`, `results.md`), with
`diagnostic_only: true`, `protected_holdout_opened: false`, `model_trained: false`.

A smoke mode (`PATHD_DRIFT_ROBUST_SMOKE=1`; 8 sessions × 20 surrogates × 2,000 permutations,
written to the session scratchpad, never to `v4/audit/`) is permitted **only** as an engineering
check that the code runs end to end; its numbers are non-evidentiary and must not be reported as
results.

*Prepared by Claude Fable 5 — 2026-08-04, under the owner's "review and execute" directive on the
Fable brief.*
