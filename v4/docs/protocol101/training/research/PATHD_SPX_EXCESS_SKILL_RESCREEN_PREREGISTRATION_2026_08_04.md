# Path-D SPX Excess-Skill Re-Screen (Option 0) — PRE-REGISTRATION (2026-08-04)

**Status: FROZEN — pre-registered before execution. DIAGNOSTIC-ONLY.**

Owner-authorized 2026-08-04 in
[`PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md`](../contracts/PATHD_NEXT_CLASS_OWNER_DECISION_2026_08_04.md)
§"Owner decision", under three binding constraints, all of which are honoured below.

Read-only local analysis on owned data: no training, no estimator, no broker, no paid download, no
holdout access. Nothing may be edited after the run starts; amendments are dated appendices.

---

## 1. What this is, and the ceiling on what it can claim

**This is a second look at the same 213 sessions, with the hypothesis chosen after seeing run 1.** It is
therefore **diagnostic-only, and no outcome of it is confirmatory.** A surviving excess feeds the Option A
decision and nothing more. A dying excess closes directional strategies on this feature set for every
instrument. Neither result licenses training, a wave, capture consumption, or ledger regeneration, and the
protected holdout is SPENT and stays closed.

**The question run 1 could not answer.** Run 1 measured a mean Spearman IC of `+0.522` and it was
~79% mechanical: the features are functions of the price level `P(t)`, the target `P(t+h) − P(t)` contains
`−P(t)`, and that shared term forces a negative rank correlation for any bounded path. This run asks the
only remaining question: **is there anything left after that artifact is removed?**

## 2. The surrogate construction — load-bearing, specified exactly

Three independent calibrations of the same artifact disagree, and the disagreement decides the answer:

| Null | omar IC @15/30/60 m |
|---|---|
| Constant-σ random walk (Fable, 2026-08-04) | −0.37 / −0.48 / −0.58 — **stronger** than measured |
| Circular block bootstrap, 30-min blocks (run 1) | −0.26 / −0.33 / −0.41 — **weaker** than measured |
| **Measured** | −0.32 / −0.42 / −0.52 |

The spread is not noise; it is **surrogate mis-specification**. A constant-σ walk gets the volatility
profile wrong. A block bootstrap resamples blocks, so it moves the loud minutes around and changes total
realized variance. Since the artifact's magnitude depends on the path's volatility profile, either choice
can flip the sign of the "excess".

**Primary null — drift-preserving Rademacher wild bootstrap.** Per session, on the one-minute log returns
`r_t` of that session's own available closes:

1. `mu = mean(r)`, `e_t = r_t − mu`
2. `s_t` iid Rademacher (`±1`, equal probability)
3. `r*_t = mu + e_t · s_t`
4. `P*_0 = P_0` (the session's true opening level), `P*_t = P*_{t−1} · exp(r*_t)`

**Preserved exactly:** the session's drift `mu`; `|e_t|` at every timestamp `t`, hence the **intraday
volatility profile** (the U-shape), the **position** of every volatility cluster, and total realized
variance; the bar count; the opening level; and the real **volume** series (unchanged, so VWAP is
volume-weighted identically). **Destroyed:** the sign structure — hence all serial predictability,
momentum and mean reversion.

This removes the degree of freedom the three calibrations disagreed on. The surrogate differs from the real
path in exactly one respect: it cannot be predicted.

**Secondary null — the run-1 circular block bootstrap** (30-minute blocks), retained unchanged so this run
is comparable to run 1. **Reported, and required to agree** (§6).

**Declared limitation.** Sign randomization makes returns exactly symmetric conditional on `|e_t|`, so it
does not preserve return skew. Skew is expected to be second-order for a rank statistic, but it is not
tested here.

## 3. Availability, features and target — identical machinery to run 1

**Boundary/bar selection is taken from the real session and reused for every surrogate.** For each boundary
the selected bar index (or "stale/absent") is computed once from the real availability clock — bar
`available_at = event_time + 60 s`, 90-second staleness cap — and the surrogate uses the **same** index
map with substituted prices. Real and surrogate therefore share an identical NaN structure by construction.

**Feature parity is asserted, not assumed.** Surrogates have no vendor provenance, so the frozen kernel
cannot run on them; a path-based implementation is required. That implementation **must reproduce the
frozen kernel's nine features bit-for-bit on the real path, at every boundary of every session**, or the
run aborts. This is the same discipline as run 1's close-parity assertion, which caught a 1000× clock
error.

**Target:** unchanged — `y_h(t)` = close available at `t + h·60 s` minus close available at `t`, horizons
15 / 30 / 60 minutes, boundaries every minute 10:00–15:00 ET, 213 development sessions.

## 4. Family — 60 members

The nine side-free features of run 1, **plus `minute_of_session`** (owner/Fable request: test the
time-of-session hypothesis directly, since `session_range_bps` is a monotone proxy for it and was the
least artifact-explained member at 55%). `minute_of_session` is a diagnostic addition and is **not** a
feature the entry model ever had.

**10 features × 3 horizons × 2 signs = 60 members.** The sign mirroring is retained from run 1 for the
reason established there: `session_blocked_max_t` is one-sided on signed `t`, and every member that
cleared run 1 did so at the negative sign.

## 5. Statistic — the correction applies to the EXCESS

Per session `s` and member `m`, with `S = 200` surrogates per session:

```
excess(s, m) = IC_measured(s, m) − mean_over_surrogates IC_surrogate(s, m)
```

Then `paired_summary` per member and `session_blocked_max_t` over the 60-member family,
`permutations = 20_000`, `seed = 5077`. The unit of inference remains the session; the family-wise
correction is applied to the excess, not to the raw IC.

Sessions with fewer than 30 valid boundaries for a member are dropped for that member. Fold map from
`entry_expanding_folds(development_sessions(...))`; per-fold mean excess reported per member.

**Surrogate estimation noise is acknowledged and is conservative.** The surrogate mean carries error
`≈ σ_surrogate / √200`, independent across sessions, which inflates the across-session standard error and
therefore makes the test harder to pass, not easier. Its magnitude is reported.

## 6. Decision rule — frozen

A member is **`EXCESS_SURVIVES`** iff **all** of:

1. `maxT_p_one_sided < 0.05` on the excess against the **primary** (wild-bootstrap) null;
2. mean excess carries the **same sign in ≥ 4 of 5 folds**;
3. **the secondary (block-bootstrap) null agrees** — same sign of mean excess, and nominal
   `p_one_sided < 0.05`. Requiring two independently-constructed nulls to agree is the direct guard
   against the surrogate mis-specification identified in §2; a member that survives one null and not the
   other is reported as `NULL_DEPENDENT`, which is a finding, not a pass;
4. the negative control (§7) behaves.

Verdict is **`EXCESS_SKILL_CANDIDATE`** if any member qualifies, otherwise **`NO_EXCESS_SKILL`**. If the
negative control breaches, the verdict is **`INVALID`**.

**Do not** add features, horizons, sessions, or nulls to reach a pass, and do not relax the two-null
agreement requirement.

## 7. Negative control — the one run 1 was missing

**Surrogate-as-measured.** One held-out surrogate draw is substituted for the real path and pushed through
the entire pipeline, with its excess computed against the remaining surrogates. It has **no predictability
by construction**, so its excess must be null and **no member may clear the §6 rule**. If any does, the
artifact-removal machinery is itself producing signal and the run is `INVALID`.

This is the control that run 1 lacked. Run 1's session-shuffle was correct against look-ahead and
multiplicity but blind to a shared-term artifact, because shuffling destroys the very pairing the artifact
lives in.

The run-1 identity check is retained: `IC(−x, y)` must equal `−IC(x, y)` exactly.

## 8. Claude's prior, recorded in advance

**`NO_EXCESS_SKILL` is more likely than not**, and more likely than it was before Fable's calibration. A
constant-σ walk already produces a *stronger* pull than the real data at every horizon, which is direct
evidence that the true excess may be zero or slightly momentum-side. Against that, run 1's block-bootstrap
excess was `−0.112` at `z ≈ −6.7` for `omar`@60 m. **The two nulls disagree about the sign of the answer,
which is precisely why the primary null was rebuilt to preserve the volatility profile exactly.** A
surviving excess must be argued for, and would still be an information coefficient — not an edge, and not
costed on any instrument.

## 9. Evidence and reproduction

- **Script:** `v4/research/pathd_spx_excess_skill_rescreen.py` (no CLI arguments; `main()`).
- **Output:** `v4/audit/autoresearch/pathd_spx_excess_skill_rescreen_2026_08_04/` — `receipt.json`
  (self-sealed via `stable_hash`), `results.csv` (60 rows), `results.md`.
- The receipt carries `protected_holdout_opened: false`, `paper_order_submitted: false`,
  `model_trained: false`, `diagnostic_only: true`, the sha256 of the script and every imported feature
  module, and the path and sha256 of **this** pre-registration.
- Output directory is created with `exist_ok=False`; the run refuses up front if it already exists.

*Signed: Claude Opus 5 — 2026-08-04 — FROZEN before execution.*
